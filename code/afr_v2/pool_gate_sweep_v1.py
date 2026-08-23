"""Pool x Gate sweep: does the learned router / rule cascade survive a change of action pool?

Design notes (deliberate deviations from summary_router_monotonic_triad_v1):
  * The learned router here is HGB-S1 only, and instead of reproducing the 24-candidate
    inner-OOF hyperparameter search we sweep the cost-discount beta over a fixed grid and
    report the WHOLE curve. Beta is never selected using held-out labels, so no leakage;
    the trade-off is that absolute numbers are not bit-comparable to the frozen run.
    The CURRENT pool is included so the offset can be measured.
  * Tier order inside a pool is by measured cost. The quality endpoint is re-derived on the
    training split only (never the most expensive by assumption).
  * Cascades run in two escalation orders: by cost and by train-split quality.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from afr_v2 import summary_router_compact16_direct_v1 as base
from afr_v2 import summary_router_compact16_targetmix_lodo_v1 as lodo

OUTPUT_RELATIVE = Path("results/pool_gate_sweep_v1")

POOLS: dict[str, tuple[str, ...]] = {
    "CURRENT": ("factkb", "granite_guardian_3_1_2b", "wecheck"),
    "TOP1": ("lettuce_v2", "minicheck_dbta", "qwen30_fast"),
    "CHEAP": ("factkb", "minicheck_dbta", "qwen30_fast"),
    "ALIGN": ("alignscore", "lettuce_v2", "qwen30_fast"),
    "GRANITE": ("alignscore", "granite_guardian_3_1_2b", "minicheck_ft5"),
}

ALL_VERIFIERS = (
    "alignscore", "factcc", "factcg", "factkb", "granite_guardian_3_1_2b",
    "granite_guardian_3_2_3b_a800m", "granite_guardian_3_2_8b_factuality",
    "granite_guardian_4_1_3b_factuality_lora", "hhem", "lettuce_v2", "minicheck_dbta",
    "minicheck_ft5", "qwen30_fast", "qwen30_judge", "wecheck",
)

DATASETS = tuple(lodo.DATASETS)
SEEDS = tuple(base.SEEDS)
BETA_GRID = (0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2)
CONF_WIDTHS = (0.02, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
BOOTSTRAP_DRAWS = 2000

# anchors that do NOT depend on our router implementation -- they must reproduce exactly
FIXED_BASELINE_ANCHORS = {
    ("wecheck", "lodo_macro_auroc"): 0.7325,
    ("granite_guardian_3_1_2b", "lodo_macro_auroc"): 0.6936,
    ("factkb", "lodo_macro_auroc"): 0.6371,
    ("factcg", "lodo_macro_auroc"): 0.7349,
}
ANCHOR_TOLERANCE = 0.002


# ---------------------------------------------------------------- data plumbing

def load_matrix(project_root: Path) -> pd.DataFrame:
    """Frozen scoring matrix joined with the frozen Compact-16 features."""
    matrix_path = project_root / base.MATRIX_RELATIVE
    feature_path = project_root / lodo.FEATURE_RELATIVE
    digest = base.sha256_file(matrix_path)
    if digest != base.EXPECTED_MATRIX_SHA256:
        raise ValueError(f"frozen matrix sha256 drift: {digest}")
    feature_digest = base.sha256_file(feature_path)
    if feature_digest != lodo.EXPECTED_FEATURE_SHA256:
        raise ValueError(f"frozen feature sha256 drift: {feature_digest}")
    frame = pd.read_parquet(matrix_path)
    features = pd.read_parquet(feature_path)
    if len(frame) != 6_850 or len(features) != 6_850:
        raise ValueError(f"unexpected row count {len(frame)}/{len(features)}")
    keep = ["episode_key", *base.FEATURE_COLUMNS]
    frame = frame.drop(columns=[c for c in base.FEATURE_COLUMNS if c in frame.columns])
    frame = frame.merge(features[keep], on="episode_key", how="inner", validate="one_to_one")
    if len(frame) != 6_850:
        raise ValueError("feature join dropped rows")
    missing = [c for c in base.FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing Compact-16 columns after join: {missing}")
    return frame.sort_values("episode_key").reset_index(drop=True)


def build_splits(matrix: pd.DataFrame) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    splits: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    folds = matrix["fold"].to_numpy(int)
    for fold in sorted(set(folds.tolist())):
        test = folds == fold
        splits.append(("pooled", str(fold), ~test, test))
    keys = matrix["dataset_key"].astype(str).to_numpy()
    for dataset in DATASETS:
        test = keys == dataset
        splits.append(("lodo", dataset, ~test, test))
    return splits


def pool_costs(matrix: pd.DataFrame, members: Sequence[str]) -> dict[str, float]:
    costs = {}
    for name in members:
        available = matrix[f"available__{name}"].astype(bool).to_numpy()
        costs[name] = float(matrix.loc[available, f"latency_ms__{name}"].mean())
    return costs


def order_by_cost(matrix: pd.DataFrame, members: Sequence[str]) -> tuple[str, ...]:
    costs = pool_costs(matrix, members)
    return tuple(sorted(members, key=lambda name: costs[name]))


def train_quality_order(train: pd.DataFrame, actions: Sequence[str]) -> tuple[tuple[str, ...], dict[str, float]]:
    """Ascending AUROC measured on the TRAIN split only."""
    scores: dict[str, float] = {}
    labels = train["label_supported"].to_numpy(int)
    for name in actions:
        available = train[f"available__{name}"].astype(bool).to_numpy()
        y = labels[available]
        s = train.loc[available, f"score__{name}"].to_numpy(float)
        scores[name] = float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else 0.5
    return tuple(sorted(actions, key=lambda name: scores[name])), scores


# ---------------------------------------------------------------- prediction frames

def selected_prediction(
    frame: pd.DataFrame,
    actions: Sequence[str],
    selected: np.ndarray,
    *,
    probability: np.ndarray | None = None,
    decision: np.ndarray | None = None,
    cumulative_latency: np.ndarray | None = None,
    router_latency: np.ndarray | float = 0.0,
    calls: np.ndarray | float = 1.0,
) -> pd.DataFrame:
    rows = np.arange(len(frame))
    scores = np.column_stack([frame[f"score__{a}"].fillna(0.5).to_numpy(float) for a in actions])
    decisions = np.column_stack([frame[f"decision__{a}"].fillna(0).to_numpy(int) for a in actions])
    latencies = np.column_stack([frame[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in actions])
    out = frame[["episode_key", "dataset_key", "group_id", "label_supported"]].copy()
    out["selected_action"] = np.asarray(actions, dtype=object)[selected]
    out["router_decision"] = decisions[rows, selected] if decision is None else np.asarray(decision, int)
    out["probability_supported"] = (
        scores[rows, selected] if probability is None else np.asarray(probability, float)
    )
    out["correct"] = (out["router_decision"].to_numpy(int) == out["label_supported"].to_numpy(int)).astype(np.int8)
    out["verifier_latency_ms"] = (
        latencies[rows, selected] if cumulative_latency is None else np.asarray(cumulative_latency, float)
    )
    out["feature_latency_ms"] = frame["feature_latency_ms"].to_numpy(float)
    out["router_latency_ms"] = router_latency
    out["end_to_end_latency_ms"] = (
        out["verifier_latency_ms"] + out["feature_latency_ms"] + out["router_latency_ms"]
    )
    out["forced_upgrade"] = False
    out["mean_calls_per_summary"] = calls
    return out


def prediction_metrics(part: pd.DataFrame, actions: Sequence[str], keys: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(keys)
    result.update(
        {
            "rows": int(len(part)),
            "source_groups": int(part["group_id"].nunique()),
            **base._quality_metrics(
                part["label_supported"].to_numpy(int),
                part["router_decision"].to_numpy(int),
                part["probability_supported"].to_numpy(float),
            ),
            "mean_end_to_end_latency_ms": float(part["end_to_end_latency_ms"].mean()),
            "mean_calls_per_summary": float(part["mean_calls_per_summary"].mean()),
        }
    )
    counts = part["selected_action"].value_counts().to_dict()
    for index, action in enumerate(actions):
        result[f"rate__tier{index}"] = float(counts.get(action, 0) / len(part))
        result[f"tier{index}_action"] = action
    return result


def macro_auroc(part: pd.DataFrame) -> float:
    values = []
    for _, chunk in part.groupby("dataset_key", sort=True):
        y = chunk["label_supported"].to_numpy(int)
        if len(set(y.tolist())) < 2:
            continue
        values.append(roc_auc_score(y, chunk["probability_supported"].to_numpy(float)))
    return float(np.mean(values)) if values else float("nan")


def scope_auroc(part: pd.DataFrame, scope: str) -> float:
    if scope == "lodo":
        return macro_auroc(part)
    y = part["label_supported"].to_numpy(int)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, part["probability_supported"].to_numpy(float)))


# ---------------------------------------------------------------- policies

def label_calibrated(train: pd.DataFrame, test: pd.DataFrame, actions: Sequence[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for action in actions:
        available = train[f"available__{action}"].astype(bool).to_numpy()
        calibrator = base._Platt.fit(
            train.loc[available, f"score__{action}"].to_numpy(float),
            train.loc[available, "label_supported"].to_numpy(int),
        )
        out[action] = calibrator.predict(test[f"score__{action}"].fillna(0.5).to_numpy(float))
    return out


def correctness_heads(
    train: pd.DataFrame, test: pd.DataFrame, actions: Sequence[str], seed: int
) -> np.ndarray:
    """P(action is correct) per action, HGB on the Compact-16 features. TRAIN only."""
    features = list(base.FEATURE_COLUMNS)
    x_train = train[features].to_numpy(float)
    x_test = test[features].to_numpy(float)
    columns = []
    for action in actions:
        available = train[f"available__{action}"].astype(bool).to_numpy()
        y = train.loc[available, f"correct__{action}"].fillna(0).to_numpy(int)
        if len(set(y.tolist())) < 2:
            columns.append(np.full(len(test), float(y.mean()) if len(y) else 0.5))
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
        model.fit(x_train[available], y)
        columns.append(model.predict_proba(x_test)[:, 1])
    return np.column_stack(columns)


def cost_discounts(beta: float, actions: Sequence[str], costs: Mapping[str, float]) -> np.ndarray:
    """exp(-beta * cost_i / cost_max), matching the frozen run's common_utility form."""
    values = np.asarray([float(costs[a]) for a in actions], dtype=float)
    denominator = float(values.max())
    if denominator <= 0:
        return np.ones(len(actions), dtype=float)
    return np.exp(-float(beta) * values / denominator)


def router_prediction(
    test: pd.DataFrame,
    actions: Sequence[str],
    correct_probability: np.ndarray,
    calibrated: Mapping[str, np.ndarray],
    costs: Mapping[str, float],
    beta: float,
) -> pd.DataFrame:
    availability = np.column_stack([test[f"available__{a}"].astype(bool).to_numpy() for a in actions])
    discount = cost_discounts(beta, actions, costs)
    utility = correct_probability * discount[None, :]
    selected = base.masked_argmax(utility, availability)
    probability = np.column_stack([calibrated[a] for a in actions])[np.arange(len(test)), selected]
    return selected_prediction(
        test, actions, selected,
        probability=probability, decision=(probability >= 0.5).astype(int), calls=1.0,
    )


def matched_random_single(
    test: pd.DataFrame,
    actions: Sequence[str],
    reference: pd.DataFrame,
    calibrated: Mapping[str, np.ndarray],
    seed_parts: Sequence[Any],
) -> pd.DataFrame:
    """Same marginal action shares as `reference`, one call, decisions randomised."""
    availability = np.column_stack([test[f"available__{a}"].astype(bool).to_numpy() for a in actions])
    shares = (
        reference["selected_action"].value_counts(normalize=True)
        .reindex(list(actions), fill_value=0.0).to_numpy(float)
    )
    weights = np.tile(shares, (len(test), 1)) * availability
    empty = weights.sum(axis=1) == 0
    weights[empty] = availability[empty].astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    generator = np.random.default_rng(base.stable_seed(int(seed_parts[0]), *seed_parts[1:]))
    draws = generator.random(len(test))
    selected = np.minimum((draws[:, None] > np.cumsum(weights, axis=1)).sum(axis=1), len(actions) - 1)
    probability = np.column_stack([calibrated[a] for a in actions])[np.arange(len(test)), selected]
    return selected_prediction(
        test, actions, selected,
        probability=probability, decision=(probability >= 0.5).astype(int), calls=1.0,
    )


def cascade_conf(
    test: pd.DataFrame, chain: Sequence[str], calibrated: Mapping[str, np.ndarray], width: float
) -> pd.DataFrame:
    availability = np.column_stack([test[f"available__{a}"].astype(bool).to_numpy() for a in chain])
    latency = np.column_stack([test[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in chain])
    probs = np.column_stack([calibrated[a] for a in chain])
    n, depth = len(test), len(chain)
    selected = np.zeros(n, dtype=int)
    total = np.zeros(n, dtype=float)
    calls = np.zeros(n, dtype=float)
    for row in range(n):
        level = depth - 1
        for k in range(depth):
            if not availability[row, k]:
                continue
            total[row] += latency[row, k]
            calls[row] += 1
            level = k
            if k == depth - 1 or abs(float(probs[row, k]) - 0.5) > float(width):
                break
        selected[row] = level
    probability = probs[np.arange(n), selected]
    return selected_prediction(
        test, chain, selected, probability=probability,
        decision=(probability >= 0.5).astype(int), cumulative_latency=total, calls=calls,
    )


def matched_random_cascade(
    test: pd.DataFrame,
    chain: Sequence[str],
    reference: pd.DataFrame,
    calibrated: Mapping[str, np.ndarray],
    seed_parts: Sequence[Any],
) -> pd.DataFrame:
    """Same final-tier shares AND the same chain cost model as the cascade."""
    availability = np.column_stack([test[f"available__{a}"].astype(bool).to_numpy() for a in chain])
    latency = np.column_stack([test[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in chain])
    shares = (
        reference["selected_action"].value_counts(normalize=True)
        .reindex(list(chain), fill_value=0.0).to_numpy(float)
    )
    weights = np.tile(shares, (len(test), 1)) * availability
    empty = weights.sum(axis=1) == 0
    weights[empty] = availability[empty].astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    generator = np.random.default_rng(base.stable_seed(int(seed_parts[0]), *seed_parts[1:]))
    draws = generator.random(len(test))
    selected = np.minimum((draws[:, None] > np.cumsum(weights, axis=1)).sum(axis=1), len(chain) - 1)
    total = np.zeros(len(test), dtype=float)
    calls = np.zeros(len(test), dtype=float)
    for row, level in enumerate(selected):
        for k in range(level + 1):
            if availability[row, k]:
                total[row] += latency[row, k]
                calls[row] += 1
    probability = np.column_stack([calibrated[a] for a in chain])[np.arange(len(test)), selected]
    return selected_prediction(
        test, chain, selected, probability=probability,
        decision=(probability >= 0.5).astype(int), cumulative_latency=total, calls=calls,
    )


def fixed_verifier_macro_auroc(matrix: pd.DataFrame, verifier: str) -> float:
    """AUROC of a single verifier's raw score, computed on its available rows only,
    averaged over datasets. Implementation independent -- used as the consistency anchor."""
    values = []
    for dataset in DATASETS:
        part = matrix[matrix["dataset_key"].astype(str).eq(dataset)]
        available = part[f"available__{verifier}"].astype(bool).to_numpy()
        y = part.loc[available, "label_supported"].to_numpy(int)
        s = part.loc[available, f"score__{verifier}"].to_numpy(float)
        if len(set(y.tolist())) > 1:
            values.append(roc_auc_score(y, s))
    return float(np.mean(values)) if values else float("nan")


def fixed_prediction(test: pd.DataFrame, actions: Sequence[str], action: str) -> pd.DataFrame:
    availability = np.column_stack([test[f"available__{a}"].astype(bool).to_numpy() for a in actions])
    preference = np.zeros_like(availability, dtype=float)
    preference[:, list(actions).index(action)] = 1.0
    return selected_prediction(test, actions, base.masked_argmax(preference, availability))


def cheapest_correct_oracle(test: pd.DataFrame, actions: Sequence[str], costs: Mapping[str, float]) -> pd.DataFrame:
    availability = np.column_stack([test[f"available__{a}"].astype(bool).to_numpy() for a in actions])
    correct = np.column_stack(
        [pd.to_numeric(test[f"correct__{a}"], errors="coerce").fillna(0).to_numpy(float) for a in actions]
    )
    cost = np.asarray([costs[a] for a in actions])
    utility = correct + (cost.max() - cost)[None, :] * 1e-9
    return selected_prediction(test, actions, base.masked_argmax(utility, availability))


# ---------------------------------------------------------------- bootstrap

def paired_bootstrap(
    treatment: pd.DataFrame, control: pd.DataFrame, scope: str, draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, float]:
    """Resample source groups with replacement, stratified by dataset."""
    merged = treatment.merge(
        control[["episode_key", "probability_supported", "router_decision"]],
        on="episode_key", suffixes=("_t", "_c"),
    )
    generator = np.random.default_rng(base.stable_seed(2026, scope, "paired-bootstrap"))
    groups = merged.groupby("group_id").indices
    keys = np.asarray(list(groups.keys()), dtype=object)
    dataset_of = {g: merged.iloc[idx[0]]["dataset_key"] for g, idx in groups.items()}
    strata: dict[str, list] = {}
    for key in keys:
        strata.setdefault(str(dataset_of[key]), []).append(key)

    deltas = []
    for _ in range(draws):
        picked: list[int] = []
        for members in strata.values():
            sample = generator.choice(np.asarray(members, dtype=object), size=len(members), replace=True)
            for key in sample:
                picked.extend(groups[key].tolist())
        chunk = merged.iloc[picked]
        t = chunk.rename(columns={"probability_supported_t": "probability_supported"})
        c = chunk.rename(columns={"probability_supported_c": "probability_supported"})
        a, b = scope_auroc(t, scope), scope_auroc(c, scope)
        if a == a and b == b:
            deltas.append(a - b)
    if not deltas:
        return {"delta_mean": float("nan"), "delta_ci_low": float("nan"), "delta_ci_high": float("nan")}
    array = np.asarray(deltas, dtype=float)
    return {
        "delta_mean": float(array.mean()),
        "delta_ci_low": float(np.percentile(array, 2.5)),
        "delta_ci_high": float(np.percentile(array, 97.5)),
        "draws": len(array),
    }
