from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from afr_v2 import summary_router_compact16_direct_v1 as base


SCHEMA_VERSION = "summary_router_compact16_targetmix_v1"
OUTPUT_RELATIVE = Path("results/summary_router_compact16_targetmix_v1")
BASE_RESULT_RELATIVE = Path("results/summary_router_compact16_direct_v1")
BASE_OOF_RELATIVE = BASE_RESULT_RELATIVE / "OOF_PREDICTIONS.parquet"
EXPECTED_BASE_OOF_SHA256 = "323ac7340a15e58575c05b42e91e264b901ea9a96dd2099946d019b8d748560d"

ACTIONS = base.ACTIONS
ACTION_COSTS_MS = base.ACTION_COSTS_MS
FEATURE_COLUMNS = base.FEATURE_COLUMNS
BETAS = base.BETAS
SEEDS = base.SEEDS
INNER_FOLDS = base.INNER_FOLDS
TARGET_SHARE = np.asarray((0.60, 0.30, 0.10), dtype=float)
PRICE_BOUNDS = (-40.0, 40.0)
PRICE_ITERATIONS = 80

REUSED_METHODS = base.METHODS
NEW_METHODS = ("LR-S2", "LR-S3", "HGB-S2", "HGB-S3")
METHODS = (
    "LR-S1",
    "LR-S2",
    "LR-S3",
    "HGB-S1",
    "HGB-S2",
    "HGB-S3",
    "MLP-S1",
    "MLP-S2",
    "MLP-S3",
)


class _ConstantProbability:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), self.probability, dtype=float)
        return np.column_stack((1.0 - positive, positive))


class _ConstantRegression:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


@dataclass
class _PairwiseBundle:
    correctness_heads: Any
    pairs: list[tuple[int, int]]
    preference_heads: list[Any]

    def predict_raw(self, features: np.ndarray, availability: np.ndarray) -> np.ndarray:
        available = np.asarray(availability, dtype=bool)
        correctness = self.correctness_heads.predict_raw(features, available)
        wins = np.zeros_like(correctness, dtype=float)
        counts = np.zeros_like(correctness, dtype=float)
        for (left, right), head in zip(self.pairs, self.preference_heads):
            probability = head.predict_proba(features)[:, 1]
            pair_available = available[:, left] & available[:, right]
            wins[pair_available, left] += probability[pair_available]
            wins[pair_available, right] += 1.0 - probability[pair_available]
            counts[pair_available, left] += 1.0
            counts[pair_available, right] += 1.0
        preference = np.divide(wins, counts, out=np.full_like(wins, 0.5), where=counts > 0)
        score = np.sqrt(np.clip(correctness, 1e-8, 1.0) * np.clip(preference, 1e-8, 1.0))
        return np.where(available, score, 0.0)


@dataclass
class _RewardBundle:
    heads: list[Any]

    def predict_raw(self, features: np.ndarray, availability: np.ndarray) -> np.ndarray:
        values = np.column_stack([head.predict(features) for head in self.heads])
        values = np.clip(values, 0.0, 1.0)
        return np.where(np.asarray(availability, dtype=bool), values, 0.0)


def _matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    correct = np.column_stack(
        [pd.to_numeric(frame[f"correct__{action}"], errors="coerce").to_numpy(float) for action in ACTIONS]
    )
    availability = np.column_stack(
        [frame[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS]
    )
    decisions = np.column_stack(
        [pd.to_numeric(frame[f"decision__{action}"], errors="coerce").fillna(0).to_numpy(int) for action in ACTIONS]
    )
    labels = frame["label_supported"].to_numpy(int)
    latency = np.column_stack(
        [pd.to_numeric(frame[f"latency_ms__{action}"], errors="coerce").fillna(0).to_numpy(float) for action in ACTIONS]
    )
    return correct, availability, decisions, labels, latency


def _candidate_grid(method: str, quick: bool) -> list[dict[str, Any]]:
    family = method.split("-", 1)[0]
    if family == "LR":
        values = (1.0,) if quick else (0.1, 1.0, 10.0)
        return [{"strength": float(value)} for value in values]
    if family == "HGB":
        if quick:
            return [{"max_leaf_nodes": 7, "learning_rate": 0.1, "min_samples_leaf": 20}]
        return [
            {"max_leaf_nodes": leaf, "learning_rate": rate, "min_samples_leaf": minimum}
            for leaf, rate, minimum in product((7, 15), (0.03, 0.1), (20, 50))
        ]
    raise ValueError(method)


def _classifier(family: str, parameters: Mapping[str, Any], seed: int) -> Any:
    if family == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(parameters["strength"]),
                max_iter=2_000,
                solver="lbfgs",
                random_state=int(seed),
            ),
        )
    if family == "HGB":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            learning_rate=float(parameters["learning_rate"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=1.0,
            max_iter=100,
            early_stopping=False,
            random_state=int(seed),
        )
    raise ValueError(family)


def _regressor(family: str, parameters: Mapping[str, Any], seed: int) -> Any:
    if family == "LR":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0 / float(parameters["strength"])))
    if family == "HGB":
        return HistGradientBoostingRegressor(
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            learning_rate=float(parameters["learning_rate"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=1.0,
            max_iter=100,
            early_stopping=False,
            random_state=int(seed),
        )
    raise ValueError(family)


def _fit_pairwise(
    family: str,
    features: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    parameters: Mapping[str, Any],
    beta: float,
    seed: int,
) -> _PairwiseBundle:
    correctness_heads = base._fit_sklearn_heads(
        f"{family}-S1", features, correct, availability, _base_parameters(family, parameters), seed
    )
    rewards = np.nan_to_num(correct, nan=0.0) * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    any_correct = ((np.nan_to_num(correct, nan=0.0) > 0) & availability).any(axis=1)
    pairs = list(combinations(range(len(ACTIONS)), 2))
    heads: list[Any] = []
    for pair_index, (left, right) in enumerate(pairs):
        gap = rewards[:, left] - rewards[:, right]
        mask = any_correct & availability[:, left] & availability[:, right] & (np.abs(gap) > 1e-12)
        target = (gap[mask] > 0).astype(int)
        weight = np.abs(gap[mask])
        if len(target) == 0:
            heads.append(_ConstantProbability(0.5))
            continue
        if len(np.unique(target)) == 1:
            heads.append(_ConstantProbability(float(target[0])))
            continue
        model = _classifier(family, parameters, base.stable_seed(seed, "pair", pair_index))
        model.fit(features[mask], target, **({"sample_weight": weight} if family == "HGB" else {"logisticregression__sample_weight": weight}))
        heads.append(model)
    return _PairwiseBundle(correctness_heads=correctness_heads, pairs=pairs, preference_heads=heads)


def _base_parameters(family: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if family == "LR":
        return {"C": float(parameters["strength"])}
    return dict(parameters)


def _fit_reward(
    family: str,
    features: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    parameters: Mapping[str, Any],
    beta: float,
    seed: int,
) -> _RewardBundle:
    rewards = np.nan_to_num(correct, nan=0.0) * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    heads: list[Any] = []
    for action_index in range(len(ACTIONS)):
        mask = availability[:, action_index]
        target = rewards[mask, action_index]
        if len(np.unique(target)) == 1:
            heads.append(_ConstantRegression(float(target[0])))
            continue
        model = _regressor(family, parameters, base.stable_seed(seed, "reward", action_index))
        model.fit(features[mask], target)
        heads.append(model)
    return _RewardBundle(heads=heads)


def _fit_model(
    method: str,
    features: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    parameters: Mapping[str, Any],
    beta: float,
    seed: int,
) -> Any:
    family, supervision = method.split("-")
    if supervision == "S2":
        return _fit_pairwise(family, features, correct, availability, parameters, beta, seed)
    if supervision == "S3":
        return _fit_reward(family, features, correct, availability, parameters, beta, seed)
    raise ValueError(method)


def _selection_metrics(
    score: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    decisions: np.ndarray,
    labels: np.ndarray,
    latency: np.ndarray,
    beta: float,
    supervision: str,
) -> tuple[float, float, float]:
    utility_score = score if supervision == "S3" else score * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    selected = base.masked_argmax(utility_score, availability)
    rows = np.arange(len(selected))
    realized = np.nan_to_num(correct, nan=0.0) * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    return (
        float(realized[rows, selected].mean()),
        float(balanced_accuracy_score(labels, decisions[rows, selected])),
        float(latency[rows, selected].mean()),
    )


def _nested_fit(
    train_frame: pd.DataFrame,
    train_features: np.ndarray,
    test_features: np.ndarray,
    test_availability: np.ndarray,
    method: str,
    beta: float,
    seed: int,
    inner_folds: int,
    quick: bool,
) -> Any:
    correct, availability, decisions, labels, latency = _matrices(train_frame)
    folds = base._inner_folds(train_frame, inner_folds, base.stable_seed(seed, method, beta, "inner"))
    candidates = _candidate_grid(method, quick)
    supervision = method.split("-")[-1]
    candidate_rows: list[dict[str, Any]] = []
    candidate_oof: list[np.ndarray] = []
    for candidate_index, parameters in enumerate(candidates):
        raw_oof = np.zeros_like(correct, dtype=float)
        for fold in range(inner_folds):
            fit = folds != fold
            validation = folds == fold
            model = _fit_model(
                method,
                train_features[fit],
                correct[fit],
                availability[fit],
                parameters,
                beta,
                base.stable_seed(seed, method, beta, candidate_index, fold),
            )
            raw_oof[validation] = model.predict_raw(train_features[validation], availability[validation])
        utility, bacc, mean_latency = _selection_metrics(
            raw_oof, correct, availability, decisions, labels, latency, beta, supervision
        )
        candidate_rows.append(
            {
                "candidate": candidate_index,
                "parameters": json.dumps(parameters, sort_keys=True),
                "primary": utility,
                "inner_utility": utility,
                "inner_bacc": bacc,
                "inner_latency_ms": mean_latency,
            }
        )
        candidate_oof.append(raw_oof)
    best = max(
        candidate_rows,
        key=lambda row: (row["primary"], row["inner_bacc"], -row["inner_latency_ms"], row["parameters"]),
    )
    best_index = int(best["candidate"])
    parameters = candidates[best_index]
    raw_oof = candidate_oof[best_index]
    calibrators = None
    if supervision == "S2":
        calibrators = []
        for action_index in range(len(ACTIONS)):
            mask = availability[:, action_index]
            calibrators.append(base._Platt.fit(raw_oof[mask, action_index], correct[mask, action_index]))
    model = _fit_model(
        method,
        train_features,
        correct,
        availability,
        parameters,
        beta,
        base.stable_seed(seed, method, beta, "outer_final"),
    )
    raw_test = model.predict_raw(test_features, test_availability)
    if calibrators is None:
        calibrated_test = raw_test
    else:
        calibrated_test = np.column_stack(
            [calibrator.predict(raw_test[:, index]) for index, calibrator in enumerate(calibrators)]
        )
        calibrated_test = np.where(test_availability, calibrated_test, 0.0)

    def callback(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raw = model.predict_raw(values, mask)
        if calibrators is not None:
            raw = np.column_stack([calibrators[index].predict(raw[:, index]) for index in range(len(ACTIONS))])
            raw = raw * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
        return base.masked_argmax(raw, mask)

    router_latency = base._measure_batch1_ms(callback, test_features[0], test_availability[0])
    selection = {
        **best,
        "method": method,
        "beta": float(beta),
        "seed": int(seed),
        "train_rows": int(len(train_frame)),
        "inner_folds": int(inner_folds),
        "candidate_count": int(len(candidates)),
        "all_candidates": candidate_rows,
        "supervision_reduction": "correctness_plus_pairwise_borda" if supervision == "S2" else "direct_conditional_reward_regression",
    }
    return base._NestedResult(
        model=model,
        calibrated_test=calibrated_test,
        raw_test=raw_test,
        calibrators=calibrators,
        selection=selection,
        router_latency_ms=router_latency,
    )


def _beta_slug(beta: float) -> str:
    return f"{float(beta):g}".replace(".", "p")


def _part_paths(output_dir: Path, method: str, beta: float, seed: int, fold: int) -> tuple[Path, Path]:
    stem = output_dir / "new_base_parts" / method.casefold().replace("-", "_") / f"seed_{seed}"
    name = f"beta_{_beta_slug(beta)}__fold_{fold}"
    return stem / f"{name}.parquet", stem / f"{name}.json"


def _prediction_part(
    test: pd.DataFrame,
    feature_frame: pd.DataFrame,
    probabilities: np.ndarray,
    result: Any,
    method: str,
    beta: float,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    availability = np.column_stack([test[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS])
    supervision = method.split("-")[-1]
    score = probabilities if supervision == "S3" else probabilities * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    selected = base.masked_argmax(score, availability)
    rows = np.arange(len(test))
    decisions = np.column_stack(
        [pd.to_numeric(test[f"decision__{action}"], errors="coerce").fillna(0).to_numpy(int) for action in ACTIONS]
    )
    verifier_scores = np.column_stack(
        [pd.to_numeric(test[f"score__{action}"], errors="coerce").fillna(0.5).to_numpy(float) for action in ACTIONS]
    )
    verifier_latency = np.column_stack(
        [pd.to_numeric(test[f"latency_ms__{action}"], errors="coerce").fillna(0).to_numpy(float) for action in ACTIONS]
    )
    output = test[["episode_key", "dataset_key", "group_id", "label_supported"]].copy()
    output["outer_fold"] = int(fold)
    output["method"] = method
    output["beta"] = float(beta)
    output["seed"] = int(seed)
    output["selected_action"] = np.asarray(ACTIONS, dtype=object)[selected]
    output["router_decision"] = decisions[rows, selected]
    output["probability_supported"] = verifier_scores[rows, selected]
    output["correct"] = (output["router_decision"].to_numpy(int) == output["label_supported"].to_numpy(int)).astype(np.int8)
    output["verifier_latency_ms"] = verifier_latency[rows, selected]
    output["feature_latency_ms"] = feature_frame["feature_latency_ms"].to_numpy(float)
    output["router_latency_ms"] = float(result.router_latency_ms)
    output["end_to_end_latency_ms"] = output["verifier_latency_ms"] + output["feature_latency_ms"] + output["router_latency_ms"]
    output["forced_upgrade"] = ~availability[:, 0]
    correct = np.column_stack(
        [pd.to_numeric(test[f"correct__{action}"], errors="coerce").to_numpy(float) for action in ACTIONS]
    )
    for index, action in enumerate(ACTIONS):
        output[f"available__{action}"] = availability[:, index]
        output[f"correct__{action}"] = correct[:, index]
        output[f"router_probability__{action}"] = probabilities[:, index]
        output[f"raw_router_probability__{action}"] = result.raw_test[:, index]
    return output.reset_index(drop=True)


def run_new_base_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    *,
    methods: Sequence[str] = NEW_METHODS,
    betas: Sequence[float] = BETAS,
    seeds: Sequence[int] = SEEDS,
    inner_folds: int = INNER_FOLDS,
    quick: bool = False,
) -> dict[str, Any]:
    values = features.loc[:, FEATURE_COLUMNS].to_numpy(np.float32)
    folds = frame["fold"].to_numpy(int)
    fold_values = sorted(np.unique(folds).tolist())
    base.validate_group_folds(frame, folds, n_splits=len(fold_values))
    for method, beta, seed, fold in product(methods, betas, seeds, fold_values):
        part_path, selection_path = _part_paths(output_dir, method, float(beta), int(seed), int(fold))
        if part_path.is_file() and selection_path.is_file():
            continue
        train_mask = folds != fold
        test_mask = folds == fold
        train = frame.loc[train_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        test_availability = np.column_stack(
            [test[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS]
        )
        print(f"[new-oof] start method={method} beta={float(beta):g} seed={int(seed)} fold={int(fold)}", flush=True)
        result = _nested_fit(
            train,
            values[train_mask],
            values[test_mask],
            test_availability,
            method,
            float(beta),
            int(seed),
            int(inner_folds),
            quick,
        )
        part = _prediction_part(
            test,
            features.loc[test_mask].reset_index(drop=True),
            result.calibrated_test,
            result,
            method,
            float(beta),
            int(seed),
            int(fold),
        )
        selection = {
            **result.selection,
            "trained_beta": float(beta),
            "decision_beta": float(beta),
            "outer_fold": int(fold),
            "test_rows": int(len(test)),
            "router_latency_ms": float(result.router_latency_ms),
        }
        base.write_json(selection_path, selection)
        base.write_parquet(part_path, part)
        print(f"[new-oof] complete method={method} beta={float(beta):g} seed={int(seed)} fold={int(fold)}", flush=True)
    parts: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for method, beta, seed, fold in product(methods, betas, seeds, fold_values):
        part_path, selection_path = _part_paths(output_dir, method, float(beta), int(seed), int(fold))
        if not part_path.is_file() or not selection_path.is_file():
            raise AssertionError(f"missing new OOF part: {part_path}")
        parts.append(pd.read_parquet(part_path))
        selections.append(json.loads(selection_path.read_text()))
    predictions = pd.concat(parts, ignore_index=True)
    expected = len(frame) * len(methods) * len(betas) * len(seeds)
    keys = ["episode_key", "method", "beta", "seed"]
    if len(predictions) != expected or predictions.duplicated(keys).any():
        raise AssertionError("new base OOF coverage failure")
    base.write_parquet(output_dir / "NEW_BASE_OOF_PREDICTIONS.parquet", predictions)
    selection_frame = pd.DataFrame([{k: v for k, v in row.items() if k != "all_candidates"} for row in selections])
    base.write_csv(output_dir / "NEW_NESTED_SELECTION.csv", selection_frame)
    return {"predictions": predictions, "selection": selection_frame}


def _decision_scores(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability = np.column_stack(
        [pd.to_numeric(part[f"router_probability__{action}"], errors="coerce").fillna(0).to_numpy(float) for action in ACTIONS]
    )
    availability = np.column_stack(
        [part[f"available__{action}"].fillna(False).to_numpy(bool) for action in ACTIONS]
    )
    supervision = str(part["method"].iloc[0]).split("-")[-1]
    beta = float(part["beta"].iloc[0])
    if supervision != "S3":
        probability = probability * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    return np.log(np.clip(probability, 1e-12, 1.0)), availability


def _select_with_prices(log_score: np.ndarray, availability: np.ndarray, low_price: float, high_price: float) -> np.ndarray:
    prices = np.asarray((float(low_price), 0.0, float(high_price)), dtype=float)
    return base.masked_argmax(log_score - prices[None, :], availability)


def _low_price_for_high(log_score: np.ndarray, availability: np.ndarray, high_price: float) -> float:
    lower, upper = PRICE_BOUNDS
    best = None
    for _ in range(PRICE_ITERATIONS):
        value = (lower + upper) / 2.0
        selected = _select_with_prices(log_score, availability, value, high_price)
        share = float((selected == 0).mean())
        candidate = (abs(share - TARGET_SHARE[0]), value)
        if best is None or candidate < best:
            best = candidate
        if share > TARGET_SHARE[0]:
            lower = value
        else:
            upper = value
    return float(best[1])


def calibrate_prices(log_score: np.ndarray, availability: np.ndarray) -> dict[str, Any]:
    lower, upper = PRICE_BOUNDS
    best = None
    for _ in range(PRICE_ITERATIONS):
        high_price = (lower + upper) / 2.0
        low_price = _low_price_for_high(log_score, availability, high_price)
        selected = _select_with_prices(log_score, availability, low_price, high_price)
        shares = np.bincount(selected, minlength=len(ACTIONS)).astype(float) / len(selected)
        objective = float(np.abs(shares - TARGET_SHARE).sum())
        candidate = (objective, abs(shares[2] - TARGET_SHARE[2]), low_price, high_price, shares)
        if best is None or candidate[:4] < best[:4]:
            best = candidate
        if shares[2] > TARGET_SHARE[2]:
            lower = high_price
        else:
            upper = high_price
    return {
        "objective_l1": float(best[0]),
        "low_price": float(best[2]),
        "mid_price": 0.0,
        "high_price": float(best[3]),
        "shares": np.asarray(best[4], dtype=float),
    }


def _reselect(part: pd.DataFrame, matrix: pd.DataFrame, price: Mapping[str, float]) -> pd.DataFrame:
    output = part.copy().reset_index(drop=True)
    log_score, availability = _decision_scores(output)
    selected = _select_with_prices(log_score, availability, float(price["low_price"]), float(price["high_price"]))
    rows = np.arange(len(output))
    aligned = matrix.reindex(output["episode_key"].astype(str)).copy()
    if aligned.index.isna().any():
        raise AssertionError("matrix alignment failed")
    decisions = np.column_stack(
        [pd.to_numeric(aligned[f"decision__{action}"], errors="coerce").fillna(0).to_numpy(int) for action in ACTIONS]
    )
    scores = np.column_stack(
        [pd.to_numeric(aligned[f"score__{action}"], errors="coerce").fillna(0.5).to_numpy(float) for action in ACTIONS]
    )
    latency = np.column_stack(
        [pd.to_numeric(aligned[f"latency_ms__{action}"], errors="coerce").fillna(0).to_numpy(float) for action in ACTIONS]
    )
    output["selected_action"] = np.asarray(ACTIONS, dtype=object)[selected]
    output["router_decision"] = decisions[rows, selected]
    output["probability_supported"] = scores[rows, selected]
    output["correct"] = (output["router_decision"].to_numpy(int) == output["label_supported"].to_numpy(int)).astype(np.int8)
    output["verifier_latency_ms"] = latency[rows, selected]
    output["end_to_end_latency_ms"] = output["verifier_latency_ms"] + output["feature_latency_ms"] + output["router_latency_ms"]
    output["forced_upgrade"] = ~availability[:, 0]
    output["price__factkb"] = float(price["low_price"])
    output["price__granite_guardian_3_1_2b"] = 0.0
    output["price__qwen30_fast"] = float(price["high_price"])
    for index, action in enumerate(ACTIONS):
        output[f"targetmix_score__{action}"] = log_score[:, index] - (float(price["low_price"]) if index == 0 else float(price["high_price"]) if index == 2 else 0.0)
    return output


def project_target_mix(base_predictions: pd.DataFrame, matrix_frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    matrix = matrix_frame.copy()
    matrix["episode_key"] = matrix["episode_key"].astype(str)
    matrix = matrix.set_index("episode_key", drop=False)
    projected: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for key, config in base_predictions.groupby(["method", "beta", "seed"], sort=True):
        method, beta, seed = key
        fold_values = sorted(config["outer_fold"].astype(int).unique().tolist())
        for fold in fold_values:
            calibrate = config[config["outer_fold"].astype(int) != int(fold)].reset_index(drop=True)
            heldout = config[config["outer_fold"].astype(int) == int(fold)].reset_index(drop=True)
            log_score, availability = _decision_scores(calibrate)
            price = calibrate_prices(log_score, availability)
            result = _reselect(heldout, matrix, price)
            shares = result["selected_action"].value_counts(normalize=True).reindex(ACTIONS, fill_value=0.0)
            projected.append(result)
            audit_rows.append(
                {
                    "method": method,
                    "beta": float(beta),
                    "seed": int(seed),
                    "heldout_outer_fold": int(fold),
                    "calibration_rows": int(len(calibrate)),
                    "heldout_rows": int(len(heldout)),
                    "price_low": float(price["low_price"]),
                    "price_mid": 0.0,
                    "price_high": float(price["high_price"]),
                    "calibration_l1_error": float(price["objective_l1"]),
                    "calibration_rate_low": float(price["shares"][0]),
                    "calibration_rate_mid": float(price["shares"][1]),
                    "calibration_rate_high": float(price["shares"][2]),
                    "heldout_rate_low": float(shares[ACTIONS[0]]),
                    "heldout_rate_mid": float(shares[ACTIONS[1]]),
                    "heldout_rate_high": float(shares[ACTIONS[2]]),
                }
            )
        log_score, availability = _decision_scores(config.reset_index(drop=True))
        price = calibrate_prices(log_score, availability)
        final_rows.append(
            {
                "method": method,
                "beta": float(beta),
                "seed": int(seed),
                "price_low": float(price["low_price"]),
                "price_mid": 0.0,
                "price_high": float(price["high_price"]),
                "full_oof_l1_error": float(price["objective_l1"]),
                "full_oof_rate_low": float(price["shares"][0]),
                "full_oof_rate_mid": float(price["shares"][1]),
                "full_oof_rate_high": float(price["shares"][2]),
            }
        )
    predictions = pd.concat(projected, ignore_index=True)
    keys = ["episode_key", "method", "beta", "seed"]
    expected = len(matrix_frame) * len(METHODS) * len(BETAS) * len(SEEDS)
    if len(predictions) != expected or predictions.duplicated(keys).any():
        raise AssertionError("target-mix OOF coverage failure")
    base.write_parquet(output_dir / "OOF_PREDICTIONS.parquet", predictions)
    base.write_csv(output_dir / "CROSSFIT_PRICE_AUDIT.csv", pd.DataFrame(audit_rows))
    base.write_csv(output_dir / "FINAL_POLICY_PRICES.csv", pd.DataFrame(final_rows))
    return {"predictions": predictions, "price_audit": pd.DataFrame(audit_rows), "final_prices": pd.DataFrame(final_rows)}


def load_reused_base(project_root: Path) -> pd.DataFrame:
    path = project_root / BASE_OOF_RELATIVE
    if base.sha256_file(path) != EXPECTED_BASE_OOF_SHA256:
        raise ValueError("reused Compact-16 OOF hash mismatch")
    predictions = pd.read_parquet(path)
    if set(predictions["method"].unique()) != set(REUSED_METHODS):
        raise ValueError("reused method set mismatch")
    return predictions


def select_best(reports: Mapping[str, pd.DataFrame], output_dir: Path) -> dict[str, Any]:
    summary = reports["OOF_SEED_SUMMARY.csv"].copy()
    summary["max_abs_share_error"] = np.maximum.reduce(
        [
            np.abs(summary["rate__factkb_mean"] - TARGET_SHARE[0]),
            np.abs(summary["rate__granite_guardian_3_1_2b_mean"] - TARGET_SHARE[1]),
            np.abs(summary["rate__qwen30_fast_mean"] - TARGET_SHARE[2]),
        ]
    )
    summary["target_mix_eligible"] = summary["max_abs_share_error"] <= 0.05
    rows: list[pd.Series] = []
    for method, part in summary.groupby("method", sort=True):
        eligible = part[part["target_mix_eligible"]]
        pool = eligible if len(eligible) else part.sort_values("max_abs_share_error").head(1)
        chosen = pool.sort_values(
            ["auroc_mean", "mean_end_to_end_latency_ms_mean", "balanced_accuracy_mean"],
            ascending=[False, True, False],
        ).iloc[0].copy()
        chosen["selection_status"] = "eligible" if bool(chosen["target_mix_eligible"]) else "closest_ineligible"
        rows.append(chosen)
    best = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False).reset_index(drop=True)
    base.write_csv(output_dir / "BEST_CONFIG_BY_METHOD.csv", best)
    eligible = best[best["target_mix_eligible"]]
    overall_pool = eligible if len(eligible) else best
    overall = overall_pool.sort_values(
        ["auroc_mean", "mean_end_to_end_latency_ms_mean", "balanced_accuracy_mean"],
        ascending=[False, True, False],
    ).iloc[0]
    payload = {key: base.json_safe(value) for key, value in overall.to_dict().items()}
    base.write_json(output_dir / "OVERALL_BEST.json", payload)
    return {"best_by_method": best, "overall": payload}


def preregistration_payload(project_root: Path, code_paths: Sequence[Path]) -> dict[str, Any]:
    matrix_path = project_root / base.MATRIX_RELATIVE
    input_path = project_root / base.INPUT_RELATIVE
    reused_path = project_root / BASE_OOF_RELATIVE
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": base.utc_now(),
        "scope": "four_dataset_mixed_pool_source_group_oof_targetmix_only",
        "assets": {
            "matrix_path": str(base.MATRIX_RELATIVE),
            "matrix_sha256": base.sha256_file(matrix_path),
            "summary_input_path": str(base.INPUT_RELATIVE),
            "summary_input_sha256": base.sha256_file(input_path),
            "reused_oof_path": str(BASE_OOF_RELATIVE),
            "reused_oof_sha256": base.sha256_file(reused_path),
        },
        "methods": list(METHODS),
        "reused_methods": list(REUSED_METHODS),
        "new_methods": list(NEW_METHODS),
        "betas": list(BETAS),
        "seeds": list(SEEDS),
        "outer_folds": base.OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "target_share": {action: float(TARGET_SHARE[index]) for index, action in enumerate(ACTIONS)},
        "targetmix_policy": "crossfit_action_log_price_calibration_on_other_outer_folds_no_labels_no_hard_quota",
        "price_anchor": "mid_price_zero",
        "price_search": {"bounds": list(PRICE_BOUNDS), "iterations": PRICE_ITERATIONS, "objective": "l1_to_60_30_10"},
        "supervision": {
            "S1": "existing_independent_correctness_probability_heads",
            "S2": "correctness_plus_utility_gap_pairwise; LR/HGB use weighted pairwise Borda reduction",
            "S3": "one_step_expected_reward; LR/HGB use direct conditional reward regression",
        },
        "expected_configurations": len(METHODS) * len(BETAS) * len(SEEDS),
        "expected_oof_rows": 6_850 * len(METHODS) * len(BETAS) * len(SEEDS),
        "success_gate": {
            "per_method_best_max_abs_share_error": 0.05,
            "primary": "auroc_mean",
            "secondary": "mean_end_to_end_latency_ms_mean",
        },
        "code_hashes": {str(path.relative_to(project_root)): base.sha256_file(path) for path in code_paths},
        "storysumm_rows_read": 0,
        "official_or_sealed_test_rows_read": 0,
    }


def validate_assets(project_root: Path) -> dict[str, Any]:
    checks = {
        "matrix_sha256": base.sha256_file(project_root / base.MATRIX_RELATIVE),
        "summary_input_sha256": base.sha256_file(project_root / base.INPUT_RELATIVE),
        "reused_oof_sha256": base.sha256_file(project_root / BASE_OOF_RELATIVE),
    }
    expected = {
        "matrix_sha256": base.EXPECTED_MATRIX_SHA256,
        "summary_input_sha256": base.EXPECTED_INPUT_SHA256,
        "reused_oof_sha256": EXPECTED_BASE_OOF_SHA256,
    }
    if checks != expected:
        raise ValueError(f"frozen asset mismatch: {checks}")
    return {"status": "PASS", **checks}


def audit_completed_run(project_root: Path, output_dir: Path) -> dict[str, Any]:
    assets = validate_assets(project_root)
    predictions = pd.read_parquet(output_dir / "OOF_PREDICTIONS.parquet")
    best = pd.read_csv(output_dir / "BEST_CONFIG_BY_METHOD.csv")
    price_audit = pd.read_csv(output_dir / "CROSSFIT_PRICE_AUDIT.csv")
    keys = ["episode_key", "method", "beta", "seed"]
    expected_rows = 6_850 * len(METHODS) * len(BETAS) * len(SEEDS)
    configurations = predictions[["method", "beta", "seed"]].drop_duplicates()
    required = [
        "OOF_CONFIG_METRICS_BY_SEED.csv",
        "OOF_DATASET_METRICS_BY_SEED.csv",
        "OOF_FAMILY_MACRO_BY_SEED.csv",
        "OOF_SEED_SUMMARY.csv",
        "PARETO_FRONTIER.csv",
        "CORRECTNESS_CALIBRATION.csv",
        "BASELINE_METRICS.csv",
        "BOOTSTRAP_CI.csv",
        "BEST_CONFIG_BY_METHOD.csv",
        "OVERALL_BEST.json",
        "CROSSFIT_PRICE_AUDIT.csv",
        "FINAL_POLICY_PRICES.csv",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    eligible_methods = int(best["target_mix_eligible"].fillna(False).astype(bool).sum())
    status = (
        len(predictions) == expected_rows
        and len(configurations) == len(METHODS) * len(BETAS) * len(SEEDS)
        and not predictions.duplicated(keys).any()
        and set(predictions["method"].unique()) == set(METHODS)
        and len(best) == len(METHODS)
        and eligible_methods == len(METHODS)
        and not missing
    )
    audit = {
        "status": "PASS" if status else "FAIL",
        "created_at_utc": base.utc_now(),
        "assets": assets,
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": int(expected_rows),
        "configurations": int(len(configurations)),
        "expected_configurations": int(len(METHODS) * len(BETAS) * len(SEEDS)),
        "duplicate_predictions": int(predictions.duplicated(keys).sum()),
        "methods": sorted(predictions["method"].unique().tolist()),
        "eligible_best_methods": eligible_methods,
        "price_audit_rows": int(len(price_audit)),
        "missing_outputs": missing,
        "storysumm_rows_read": 0,
        "official_or_sealed_test_rows_read": 0,
    }
    base.write_json(output_dir / "AUDIT.json", audit)
    if not status:
        raise AssertionError(f"target-mix audit failed: {audit}")
    return audit


def run_formal(project_root: Path, output_dir: Path, code_paths: Sequence[Path], *, quick: bool = False) -> dict[str, Any]:
    validate_assets(project_root)
    prereg_path = output_dir / "PREREG.json"
    expected_prereg = preregistration_payload(project_root, code_paths)
    if prereg_path.is_file():
        existing = json.loads(prereg_path.read_text())
        for key in ("assets", "methods", "reused_methods", "new_methods", "betas", "seeds", "target_share", "targetmix_policy", "code_hashes"):
            if existing.get(key) != expected_prereg.get(key):
                raise ValueError(f"preregistration drift in {key}")
    else:
        base.write_json(prereg_path, expected_prereg)
    frame, source_input, input_audit = base.load_frozen_training_inputs(project_root)
    feature_path = project_root / BASE_RESULT_RELATIVE / "COMPACT16_FEATURES.parquet"
    feature_audit_path = project_root / BASE_RESULT_RELATIVE / "FEATURE_AUDIT.json"
    features = base.load_feature_asset(frame, project_root / BASE_RESULT_RELATIVE)
    new = run_new_base_oof(frame, features, output_dir, quick=quick)
    reused = load_reused_base(project_root)
    combined = pd.concat([reused, new["predictions"]], ignore_index=True)
    base.write_parquet(output_dir / "COMBINED_BASE_OOF_PREDICTIONS.parquet", combined)
    projected = project_target_mix(combined, frame, output_dir)
    reports = base.build_oof_reports(
        projected["predictions"],
        frame.reset_index(drop=True),
        features.reset_index(drop=True),
        output_dir,
        bootstrap_draws=0 if quick else base.BOOTSTRAP_DRAWS,
    )
    selected = select_best(reports, output_dir)
    audit = audit_completed_run(project_root, output_dir)
    return {
        "input_audit": input_audit,
        "feature_path": str(feature_path),
        "feature_audit_path": str(feature_audit_path),
        "selection": selected,
        "audit": audit,
    }
