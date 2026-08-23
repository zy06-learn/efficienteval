#!/usr/bin/env python3
"""Shared machinery. One implementation of each step, imported by every experiment.

The point of this module is that there is exactly one place where a head gets trained, one place
where beta gets chosen, and one place where a metric is computed. In the previous round each
script re-implemented these and four of them drifted onto a different RandomForest and a beta
rule we had already retired -- a class of bug that is invisible in any single script's output and
only shows up when two tables disagree.

Nothing here reads a test fold for any purpose other than producing final numbers, and nothing
here calls a verifier: every score comes from the frozen matrix.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import json, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
                             f1_score, average_precision_score)

import config as C

sys.path.insert(0, str(C.SRC_ROOT))
sys.path.insert(0, str(C.SRC_ROOT / "scripts"))
from afr_v2 import tenfold_v1 as tenfold          # noqa: E402
from afr_v2 import pool_gate_sweep_v1 as v1       # noqa: E402
from afr_v2 import summary_router_compact16_direct_v1 as base  # noqa: E402

for _p in (C.RESULTS, C.STATUS, C.LOGS, C.NOT_REPORTED):
    _p.mkdir(parents=True, exist_ok=True)

_T0 = time.time()


class HeadPredictions(np.ndarray):
    """ndarray carrying measured batch-amortized head inference latency."""

    def __new__(cls, values, predict_ms=0.0):
        obj = np.asarray(values, float).view(cls)
        obj.predict_ms = float(predict_ms)
        return obj

    def __array_finalize__(self, parent):
        if parent is not None:
            self.predict_ms = getattr(parent, "predict_ms", 0.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-_T0:6.0f}s] {msg}", flush=True)


def done(tag: str) -> bool:
    return (C.STATUS / f"{tag}.done").exists()


def mark(tag: str, payload=None) -> None:
    (C.STATUS / f"{tag}.done").write_text(json.dumps(payload or {"ok": True}, default=str))


def save(name: str, frame: pd.DataFrame, index: bool = False, directory=None) -> pd.DataFrame:
    destination = Path(directory or C.RESULTS)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / name, index=index)
    return frame


# ------------------------------------------------------------------ data
MATRIX = v1.load_matrix(C.SRC_ROOT)
# POOL is frozen in cheap-to-expensive order. Routing never derives an action order from
# validation/test rows; per-rotation discounts use fold_costs(train) below.
ACTIONS = list(C.POOL)
COSTS = v1.pool_costs(MATRIX, C.POOL)                      # descriptive reference only
CVEC = np.array([COSTS[a] for a in ACTIONS], float)
ALL_VERIFIERS = [v for v in v1.ALL_VERIFIERS if f"score__{v}" in MATRIX.columns]
VCOST = {v: float(MATRIX[f"latency_ms__{v}"].mean()) for v in ALL_VERIFIERS}
DATASETS = sorted(MATRIX["dataset_key"].astype(str).unique())


def fold_costs(train: pd.DataFrame, actions=None) -> dict[str, float]:
    """Mean action costs estimated only from the current training rows."""
    return v1.pool_costs(train, list(actions or ACTIONS))


def verifier_cost(train: pd.DataFrame, verifier: str) -> float:
    """Training-only fallback cost for a fixed verifier."""
    value = float(train[f"latency_ms__{verifier}"].mean())
    if not np.isfinite(value):
        raise ValueError(f"no finite training latency for {verifier}")
    return value


def pooled_folds(seed: int) -> np.ndarray:
    """10 group-disjoint folds; rotations() yields (test, val, train) triples."""
    return tenfold.build_folds(MATRIX, seed)


def pooled_rotations():
    return tenfold.rotations()


def dataset_folds(sub: pd.DataFrame, dataset: str, seed: int, k: int = C.PERDATASET_FOLDS):
    """Group-disjoint k-fold WITHIN one corpus, for the 3/1/1 per-corpus protocol."""
    groups = sub["group_id"].astype(str).to_numpy()
    uniq = np.unique(groups)
    rng = np.random.default_rng(base.stable_seed(seed, "perdataset", dataset))
    rng.shuffle(uniq)
    assign = {g: i % k for i, g in enumerate(uniq)}
    return np.array([assign[g] for g in groups])


def dataset_rotations(k: int = C.PERDATASET_FOLDS):
    return [(t, (t + 1) % k, [f for f in range(k) if f not in (t, (t + 1) % k)]) for t in range(k)]


# ------------------------------------------------------------------ calibration
def platt(train: pd.DataFrame, actions=None):
    """Stage 1: per-verifier Platt, fit on training rows only."""
    return tenfold.fit_calibrators_on_val(train, list(actions or ACTIONS))


def apply_platt(frame: pd.DataFrame, cals, actions=None):
    return tenfold.apply_calibrators(frame, list(actions or ACTIONS), cals)


def isotonic(p_val: np.ndarray, y_val: np.ndarray, p_test: np.ndarray) -> np.ndarray:
    """Stage 2: monotone recalibration of a system's own output, fit on the validation fold.

    Needed because a router emits whichever member it selected, and three separately-calibrated
    verifiers do not share one probability scale. Monotone, so within a fold the ranking is
    untouched; the pooled gain comes from putting the ten test folds on a common scale.
    Applied identically to every system wherever it is used.
    """
    m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    m.fit(np.asarray(p_val, float), np.asarray(y_val, int))
    return np.clip(m.predict(np.asarray(p_test, float)), 1e-6, 1 - 1e-6)


# ------------------------------------------------------------------ supervision target
def targets(frame: pd.DataFrame, cal, actions=None, kind: str = None):
    """Y[:, j] is what head j should learn. See config for why `regret` is the locked choice."""
    actions = list(actions or ACTIONS)
    kind = kind or C.TARGET
    y = frame["label_supported"].to_numpy(float)
    Cm = np.column_stack([np.asarray(cal[a], float) for a in actions])
    Q = np.where(y[:, None] > 0.5, Cm, 1.0 - Cm)           # mass placed on the truth
    k = len(actions)
    if kind == "regret":
        return 1.0 + (Q - Q.max(1, keepdims=True)), False
    if kind == "prob_quality":
        return Q, False
    if kind == "brier":
        return 1.0 - (Cm - y[:, None]) ** 2, False
    if kind == "pairwise_rank":
        return (Q[:, :, None] > Q[:, None, :]).sum(2) / max(k - 1, 1), False
    if kind == "correctness":
        return np.column_stack([frame[f"correct__{a}"].fillna(0).to_numpy(int) for a in actions]), True
    if kind == "is_best":
        return (Q >= Q.max(1, keepdims=True) - 1e-12).astype(int), True
    raise ValueError(f"unknown target {kind}")


# ------------------------------------------------------------------ heads
def make_regressor(seed: int, hp=None, criterion=None):
    hp = dict(hp or C.HP)
    return RandomForestRegressor(criterion=criterion or C.CRITERION, n_jobs=8,
                                 random_state=seed, **hp)


def make_classifier(learner: str, seed: int, hp=None):
    """Classification counterpart used by correctness/is_best target sweeps.

    Refuse unsupported learner names instead of silently substituting RandomForest.
    """
    if learner == "rf":
        return RandomForestClassifier(n_jobs=8, random_state=seed, **dict(hp or C.HP))
    if learner == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                              max_leaf_nodes=31, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    raise ValueError(f"unsupported classification learner: {learner}")


def fit_heads(train, out_frames, seed, cal_train, *, actions=None, features=None,
              hp=None, criterion=None, target=None, learner=None):
    """One head per action, fit only on rows where that action is available."""
    actions = list(actions or ACTIONS)
    features = list(features or C.FEATURES)
    learner = learner or C.LEARNER
    Y, is_clf = targets(train, cal_train, actions, target)
    xt = train[features].to_numpy(float)
    xs = [f[features].to_numpy(float) for f in out_frames]
    preds = [HeadPredictions(np.zeros((len(f), len(actions)))) for f in out_frames]
    predict_ms = np.zeros(len(out_frames), float)
    for j, a in enumerate(actions):
        av = train[f"available__{a}"].astype(bool).to_numpy()
        yv = np.asarray(Y[av, j])
        if is_clf:
            if len(set(yv.astype(int).tolist())) < 2:       # degenerate column, emit the prior
                fill = float(yv.mean()) if len(yv) else 0.5
                for i in range(len(preds)):
                    preds[i][:, j] = fill
                continue
            model = make_classifier(learner, seed, hp)
            model.fit(xt[av], yv.astype(int))
            for i in range(len(preds)):
                started = time.perf_counter()
                preds[i][:, j] = model.predict_proba(xs[i])[:, 1]
                predict_ms[i] += (time.perf_counter() - started) * 1000.0 / max(len(xs[i]), 1)
            continue
        model = (_ALT_LEARNERS[learner](seed) if learner in _ALT_LEARNERS
                 else make_regressor(seed, hp, criterion))
        model.fit(xt[av], yv.astype(float))
        for i in range(len(preds)):
            started = time.perf_counter()
            preds[i][:, j] = np.clip(model.predict(xs[i]), 0.0, 1.0)
            predict_ms[i] += (time.perf_counter() - started) * 1000.0 / max(len(xs[i]), 1)
    for pred, latency in zip(preds, predict_ms):
        pred.predict_ms = float(latency)
    return preds


# Alternatives used only by the learner-sensitivity experiment; the locked learner is RF.
_ALT_LEARNERS = {
    "extra_trees": lambda s: ExtraTreesRegressor(n_estimators=800, min_samples_leaf=1,
                                                 max_features="sqrt", max_depth=10,
                                                 n_jobs=8, random_state=s),
    "hgb": lambda s: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                   max_leaf_nodes=31, l2_regularization=1.0,
                                                   early_stopping=False, random_state=s),
    "gbr": lambda s: GradientBoostingRegressor(n_estimators=300, learning_rate=0.06,
                                               max_depth=4, random_state=s),
    "ridge": lambda s: Ridge(alpha=1.0),
    "knn": lambda s: KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=8),
    "mlp": lambda s: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400,
                                  early_stopping=True, random_state=s),
    "tree": lambda s: DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=s),
}


# ------------------------------------------------------------------ beta
def choose_beta(val, heads_val, cal_val, costs=None, actions=None, eps=None):
    """Budget rule: the cheapest beta whose validation AUROC is within eps of the best.

    This replaced a `quality_gain x speed_gain` product whose speed term was normalised by
    (most expensive - cheapest action). Making the strongest verifier cheaper collapsed that
    denominator, the product became dominated by cost, and the router degenerated to always
    choosing the weakest action -- AUROC .7931 -> .6734 when Granite was halved. The budget rule
    references no cost endpoint and so cannot degenerate that way.
    """
    actions = list(actions or ACTIONS)
    costs = costs or COSTS
    eps = C.EPS if eps is None else eps
    ledger = []
    for b in tenfold.BETA_GRID:
        p = v1.router_prediction(val, actions, heads_val, cal_val, costs, b)
        ledger.append((b, v1.scope_auroc(p, "pooled"),
                       float(p["end_to_end_latency_ms"].mean())))
    best = max(r[1] for r in ledger)
    ok = [r for r in ledger if r[1] >= best - eps]
    return min(ok, key=lambda r: r[2])[0]


def route(frame, heads, cal, beta, actions=None, costs=None):
    started = time.perf_counter()
    out = v1.router_prediction(frame, list(actions or ACTIONS), heads, cal,
                               costs or COSTS, beta)
    policy_ms = (time.perf_counter() - started) * 1000.0 / max(len(frame), 1)
    router_ms = float(getattr(heads, "predict_ms", 0.0)) + policy_ms
    component_ms = out["end_to_end_latency_ms"].to_numpy(float).copy()
    out["component_latency_ms"] = component_ms
    out["router_latency_ms"] = router_ms
    out["end_to_end_latency_ms"] = component_ms + router_ms
    return out


def make_frame(test, prob, decision, label, verifier_ms, calls, charge_features):
    """Prediction frame for an arm that is not produced by router_prediction.

    charge_features must be False for anything that does not compute the cheap features -- a
    fixed verifier and an ensemble do not, and folding our 8.6ms into their cost would flatter us.
    """
    n = len(test)
    out = test[["episode_key", "dataset_key", "group_id", "label_supported"]].copy()
    out["selected_action"] = label
    out["router_decision"] = np.asarray(decision, int)
    out["probability_supported"] = np.asarray(prob, float)
    out["correct"] = (out["router_decision"].to_numpy(int)
                      == out["label_supported"].to_numpy(int)).astype(np.int8)
    out["verifier_latency_ms"] = np.broadcast_to(np.asarray(verifier_ms, float), (n,)).copy()
    out["feature_latency_ms"] = (test["feature_latency_ms"].to_numpy(float)
                                 if charge_features else np.zeros(n))
    out["router_latency_ms"] = 0.0
    out["end_to_end_latency_ms"] = out["verifier_latency_ms"] + out["feature_latency_ms"]
    out["component_latency_ms"] = out["end_to_end_latency_ms"]
    out["mean_calls_per_summary"] = np.broadcast_to(np.asarray(calls, float), (n,)).copy()
    return out


# ------------------------------------------------------------------ metrics
def ece(y, p, bins=None):
    bins = bins or C.ECE_BINS
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total, worst = 0.0, 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            gap = abs(p[m].mean() - y[m].mean())
            total += m.mean() * gap
            worst = max(worst, gap)
    return float(total), float(worst)


def risk_coverage(y, p, decision):
    """Sort by |p-.5| descending; selective error at each coverage, plus AURC."""
    ok = (np.asarray(decision, int) == np.asarray(y, int)).astype(int)
    order = np.argsort(-np.abs(np.asarray(p, float) - 0.5), kind="stable")
    ok = ok[order]
    run = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    out = {f"selerr@{c:.1f}": float(1.0 - run[max(int(round(c * len(ok))), 1) - 1])
           for c in C.COVERAGE}
    dense = np.maximum((np.linspace(0.02, 1.0, 50) * len(ok)).round().astype(int), 1)
    out["aurc"] = float(np.mean(1.0 - run[dense - 1]))
    return out


def metrics(y, p, ms, threshold=0.5, with_risk=True):
    """Every number the paper reports for one system on one seed."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    ms = np.asarray(ms, float)
    d = (p >= threshold).astype(int)
    unsup = y == 0
    flag = d == 0
    e, mx = ece(y, p)
    out = {
        "n": int(len(y)),
        "auroc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else np.nan,
        "auprc_unsup": float(average_precision_score(unsup.astype(int), 1.0 - p)),
        "acc": float((d == y).mean()),
        "bacc": float(balanced_accuracy_score(y, d)),
        "mcc": float(matthews_corrcoef(y, d)) if len(set(d.tolist())) > 1 else 0.0,
        "f1_macro": float(f1_score(y, d, average="macro")),
        "f1_unsup": float(f1_score(unsup.astype(int), flag.astype(int))),
        "prec_unsup": float(unsup[flag].mean()) if flag.any() else np.nan,
        "rec_unsup": float(flag[unsup].mean()) if unsup.any() else np.nan,
        "ece": e, "mce": mx,
        "brier": float(np.mean((p - y) ** 2)),
        "ms": float(ms.mean()),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "qps": float(1000.0 / ms.mean()),
    }
    if with_risk:
        out.update(risk_coverage(y, p, d))
    return out


def balance(quality, cost):
    q = (quality - C.BALANCE_Q_FLOOR) / (C.BALANCE_Q_CEIL - C.BALANCE_Q_FLOOR)
    c = ((np.log(max(cost, 1e-6)) - np.log(C.BALANCE_C_MIN))
         / (np.log(C.BALANCE_C_MAX) - np.log(C.BALANCE_C_MIN)))
    return float(1.0 - np.sqrt((1 - q) ** 2 + c ** 2) / np.sqrt(2.0))


def conformal_tau(p_unsupported, delta):
    """Finite-sample upper quantile for the inclusive flag rule ``p <= tau``."""
    p = np.sort(np.asarray(p_unsupported, float))
    n = len(p)
    if n == 0:
        return 0.5
    k = int(np.ceil((n + 1) * (1.0 - delta)))
    return np.inf if k > n else float(p[k - 1])


def group_conformal_tau(y, p_supported, groups, delta):
    """Calibrate on one worst-case unsupported score per source group.

    Taking the maximum supported-probability among unsupported rows makes the calibration unit
    the independent source group rather than a correlated summary row.
    """
    y = np.asarray(y, int)
    p_supported = np.asarray(p_supported, float)
    groups = np.asarray(groups).astype(str)
    frame = pd.DataFrame({"group": groups[y == 0], "score": p_supported[y == 0]})
    group_scores = frame.groupby("group", sort=True)["score"].max().to_numpy(float)
    return conformal_tau(group_scores, delta), int(len(group_scores))


def paired_bootstrap(y, p_a, p_b, seed, draws=None):
    """Delta AUROC (a - b) on identical rows, with a paired bootstrap interval."""
    draws = draws or C.BOOTSTRAP
    y = np.asarray(y, int)
    rng = np.random.default_rng(base.stable_seed(seed, "paired", "boot"))
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        vals.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _as_prediction_matrix(p, n):
    p = np.asarray(p, float)
    if p.ndim == 1:
        p = p[:, None]
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError(f"predictions must have shape (n,) or (n,seeds); got {p.shape}")
    return p


def _cluster_indices(by_group, sampled_groups):
    return np.concatenate([by_group[g] for g in sampled_groups])


def paired_cluster_bootstrap(y, p_a, p_b, groups, seed, draws=None):
    """Mean-over-seeds AUROC delta with a paired source-group cluster bootstrap."""
    draws = draws or C.BOOTSTRAP
    y = np.asarray(y, int)
    groups = np.asarray(groups).astype(str)
    if len(groups) != len(y):
        raise ValueError("groups and labels must have identical length")
    p_a = _as_prediction_matrix(p_a, len(y))
    p_b = _as_prediction_matrix(p_b, len(y))
    if p_a.shape != p_b.shape:
        raise ValueError("paired prediction matrices must have identical shape")

    def delta(idx):
        yy = y[idx]
        return float(np.mean([
            roc_auc_score(yy, p_a[idx, s]) - roc_auc_score(yy, p_b[idx, s])
            for s in range(p_a.shape[1])
        ]))

    point = delta(np.arange(len(y)))
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(base.stable_seed(seed, "paired_group", "boot"))
    vals = []
    for _ in range(draws):
        sampled = rng.choice(uniq, len(uniq), replace=True)
        idx = _cluster_indices(by_group, sampled)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(delta(idx))
    if not vals:
        raise ValueError("no valid clustered bootstrap draws")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def cluster_bootstrap_auroc(y, predictions, groups, seed, draws=None):
    """Mean-over-seeds AUROC with uncertainty from resampling source groups."""
    draws = draws or C.BOOTSTRAP
    y = np.asarray(y, int)
    groups = np.asarray(groups).astype(str)
    predictions = _as_prediction_matrix(predictions, len(y))

    def score(idx):
        yy = y[idx]
        return float(np.mean([
            roc_auc_score(yy, predictions[idx, s]) for s in range(predictions.shape[1])
        ]))

    point = score(np.arange(len(y)))
    uniq = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(base.stable_seed(seed, "group", "auroc"))
    vals = []
    for _ in range(draws):
        sampled = rng.choice(uniq, len(uniq), replace=True)
        idx = _cluster_indices(by_group, sampled)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(score(idx))
    if not vals:
        raise ValueError("no valid clustered bootstrap draws")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def contamination_flag(system, dataset):
    if (system, dataset) in C.CONFIRMED_TRAIN:
        return "CONFIRMED_TRAIN"
    return "DEV_EXPOSED" if system in C.DEV_EXPOSED else "clean"
