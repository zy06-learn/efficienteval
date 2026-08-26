"""10-fold rotation protocol: 8 train / 1 val / 1 test, source-group disjoint.

Replaces the 5-fold + inner-fold arrangement with an explicit validation fold:
  * 8 folds  -> fit the three HGB correctness heads
  * 1 fold   -> fit the Platt calibrators AND select beta   (never used for fitting trees)
  * 1 fold   -> evaluate once, everything frozen            (never touched before this)
Rotating 10 times gives every summary exactly one test-fold prediction.

Two fixes carried over from the audit:
  * the budget-matched random control is scored with the SAME calibrated probabilities
    as the router (the previous control used raw verifier scores, which depressed it);
  * the control is drawn CONTROL_DRAWS times and the bootstrap redraws it each iteration.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from verifier_wrappers import pool_gate_sweep_v1 as v1
from verifier_wrappers import pool_gate_sweep_v2 as v2
from verifier_wrappers import summary_router_compact16_direct_v1 as base

OUTPUT_RELATIVE = Path("results/tenfold_pooled_v1")

N_FOLDS = 10
CONTROL_DRAWS = 16
BOOTSTRAP_DRAWS = 2000
BETA_GRID = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2)
SEEDS = (17, 29, 43)
POOL = ("factkb", "granite_guardian_3_1_2b", "wecheck")


def build_folds(matrix: pd.DataFrame, seed: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """Source-group disjoint folds, stratified by (dataset, majority label of the group)."""
    groups = matrix.groupby("group_id", sort=True)
    rows = []
    for gid, part in groups:
        rows.append({
            "group_id": gid,
            "dataset_key": str(part["dataset_key"].iloc[0]),
            "label_bucket": int(round(float(part["label_supported"].mean()))),
            "size": len(part),
        })
    frame = pd.DataFrame(rows)
    generator = np.random.default_rng(base.stable_seed(seed, "tenfold", "assign"))
    assignment: dict[Any, int] = {}
    for _, stratum in frame.groupby(["dataset_key", "label_bucket"], sort=True):
        order = stratum.sample(frac=1.0, random_state=int(generator.integers(1 << 31)))
        # greedy balance: walk the shuffled groups and hand them to folds round-robin
        for position, (_, row) in enumerate(order.iterrows()):
            assignment[row["group_id"]] = position % n_folds
    return matrix["group_id"].map(assignment).to_numpy(int)


def rotations(n_folds: int = N_FOLDS) -> list[tuple[int, int, list[int]]]:
    out = []
    for test in range(n_folds):
        val = (test + 1) % n_folds
        train = [f for f in range(n_folds) if f not in (test, val)]
        out.append((test, val, train))
    return out


def fit_calibrators_on_val(
    val: pd.DataFrame, actions: Sequence[str]
) -> dict[str, Any]:
    """Platt calibrators for the label probability, fitted on the validation fold only."""
    out = {}
    for action in actions:
        available = val[f"available__{action}"].astype(bool).to_numpy()
        out[action] = base._Platt.fit(
            val.loc[available, f"score__{action}"].to_numpy(float),
            val.loc[available, "label_supported"].to_numpy(int),
        )
    return out


def apply_calibrators(frame: pd.DataFrame, actions: Sequence[str], calibrators) -> dict[str, np.ndarray]:
    return {
        a: calibrators[a].predict(frame[f"score__{a}"].fillna(0.5).to_numpy(float))
        for a in actions
    }


def select_beta(
    val: pd.DataFrame,
    actions: Sequence[str],
    heads_val: np.ndarray,
    calibrated_val: Mapping[str, np.ndarray],
    costs: Mapping[str, float],
) -> tuple[float, list[dict]]:
    """Pick beta on the validation fold: maximise quality gain x speed gain.

    Quality/speed are normalised between the cheapest and the best-on-val endpoint,
    so nothing here assumes 'more expensive is better'.
    """
    fast = min(actions, key=lambda a: costs[a])
    aurocs = {}
    for a in actions:
        ok = val[f"available__{a}"].astype(bool).to_numpy()
        y = val.loc[ok, "label_supported"].to_numpy(int)
        aurocs[a] = roc_auc_score(y, val.loc[ok, f"score__{a}"].to_numpy(float)) if len(set(y)) > 1 else 0.5
    quality = max(actions, key=lambda a: aurocs[a])
    if quality == fast:
        quality = sorted(actions, key=lambda a: -aurocs[a])[1]

    fast_auc, quality_auc = aurocs[fast], aurocs[quality]
    fast_ms, quality_ms = costs[fast], costs[quality]
    ledger = []
    best = (float("-inf"), BETA_GRID[0])
    for beta in BETA_GRID:
        pred = v1.router_prediction(val, actions, heads_val, calibrated_val, costs, beta)
        auc = v1.scope_auroc(pred, "pooled")
        ms = float(pred["end_to_end_latency_ms"].mean())
        q = (auc - fast_auc) / (quality_auc - fast_auc) if quality_auc > fast_auc else float("nan")
        s = (quality_ms - ms) / (quality_ms - fast_ms) if quality_ms > fast_ms else float("nan")
        score = max(q, 0.0) * max(s, 0.0) if q == q and s == s else float("nan")
        ledger.append({"beta": beta, "val_auroc": auc, "val_ms": ms,
                       "quality_gain": q, "speed_gain": s, "select_score": score})
        if score == score and score > best[0]:
            best = (score, beta)
    return best[1], ledger
