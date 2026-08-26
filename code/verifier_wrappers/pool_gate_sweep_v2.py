"""v2: fixes the matched-random sampling-variance flaw in v1.

v1 drew the budget-matched random policy ONCE per configuration. A single draw carries
sd ~= 0.02-0.03 AUROC, which is the same order as the effects being measured, and the
paired bootstrap resampled source groups only -- it never redrew the random policy, so
its interval understated the true uncertainty.

v2:
  * the control is N independent draws instead of one;
  * the point estimate compares against the MEAN control AUROC over those draws;
  * every bootstrap iteration resamples source groups AND picks a draw uniformly at
    random, so the interval integrates both variance sources at the same cost as v1;
  * the raw draw-to-draw spread is reported so the size of the v1 flaw is visible.
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
from verifier_wrappers import summary_router_compact16_direct_v1 as base

OUTPUT_RELATIVE = Path("results/pool_gate_sweep_v2")

CONTROL_DRAWS = 16
BOOTSTRAP_DRAWS = 2000

POOLS = v1.POOLS
DATASETS = v1.DATASETS
SEEDS = v1.SEEDS
BETA_GRID = v1.BETA_GRID
CONF_WIDTHS = v1.CONF_WIDTHS


def _auroc(labels: np.ndarray, probability: np.ndarray, datasets: np.ndarray, scope: str) -> float:
    if scope == "lodo":
        values = []
        for key in np.unique(datasets):
            mask = datasets == key
            y = labels[mask]
            if len(np.unique(y)) < 2:
                continue
            values.append(roc_auc_score(y, probability[mask]))
        return float(np.mean(values)) if values else float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probability))


class Comparison:
    """One treatment policy against N budget-matched random draws, on one scope."""

    def __init__(self, treatment: pd.DataFrame, controls: Sequence[pd.DataFrame], scope: str) -> None:
        self.scope = scope
        self.labels = treatment["label_supported"].to_numpy(int)
        self.datasets = treatment["dataset_key"].astype(str).to_numpy()
        self.groups = treatment["group_id"].astype(str).to_numpy()
        self.treatment = treatment["probability_supported"].to_numpy(np.float32)
        self.controls = np.column_stack(
            [c["probability_supported"].to_numpy(np.float32) for c in controls]
        )
        self.treatment_latency = float(treatment["end_to_end_latency_ms"].mean())
        self.control_latency = float(np.mean([c["end_to_end_latency_ms"].mean() for c in controls]))
        self.treatment_calls = float(treatment["mean_calls_per_summary"].mean())
        self.control_calls = float(np.mean([c["mean_calls_per_summary"].mean() for c in controls]))

    def point(self) -> dict[str, float]:
        t = _auroc(self.labels, self.treatment, self.datasets, self.scope)
        per_draw = np.asarray(
            [_auroc(self.labels, self.controls[:, j], self.datasets, self.scope)
             for j in range(self.controls.shape[1])], dtype=float
        )
        return {
            "treatment_auroc": t,
            "control_auroc_mean": float(np.nanmean(per_draw)),
            "control_auroc_sd_across_draws": float(np.nanstd(per_draw, ddof=1)),
            "control_auroc_min": float(np.nanmin(per_draw)),
            "control_auroc_max": float(np.nanmax(per_draw)),
            "delta_point": t - float(np.nanmean(per_draw)),
            "treatment_latency_ms": self.treatment_latency,
            "control_latency_ms": self.control_latency,
            "treatment_calls": self.treatment_calls,
            "control_calls": self.control_calls,
            "control_draws": int(self.controls.shape[1]),
        }

    def bootstrap(self, tag: str, draws: int = BOOTSTRAP_DRAWS) -> dict[str, float]:
        """Resample source groups (stratified by dataset) AND redraw the control policy."""
        generator = np.random.default_rng(base.stable_seed(2026, self.scope, tag, "v2-bootstrap"))
        order = np.argsort(self.groups, kind="stable")
        sorted_groups = self.groups[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
        index_blocks = np.split(order, boundaries[1:])
        block_dataset = np.asarray([self.datasets[b[0]] for b in index_blocks])
        strata = {key: np.flatnonzero(block_dataset == key) for key in np.unique(block_dataset)}
        n_controls = self.controls.shape[1]

        deltas = []
        for _ in range(draws):
            picks = [
                generator.choice(members, size=len(members), replace=True)
                for members in strata.values()
            ]
            chosen = np.concatenate(picks)
            rows = np.concatenate([index_blocks[b] for b in chosen])
            j = int(generator.integers(n_controls))
            a = _auroc(self.labels[rows], self.treatment[rows], self.datasets[rows], self.scope)
            b = _auroc(self.labels[rows], self.controls[rows, j], self.datasets[rows], self.scope)
            if a == a and b == b:
                deltas.append(a - b)
        if not deltas:
            return {"delta_mean": float("nan"), "delta_ci_low": float("nan"),
                    "delta_ci_high": float("nan"), "bootstrap_draws": 0}
        array = np.asarray(deltas, dtype=float)
        return {
            "delta_mean": float(array.mean()),
            "delta_ci_low": float(np.percentile(array, 2.5)),
            "delta_ci_high": float(np.percentile(array, 97.5)),
            "bootstrap_draws": len(array),
        }


def control_bank_single(
    test: pd.DataFrame, actions: Sequence[str], reference: pd.DataFrame,
    calibrated: Mapping[str, np.ndarray], tag: Sequence[Any], draws: int = CONTROL_DRAWS,
) -> list[pd.DataFrame]:
    return [
        v1.matched_random_single(test, actions, reference, calibrated, (*tag, f"draw{j}"))
        for j in range(draws)
    ]


def control_bank_cascade(
    test: pd.DataFrame, chain: Sequence[str], reference: pd.DataFrame,
    calibrated: Mapping[str, np.ndarray], tag: Sequence[Any], draws: int = CONTROL_DRAWS,
) -> list[pd.DataFrame]:
    return [
        v1.matched_random_cascade(test, chain, reference, calibrated, (*tag, f"draw{j}"))
        for j in range(draws)
    ]
