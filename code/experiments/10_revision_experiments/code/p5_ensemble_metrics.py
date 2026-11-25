#!/usr/bin/env python3
"""Equal-budget ensembles of cheap verifiers on every main-table metric, both protocols.

Replicates p2_checks.py part C for the seven non-empty subsets of FactCC, FactKB and Lettuce-v2:
per-member Platt on the fit partition, then either the mean of the calibrated scores or the
fraction of members voting "supported", then isotonic fitted on the validation partition.
Latency is the sum of the members' verifier latencies, as in p2_checks. Added: part1c's
MCC-selected threshold on the validation side, so BAcc and F1 can be reported.

Gate: Protocol B pooled AUROC and ms reproduce results_p2checks/C_equal_budget_ensemble.csv (<= 1e-9).
Output: results_p5align/ENSEMBLE_{B,A}.csv, ENSEMBLE_GATES.json
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as V  # noqa: E402
import core  # noqa: E402
from p5_front_metrics import rule_mcc_fast, rule_mcc_ref  # noqa: E402

SEEDS = list(V.SEEDS)
OUT = HERE / "results_p5align"
OUT.mkdir(exist_ok=True)
MEMBERS = ["factcc", "factkb", "lettuce_v2"]
NICE = {"factcc": "FactCC", "factkb": "FactKB", "lettuce_v2": "Lettuce-v2"}
COMBOS = [c for r in (1, 2, 3) for c in itertools.combinations(MEMBERS, r)]
RULES = ("mean of calibrated scores", "majority vote")
KEYS = ["auroc", "auprc_unsup", "bacc", "f1_unsup", "ms"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iso(p_val, y_val, p_ev):
    m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    return np.clip(m.predict(p_ev), 1e-6, 1 - 1e-6)


def one_split(fit, val, ev, combo, check):
    yv = val["label_supported"].to_numpy(int)
    cals = core.platt(fit, actions=list(combo))
    cv = core.apply_platt(val, cals, actions=list(combo))
    ce = core.apply_platt(ev, cals, actions=list(combo))
    raw = {RULES[0]: (np.mean([cv[v] for v in combo], axis=0), np.mean([ce[v] for v in combo], axis=0)),
           RULES[1]: (np.mean([(np.asarray(cv[v]) >= 0.5).astype(float) for v in combo], axis=0),
                      np.mean([(np.asarray(ce[v]) >= 0.5).astype(float) for v in combo], axis=0))}
    out = {}
    for rule, (pv, pe) in raw.items():
        q_val = iso(pv, yv, pv)
        thr = rule_mcc_fast(q_val, yv)
        if check and thr != rule_mcc_ref(q_val, yv):
            raise AssertionError("fast MCC threshold differs from part1c's rule")
        out[rule] = (iso(pv, yv, pe), thr)
    ms = sum(ev[f"latency_ms__{v}"].fillna(0).to_numpy(float) for v in combo)
    return out, ms


def run(proto):
    TRAIN, TEST, ALL, _v = V.load(with_test_labels=True)
    frame = TEST if proto == "B" else ALL.reset_index(drop=True)
    y, n, S = frame["label_supported"].to_numpy(int), len(frame), len(SEEDS)
    rows = []
    for ci, combo in enumerate(COMBOS):
        acc = {r: (np.zeros((n, S)), np.zeros((n, S))) for r in RULES}
        MS = np.zeros((n, S))
        for si, seed in enumerate(SEEDS):
            if proto == "B":
                held = V.stratified_group_split(TRAIN, seed)
                splits = [(TRAIN.loc[~held].reset_index(drop=True),
                           TRAIN.loc[held].reset_index(drop=True), np.arange(n))]
            else:
                fold = V.folds_stratified(frame, seed)
                splits = [(frame.loc[np.isin(fold, tr)].reset_index(drop=True),
                           frame.loc[fold == va].reset_index(drop=True), np.flatnonzero(fold == te))
                          for te, va, tr in V.rotations()]
            for fit, val, pos in splits:
                ev = frame.iloc[pos].reset_index(drop=True)
                out, ms = one_split(fit, val, ev, combo, check=(ci == 0 and si == 0))
                for r in RULES:
                    acc[r][0][pos, si], acc[r][1][pos, si] = out[r]
                MS[pos, si] = ms
        for r in RULES:
            P, THR = acc[r]
            per = [core.metrics(y, P[:, s], MS[:, s], threshold=THR[:, s], with_risk=False)
                   for s in range(S)]
            rows.append({"members": "+".join(NICE[v] for v in combo), "k": len(combo), "rule": r,
                         **{k: float(np.nanmean([m[k] for m in per])) for k in KEYS}})
        log(f"{proto} {'+'.join(combo)} done")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    B = run("B")
    orig = pd.read_csv(HERE / "results_p2checks" / "C_equal_budget_ensemble.csv")
    m = B.merge(orig, on=["members", "k", "rule"], suffixes=("", "_orig"))
    if len(m) != len(B):
        raise AssertionError("gate: not every ensemble row matched the original table")
    gate = {"auroc": float((m["auroc"] - m["pooled"]).abs().max()),
            "ms": float((m["ms"] - m["ms_orig"]).abs().max())}
    if max(gate.values()) > 1e-9:
        raise AssertionError(f"gate: does not reproduce C_equal_budget_ensemble.csv: {gate}")
    B.to_csv(OUT / "ENSEMBLE_B.csv", index=False)
    log(f"B gate PASS {gate}")
    A = run("A")
    A.to_csv(OUT / "ENSEMBLE_A.csv", index=False)
    (OUT / "ENSEMBLE_GATES.json").write_text(json.dumps({"B_max_abs_diff": gate, "rows": len(B)}, indent=2))
    print(B.sort_values("auroc", ascending=False).to_string(index=False))
    print(A.sort_values("auroc", ascending=False).to_string(index=False))
