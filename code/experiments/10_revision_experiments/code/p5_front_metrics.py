#!/usr/bin/env python3
"""The cascade-frontier sweep of p0_cascade_v1 / p0_cascade_a_v1, re-scored on every main-table metric.

The sweep is unchanged: same fitted splits, router beta grid, cascade triggers, escalation targets,
threshold grid and latency definition (verifier calls plus both feature extractors). Added per
configuration and split: the validation-side stage-2 probabilities and part1c's MCC-selected
threshold, so AUPRC, BAcc and F1 can be reported beside AUROC.

Gates (abort on failure):
  G1  the vectorised MCC threshold equals part1c's rule on every split of the first configuration
  G2  every configuration's AUROC, ms and calls reproduce E5_cascade_sweep{,_A}.csv (<= 1e-9)

Output: results_p5align/{B,A}_CASCADE_SWEEP_FULL.csv, {B,A}_CASCADE_FRONTIER.csv, {B,A}_FRONT_GATES.json
Usage:  p5_front_metrics.py B|A
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
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as V  # noqa: E402
import core  # noqa: E402
import p0_cascade_v1 as CB  # noqa: E402
from p0_cascade_v1 import Fitted, NICE, POOL, GRID  # noqa: E402

PROTO = sys.argv[1] if __name__ == "__main__" else None  # importable for its threshold rules
SEEDS = list(V.SEEDS)
OUT = HERE / "results_p5align"
OUT.mkdir(exist_ok=True)
KEYS = ["auroc", "auprc_unsup", "bacc", "f1_unsup", "ms"]
BUDGETS = {"B": (100.0, 106.0, 120.0, float("inf")), "A": (100.0, 112.0, 130.0, float("inf"))}
TAU_Q = np.linspace(0.0, 0.95, 20)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rule_mcc_ref(q, y):
    """part1c_main_full_v1._rule_mcc, verbatim."""
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def rule_mcc_fast(q, y):
    """Same grid and first-maximum tie-break; sklearn's MCC formula on exact integer counts."""
    q, y = np.asarray(q, float), np.asarray(y).astype(int)
    grid = np.unique(np.quantile(q, np.linspace(0.02, 0.98, 97)))
    d = q[None, :] >= grid[:, None]
    tp = (d & (y == 1)[None, :]).sum(1).astype(np.int64)
    fp = (d & (y == 0)[None, :]).sum(1).astype(np.int64)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    n = n0 + n1
    p1 = tp + fp
    p0 = n - p1
    cov_ytyp = (tp + (n0 - fp)) * n - (n0 * p0 + n1 * p1)
    cov_ypyp = n * n - (p0 * p0 + p1 * p1)
    cov_ytyt = n * n - (n0 * n0 + n1 * n1)
    den = cov_ytyt * cov_ypyp
    mcc = np.where(den == 0, 0.0, cov_ytyp / np.sqrt(np.where(den == 0, 1, den)))
    return float(grid[int(np.argmax(mcc))])


def finish(F, p_val, p_ev, ms_rows, calls_rows):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, F.yv)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    return q_ev, np.asarray(ms_rows, float), float(np.mean(calls_rows)), q_val


def router_full(F, beta):
    """Fitted.router, returning the validation side as well."""
    uv, ue = F.Hv * np.exp(-beta * F.cn), F.He * np.exp(-beta * F.cn)
    sv, se = uv.argmax(1), ue.argmax(1)
    rv, re_ = np.arange(len(F.Cv)), np.arange(len(F.Ce))
    return finish(F, F.Cv[rv, sv], F.Ce[re_, se], F.Le[re_, se] + F.pe, np.ones(len(re_)))


def cascade_full(F, first, second, tau, signal):
    """Fitted.cascade, returning the validation side as well."""
    got = F.signals(first, signal)
    if got is None:
        return None
    sv, se = got
    ev_v, ev_e = sv < tau, se < tau
    rv, re_ = np.arange(len(F.Cv)), np.arange(len(F.Ce))
    if second < 0:
        hv, he = F.Hv.copy(), F.He.copy()
        hv[:, first] = -np.inf
        he[:, first] = -np.inf
        tv, te = hv.argmax(1), he.argmax(1)
    else:
        tv, te = np.full(len(rv), second), np.full(len(re_), second)
    p_val = np.where(ev_v, F.Cv[rv, tv], F.Cv[:, first])
    p_ev = np.where(ev_e, F.Ce[re_, te], F.Ce[:, first])
    return finish(F, p_val, p_ev, F.Le[:, first] + ev_e * F.Le[re_, te] + F.pe, 1.0 + ev_e)


def build_blocks():
    """(seed index, evaluation positions, Fitted) exactly as the original sweeps build them."""
    if PROTO == "B":
        TRAIN, TEST, _a, _v = V.load(with_test_labels=True)
        blocks = []
        for si, seed in enumerate(SEEDS):
            held = V.stratified_group_split(TRAIN, seed)
            blocks.append((si, np.arange(len(TEST)),
                           Fitted(TRAIN.loc[~held].reset_index(drop=True),
                                  TRAIN.loc[held].reset_index(drop=True), TEST, seed)))
            log(f"fitted seed {seed}")
        return TEST["label_supported"].to_numpy(int), blocks
    _tr, _te, ALL, _v = V.load(with_test_labels=True)
    ALL = ALL.reset_index(drop=True)
    blocks = []
    for si, seed in enumerate(SEEDS):
        fold = V.folds_stratified(ALL, seed)
        for test_fold, val_fold, train_folds in V.rotations():
            m_fit, m_val, m_ev = np.isin(fold, train_folds), fold == val_fold, fold == test_fold
            blocks.append((si, np.flatnonzero(m_ev),
                           Fitted(ALL.loc[m_fit].reset_index(drop=True),
                                  ALL.loc[m_val].reset_index(drop=True),
                                  ALL.loc[m_ev].reset_index(drop=True), seed)))
        log(f"fitted seed {seed} ({len(blocks)} blocks)")
    return ALL["label_supported"].to_numpy(int), blocks


def main():
    if PROTO == "A":  # as in p0_cascade_a_v1: Protocol A selected its own hyperparameters on TRAIN
        CB.HP_B = dict(json.loads((HERE / "artifacts" / "part1c_main_full_v1" / "00_contract" /
                                   "HP_SELECTED.json").read_text())["hyperparameters_A"])
    y, blocks = build_blocks()
    n, S = len(y), len(SEEDS)
    gate_thr = []

    def agg(fn, check_thr=False):
        P, MS, THR, CA = np.zeros((n, S)), np.zeros((n, S)), np.zeros((n, S)), []
        for si, pos, F in blocks:
            r = fn(F)
            if r is None:
                return None
            q_ev, ms, calls, q_val = r
            thr = rule_mcc_fast(q_val, F.yv)
            if check_thr:
                ref = rule_mcc_ref(q_val, F.yv)
                gate_thr.append(abs(thr - ref))
                if thr != ref:
                    raise AssertionError(f"G1: fast MCC threshold {thr} != part1c rule {ref}")
            P[pos, si], MS[pos, si], THR[pos, si] = q_ev, ms, thr
            CA.append(calls)
        per = [core.metrics(y, P[:, s], MS[:, s], threshold=THR[:, s], with_risk=False)
               for s in range(S)]
        out = {k: float(np.nanmean([m[k] for m in per])) for k in KEYS}
        out["calls"] = float(np.mean(CA))
        return out

    rows = []
    for bi, beta in enumerate(GRID):
        m = agg(lambda F, b=beta: router_full(F, b), check_thr=(bi == 0))
        rows.append({"family": "router (pre-call)", "config": f"beta={beta}", **m})
    log("router grid done; G1 passed on every split of the first configuration")

    pairs = list(itertools.permutations(range(len(POOL)), 2)) + [(f, -1) for f in range(len(POOL))]
    for signal in ("confidence", "learned", "learned+score"):
        for first, second in pairs:
            for qi in TAU_Q:
                qs = [F.signals(first, signal) for _si, _p, F in blocks]
                if any(q is None for q in qs):
                    continue
                tau = float(np.mean([np.quantile(q[0], qi) for q in qs]))
                m = agg(lambda F, f=first, s=second, t=tau, g=signal: cascade_full(F, f, s, t, g))
                if m is None:
                    continue
                rows.append({"family": f"cascade ({signal})",
                             "config": (f"{NICE[POOL[first]]}->"
                                        f"{'best remaining' if second < 0 else NICE[POOL[second]]}"
                                        f" q={qi:.2f}"), **m})
        log(f"swept {signal}")

    full = pd.DataFrame(rows)
    orig = pd.read_csv(HERE / "results_p0exp" / ("E5_cascade_sweep.csv" if PROTO == "B"
                                                 else "E5_cascade_sweep_A.csv"))
    if len(orig) != len(full) or not (orig["config"].values == full["config"].values).all():
        raise AssertionError("G2: configuration list differs from the original sweep")
    g2 = {k: float(np.max(np.abs(orig[k].to_numpy(float) - full[k].to_numpy(float))))
          for k in ("auroc", "ms", "calls")}
    if max(g2.values()) > 1e-9:
        raise AssertionError(f"G2: does not reproduce the original sweep: {g2}")
    full.to_csv(OUT / f"{PROTO}_CASCADE_SWEEP_FULL.csv", index=False)

    front = []
    for fam in full["family"].unique():
        sub = full[full.family == fam]
        for budget in BUDGETS[PROTO]:
            ok = sub[sub.ms <= budget]
            if len(ok):
                best = ok.loc[ok.auroc.idxmax()]
                front.append({"family": fam, "budget_ms": budget, **best.drop("family").to_dict()})
    pd.DataFrame(front).to_csv(OUT / f"{PROTO}_CASCADE_FRONTIER.csv", index=False)
    (OUT / f"{PROTO}_FRONT_GATES.json").write_text(json.dumps(
        {"G1_thr_checked_splits": len(gate_thr), "G1_max_abs_diff": max(gate_thr),
         "G2_max_abs_diff": g2, "configs": len(full)}, indent=2))
    log(f"{PROTO} gates PASS: G2 {g2}")
    print(pd.DataFrame(front).to_string(index=False))


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"done in {time.time() - t0:.0f}s")
