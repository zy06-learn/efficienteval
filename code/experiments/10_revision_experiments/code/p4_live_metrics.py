#!/usr/bin/env python3
"""Every main-table metric for the router, recomputed on the live re-execution.

The live runs (runs/live_main_v1 for Protocol B, runs/live_main_a_v1 for Protocol A) stored the
routing plan (plan.npz: selected action per row and seed) and the score the selected verifier
actually returned (live_raw.npz). Their finish stage reported AUROC only. This script refits
everything exactly as part1c and live_main do -- Platt, heads, beta, routing, isotonic, the MCC
threshold -- and swaps only the test-side score of the selected verifier for its live value.

Gates, all checked on the matrix side before any live number is trusted:
  G1  routing recomputed here == live plan == part1c sel__OURS
  G2  stage-2 probabilities == part1c prob__OURS          (max abs diff <= 1e-12)
  G3  MCC thresholds == part1c thr_mcc__OURS             (max abs diff <= 1e-12)
  G4  aggregate metrics == part1c {A,B}_MAIN.csv OURS    (abs diff <= 1e-9)

Latency is part1c's per-row end-to-end ms_full for the same selections (G1 makes it the
latency of exactly the verifiers the live run called).

Output: results_p4live/{A,B}_LIVE_METRICS.csv, {A,B}_LIVE_PER_SEED.csv, {A,B}_LIVE_PROBS.npz
Usage:  p4_live_metrics.py B|A
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as V  # noqa: E402
import core  # noqa: E402

P1C = HERE / "artifacts" / "part1c_main_full_v1"
HP = json.loads((P1C / "00_contract" / "HP_SELECTED.json").read_text())
FEATURES = json.loads((P1C / "00_contract" / "INHERITED_FROZEN_v3.json").read_text())["features"]
POOL = list(core.ACTIONS)
SEEDS = list(V.SEEDS)
OUT = HERE / "results_p4live"
OUT.mkdir(exist_ok=True)
KEYS = ["auroc", "auprc_unsup", "acc", "bacc", "mcc", "f1_macro", "f1_unsup", "prec_unsup",
        "rec_unsup", "ece", "mce", "brier", "aurc", "ms", "p50_ms", "p95_ms", "p99_ms", "qps"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rule_mcc(q, y):
    """part1c_main_full_v1._rule_mcc, verbatim."""
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def run_one(fit, val, ev, seed, hp, raw_ev, sel_plan):
    """One (fit, val, eval) triple: matrix-side and live-side stage-2 outputs."""
    cals = core.platt(fit, actions=POOL)
    c_fit = core.apply_platt(fit, cals, actions=POOL)
    c_val = core.apply_platt(val, cals, actions=POOL)
    c_ev = core.apply_platt(ev, cals, actions=POOL)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=hp, target="regret", learner="rf")
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
    sel, p_ev_mat, _m = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
    sel = np.asarray(sel, int)
    if not np.array_equal(sel, np.asarray(sel_plan, int)):
        raise AssertionError(f"G1 seed {seed}: recomputed routing differs from the live plan")

    ev_live = ev.copy()
    for k, a in enumerate(POOL):
        j = np.where(sel == k)[0]
        col = ev_live[f"score__{a}"].to_numpy(float).copy()
        col[j] = raw_ev[j]
        ev_live[f"score__{a}"] = col
    c_live = core.apply_platt(ev_live, cals, actions=POOL)
    rows = np.arange(len(ev))
    p_ev_live = np.column_stack([c_live[a] for a in POOL])[rows, sel]

    y_val = val["label_supported"].to_numpy(int)
    q_val = core.isotonic(p_val, y_val, p_val)
    return {"sel": sel, "thr": rule_mcc(q_val, y_val),
            "q_mat": core.isotonic(p_val, y_val, p_ev_mat),
            "q_live": core.isotonic(p_val, y_val, p_ev_live),
            "n_changed": int(np.sum(p_ev_live != p_ev_mat)),
            "max_abs_dp1": float(np.max(np.abs(p_ev_live - p_ev_mat)))}


def aggregate(y, Q, MS, THR):
    """part1c_main_full_v1._agg, plus a per-metric across-seed SD."""
    per = [core.metrics(y, Q[:, s], MS[:, s], threshold=THR[:, s]) for s in range(Q.shape[1])]
    out = {k: float(np.nanmean([r[k] for r in per])) for k in per[0]}
    for k in KEYS:
        out[f"{k}_sd"] = float(np.std([r[k] for r in per]))
    out["auroc_sd"] = float(np.std([r["auroc"] for r in per]))
    return out, per


def gate_close(name, got, want, tol):
    d = float(np.max(np.abs(np.asarray(got, float) - np.asarray(want, float))))
    if not d <= tol:
        raise AssertionError(f"{name}: max abs diff {d:.3e} > {tol:.0e}")
    return d


def finish(protocol, y, Q_live, Q_mat, THR, SEL, ref, stats):
    gates = {
        "G1_sel": bool(np.array_equal(SEL, ref["sel__OURS"].astype(int))),
        "G2_prob_maxdiff": gate_close("G2", Q_mat, ref["prob__OURS"], 1e-12),
        "G3_thr_maxdiff": gate_close("G3", THR, ref["thr_mcc__OURS"], 1e-12),
    }
    if not gates["G1_sel"]:
        raise AssertionError("G1: live selections differ from part1c sel__OURS")
    MS = ref["ms_full__OURS"]
    mat, _ = aggregate(y, Q_mat, MS, THR)
    live, per_live = aggregate(y, Q_live, MS, THR)

    main = pd.read_csv(P1C / "01_main_tables" / "publication" / f"{protocol}_MAIN.csv")
    want = main[main.system == "OURS"].iloc[0]
    g4 = {k: abs(mat[k] - float(want[k])) for k in KEYS if k in want and pd.notna(want[k])}
    gates["G4_main_maxdiff"] = max(g4.values())
    if gates["G4_main_maxdiff"] > 1e-9:
        bad = {k: v for k, v in g4.items() if v > 1e-9}
        raise AssertionError(f"G4: matrix side does not reproduce {protocol}_MAIN.csv: {bad}")

    table = pd.DataFrame([{"system": "OURS_live", **live}, {"system": "OURS_matrix", **mat}])
    table.to_csv(OUT / f"{protocol}_LIVE_METRICS.csv", index=False)
    pd.DataFrame([{"seed": s, **r} for s, r in zip(SEEDS, per_live)]).to_csv(
        OUT / f"{protocol}_LIVE_PER_SEED.csv", index=False)
    np.savez_compressed(OUT / f"{protocol}_LIVE_PROBS.npz", y=y.astype(np.int8), q_live=Q_live,
                        q_matrix=Q_mat, thr_mcc=THR, sel=SEL.astype(np.int8), ms_full=MS,
                        episode_key=ref["episode_key"], corpus=ref["corpus"], seeds=np.array(SEEDS))
    (OUT / f"{protocol}_GATES.json").write_text(json.dumps({**gates, **stats}, indent=2))
    log(f"{protocol} gates PASS: {gates}")
    for k in ("auroc", "auprc_unsup", "acc", "bacc", "mcc", "f1_macro", "f1_unsup", "ece",
              "brier", "aurc", "ms", "p95_ms"):
        log(f"  {k:12s} live {live[k]:.6f}   matrix {mat[k]:.6f}   d {live[k]-mat[k]:+.6f}")


def proto_b():
    TRAIN, TEST, _all, _v = V.load(with_test_labels=True)
    ev = TEST.reset_index(drop=True)
    n, S = len(ev), len(SEEDS)
    run = HERE / "runs" / "live_main_v1"
    plan, raw = np.load(run / "plan.npz")["sel"], np.load(run / "live_raw.npz")["raw"]
    ref = np.load(P1C / "10_row_level" / "probs_B.npz", allow_pickle=False)
    if not np.array_equal(ref["episode_key"].astype(str), ev["episode_key"].astype(str).to_numpy()):
        raise AssertionError("row order of part1c probs_B differs from TEST")
    y = ev["label_supported"].to_numpy(int)
    Q_live, Q_mat, THR = (np.zeros((n, S)) for _ in range(3))
    SEL = np.zeros((n, S), int)
    changed, maxd = 0, 0.0
    for si, seed in enumerate(SEEDS):
        held = V.stratified_group_split(TRAIN, seed)
        fit = TRAIN.loc[~held].reset_index(drop=True)
        val = TRAIN.loc[held].reset_index(drop=True)
        r = run_one(fit, val, ev, seed, HP["hyperparameters_B"], raw[:, si], plan[:, si])
        Q_live[:, si], Q_mat[:, si], THR[:, si], SEL[:, si] = r["q_live"], r["q_mat"], r["thr"], r["sel"]
        changed, maxd = changed + r["n_changed"], max(maxd, r["max_abs_dp1"])
        log(f"B seed {seed}: thr {r['thr']:.4f}  rows whose stage-1 prob changed {r['n_changed']}")
    finish("B", y, Q_live, Q_mat, THR, SEL, ref,
           {"live_calls": int(np.isfinite(raw).sum()), "rows_changed": changed,
            "max_abs_dp1": maxd})


def proto_a():
    _tr, _te, ALL, _v = V.load(with_test_labels=True)
    ALL = ALL.reset_index(drop=True)
    n, S = len(ALL), len(SEEDS)
    run = HERE / "runs" / "live_main_a_v1"
    plan, raw = np.load(run / "plan.npz")["sel"], np.load(run / "live_raw.npz")["raw"]
    ref = np.load(P1C / "10_row_level" / "probs_A.npz", allow_pickle=False)
    ek = ALL["episode_key"].astype(str).to_numpy()
    if len(set(ek.tolist())) != n:
        raise AssertionError("episode keys are not unique")
    order = np.argsort(ek, kind="stable")
    if not np.array_equal(ref["episode_key"].astype(str), ek[order]):
        raise AssertionError("part1c probs_A canonical order is not the sorted episode key")
    y = ALL["label_supported"].to_numpy(int)[order]
    Q_live, Q_mat, THR = (np.zeros((n, S)) for _ in range(3))
    SEL = np.zeros((n, S), int)
    changed, maxd = 0, 0.0
    for si, seed in enumerate(SEEDS):
        t0 = time.time()
        fold = V.folds_stratified(ALL, seed)
        ql, qm, th, sl = (np.full(n, np.nan) for _ in range(4))
        for test_fold, val_fold, train_folds in V.rotations():
            m_ev = fold == test_fold
            fit = ALL.loc[np.isin(fold, train_folds)].reset_index(drop=True)
            val = ALL.loc[fold == val_fold].reset_index(drop=True)
            ev = ALL.loc[m_ev].reset_index(drop=True)
            idx = np.where(m_ev)[0]
            r = run_one(fit, val, ev, seed, HP["hyperparameters_A"], raw[idx, si], plan[idx, si])
            ql[idx], qm[idx], th[idx], sl[idx] = r["q_live"], r["q_mat"], r["thr"], r["sel"]
            changed, maxd = changed + r["n_changed"], max(maxd, r["max_abs_dp1"])
        if not np.isfinite(sl).all():
            raise AssertionError(f"A seed {seed}: a row was never evaluated")
        Q_live[:, si], Q_mat[:, si], THR[:, si] = ql[order], qm[order], th[order]
        SEL[:, si] = sl[order].astype(int)
        log(f"A seed {seed}: {time.time() - t0:.0f}s")
    finish("A", y, Q_live, Q_mat, THR, SEL, ref,
           {"live_calls": int(np.isfinite(raw).sum()), "rows_changed": changed,
            "max_abs_dp1": maxd})


if __name__ == "__main__":
    {"B": proto_b, "A": proto_a}[sys.argv[1]]()
