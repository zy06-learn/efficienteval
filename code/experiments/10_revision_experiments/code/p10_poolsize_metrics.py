#!/usr/bin/env python3
"""Appendix J (tab:poolprov): complete the pool-size sweep to the Table-1 metric set.

The published E1_poolsize tables report pooled AUROC and mean latency only, because run_arm
never fitted a classification threshold. This replays the identical fitting path for both
protocols, reproduces the published AUROC and ms cell by cell, and only then emits AUPRC, BAcc
and F1.

The greedy pool order is read from the published E1_greedy_path.csv, exactly as the Protocol A
script already does, so no selection is redone and nothing new touches a test label.

Threshold rule follows the global convention: MCC-maximising on the isotonic-calibrated
validation scores. Under Protocol A that is per rotation, matching the main pipeline.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as _V                                # noqa: E402  (sets sys.path for core)
import core                                        # noqa: E402
import p0_exp_v1 as PB                             # noqa: E402
import p0_exp_a_v1 as PA                           # noqa: E402

OUT = Path(os.environ['V3_RUN_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ['auroc', 'auprc_unsup', 'bacc', 'f1_unsup', 'ms']
GATE = 1e-9
KS = (1, 2, 3, 5, 8, 11, 15)


def mcc_threshold(q, y):
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def stage2_with_threshold(p_val, y_val, p_ev):
    """The published stage2, plus the threshold it never computed."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(p_val, y_val)
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    return q_ev, mcc_threshold(q_val, y_val)


def metrics_of(y, q, ms, thr):
    r = core.metrics(y, q, ms, threshold=thr, with_risk=False)
    return {k: float(r[k]) for k in METRICS}


def run_b(actions):
    """Protocol B: one split per seed, one threshold per seed."""
    per = []
    for seed, fit, val, ev in PB.protob_splits():
        cals = core.platt(fit, actions=actions)
        c_fit = core.apply_platt(fit, cals, actions=actions)
        c_val = core.apply_platt(val, cals, actions=actions)
        c_ev = core.apply_platt(ev, cals, actions=actions)
        h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=actions,
                                     features=PB.FEATURES, hp=PB.HP_B, target='regret',
                                     learner='rf')
        cost = core.fold_costs(fit, actions=actions)
        cv = np.array([cost[a] for a in actions], float)
        cn = cv / cv.max()

        def route(frame, heads, cal, mult):
            avail = np.column_stack([frame[f'available__{a}'].astype(bool).to_numpy()
                                     for a in actions])
            util = np.asarray(heads, float) * np.exp(-mult * cn)
            sel = np.argmax(np.where(avail, util, -np.inf), axis=1)
            r = np.arange(len(frame))
            lat = np.column_stack([frame[f'latency_ms__{a}'].fillna(0).to_numpy(float)
                                   for a in actions])[r, sel]
            return np.column_stack([cal[a] for a in actions])[r, sel], lat + PB.pre_ms(frame)

        y_val = val['label_supported'].to_numpy(int)
        scored = []
        for m in PB.GRID:
            p, ms = route(val, h_val, c_val, m)
            scored.append((m, PB.au(y_val, p), float(ms.mean())))
        mult = PB.select_from_grid(scored)
        p_val, _ = route(val, h_val, c_val, mult)
        p_ev, ms_ev = route(ev, h_ev, c_ev, mult)
        q, thr = stage2_with_threshold(p_val, y_val, p_ev)
        per.append(metrics_of(ev['label_supported'].to_numpy(int), q, ms_ev, thr))
    f = pd.DataFrame(per)
    return {m: float(f[m].mean()) for m in METRICS}


def run_a(actions):
    """Protocol A: ten rotations per seed; each test fold keeps its own rotation threshold."""
    frame = PA.load_all()
    n = len(frame)
    y = frame['label_supported'].to_numpy(int)
    Q = np.zeros((n, len(PA.SEEDS)))
    MS = np.zeros((n, len(PA.SEEDS)))
    THR = np.zeros((n, len(PA.SEEDS)))
    for si, seed in enumerate(PA.SEEDS):
        for fit, val, ev, pos in PA.rotations_for(seed):
            cals = core.platt(fit, actions=actions)
            c_fit = core.apply_platt(fit, cals, actions=actions)
            c_val = core.apply_platt(val, cals, actions=actions)
            c_ev = core.apply_platt(ev, cals, actions=actions)
            h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=actions,
                                         features=PA.FEATURES, hp=PA.HP_A, target='regret',
                                         learner='rf')
            cost = core.fold_costs(fit, actions=actions)
            cn = np.array([cost[a] for a in actions], float)
            cn = cn / cn.max()

            def go(f_, heads, cal, mult):
                avail = np.column_stack([f_[f'available__{a}'].astype(bool).to_numpy()
                                         for a in actions])
                util = np.asarray(heads, float) * np.exp(-mult * cn)
                sel = np.argmax(np.where(avail, util, -np.inf), axis=1)
                r = np.arange(len(f_))
                return (np.column_stack([cal[a] for a in actions])[r, sel],
                        PA.lat_matrix(f_, actions)[r, sel] + PA.pre_ms(f_))

            yv = val['label_supported'].to_numpy(int)
            scored = []
            for m in PA.GRID:
                p, ms = go(val, h_val, c_val, m)
                scored.append((m, PA.au(yv, p), float(ms.mean())))
            best = max(r[1] for r in scored)
            mult = min([r for r in scored if r[1] >= best - PA.EPS], key=lambda r: r[2])[0]
            p_val, _ = go(val, h_val, c_val, mult)
            p_ev, ms_ev = go(ev, h_ev, c_ev, mult)
            q, thr = stage2_with_threshold(p_val, yv, p_ev)
            Q[pos, si], MS[pos, si], THR[pos, si] = q, ms_ev, thr
    per = [metrics_of(y, Q[:, s], MS[:, s], THR[:, s]) for s in range(len(PA.SEEDS))]
    f = pd.DataFrame(per)
    return {m: float(f[m].mean()) for m in METRICS}


def main():
    path = pd.read_csv(PB.OUT / 'E1_greedy_path.csv')
    chosen = list(path['added'])
    arms = [(f'greedy k={k}', chosen[:k]) for k in KS] + [('frozen pool (paper)', list(PB.POOL))]
    rows, checks = [], []
    published = {'A': pd.read_csv(PB.OUT / 'E1_poolsize_A.csv').set_index('arm'),
                 'B': pd.read_csv(PB.OUT / 'E1_poolsize.csv').set_index('arm')}
    for label, actions in arms:
        for proto, fn in (('B', run_b), ('A', run_a)):
            got = fn(actions)
            ref = published[proto].loc[label]
            for col, key in (('auroc', 'pooled'), ('ms', 'ms')):
                diff = abs(got[col] - float(ref[key]))
                checks.append({'protocol': proto, 'arm': label, 'metric': col,
                               'published': float(ref[key]), 'recomputed': got[col],
                               'abs_diff': diff, 'pass': bool(diff <= GATE)})
            rows.append({'protocol': proto, 'arm': label, 'k': len(actions),
                         'actions': '+'.join(actions), **got})
            print(f'{proto} {label:22s} AUROC {got["auroc"]:.6f} '
                  f'(pub {float(ref["pooled"]):.6f})  ms {got["ms"]:.2f} '
                  f'(pub {float(ref["ms"]):.2f})  F1 {got["f1_unsup"]:.4f}', flush=True)
    bad = [c for c in checks if not c['pass']]
    pd.DataFrame(checks).to_csv(OUT / 'POOLSIZE_REPRODUCTION.csv', index=False)
    (OUT / 'POOLSIZE_GATE.json').write_text(json.dumps(
        {'gate': GATE, 'n_checks': len(checks), 'pass': not bad, 'failures': bad},
        indent=2, sort_keys=True, default=float))
    if bad:
        print(f'GATE FAILED on {len(bad)}/{len(checks)}; new metrics NOT published')
        raise SystemExit(1)
    pd.DataFrame(rows).to_csv(OUT / 'POOLSIZE_METRICS.csv', index=False)
    (OUT / 'DONE.json').write_text(json.dumps({'status': 'complete', 'checks': len(checks)}))
    print(f'gate passed on {len(checks)} published cells')


if __name__ == '__main__':
    main()
