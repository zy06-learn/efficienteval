#!/usr/bin/env python3
"""Matched-quality efficiency: EfficientEval at every point of its frozen beta grid, end to end.

The appendix beta sweep came from a separate re-implementation that charges the verifier call
and the feature extractors only, so its points are not on the main table's footing. This
replays the part1c fitting path itself -- same splits, Platt, heads, cost vector, stage-2
isotonic and MCC threshold -- with beta forced to each grid value instead of chosen on
validation.

Gate, per fit: at the beta that fit selected on validation, the replay must reproduce part1c's
stored stage-2 probabilities, MCC threshold, routing decisions and end-to-end latency to 1e-9,
and the selected beta must equal the stored ledger. Then, per protocol, the replayed deployed
router must reproduce the published OURS row of {A,B}_MAIN.csv to 1e-9.

Latency per row = selected verifier call + both cheap feature extractors + the head, routing,
Platt and isotonic overheads. The verifier call and the features are read from data, so they
are exact at every beta. The four overheads are wall-clock and do not depend on beta -- the
heads, the routing arithmetic and both calibration maps run identically at every grid point --
so they are taken from part1c's measurement of the same fit, which is what the main table
charges.

Stages: B | A | all.  Output: $V3_RUN_DIR/{A,B}_BETA_GRID.csv, {A,B}_PER_SEED.csv,
{A,B}_DEPLOYED.csv, GATES_{A,B}.json, rows_{A,B}.npz.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import part1c_main_full_v1 as P                    # noqa: E402  (reads V3_RUN_DIR)

V, core = P.V, P.core
OUT = Path(os.environ['V3_RUN_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
ART = HERE / 'artifacts' / 'part1c_main_full_v1'
HPS = json.loads((ART / '00_contract' / 'HP_SELECTED.json').read_text())
GRID = tuple(float(b) for b in V.BETA_GRID)
GATE = 1e-9
REPORT = ['auroc', 'auroc_sd', 'auprc_unsup', 'bacc', 'f1_unsup', 'mcc', 'ms']


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def replay(fit, val, ev, seed, hp):
    """One fit, every beta. Mirrors part1c.router_raw + _stage2 (MCC rule only)."""
    cals = core.platt(fit, actions=P.POOL)
    c_fit = core.apply_platt(fit, cals, actions=P.POOL)
    c_val = core.apply_platt(val, cals, actions=P.POOL)
    c_ev = core.apply_platt(ev, cals, actions=P.POOL)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=P.POOL,
                                 features=P.FEATURES, hp=hp, target=P.TARGET,
                                 learner=P.LEARNER)
    cvec = np.array([core.fold_costs(fit, actions=P.POOL)[a] for a in P.POOL], float)
    chosen, _ledger = V.choose_beta(val, h_val, c_val, P.POOL, cvec)
    y_val = val['label_supported'].to_numpy(int)
    out = {}
    for b in GRID:
        _s, p_val, _m = V.route(val, h_val, c_val, b, P.POOL, cvec)
        sel, p_ev, _m = V.route(ev, h_ev, c_ev, b, P.POOL, cvec)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(p_val, y_val)
        q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
        q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
        out[b] = {'q': q_ev, 'thr': float(P._rule_mcc(q_val, y_val, None)),
                  'sel': np.asarray(sel, int), 'ver': P._verifier_ms(ev, P.POOL, sel)}
    return float(chosen), out


def end_to_end(ver, pre, c):
    # same association order as part1c._assemble
    return ver + pre + c['heads'] + c['routing'] + (c['platt'] + c['isotonic'])


def aggregate(y, Q, MS, THR, SEL):
    """Per-beta metrics, averaged over seeds exactly as the main table (part1c._agg)."""
    rows, per_seed = [], []
    for b in GRID:
        m = P._agg(y, Q[b], MS[b], THR[b])
        shares = {f'share__{a}': float((SEL[b] == i).mean()) for i, a in enumerate(P.POOL)}
        rows.append({'beta': b, **{k: m[k] for k in REPORT}, **shares})
        for s in range(Q[b].shape[1]):
            per_seed.append({'beta': b, 'seed': P.SEEDS[s],
                             'auroc': float(roc_auc_score(y, Q[b][:, s])),
                             'ms': float(MS[b][:, s].mean())})
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


def gate_fit(tag, chosen, want_beta, out, ms_row, z, col, rows=None):
    """Replay at the selected beta against part1c's stored arrays for this fit."""
    r = slice(None) if rows is None else rows
    o = out[chosen]
    d = {'beta': abs(chosen - want_beta),
         'prob': float(np.max(np.abs(o['q'] - z['prob__OURS'][r, col]))),
         'thr': float(np.max(np.abs(o['thr'] - z['thr_mcc__OURS'][r, col]))),
         'ms': float(np.max(np.abs(ms_row - z['ms_full__OURS'][r, col]))),
         'sel': int(np.sum(o['sel'] != z['sel__OURS'][r, col]))}
    ok = d['beta'] == 0 and d['sel'] == 0 and max(d['prob'], d['thr'], d['ms']) <= GATE
    if not ok:
        raise AssertionError(f'{tag}: replay does not reproduce part1c at the selected beta: {d}')
    return d


def gate_main(protocol, y, Qd, MSd, THRd):
    """The replayed deployed router must land on the published OURS row."""
    got = P._agg(y, Qd, MSd, THRd)
    main = pd.read_csv(ART / '01_main_tables' / 'publication' / f'{protocol}_MAIN.csv')
    want = main[main.system == 'OURS'].iloc[0]
    diff = {k: abs(float(got[k]) - float(want[k]))
            for k in ('auroc', 'auprc_unsup', 'bacc', 'f1_unsup', 'mcc', 'ms')}
    if max(diff.values()) > GATE:
        raise AssertionError(f'{protocol}: deployed replay misses the published row: {diff}')
    return got, diff


def run_b():
    log('===== Protocol B =====')
    hp = dict(HPS['hyperparameters_B'])
    TRAIN, TEST, _ALL, _v = V.load(with_test_labels=True)
    y = TEST['label_supported'].to_numpy(int)
    n, S = len(TEST), len(P.SEEDS)
    pre = P._pre_call_feature_ms(TEST)
    z = np.load(ART / '10_row_level' / 'probs_B.npz', allow_pickle=False)
    if not np.array_equal(z['episode_key'].astype(str), TEST['episode_key'].astype(str).to_numpy()):
        raise AssertionError('B: stored row order differs from the loaded TEST frame')
    comp = pd.read_csv(ART / '02_latency' / 'B_LATENCY_COMPONENTS.csv')
    comp = comp[comp.system == 'OURS'].set_index('seed')
    chosen_ref = pd.read_csv(ART / '04_beta_evidence' / 'B_BETA_SELECTED.csv').set_index('seed')

    Q = {b: np.zeros((n, S)) for b in GRID}
    MS = {b: np.zeros((n, S)) for b in GRID}
    THR = {b: np.zeros((n, S)) for b in GRID}
    SEL = {b: np.zeros((n, S), np.int8) for b in GRID}
    Qd, MSd, THRd = np.zeros((n, S)), np.zeros((n, S)), np.zeros((n, S))
    fit_gates = []
    for si, seed in enumerate(P.SEEDS):
        t0 = time.time()
        held = V.stratified_group_split(TRAIN, seed)
        fit = TRAIN.loc[~held].reset_index(drop=True)
        val = TRAIN.loc[held].reset_index(drop=True)
        chosen, out = replay(fit, val, TEST, seed, hp)
        c = comp.loc[seed]
        for b in GRID:
            Q[b][:, si], THR[b][:, si] = out[b]['q'], out[b]['thr']
            SEL[b][:, si] = out[b]['sel']
            MS[b][:, si] = end_to_end(out[b]['ver'], pre, c)
        Qd[:, si], THRd[:, si], MSd[:, si] = Q[chosen][:, si], THR[chosen][:, si], MS[chosen][:, si]
        d = gate_fit(f'B seed {seed}', chosen, float(chosen_ref.loc[seed, 'beta']), out,
                     MS[chosen][:, si], z, si)
        fit_gates.append({'seed': seed, 'beta': chosen, **{f'd_{k}': v for k, v in d.items()}})
        log(f'  seed {seed}: selected beta {chosen} reproduced (max d prob {d["prob"]:.1e}, '
            f'ms {d["ms"]:.1e}) ({time.time() - t0:.0f}s)')
    return finish('B', y, Q, MS, THR, SEL, Qd, MSd, THRd, fit_gates)


def run_a():
    log('===== Protocol A =====')
    hp = dict(HPS['hyperparameters_A'])
    _TR, _TE, ALL, _v = V.load(with_test_labels=True)
    n, S = len(ALL), len(P.SEEDS)
    z = np.load(ART / '10_row_level' / 'probs_A.npz', allow_pickle=False)
    ek_ref = z['episode_key'].astype(str)
    comp = pd.read_csv(ART / '02_latency' / 'A_LATENCY_COMPONENTS.csv')
    comp = comp[comp.system == 'OURS'].set_index(['seed', 'rotation'])
    chosen_ref = (pd.read_csv(ART / '04_beta_evidence' / 'A_BETA_SELECTED.csv')
                  .set_index(['seed', 'rotation']))
    pos_of = {k: i for i, k in enumerate(ek_ref)}

    Q = {b: np.zeros((n, S)) for b in GRID}
    MS = {b: np.zeros((n, S)) for b in GRID}
    THR = {b: np.zeros((n, S)) for b in GRID}
    SEL = {b: np.zeros((n, S), np.int8) for b in GRID}
    Qd, MSd, THRd = np.zeros((n, S)), np.zeros((n, S)), np.zeros((n, S))
    y = z['y'].astype(int)
    fit_gates = []
    for si, seed in enumerate(P.SEEDS):
        t0 = time.time()
        fold = V.folds_stratified(ALL, seed)
        seen = np.zeros(n, bool)
        for test_fold, val_fold, train_folds in V.rotations():
            fit = ALL.loc[np.isin(fold, train_folds)].reset_index(drop=True)
            val = ALL.loc[fold == val_fold].reset_index(drop=True)
            ev = ALL.loc[fold == test_fold].reset_index(drop=True)
            rows = np.array([pos_of[k] for k in ev['episode_key'].astype(str)])
            if seen[rows].any():
                raise AssertionError(f'A seed {seed}: a row is evaluated twice')
            seen[rows] = True
            if not np.array_equal(y[rows], ev['label_supported'].to_numpy(int)):
                raise AssertionError(f'A seed {seed}: label mismatch against stored order')
            chosen, out = replay(fit, val, ev, seed, hp)
            c = comp.loc[(seed, test_fold)]
            pre = P._pre_call_feature_ms(ev)
            for b in GRID:
                Q[b][rows, si], THR[b][rows, si] = out[b]['q'], out[b]['thr']
                SEL[b][rows, si] = out[b]['sel']
                MS[b][rows, si] = end_to_end(out[b]['ver'], pre, c)
            Qd[rows, si] = Q[chosen][rows, si]
            THRd[rows, si] = THR[chosen][rows, si]
            MSd[rows, si] = MS[chosen][rows, si]
            d = gate_fit(f'A seed {seed} rotation {test_fold}', chosen,
                         float(chosen_ref.loc[(seed, test_fold), 'beta']), out,
                         MS[chosen][rows, si], z, si, rows)
            fit_gates.append({'seed': seed, 'rotation': test_fold, 'beta': chosen,
                              **{f'd_{k}': v for k, v in d.items()}})
        if not seen.all():
            raise AssertionError(f'A seed {seed}: {int((~seen).sum())} rows never evaluated')
        log(f'  seed {seed}: 10 rotations reproduced ({time.time() - t0:.0f}s)')
    return finish('A', y, Q, MS, THR, SEL, Qd, MSd, THRd, fit_gates)


def finish(protocol, y, Q, MS, THR, SEL, Qd, MSd, THRd, fit_gates):
    got, diff = gate_main(protocol, y, Qd, MSd, THRd)
    log(f'  deployed replay vs published OURS row: max diff {max(diff.values()):.1e}')
    grid, per_seed = aggregate(y, Q, MS, THR, SEL)
    grid.to_csv(OUT / f'{protocol}_BETA_GRID.csv', index=False)
    per_seed.to_csv(OUT / f'{protocol}_PER_SEED.csv', index=False)
    pd.DataFrame([{'system': 'OURS (validation-selected beta)',
                   **{k: float(got[k]) for k in REPORT}}]).to_csv(
        OUT / f'{protocol}_DEPLOYED.csv', index=False)
    pd.DataFrame(fit_gates).to_csv(OUT / f'{protocol}_FIT_GATES.csv', index=False)
    np.savez_compressed(OUT / f'rows_{protocol}.npz', y=y.astype(np.int8),
                        betas=np.array(GRID), seeds=np.array(P.SEEDS),
                        **{f'q__{b}': Q[b] for b in GRID}, **{f'ms__{b}': MS[b] for b in GRID},
                        **{f'thr__{b}': THR[b] for b in GRID},
                        **{f'sel__{b}': SEL[b] for b in GRID})
    fg = pd.DataFrame(fit_gates)
    (OUT / f'GATES_{protocol}.json').write_text(json.dumps({
        'protocol': protocol, 'tolerance': GATE, 'fits': len(fg),
        'max_d_prob': float(fg.d_prob.max()), 'max_d_thr': float(fg.d_thr.max()),
        'max_d_ms': float(fg.d_ms.max()), 'sel_mismatches': int(fg.d_sel.sum()),
        'beta_mismatches': int((fg.d_beta > 0).sum()),
        'deployed_vs_published': diff, 'pass': True}, indent=2))
    print(grid.to_string(index=False, float_format=lambda v: f'{v:.4f}'), flush=True)
    return grid


if __name__ == '__main__':
    want = sys.argv[1:] or ['all']
    stages = ['B', 'A'] if want == ['all'] else want
    for st in stages:
        t0 = time.time()
        {'B': run_b, 'A': run_a}[st]()
        log(f'===== {st} done in {time.time() - t0:.0f}s =====')
    (OUT / 'DONE.json').write_text(json.dumps({'stages': stages, 'status': 'complete',
                                               'finished': time.strftime('%Y-%m-%d %H:%M:%S')}))
