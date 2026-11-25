#!/usr/bin/env python3
"""§6.4 / Appendix I: corpus-mixture sensitivity on the Table-1 metric set -- threshold fix.

Copy of p8_mixture_metrics.py (2026-09-17) with one bug fixed and one row group added.

Bug: p8 flattened the (n, seeds) MCC-threshold matrix and indexed it by seed, so every
subsample was classified with subsample 0's threshold. Under Protocol B one threshold is fitted
per seed, so nothing changed there; under Protocol A one is fitted per rotation (76 distinct
values for EfficientEval), so BAcc and F1 of every Protocol A row were computed with the wrong
threshold. Here each resampled subsample keeps its own threshold, thr[ii, s].

Added: mixture == 'natural_exact', the test frame itself with no resampling. It must equal the
Table-1 values in E01_main_part1c/{A,B}_MAIN.csv on every metric, which checks the fix.

Gates (1e-9), all must pass before anything is written:
  G1  every AUROC of the published E7 tables, as in p8
  G2  every threshold-free column of p8's MIXTURE_METRICS.csv (auroc, auprc_unsup, ms,
      ms_mixture), and under Protocol B also bacc and f1_unsup
  G3  natural_exact == E01 {A,B}_MAIN.csv for auroc, auprc_unsup, bacc, f1_unsup, ms
  G4  the Dirichlet summary == p8's MIXTURE_DIRICHLET.json
The resampling streams are the ones p8 used; natural_exact draws nothing.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
OUT = Path(os.environ['V3_RUN_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
OLD = HERE / 'results_p0exp'
P8 = HERE / 'runs' / 'p8_mixture_20260917'
E01 = ART / 'part1c_main_full_v1'
CORPORA = ('cogensumm', 'frank', 'ragtruth', 'unisumeval')
NDRAW, NREP = 200, 50
GATE = 1e-9


def au(y, p):
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 else float('nan')


def quality(y, p, thr):
    """AUROC/AUPRC ignore the threshold; BAcc/F1 use each subsample's own validation MCC
    threshold (one per seed under B, one per rotation under A)."""
    y = np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        return dict(auroc=np.nan, auprc_unsup=np.nan, bacc=np.nan, f1_unsup=np.nan)
    d = (np.asarray(p, float) >= np.asarray(thr, float)).astype(int)
    return dict(auroc=float(roc_auc_score(y, p)),
                auprc_unsup=float(average_precision_score((y == 0).astype(int), 1.0 - p)),
                bacc=float(balanced_accuracy_score(y, d)),
                f1_unsup=float(f1_score((y == 0).astype(int), (d == 0).astype(int),
                                        zero_division=0)))


def load(protocol):
    d = np.load(ART / 'part1c_main_full_v1' / '10_row_level' / f'probs_{protocol}.npz',
                allow_pickle=True)
    y = d['y']
    fam = np.array([str(c).split('_')[0] for c in d['corpus']])
    names, ms, thr = {'EfficientEval': d['prob__OURS']}, {}, {}
    ms['EfficientEval'] = d['ms_full__OURS']
    thr['EfficientEval'] = np.asarray(d['thr_mcc__OURS'], float)
    for s in d['systems']:
        s = str(s)
        if s.startswith('fixed::'):
            k = s.split('::', 1)[1]
            names[k] = d[f'prob__fixed__{k}']
            ms[k] = d[f'ms_full__fixed__{k}']
            thr[k] = np.asarray(d[f'thr_mcc__fixed__{k}'], float)
    for k, t in thr.items():
        assert t.shape == names[k].shape, (protocol, k, t.shape, names[k].shape)
    return y, fam, names, ms, thr


def run(protocol):
    y, fam, names, ms, thr = load(protocol)
    n_rows = len(y)
    idx = {c: np.flatnonzero(fam == c) for c in CORPORA}
    base = np.array([len(idx[c]) for c in CORPORA], float)
    base /= base.sum()

    def resample(w, r):
        return np.concatenate([r.choice(idx[c], size=max(int(round(n_rows * wi)), 2), replace=True)
                               for c, wi in zip(CORPORA, w)])

    rows = []
    for name, P in names.items():   # the test frame as it is, no resampling
        per = [quality(y, P[:, s], thr[name][:, s]) for s in range(P.shape[1])]
        rows.append({'protocol': protocol, 'mixture': 'natural_exact', 'system': name,
                     **{k: float(np.nanmean([q[k] for q in per]))
                        for k in ('auroc', 'auprc_unsup', 'bacc', 'f1_unsup')},
                     'ms': float(ms[name].mean()), 'ms_mixture': float(ms[name].mean())})
    for label, w in (('natural', base), ('equal', np.full(4, 0.25))):
        # One shared generator consumed in system order, exactly as the published script did.
        r = np.random.default_rng(1)
        for name, P in names.items():
            n_seeds = P.shape[1]
            acc = {k: [] for k in ('auroc', 'auprc_unsup', 'bacc', 'f1_unsup')}
            ms_mix = []
            for _ in range(NREP):
                ii = resample(w, r)
                per = [quality(y[ii], P[ii, s], thr[name][ii, s]) for s in range(n_seeds)]
                for k in acc:
                    acc[k].append(np.nanmean([q[k] for q in per]))
                ms_mix.append(float(np.mean(ms[name][ii])))
            rows.append({'protocol': protocol, 'mixture': label, 'system': name,
                         **{k: float(np.mean(v)) for k, v in acc.items()},
                         'ms': float(ms[name].mean()),
                         'ms_mixture': float(np.mean(ms_mix))})

    rng = np.random.default_rng(0)
    draws = [rng.dirichlet(np.ones(4)) for _ in range(NDRAW)]
    r = np.random.default_rng(2)
    hits, wins = 0, {k: 0 for k in names}
    for w in draws:
        ii = resample(w, r)
        a = {k: float(np.nanmean([au(y[ii], P[ii, s]) for s in range(P.shape[1])]))
             for k, P in names.items()}
        om = ms['EfficientEval'].mean()
        hits += not any(a[k] > a['EfficientEval'] and ms[k].mean() < om
                        for k in names if k != 'EfficientEval')
        wins[max(a, key=a.get)] += 1
    dirichlet = {'on_frontier': hits / NDRAW,
                 'best_auroc_share': {k: v / NDRAW for k, v in wins.items() if v}}
    return pd.DataFrame(rows), dirichlet


def gate(protocol, table, dirichlet):
    """Every published value must come back identical before a new metric is published."""
    old = pd.read_csv(OLD / ('E7_mixture.csv' if protocol == 'B' else 'E7_mixture_A.csv'))
    checks = []
    for label in ('natural', 'equal'):
        ref = old[old.mixture == label].iloc[0]
        for _, row in table[table.mixture == label].iterrows():
            for col, key in [('auroc', row.system)] + (
                    [('ms', row.system + '__ms')] if protocol == 'B' else []):
                if key not in old.columns or pd.isna(ref[key]):
                    continue
                diff = abs(float(row[col]) - float(ref[key]))
                checks.append({'protocol': protocol, 'mixture': label, 'system': row.system,
                               'metric': col, 'abs_diff': diff, 'pass': bool(diff <= GATE)})
    fr = old[old.mixture.astype(str).str.contains('on frontier')]
    if len(fr):
        diff = abs(dirichlet['on_frontier'] - float(fr.iloc[0]['EfficientEval']))
        checks.append({'protocol': protocol, 'mixture': 'dirichlet', 'system': 'EfficientEval',
                       'metric': 'on_frontier', 'abs_diff': diff, 'pass': bool(diff <= GATE)})
    sh = old[old.mixture.astype(str).str.contains('best-AUROC')]
    if len(sh):
        ref = sh.iloc[0]
        for k, v in dirichlet['best_auroc_share'].items():
            if k in old.columns and not pd.isna(ref[k]):
                diff = abs(v - float(ref[k]))
                checks.append({'protocol': protocol, 'mixture': 'dirichlet', 'system': k,
                               'metric': 'best_auroc_share', 'abs_diff': diff,
                               'pass': bool(diff <= GATE)})
    return checks


def gate_fix(protocol, table, dirichlet):
    """G2-G4: what the fix must leave alone, and the exact frame against Table 1."""
    checks = []

    def chk(mix, system, metric, got, ref):
        diff = abs(float(got) - float(ref))
        checks.append({'protocol': protocol, 'mixture': mix, 'system': system,
                       'metric': metric, 'abs_diff': diff, 'pass': bool(diff <= GATE)})

    old = pd.read_csv(P8 / 'MIXTURE_METRICS.csv')
    old = old[old.protocol == protocol].set_index(['mixture', 'system'])
    cols = ['auroc', 'auprc_unsup', 'ms', 'ms_mixture'] + (
        ['bacc', 'f1_unsup'] if protocol == 'B' else [])
    for _, row in table[table.mixture.isin(['natural', 'equal'])].iterrows():
        for c in cols:
            chk(row.mixture, row.system, 'G2:' + c, row[c], old.loc[(row.mixture, row.system), c])
    main = pd.read_csv(E01 / '01_main_tables' / 'publication' / f'{protocol}_MAIN.csv')
    main = main.set_index('system')
    for _, row in table[table.mixture == 'natural_exact'].iterrows():
        key = 'OURS' if row.system == 'EfficientEval' else 'fixed::' + row.system
        for c in ('auroc', 'auprc_unsup', 'bacc', 'f1_unsup', 'ms'):
            chk('natural_exact', row.system, 'G3:' + c, row[c], main.loc[key, c])
    ref = json.loads((P8 / 'MIXTURE_DIRICHLET.json').read_text())[protocol]
    chk('dirichlet', 'EfficientEval', 'G4:on_frontier', dirichlet['on_frontier'],
        ref['on_frontier'])
    assert set(dirichlet['best_auroc_share']) == set(ref['best_auroc_share']), protocol
    for k, v in dirichlet['best_auroc_share'].items():
        chk('dirichlet', k, 'G4:best_auroc_share', v, ref['best_auroc_share'][k])
    return checks


def main():
    all_checks, frames, dirs = [], [], {}
    for protocol in ('B', 'A'):
        table, dirichlet = run(protocol)
        all_checks += gate(protocol, table, dirichlet)
        all_checks += gate_fix(protocol, table, dirichlet)
        frames.append(table)
        dirs[protocol] = dirichlet
        print(f'{protocol}: {len(table)} rows', flush=True)
    bad = [c for c in all_checks if not c['pass']]
    (OUT / 'MIXTURE_GATE.json').write_text(json.dumps(
        {'gate': GATE, 'pass': not bad, 'n_checks': len(all_checks), 'failures': bad},
        indent=2, sort_keys=True, default=float))
    (OUT / 'MIXTURE_DIRICHLET.json').write_text(json.dumps(dirs, indent=2, sort_keys=True,
                                                           default=float))
    assert not bad, f'mixture reproduction failed on {len(bad)} cells: {json.dumps(bad[:8])}'
    pd.concat(frames, ignore_index=True).to_csv(OUT / 'MIXTURE_METRICS.csv', index=False)
    (OUT / 'DONE.json').write_text(json.dumps({'status': 'complete',
                                               'checks_passed': len(all_checks)}, indent=2))
    print(f'gate passed on {len(all_checks)} published cells', flush=True)


if __name__ == '__main__':
    main()
