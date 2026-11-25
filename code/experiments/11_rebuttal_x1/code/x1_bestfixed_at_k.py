#!/usr/bin/env python3
"""Rebuttal X1 (83vN-1, DzUN-2, DzUN-6): Best-fixed@k against the router on the adaptation sweep.

Question: given k labelled rows from a new corpus, does adapting the router beat simply using
those same k rows to pick one fixed verifier and deploying it?

The router column is the published transfer/adaptation sweep, re-run on its exact fitting path
(same split ladder, same seeds, same per-(seed, k) sample of the held-out corpus's training pool)
and gated cell by cell against the published AUROC and ms. Nothing new is fitted for the router.

For every (corpus, fraction, seed) the SAME k target rows then select a fixed verifier:
  all     -- highest AUROC among every verifier in the frozen matrix
  budget  -- same, restricted to verifiers whose mean latency on the validation split is no
             higher than the router's mean routed latency on that split (label-free, no test data)
  pool    -- same, restricted to the router's own actions
Selection uses the k target rows when they hold at least five rows of each class; otherwise
(k = 0 or too few rows) it falls back to the source-corpus training rows, and `select_on` says so.
The selected verifier is Platt-calibrated on the fit split exactly as the per-corpus fixed arm is,
and scored on the held-out corpus's test split. No verifier is re-run.

Protocol B splits only, as in the published sweep.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
HERE = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "ingest_and_scoring", ROOT / "experiments",
          ROOT / "experiments" / "08_routing_code"):
    sys.path.insert(0, str(p))
OUT = Path(os.environ.get("V3_RUN_DIR", HERE / "runs" / "x1_bestfixed_at_k")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("V3_RUN_DIR", str(OUT))

import v3core as V          # noqa: E402
import core                 # noqa: E402

PUBLISHED = HERE / 'inputs' / 'FEWSHOT_FRAC_MS.csv'   # the published adaptation sweep
POOL = list(core.ACTIONS)
SEEDS = list(V.SEEDS)
CONTRACT = ROOT / 'experiments' / '01_main_experiment' / '00_contract'
HPB = json.loads((CONTRACT / 'HP_SELECTED.json').read_text())['hyperparameters_B']
FEATURES = json.loads((CONTRACT / 'INHERITED_FROZEN_v3.json').read_text())['features']
PAIRS = {'cogensumm': ('cogensumm_val', 'cogensumm_test'),
         'frank': ('frank_valid', 'frank_test'),
         'ragtruth': ('ragtruth_train', 'ragtruth_test'),
         'unisumeval': ('unisumeval_train', 'unisumeval_dev')}
FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 1.00]
if os.environ.get('X1_SMOKE'):   # smoke test: one corpus, two fractions, all seeds (gate still valid)
    PAIRS = {'ragtruth': PAIRS['ragtruth']}
    FRACS = [0.0, 0.10]
MIN_PER_CLASS = 5
# Bit-for-bit on the recorded environment (AFR_STRICT_GATE=1). Elsewhere the random forest's
# floating-point path differs (see docs/reproducibility.md), so the router cells are compared
# at the portable tolerance and the run still reports every difference.
GATE = 1e-9 if os.environ.get('AFR_STRICT_GATE') == '1' else 2e-3


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def auroc(y, p):
    y = np.asarray(y, int)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else np.nan


def router(fit_all, ev, seed):
    """The published fitting path, verbatim; also returns the router's validation latency."""
    held = V.stratified_group_split(fit_all, seed)
    fit = fit_all.loc[~held].reset_index(drop=True)
    val = fit_all.loc[held].reset_index(drop=True)
    yv = val['label_supported'].to_numpy(int)
    if len(set(yv.tolist())) < 2 or len(fit) < 30:
        return None
    cals = core.platt(fit, actions=POOL)
    c_fit = core.apply_platt(fit, cals, POOL)
    c_val = core.apply_platt(val, cals, POOL)
    c_ev = core.apply_platt(ev, cals, POOL)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=HPB, target='regret', learner='rf')
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, ms_val = V.route(val, h_val, c_val, beta, POOL, cvec)
    _s2, p_ev, ms_ev = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
    q_ev = core.isotonic(p_val, yv, p_ev)
    y = ev['label_supported'].to_numpy(int)
    return {'auroc': auroc(y, q_ev), 'ms': float(np.mean(ms_ev)),
            'val_ms': float(np.mean(ms_val)), 'fit': fit, 'val': val}


def pick(fit, select, ev, candidates):
    """Highest AUROC on `select`; Platt fitted on `fit`, as the per-corpus fixed arm does."""
    ys = select['label_supported'].to_numpy(int)
    best, best_au = None, -np.inf
    for v in candidates:
        cal = core.platt(fit, actions=[v])
        au = auroc(ys, core.apply_platt(select, cal, actions=[v])[v])
        if np.isfinite(au) and au > best_au:
            best, best_au = v, au
    cal = core.platt(fit, actions=[best])
    pe = core.apply_platt(ev, cal, actions=[best])[best]
    ms = ev[f'latency_ms__{best}'].to_numpy(float)
    assert np.isfinite(ms).all(), f'non-finite latency for {best}'
    return {'selected': best, 'select_auroc': best_au,
            'auroc': auroc(ev['label_supported'], pe), 'ms': float(ms.mean())}


def main():
    train, test, _all, verifiers = V.load(with_test_labels=True)
    log(f'{len(verifiers)} verifiers in the frozen matrix: {verifiers}')
    pub = pd.read_csv(PUBLISHED).set_index(['corpus', 'frac'])
    per_seed, checks = [], []
    for corpus, (trkey, tekey) in PAIRS.items():
        in_c = train['dataset_key'].astype(str) == trkey
        base, pool = train.loc[~in_c], train.loc[in_c]
        ev = test[test['dataset_key'].astype(str) == tekey].reset_index(drop=True)
        ks = [min(int(round(f * len(pool))), len(pool)) for f in FRACS]
        for frac, k in zip(FRACS, ks):
            got = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + k)   # identical to the published sweep
                take = (pool.sample(k, random_state=int(rng.integers(0, 2 ** 31 - 1)))
                        if k else pool.iloc[:0])
                fit_all = pd.concat([base, take], ignore_index=True)
                r = router(fit_all, ev, seed)
                if r is None:
                    continue
                yt = take['label_supported'].to_numpy(int)
                enough = k > 0 and min((yt == 0).sum(), (yt == 1).sum()) >= MIN_PER_CLASS
                select = take if enough else base
                lat_val = {v: float(r['val'][f'latency_ms__{v}'].astype(float).mean())
                           for v in verifiers}
                budget = [v for v in verifiers if lat_val[v] <= r['val_ms']]
                row = {'corpus': corpus, 'frac': frac, 'k': k, 'seed': seed,
                       'select_on': 'target_k' if enough else 'source',
                       'router_auroc': r['auroc'], 'router_ms': r['ms'],
                       'router_val_ms': r['val_ms'], 'n_budget': len(budget)}
                for name, cands in (('all', verifiers), ('budget', budget), ('pool', POOL)):
                    if not cands:
                        continue
                    b = pick(r['fit'], select, ev, cands)
                    row.update({f'{name}_{key}': val for key, val in b.items()})
                per_seed.append(row)
                got.append(row)
            f = pd.DataFrame(got)
            ref = pub.loc[(corpus, frac)]
            for col in ('auroc', 'ms'):
                mine = float(f[f'router_{col}'].mean())
                diff = abs(mine - float(ref[col]))
                checks.append({'corpus': corpus, 'frac': frac, 'metric': col,
                               'published': float(ref[col]), 'recomputed': mine,
                               'abs_diff': diff, 'pass': bool(diff <= GATE)})
            log(f'{corpus} frac={frac:.2f} k={k:5d}  router {f["router_auroc"].mean():.4f}/'
                f'{f["router_ms"].mean():6.1f}ms  all {f["all_auroc"].mean():.4f}/'
                f'{f["all_ms"].mean():6.1f}ms  budget {f.get("budget_auroc", pd.Series([np.nan])).mean():.4f}  '
                f'pool {f["pool_auroc"].mean():.4f}')
    ps = pd.DataFrame(per_seed)
    ps.to_csv(OUT / 'X1_PER_SEED.csv', index=False)
    bad = [c for c in checks if not c['pass']]
    pd.DataFrame(checks).to_csv(OUT / 'X1_ROUTER_REPRODUCTION.csv', index=False)
    (OUT / 'X1_GATE.json').write_text(json.dumps(
        {'gate': GATE, 'n_checks': len(checks), 'pass': not bad, 'failures': bad},
        indent=2, default=float))
    if bad:
        log(f'GATE FAILED on {len(bad)}/{len(checks)} router cells; summary NOT written')
        raise SystemExit(1)
    num = [c for c in ps.columns if c.endswith(('_auroc', '_ms')) or c == 'n_budget']
    agg = ps.groupby(['corpus', 'frac', 'k'])[num].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
    picks = (ps.melt(id_vars=['corpus', 'frac', 'k'],
                     value_vars=[c for c in ('all_selected', 'budget_selected', 'pool_selected')
                                 if c in ps.columns], var_name='arm', value_name='verifier')
             .groupby(['corpus', 'frac', 'k', 'arm'])['verifier']
             .agg(lambda s: ';'.join(f'{v}:{n}' for v, n in s.value_counts().items()))
             .reset_index())
    agg.to_csv(OUT / 'X1_SUMMARY.csv', index=False)
    picks.to_csv(OUT / 'X1_PICKS.csv', index=False)
    (OUT / 'DONE.json').write_text(json.dumps({'status': 'complete', 'checks': len(checks)}))
    log(f'gate passed on {len(checks)} router cells; wrote X1_SUMMARY.csv and X1_PICKS.csv')


if __name__ == '__main__':
    main()
