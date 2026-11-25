#!/usr/bin/env python3
"""Rebuttal X1, mixed-traffic version (83vN-1, DzUN-2, DzUN-6).

Scenario: a router is fitted on three corpora; a fourth corpus arrives with k labelled rows and
joins the stream. Both systems are scored on the four-corpus test mixture at its natural
proportions (all four test splits), which is the deployment setting of the main results.

Router@k: the published adaptation-sweep fit (three corpora + k rows of the fourth, same seeds,
same per-(seed, k) sample), unchanged. The same fitted router is routed over the whole test
mixture. Its held-out-corpus slice is gated cell by cell against the published sweep, which
checks that the fit is identical; the mixture numbers are new.

Best-fixed@k: with the same labelled data the router sees (three source corpora + the k rows),
pick one verifier by AUROC on that pooled labelled set and deploy it on the whole mixture:
  all     -- any verifier in the frozen matrix
  budget  -- verifiers whose mean latency on the router's validation split is no higher than the
             router's mean routed latency there (label-free, no test data)
  pool    -- the router's own actions
Platt is fitted on the fit split, as the per-corpus fixed arm does. No verifier is re-run.

Protocol B splits only.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('AFR_ROOT', '/home/zeyu/projects/adaptive-faithfulness-router-v2'))
for q in (ROOT, ROOT / 'paper_v2', ROOT / 'paper_v3', ROOT / 'paper_v3' / 'DELIVERABLE' / '08_scripts'):
    sys.path.insert(0, str(q))
OUT = Path(os.environ['V3_RUN_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)

import v3core as V          # noqa: E402
import core                 # noqa: E402

PUBLISHED = ROOT / 'paper_v3' / 'runs' / 'fewshot_v1' / 'FEWSHOT_FRAC_MS.csv'
POOL = list(core.ACTIONS)
SEEDS = list(V.SEEDS)
HPB = json.loads((ROOT / 'paper_v3' / 'artifacts' / 'part1c_main_full_v1' / '00_contract' /
                  'HP_SELECTED.json').read_text())['hyperparameters_B']
FEATURES = json.loads((ROOT / 'paper_v3' / 'artifacts' / 'part1c_main_full_v1' / '00_contract' /
                       'INHERITED_FROZEN_v3.json').read_text())['features']
PAIRS = {'cogensumm': ('cogensumm_val', 'cogensumm_test'),
         'frank': ('frank_valid', 'frank_test'),
         'ragtruth': ('ragtruth_train', 'ragtruth_test'),
         'unisumeval': ('unisumeval_train', 'unisumeval_dev')}
FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 1.00]
if os.environ.get('X1_SMOKE'):   # smoke test: one corpus, two fractions, all seeds (gate still valid)
    PAIRS = {'ragtruth': PAIRS['ragtruth']}
    FRACS = [0.0, 0.10]
GATE = 1e-9


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def auroc(y, p):
    y = np.asarray(y, int)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else np.nan


def router(fit_all, test, seed):
    """The published fitting path, verbatim, routed over the whole test mixture."""
    held = V.stratified_group_split(fit_all, seed)
    fit = fit_all.loc[~held].reset_index(drop=True)
    val = fit_all.loc[held].reset_index(drop=True)
    yv = val['label_supported'].to_numpy(int)
    if len(set(yv.tolist())) < 2 or len(fit) < 30:
        return None
    cals = core.platt(fit, actions=POOL)
    c_fit = core.apply_platt(fit, cals, POOL)
    c_val = core.apply_platt(val, cals, POOL)
    c_ev = core.apply_platt(test, cals, POOL)
    h_val, h_ev = core.fit_heads(fit, [val, test], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=HPB, target='regret', learner='rf')
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, ms_val = V.route(val, h_val, c_val, beta, POOL, cvec)
    _s2, p_ev, ms_ev = V.route(test, h_ev, c_ev, beta, POOL, cvec)
    q_ev = core.isotonic(p_val, yv, p_ev)
    return {'q': q_ev, 'ms': np.asarray(ms_ev, float), 'val_ms': float(np.mean(ms_val)),
            'fit': fit, 'val': val}


def pick(fit, select, test, candidates):
    """Highest AUROC on the pooled labelled set; Platt fitted on `fit`."""
    ys = select['label_supported'].to_numpy(int)
    best, best_au = None, -np.inf
    for v in candidates:
        cal = core.platt(fit, actions=[v])
        au = auroc(ys, core.apply_platt(select, cal, actions=[v])[v])
        if np.isfinite(au) and au > best_au:
            best, best_au = v, au
    cal = core.platt(fit, actions=[best])
    pe = np.asarray(core.apply_platt(test, cal, actions=[best])[best], float)
    ms = test[f'latency_ms__{best}'].to_numpy(float)
    assert np.isfinite(ms).all(), f'non-finite latency for {best}'
    return best, best_au, pe, ms


def main():
    train, test, _all, verifiers = V.load(with_test_labels=True)
    test = test.reset_index(drop=True)
    y_all = test['label_supported'].to_numpy(int)
    log(f'{len(verifiers)} verifiers; test mixture {len(test)} rows: '
        f'{test["dataset_key"].value_counts().to_dict()}')
    pub = pd.read_csv(PUBLISHED).set_index(['corpus', 'frac'])
    per_seed, checks = [], []
    for corpus, (trkey, tekey) in PAIRS.items():
        in_c = train['dataset_key'].astype(str) == trkey
        base, pool = train.loc[~in_c], train.loc[in_c]
        tgt = (test['dataset_key'].astype(str) == tekey).to_numpy()
        ks = [min(int(round(f * len(pool))), len(pool)) for f in FRACS]
        for frac, k in zip(FRACS, ks):
            got = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + k)   # identical to the published sweep
                take = (pool.sample(k, random_state=int(rng.integers(0, 2 ** 31 - 1)))
                        if k else pool.iloc[:0])
                fit_all = pd.concat([base, take], ignore_index=True)
                r = router(fit_all, test, seed)
                if r is None:
                    continue
                lat_val = {v: float(r['val'][f'latency_ms__{v}'].astype(float).mean())
                           for v in verifiers}
                budget = [v for v in verifiers if lat_val[v] <= r['val_ms']]
                row = {'corpus': corpus, 'frac': frac, 'k': k, 'seed': seed,
                       'router_auroc': auroc(y_all, r['q']), 'router_ms': float(r['ms'].mean()),
                       'router_tgt_auroc': auroc(y_all[tgt], r['q'][tgt]),
                       'router_tgt_ms': float(r['ms'][tgt].mean()),
                       'router_val_ms': r['val_ms'], 'n_budget': len(budget)}
                for name, cands in (('all', verifiers), ('budget', budget), ('pool', POOL)):
                    if not cands:
                        continue
                    best, sel_au, pe, ms = pick(r['fit'], fit_all, test, cands)
                    row.update({f'{name}_selected': best, f'{name}_select_auroc': sel_au,
                                f'{name}_auroc': auroc(y_all, pe), f'{name}_ms': float(ms.mean()),
                                f'{name}_tgt_auroc': auroc(y_all[tgt], pe[tgt]),
                                f'{name}_tgt_ms': float(ms[tgt].mean())})
                per_seed.append(row)
                got.append(row)
            f = pd.DataFrame(got)
            ref = pub.loc[(corpus, frac)]
            for col in ('auroc', 'ms'):
                mine = float(f[f'router_tgt_{col}'].mean())
                diff = abs(mine - float(ref[col]))
                checks.append({'corpus': corpus, 'frac': frac, 'metric': col,
                               'published': float(ref[col]), 'recomputed': mine,
                               'abs_diff': diff, 'pass': bool(diff <= GATE)})
            log(f'{corpus} frac={frac:.2f} k={k:5d}  MIXED router {f["router_auroc"].mean():.4f}/'
                f'{f["router_ms"].mean():6.1f}ms  all {f["all_auroc"].mean():.4f}/'
                f'{f["all_ms"].mean():6.1f}ms  budget {f["budget_auroc"].mean():.4f}/'
                f'{f["budget_ms"].mean():6.1f}ms  pool {f["pool_auroc"].mean():.4f}/'
                f'{f["pool_ms"].mean():6.1f}ms')
    ps = pd.DataFrame(per_seed)
    ps.to_csv(OUT / 'X1M_PER_SEED.csv', index=False)
    bad = [c for c in checks if not c['pass']]
    pd.DataFrame(checks).to_csv(OUT / 'X1M_ROUTER_REPRODUCTION.csv', index=False)
    (OUT / 'X1M_GATE.json').write_text(json.dumps(
        {'gate': GATE, 'n_checks': len(checks), 'pass': not bad, 'failures': bad},
        indent=2, default=float))
    if bad:
        log(f'GATE FAILED on {len(bad)}/{len(checks)} router cells; summary NOT written')
        raise SystemExit(1)
    num = [c for c in ps.columns if c.endswith(('_auroc', '_ms')) or c == 'n_budget']
    agg = ps.groupby(['corpus', 'frac', 'k'])[num].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
    picks = (ps.melt(id_vars=['corpus', 'frac', 'k'],
                     value_vars=['all_selected', 'budget_selected', 'pool_selected'],
                     var_name='arm', value_name='verifier')
             .groupby(['corpus', 'frac', 'k', 'arm'])['verifier']
             .agg(lambda s: ';'.join(f'{v}:{n}' for v, n in s.value_counts().items()))
             .reset_index())
    agg.to_csv(OUT / 'X1M_SUMMARY.csv', index=False)
    picks.to_csv(OUT / 'X1M_PICKS.csv', index=False)
    (OUT / 'DONE.json').write_text(json.dumps({'status': 'complete', 'checks': len(checks)}))
    log(f'gate passed on {len(checks)} router cells; wrote X1M_SUMMARY.csv and X1M_PICKS.csv')


if __name__ == '__main__':
    main()
