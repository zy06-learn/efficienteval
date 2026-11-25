#!/usr/bin/env python3
"""§6.2 / §6.3: complete the transfer-and-adaptation sweep to the Table-1 metric set.

The published sweep (DELIVERABLE/09_live_and_controls/code/fewshot_frac_ms.py) reported AUROC
and mean latency and nothing else: it never fitted a classification threshold. This re-runs the
identical fitting path -- same split ladder, same Platt, same heads, same beta rule, same
isotonic -- reproduces the published AUROC and ms cell by cell, and only then emits AUPRC, BAcc
and F1.

Threshold rule, following the global convention: the isotonic-calibrated validation scores are
thresholded at the value that maximises MCC on that validation split. At fraction 0 the
validation split contains no target-corpus row, so the threshold is chosen entirely out of
domain. That is a real property of the leave-one-corpus-out setting, not an artefact, and it
must be stated wherever BAcc and F1 from this table are quoted.

Protocol B splits only, as in the published sweep. A Protocol A version would be a new
experiment whose source/target boundaries have to be agreed first.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

ROOT = Path(os.environ.get('AFR_ROOT', '/home/zeyu/projects/adaptive-faithfulness-router-v2'))
for q in (ROOT, ROOT / 'paper_v2', ROOT / 'paper_v3', ROOT / 'paper_v3' / 'DELIVERABLE' / '08_scripts'):
    sys.path.insert(0, str(q))
OUT = Path(os.environ['V3_RUN_DIR']).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('V3_RUN_DIR', str(OUT))

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
METRICS = ['auroc', 'auprc_unsup', 'bacc', 'f1_unsup', 'ms']
GATE = 1e-9


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def mcc_threshold(q, y):
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def fit_eval(fit_all, ev, seed):
    """The published fitting path, verbatim, plus a validation threshold at the end."""
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
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
    _s2, p_ev, ms_ev = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
    q_ev = core.isotonic(p_val, yv, p_ev)
    # New: the same isotonic map applied in-sample to the validation split gives the scale the
    # threshold is chosen on, exactly as part3_extended_v1._stage2 does for every other table.
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(p_val, yv)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    thr = mcc_threshold(q_val, yv)
    y = ev['label_supported'].to_numpy(int)
    m = core.metrics(y, q_ev, ms_ev, threshold=thr, with_risk=False)
    out = {k: float(m[k]) for k in METRICS}
    # the published columns, recomputed on exactly the arrays the published run used
    out['auroc_published_path'] = float(roc_auc_score(y, q_ev)) if len(set(y.tolist())) > 1 else np.nan
    out['ms_published_path'] = float(np.mean(ms_ev))
    out['threshold'] = thr
    return out


def main():
    train, test, _all, _v = V.load(with_test_labels=True)
    pub = pd.read_csv(PUBLISHED).set_index(['corpus', 'frac'])
    rows, per_seed, checks = [], [], []
    for corpus, (trkey, tekey) in PAIRS.items():
        in_c = train['dataset_key'].astype(str) == trkey
        base, pool = train.loc[~in_c], train.loc[in_c]
        ev = test[test['dataset_key'].astype(str) == tekey].reset_index(drop=True)
        ks = [min(int(round(f * len(pool))), len(pool)) for f in FRACS]
        for frac, k in zip(FRACS, ks):
            got = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + k)
                take = (pool.sample(k, random_state=int(rng.integers(0, 2 ** 31 - 1)))
                        if k else pool.iloc[:0])
                fit_all = pd.concat([base, take], ignore_index=True)
                r = fit_eval(fit_all, ev, seed)
                if r is None:
                    continue
                got.append(r)
                per_seed.append({'corpus': corpus, 'frac': frac, 'k': k, 'seed': seed, **r})
            f = pd.DataFrame(got)
            row = {'corpus': corpus, 'frac': frac, 'k': int(k), 'n_seeds': len(f),
                   **{m: float(f[m].mean()) for m in METRICS},
                   **{m + '_sd': float(f[m].std(ddof=0)) for m in METRICS},
                   'base_rows': int(len(base)), 'pool_rows': int(len(pool)),
                   'eval_rows': int(len(ev))}
            ref = pub.loc[(corpus, frac)]
            for col, mine in (('auroc', float(f['auroc_published_path'].mean())),
                              ('ms', float(f['ms_published_path'].mean()))):
                diff = abs(mine - float(ref[col]))
                checks.append({'corpus': corpus, 'frac': frac, 'metric': col,
                               'published': float(ref[col]), 'recomputed': mine,
                               'abs_diff': diff, 'pass': bool(diff <= GATE)})
            rows.append(row)
            log(f'{corpus} frac={frac:.2f} k={k:5d}  AUROC {row["auroc"]:.5f}  '
                f'F1 {row["f1_unsup"]:.5f}  ms {row["ms"]:.2f}')
    bad = [c for c in checks if not c['pass']]
    (OUT / 'FEWSHOT_GATE.json').write_text(json.dumps(
        {'gate': GATE, 'n_checks': len(checks), 'pass': not bad, 'failures': bad},
        indent=2, sort_keys=True, default=float))
    pd.DataFrame(checks).to_csv(OUT / 'FEWSHOT_REPRODUCTION.csv', index=False)
    pd.DataFrame(per_seed).to_csv(OUT / 'FEWSHOT_PER_SEED.csv', index=False)
    if bad:
        log(f'GATE FAILED on {len(bad)}/{len(checks)} published cells; new metrics NOT published')
        raise SystemExit(1)
    pd.DataFrame(rows).to_csv(OUT / 'FEWSHOT_METRICS.csv', index=False)
    (OUT / 'DONE.json').write_text(json.dumps({'status': 'complete', 'checks': len(checks)}))
    log(f'gate passed on {len(checks)} published cells; wrote FEWSHOT_METRICS.csv')


if __name__ == '__main__':
    main()
