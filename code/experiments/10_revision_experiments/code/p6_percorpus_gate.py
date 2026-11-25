#!/usr/bin/env python3
"""Reproduce Table 7's existing B rows before any added metrics or A runs.

Uses the original per-corpus router unchanged, including measured overhead.
Caches predictions for recovery; emits only existing metrics and gate evidence.
Run with V3_RUN_DIR pointing to a new, unique directory.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import v3core as V
import core
import part3_extended_v1 as P
import part3_percorpus_selected_v1 as S

OUT = Path(os.environ['V3_RUN_DIR']).resolve()
OLD = Path(__file__).resolve().parent / 'artifacts/part3_extended_v1/01_tables'


def main():
    gatefile = OUT / 'B_GATE.json'
    if gatefile.exists():
        raise RuntimeError('Existing gate report: do not overwrite or restart.')
    sources = [Path(__file__), Path(P.__file__), Path(S.__file__), Path(V.__file__), Path(core.__file__)]
    provenance = {
        'python': sys.version, 'platform': platform.platform(), 'seeds': P.SEEDS,
        'pool': P.POOL, 'features': P.FEATURES, 'hp': P.HP,
        'argv': sys.argv, 'cwd': os.getcwd(),
        'env': {k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'V3_RUN_DIR']},
        'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
    }
    (OUT / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2))
    train, test, _, _ = V.load(with_test_labels=True)
    original = pd.read_csv(OLD / 'E1_PERCORPUS.csv')
    selected = pd.read_csv(OLD / 'E1b_PERCORPUS_SELECTED.csv')
    picks_old = pd.read_csv(OLD / 'E1b_PERCORPUS_SELECTED_PICKS.csv')
    rows, gates = [], []
    for corpus, (trkey, tekey) in P.CORPUS_PAIRS.items():
        tr = train[train.dataset_key.astype(str) == trkey].reset_index(drop=True)
        te = test[test.dataset_key.astype(str) == tekey].reset_index(drop=True)
        y = te.label_supported.to_numpy(int)
        values = {'OURS': [], 'pool3': []}
        for seed in P.SEEDS:
            cache = OUT / f'B_{corpus}_{seed}.npz'
            if cache.exists():
                raise RuntimeError(f'Existing cache, resume requires explicit handling: {cache}')
            for fraction in (.20, .30, .40):
                held = V.stratified_group_split(tr, seed, fraction=fraction)
                fit, val = tr.loc[~held].reset_index(drop=True), tr.loc[held].reset_index(drop=True)
                yv = val.label_supported.to_numpy(int)
                if min((yv == 0).sum(), (yv == 1).sum()) >= 5:
                    break
            else:
                raise RuntimeError(f'No valid split for {corpus}/{seed}')
            assert not (set(fit.content_doc_key) & set(val.content_doc_key))
            assert not (set(tr.content_doc_key) & set(te.content_doc_key))
            router = P.route_and_score(fit, val, te, seed, 'B')
            best, best_auc, scores = None, -np.inf, {}
            for v in P.POOL:
                cal = core.platt(fit, actions=[v])
                pv = core.apply_platt(val, cal, actions=[v])[v]
                pe = core.apply_platt(te, cal, actions=[v])[v]
                auc = roc_auc_score(yv, pv)
                scores[v] = (pv, pe)
                if auc > best_auc:
                    best, best_auc = v, auc
            expected = picks_old[(picks_old.candidates == 'pool3') & (picks_old.corpus == corpus) & (picks_old.seed == seed)].iloc[0]
            assert best == expected.selected
            q, thr = S._stage2(*[scores[best][0], yv, scores[best][1]])
            ms = te[f'latency_ms__{best}'].to_numpy(float)
            np.savez_compressed(cache, episode_key=te.episode_key.astype(str).to_numpy(dtype=str),
                                y=y, router_prob=router['prob'], router_thr=router['thr'],
                                router_ms=router['ms'], router_ms_det=router['ms_det'],
                                router_sel=router['sel'], beta=router['beta'],
                                selected_prob=q, selected_thr=thr, selected_ms=ms,
                                selected_verifier=np.array(best))
            rm = core.metrics(y, router['prob'], router['ms'], threshold=router['thr'], with_risk=False)
            values['OURS'].append({k: rm[k] for k in ['auroc', 'bacc', 'ms']})
            values['pool3'].append({'auroc': float(roc_auc_score(y, q)), 'ms': float(ms.mean())})
            print(f'{corpus} seed={seed} cached; selected={best}', flush=True)
        for arm, vals in values.items():
            avg = pd.DataFrame(vals).mean().to_dict()
            ref = (original[(original.corpus == corpus) & (original.system == 'OURS')].iloc[0]
                   if arm == 'OURS' else selected[(selected.corpus == corpus) & (selected.candidates == 'pool3')].iloc[0])
            rows.append({'corpus': corpus, 'system': arm, **avg})
            for key, value in avg.items():
                diff = abs(value - float(ref[key]))
                gates.append({'corpus': corpus, 'system': arm, 'metric': key,
                              'original': float(ref[key]), 'reproduced': value,
                              'abs_diff': diff, 'pass': diff <= 1e-9})
    passed = all(g['pass'] for g in gates)
    pd.DataFrame(rows).to_csv(OUT / 'B_REPRODUCTION.csv', index=False)
    gatefile.write_text(json.dumps({'pass': passed, 'tolerance': 1e-9, 'cells': gates}, indent=2))
    print(pd.DataFrame(gates).to_string(index=False), flush=True)
    if not passed:
        raise RuntimeError('B reproduction gate failed; no new metrics or A evaluation released.')


if __name__ == '__main__':
    main()
