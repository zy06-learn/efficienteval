#!/usr/bin/env python3
"""Complete the approved two-arm Table 7; preserve B timing, add in-corpus A.

B: consume the 40 gated caches, never refit. A: original 8/1/1 rotations
within each corpus, original A hyperparameters, 10 seeds, two arms. Quality
metrics pool out-of-fold predictions within each seed, then average seeds.
Router time includes measured overhead; selected fixed time is verifier-only,
matching the original E1/E1b arm-specific timing definitions.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import v3core as V
import core
import part3_extended_v1 as P
import part3_percorpus_selected_v1 as S

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ['V3_RUN_DIR']).resolve()
BASE = HERE / 'runs/p6_table7_20260917'
OLD = HERE / 'artifacts/part3_extended_v1/01_tables'
METRICS = ['auroc', 'auprc_unsup', 'bacc', 'f1_unsup', 'ms']


def save_text(path, content):
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f'Refusing to overwrite different content: {path}')
        return
    path.write_text(content)


def save_json(name, obj):
    save_text(OUT / name, json.dumps(obj, indent=2, sort_keys=True))


def save_csv(name, rows):
    save_text(OUT / name, pd.DataFrame(rows).to_csv(index=False))


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def provenance():
    files = [Path(__file__), Path(P.__file__), Path(S.__file__), Path(V.__file__), Path(core.__file__)]
    prior = json.loads((BASE / 'PROVENANCE.json').read_text())
    for file in files[1:]:
        assert hashlib.sha256(file.read_bytes()).hexdigest() == prior['sha256'][str(file)]
    contract = {'seeds': P.SEEDS, 'hp': P.HP, 'pool': P.POOL, 'features': P.FEATURES,
                'python': sys.version, 'platform': platform.platform(),
                'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                'B_timing': 'historical E1/E1b means, user approved 2026-09-17',
                'A_timing': 'new router overhead plus cached verifier/features; pool3 verifier only',
                'A_split': 'V.folds_stratified within corpus, V.rotations 8/1/1',
                'threads': 8, 'GPU': False}
    save_json('CONTRACT.json', contract)
    if not (OUT / 'GIT_STATE.txt').exists():
        save_text(OUT / 'GIT_STATE.txt', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True)
                  + subprocess.check_output(['git', 'status', '--porcelain'], text=True))


def metrics(y, prob, ms, threshold):
    result = core.metrics(y, prob, ms, threshold=threshold, with_risk=False)
    assert all(np.isfinite(result[k]) for k in METRICS)
    return {k: float(result[k]) for k in METRICS}


def summary(corpus, arm, per, n):
    frame = pd.DataFrame(per)
    return {'corpus': corpus, 'system': arm, 'seeds_used': len(per), 'test_rows': n,
            **{k: float(frame[k].mean()) for k in METRICS},
            **{k + '_sd': float(frame[k].std(ddof=0)) for k in METRICS}}


def complete_b(test):
    gate = json.loads((BASE / 'B_GATE.json').read_text())
    assert all(cell['pass'] for cell in gate['cells'] if cell['metric'] != 'ms' or cell['system'] != 'OURS')
    original = pd.read_csv(OLD / 'E1_PERCORPUS.csv')
    selected = pd.read_csv(OLD / 'E1b_PERCORPUS_SELECTED.csv')
    rows, per_seed, checks = [], [], []
    for corpus, (_, tekey) in P.CORPUS_PAIRS.items():
        te = test[test.dataset_key.astype(str) == tekey].reset_index(drop=True)
        vals = {'OURS': [], 'pool3': []}
        for seed in P.SEEDS:
            with np.load(BASE / f'B_{corpus}_{seed}.npz', allow_pickle=False) as z:
                assert np.array_equal(z['episode_key'], te.episode_key.astype(str).to_numpy())
                assert np.array_equal(z['y'], te.label_supported.to_numpy(int))
                for arm, prefix in [('OURS', 'router'), ('pool3', 'selected')]:
                    m = metrics(z['y'], z[prefix+'_prob'], z[prefix+'_ms'], z[prefix+'_thr'])
                    vals[arm].append(m)
                    per_seed.append({'corpus': corpus, 'system': arm, 'seed': seed, **m})
        for arm, per in vals.items():
            r = summary(corpus, arm, per, len(te))
            ref = (original[(original.corpus == corpus) & (original.system == 'OURS')].iloc[0]
                   if arm == 'OURS' else selected[(selected.corpus == corpus) & (selected.candidates == 'pool3')].iloc[0])
            for k in (['auroc', 'bacc'] if arm == 'OURS' else ['auroc', 'ms']):
                diff = abs(r[k] - float(ref[k]))
                assert diff <= 1e-9, (corpus, arm, k, diff)
                checks.append({'corpus': corpus, 'system': arm, 'metric': k, 'abs_diff': diff})
            r['ms_current_reproduction'] = r['ms']
            r['ms'] = float(ref['ms'])
            # Historical per-seed timings were not saved, so do not mislabel new SD as historical.
            r.pop('ms_sd')
            r['ms_source'] = 'historical E1' if arm == 'OURS' else 'historical E1b'
            rows.append(r)
    save_csv('B_TABLE7.csv', rows)
    save_csv('B_PER_SEED_REPRODUCTION_TIMING.csv', per_seed)
    save_json('B_ACCEPTED_GATE.json', {'quality_gate_pass': True, 'checks': checks,
              'historical_timing_retained': True, 'original_full_gate_pass': gate['pass'],
              'authorization': 'User approved timing exception 2026-09-17'})
    log('B metrics complete: 8 rows, historical timing retained, quality gate passed')


def frames_by_corpus(all_frame):
    return {c: all_frame[all_frame.dataset_key.astype(str).isin(keys)].reset_index(drop=True)
            for c, keys in P.CORPUS_PAIRS.items()}


def preflight(frames):
    rows = []
    for corpus, frame in frames.items():
        for seed in P.SEEDS:
            fold = V.folds_stratified(frame, seed)
            seen = np.zeros(len(frame), int)
            for te, va, tr in V.rotations():
                masks = [np.isin(fold, tr), fold == va, fold == te]
                parts = [frame.loc[m] for m in masks]
                groups = [set(f.content_doc_key.astype(str)) for f in parts]
                assert all(not (groups[i] & groups[j]) for i in range(3) for j in range(i))
                assert all(len(f) and f.label_supported.nunique() == 2 for f in parts)
                for action in P.POOL:
                    fit_av = parts[0][parts[0][f'available__{action}'].astype(bool)]
                    assert fit_av.label_supported.nunique() == 2
                    assert all(np.isfinite(f[f'latency_ms__{action}']).all() for f in parts)
                # Match the original availability policy: router masks unavailable
                # actions; fixed baselines calibrate missing raw scores as 0.5.
                assert all(f[[f'available__{a}' for a in P.POOL]].any(axis=1).all() for f in parts)
                seen[masks[2]] += 1
                rows.append({'corpus': corpus, 'seed': seed, 'test_fold': te,
                             'fit_rows': len(parts[0]), 'val_rows': len(parts[1]), 'test_rows': len(parts[2]),
                             'val_minority': int(parts[1].label_supported.value_counts().min()),
                             'missing_member_scores': int(sum((~frame[f'available__{a}']).sum() for a in P.POOL))})
            assert (seen == 1).all()
    save_csv('A_PREFLIGHT.csv', rows)
    log(f'A preflight passed: {len(rows)} splits; group isolation, classes, availability, coverage')


def one_a_split(fit, val, ev, seed):
    router = P.route_and_score(fit, val, ev, seed, 'A')
    available = ev[[f'available__{a}' for a in P.POOL]].to_numpy(bool)
    assert available[np.arange(len(ev)), router['sel']].all()
    yv = val.label_supported.to_numpy(int)
    best, best_auc, scores = None, -np.inf, {}
    for v in P.POOL:
        cal = core.platt(fit, actions=[v])
        pv = core.apply_platt(val, cal, actions=[v])[v]
        pe = core.apply_platt(ev, cal, actions=[v])[v]
        auc = float(roc_auc_score(yv, pv))
        scores[v] = (pv, pe)
        if auc > best_auc:
            best, best_auc = v, auc
    q, thr = S._stage2(scores[best][0], yv, scores[best][1])
    return {'router_prob': router['prob'], 'router_thr': router['thr'], 'router_ms': router['ms'],
            'router_sel': router['sel'], 'router_ms_det': router['ms_det'], 'beta': router['beta'],
            'selected_prob': q, 'selected_thr': thr,
            'selected_ms': ev[f'latency_ms__{best}'].to_numpy(float),
            'selected_verifier': np.array(best), 'selected_val_auroc': best_auc}


def complete_a(frames, smoke):
    rows, per_seed, picks = [], [], []
    for corpus, frame in frames.items():
        y = frame.label_supported.to_numpy(int)
        vals = {'OURS': [], 'pool3': []}
        for seed in P.SEEDS:
            fold = V.folds_stratified(frame, seed)
            arrays = {prefix: {k: np.full(len(frame), np.nan) for k in ['prob', 'thr', 'ms']}
                      for prefix in ['router', 'selected']}
            coverage = np.zeros(len(frame), int)
            for te, va, tr in V.rotations():
                pos = np.flatnonzero(fold == te)
                ev = frame.iloc[pos].reset_index(drop=True)
                cache = OUT / f'A_{corpus}_{seed}_{te}.npz'
                if not cache.exists():
                    got = one_a_split(frame.loc[np.isin(fold, tr)].reset_index(drop=True),
                                      frame.loc[fold == va].reset_index(drop=True), ev, seed)
                    tmp = cache.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp, episode_key=ev.episode_key.astype(str).to_numpy(dtype=str),
                                        y=y[pos], pos=pos, **got)
                    tmp.replace(cache)
                with np.load(cache, allow_pickle=False) as z:
                    assert np.array_equal(z['pos'], pos) and np.array_equal(z['y'], y[pos])
                    assert np.array_equal(z['episode_key'], ev.episode_key.astype(str).to_numpy())
                    for prefix in arrays:
                        for k in arrays[prefix]:
                            arrays[prefix][k][pos] = z[prefix+'_'+k]
                    picks.append({'corpus': corpus, 'seed': seed, 'test_fold': te,
                                  'selected': str(z['selected_verifier']),
                                  'val_auroc': float(z['selected_val_auroc']), 'beta': float(z['beta'])})
                    for arm, prefix in [('OURS', 'router'), ('pool3', 'selected')]:
                        metrics(y[pos], z[prefix+'_prob'], z[prefix+'_ms'], z[prefix+'_thr'])
                coverage[pos] += 1
                if smoke:
                    save_json('SMOKE.json', {'pass': True, 'cache': cache.name,
                              'note': 'First formal split cached; not a full A performance result'})
                    log('A first-split smoke passed; resumable cache retained')
                    return
            assert (coverage == 1).all()
            for arm, prefix in [('OURS', 'router'), ('pool3', 'selected')]:
                a = arrays[prefix]
                assert all(np.isfinite(v).all() for v in a.values())
                m = metrics(y, a['prob'], a['ms'], a['thr'])
                vals[arm].append(m)
                per_seed.append({'corpus': corpus, 'system': arm, 'seed': seed, **m})
            log(f'A {corpus} seed={seed}: 10 folds completed')
        for arm, per in vals.items():
            rows.append(summary(corpus, arm, per, len(frame)))
    save_csv('A_TABLE7.csv', rows)
    save_csv('A_PER_SEED.csv', per_seed)
    save_csv('A_PICKS.csv', picks)
    save_json('DONE.json', {'status': 'complete', 'A_cached_splits': 400, 'B_cached_seeds': 40,
                          'A_rows': len(rows), 'B_rows': 8, 'quality_gate_pass': True})
    log('DONE: A/B Table 7 complete')


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--smoke', action='store_true')
    opt = args.parse_args()
    provenance()
    _, test, all_frame, _ = V.load(with_test_labels=True)
    frames = frames_by_corpus(all_frame)
    preflight(frames)
    complete_b(test)
    complete_a(frames, opt.smoke)


if __name__ == '__main__':
    main()
