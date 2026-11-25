#!/usr/bin/env python3
"""Per-corpus results, all systems, both protocols, Table-1 metric set.

Covers §6.1 Table 7 (in-corpus EfficientEval vs a validation-selected fixed verifier)
and Appendix G (every fixed verifier, per corpus).

Protocol B reuses the original E1 / E1b split ladder and must reproduce their published
AUROC / BAcc / ms before any new metric is emitted. Protocol A is a new experiment:
within-corpus 8/1/1 rotations over ten folds, ten seeds, checked for exact-once coverage
and for fit/val/test document-group disjointness.

Arms: OURS (in-corpus router), no_verifier (cheap features only, no verifier call),
fixed::<v> for every verifier, sel::all15 and sel::pool3 (validation-selected single verifier).

Timing follows each arm's original definition, which differ; the deliverables must say so.
  OURS / no_verifier : features + head + routing + both calibration stages
  fixed::<v> under B : verifier call + isotonic apply (original E1 definition)
  fixed::<v> under A : verifier call only
  sel::*             : selected verifier call only (original E1b definition)
"""
import argparse
import hashlib
import json
import os
import platform
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
OLD = HERE / 'artifacts/part3_extended_v1/01_tables'
E15 = HERE / 'runs/p6_table7_complete_20260917_v2'
METRICS = ['auroc', 'auprc_unsup', 'bacc', 'f1_unsup', 'ms']
GATE = 1e-9
# `no_verifier` (cheap features predict the label directly) does not reproduce E1's published
# AUROC/BAcc to 1e-9: its score IS a RandomForest output, so last-bit prediction differences
# reorder near-ties and move AUROC by 1e-7..8e-4. The router's score survives this because a
# routing argmax then a calibrated verifier score absorbs last-bit noise. The arm is not
# required by any table in scope, so it is computed, gated, reported, and excluded rather than
# published on a weaker gate. See B_NO_VERIFIER_REPRODUCTION.json.
EXCLUDED_ARMS = ('no_verifier',)


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def save_text(path, content):
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f'Refusing to overwrite different content: {path}')
    path.write_text(content)


def save_json(name, obj):
    save_text(OUT / name, json.dumps(obj, indent=2, sort_keys=True, default=float))


def save_csv(name, rows):
    save_text(OUT / name, pd.DataFrame(rows).to_csv(index=False))


def metrics(y, prob, ms, threshold):
    r = core.metrics(y, prob, ms, threshold=threshold, with_risk=False)
    assert all(np.isfinite(r[k]) for k in METRICS), 'non-finite metric'
    return {k: float(r[k]) for k in METRICS}


def summarise(corpus, arm, per, n, extra=None):
    f = pd.DataFrame(per)
    return {'corpus': corpus, 'system': arm, 'seeds_used': len(per), 'test_rows': int(n),
            **{k: float(f[k].mean()) for k in METRICS},
            **{k + '_sd': float(f[k].std(ddof=0)) for k in METRICS}, **(extra or {})}


def b_split(sub_tr, seed):
    """The original E1 / E1b validation ladder, copied exactly."""
    for frac in (0.20, 0.30, 0.40):
        held = V.stratified_group_split(sub_tr, seed, fraction=frac)
        f_, v_ = sub_tr.loc[~held], sub_tr.loc[held]
        yv = v_['label_supported'].to_numpy(int)
        if len(set(yv.tolist())) > 1 and min((yv == 0).sum(), (yv == 1).sum()) >= 5:
            return f_.reset_index(drop=True), v_.reset_index(drop=True), frac
    return None, None, None


def fixed_arm(fit, val, ev, verifier, *, with_stage2_ms):
    cal = core.platt(fit, actions=[verifier])
    pv = core.apply_platt(val, cal, actions=[verifier])[verifier]
    pe = core.apply_platt(ev, cal, actions=[verifier])[verifier]
    q, thr, s2 = P._stage2(pv, val['label_supported'].to_numpy(int), pe)
    ms = ev[f'latency_ms__{verifier}'].to_numpy(float)
    return {'prob': q, 'thr': thr, 'ms': ms + (s2 if with_stage2_ms else 0.0),
            'val_auroc': float(roc_auc_score(val['label_supported'].to_numpy(int), pv))
            if val['label_supported'].nunique() > 1 else 0.5}


def selected_arm(fit, val, ev, candidates):
    """E1b: pick the highest-validation-AUROC verifier; latency is that verifier only."""
    yv = val['label_supported'].to_numpy(int)
    best, best_au, cache = None, -np.inf, {}
    for v in candidates:
        cal = core.platt(fit, actions=[v])
        pv = core.apply_platt(val, cal, actions=[v])[v]
        pe = core.apply_platt(ev, cal, actions=[v])[v]
        au = float(roc_auc_score(yv, pv)) if len(set(yv.tolist())) > 1 else 0.5
        cache[v] = (pv, pe)
        if au > best_au:
            best, best_au = v, au
    pv, pe = cache[best]
    q, thr = S._stage2(pv, yv, pe)
    return {'prob': q, 'thr': thr, 'ms': ev[f'latency_ms__{best}'].to_numpy(float),
            'selected': best, 'val_auroc': best_au}


# ------------------------------------------------------------------ Protocol B
def run_b(train, test, verifiers):
    old1 = pd.read_csv(OLD / 'E1_PERCORPUS.csv')
    old1b = pd.read_csv(OLD / 'E1b_PERCORPUS_SELECTED.csv')
    rows, per_seed, checks, picks = [], [], [], []
    for corpus, (trkey, tekey) in P.CORPUS_PAIRS.items():
        sub_tr = train[train.dataset_key.astype(str) == trkey].reset_index(drop=True)
        ev = test[test.dataset_key.astype(str) == tekey].reset_index(drop=True)
        y = ev.label_supported.to_numpy(int)
        got, skipped = {}, 0
        for seed in P.SEEDS:
            cache = OUT / f'B_{corpus}_{seed}.npz'
            if not cache.exists():
                fit, val, frac = b_split(sub_tr, seed)
                if fit is None:
                    skipped += 1
                    continue
                store = {'val_fraction': np.array(frac)}
                r = P.route_and_score(fit, val, ev, seed, 'B')
                store.update(OURS_prob=r['prob'], OURS_thr=np.array(r['thr']), OURS_ms=r['ms'])
                r = P.no_verifier_arm(fit, val, ev, seed, 'B')
                store.update(no_verifier_prob=r['prob'], no_verifier_thr=np.array(r['thr']),
                             no_verifier_ms=r['ms'])
                for v in verifiers:
                    a = fixed_arm(fit, val, ev, v, with_stage2_ms=True)
                    store[f'fixed::{v}_prob'] = a['prob']
                    store[f'fixed::{v}_thr'] = np.array(a['thr'])
                    store[f'fixed::{v}_ms'] = a['ms']
                for name, cands in [('sel::all15', list(verifiers)), ('sel::pool3', list(P.POOL))]:
                    a = selected_arm(fit, val, ev, cands)
                    store[f'{name}_prob'] = a['prob']
                    store[f'{name}_thr'] = np.array(a['thr'])
                    store[f'{name}_ms'] = a['ms']
                    store[f'{name}_pick'] = np.array(a['selected'])
                    store[f'{name}_val_auroc'] = np.array(a['val_auroc'])
                tmp = cache.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, episode_key=ev.episode_key.astype(str).to_numpy(dtype=str),
                                    y=y, **store)
                tmp.replace(cache)
            with np.load(cache, allow_pickle=False) as z:
                assert np.array_equal(z['episode_key'], ev.episode_key.astype(str).to_numpy())
                assert np.array_equal(z['y'], y)
                arms = sorted({k.rsplit('_', 1)[0] for k in z.files if k.endswith('_prob')})
                for arm in arms:
                    m = metrics(y, z[arm + '_prob'], z[arm + '_ms'], float(z[arm + '_thr']))
                    got.setdefault(arm, []).append(m)
                    per_seed.append({'corpus': corpus, 'system': arm, 'seed': seed, **m})
                for name in ('sel::all15', 'sel::pool3'):
                    picks.append({'protocol': 'B', 'candidates': name.split('::')[1],
                                  'corpus': corpus, 'seed': seed,
                                  'selected': str(z[f'{name}_pick']),
                                  'val_auroc': float(z[f'{name}_val_auroc'])})
        if skipped:
            log(f'  B {corpus}: {skipped} seed(s) skipped by the original ladder')
        for arm, per in got.items():
            r = summarise(corpus, arm, per, len(ev))
            ref, keys = None, []
            o = old1[(old1.corpus == corpus) & (old1.system == arm)]
            if len(o):
                ref, keys = o.iloc[0], ['auroc', 'bacc', 'ms']
            elif arm.startswith('sel::'):
                cand = arm.split('::')[1]
                o = old1b[(old1b.corpus == corpus) & (old1b.candidates == cand)]
                if len(o):
                    ref, keys = o.iloc[0], ['auroc', 'ms']
            for k in keys:
                diff = abs(r[k] - float(ref[k]))
                checks.append({'corpus': corpus, 'system': arm, 'metric': k,
                               'abs_diff': diff, 'pass': bool(diff <= GATE),
                               'reference': float(ref[k]), 'recomputed': r[k]})
            if arm not in EXCLUDED_ARMS:
                rows.append(r)
        log(f'B {corpus}: {len(got)} arms')
    save_csv('B_PERCORPUS.csv', rows)
    save_csv('B_PER_SEED.csv', per_seed)
    save_csv('B_PICKS.csv', picks)
    excluded = [c for c in checks if c['system'] in EXCLUDED_ARMS]
    if excluded:
        save_json('B_NO_VERIFIER_REPRODUCTION.json', {
            'gate': GATE, 'pass': bool(all(c['pass'] for c in excluded)), 'checks': excluded,
            'action': 'arm excluded from every deliverable; not published on a weaker gate',
            'reason': 'RandomForest output is the score itself, so last-bit prediction '
                      'differences reorder near-ties; not required by any table in scope'})
    checks = [c for c in checks if c['system'] not in EXCLUDED_ARMS]
    quality = [c for c in checks if c['metric'] != 'ms']
    timing = [c for c in checks if c['metric'] == 'ms']
    assert all(c['pass'] for c in quality), \
        'B quality reproduction failed: ' + json.dumps([c for c in quality if not c['pass']])
    save_json('B_GATE.json', {
        'quality_pass': True, 'quality_checks': len(quality),
        'timing_checks': len(timing),
        'timing_pass': bool(all(c['pass'] for c in timing)),
        'timing_failures': [c for c in timing if not c['pass']],
        'note': 'Quality gated at 1e-9. Live-measured calibration/head time is not bit-reproducible; '
                'timing differences are reported, never silently accepted.'})
    log(f'B gate: quality {len(quality)}/{len(quality)} pass; '
        f'timing {sum(c["pass"] for c in timing)}/{len(timing)} pass')


# ------------------------------------------------------------------ Protocol A
def a_frames(all_frame):
    return {c: all_frame[all_frame.dataset_key.astype(str).isin(keys)].reset_index(drop=True)
            for c, keys in P.CORPUS_PAIRS.items()}


def a_preflight(frames):
    rows = []
    for corpus, frame in frames.items():
        for seed in P.SEEDS:
            fold = V.folds_stratified(frame, seed)
            seen = np.zeros(len(frame), int)
            for te, va, tr in V.rotations():
                masks = [np.isin(fold, tr), fold == va, fold == te]
                parts = [frame.loc[m] for m in masks]
                groups = [set(f.content_doc_key.astype(str)) for f in parts]
                assert all(not (groups[i] & groups[j]) for i in range(3) for j in range(i)), \
                    (corpus, seed, te, 'document groups overlap')
                assert all(len(f) and f.label_supported.nunique() == 2 for f in parts)
                assert all(f[[f'available__{a}' for a in P.POOL]].any(axis=1).all() for f in parts)
                seen[masks[2]] += 1
                rows.append({'corpus': corpus, 'seed': seed, 'test_fold': te,
                             'fit_rows': len(parts[0]), 'val_rows': len(parts[1]),
                             'test_rows': len(parts[2]),
                             'val_minority': int(parts[1].label_supported.value_counts().min())})
            assert (seen == 1).all(), (corpus, seed, 'coverage is not exactly once')
    save_csv('A_PREFLIGHT.csv', rows)
    log(f'A preflight passed: {len(rows)} splits (group isolation, classes, coverage==1)')


def a_split(fit, val, ev, seed, verifiers):
    store = {}
    r = P.route_and_score(fit, val, ev, seed, 'A')
    avail = ev[[f'available__{a}' for a in P.POOL]].to_numpy(bool)
    assert avail[np.arange(len(ev)), r['sel']].all(), 'router selected an unavailable action'
    store.update(OURS_prob=r['prob'], OURS_thr=np.array(r['thr']), OURS_ms=r['ms'],
                 OURS_sel=r['sel'], beta=np.array(r['beta']))
    for v in verifiers:
        a = fixed_arm(fit, val, ev, v, with_stage2_ms=False)
        store[f'fixed::{v}_prob'] = a['prob']
        store[f'fixed::{v}_thr'] = np.array(a['thr'])
        store[f'fixed::{v}_ms'] = a['ms']
    for name, cands in [('sel::all15', list(verifiers)), ('sel::pool3', list(P.POOL))]:
        a = selected_arm(fit, val, ev, cands)
        store[f'{name}_prob'] = a['prob']
        store[f'{name}_thr'] = np.array(a['thr'])
        store[f'{name}_ms'] = a['ms']
        store[f'{name}_pick'] = np.array(a['selected'])
        store[f'{name}_val_auroc'] = np.array(a['val_auroc'])
    return store


def run_a(frames, verifiers, smoke):
    rows, per_seed, picks = [], [], []
    for corpus, frame in frames.items():
        y = frame.label_supported.to_numpy(int)
        got = {}
        for seed in P.SEEDS:
            fold = V.folds_stratified(frame, seed)
            pooled, coverage = {}, np.zeros(len(frame), int)
            for te, va, tr in V.rotations():
                pos = np.flatnonzero(fold == te)
                ev = frame.iloc[pos].reset_index(drop=True)
                cache = OUT / f'A_{corpus}_{seed}_{te}.npz'
                if not cache.exists():
                    store = a_split(frame.loc[np.isin(fold, tr)].reset_index(drop=True),
                                    frame.loc[fold == va].reset_index(drop=True), ev, seed,
                                    verifiers)
                    tmp = cache.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp, pos=pos, y=y[pos],
                                        episode_key=ev.episode_key.astype(str).to_numpy(dtype=str),
                                        **store)
                    tmp.replace(cache)
                with np.load(cache, allow_pickle=False) as z:
                    assert np.array_equal(z['pos'], pos) and np.array_equal(z['y'], y[pos])
                    assert np.array_equal(z['episode_key'], ev.episode_key.astype(str).to_numpy())
                    for arm in sorted({k.rsplit('_', 1)[0] for k in z.files if k.endswith('_prob')}):
                        d = pooled.setdefault(arm, {k: np.full(len(frame), np.nan)
                                                    for k in ('prob', 'thr', 'ms')})
                        d['prob'][pos] = z[arm + '_prob']
                        d['ms'][pos] = z[arm + '_ms']
                        d['thr'][pos] = float(z[arm + '_thr'])
                    for name in ('sel::all15', 'sel::pool3'):
                        picks.append({'protocol': 'A', 'candidates': name.split('::')[1],
                                      'corpus': corpus, 'seed': seed, 'test_fold': int(te),
                                      'selected': str(z[f'{name}_pick']),
                                      'val_auroc': float(z[f'{name}_val_auroc'])})
                coverage[pos] += 1
                if smoke:
                    save_json('SMOKE.json', {'pass': True, 'cache': cache.name,
                              'arms': len(pooled), 'note': 'first formal split only'})
                    log(f'A smoke passed on {cache.name}: {len(pooled)} arms cached')
                    return
            assert (coverage == 1).all(), (corpus, seed, 'coverage is not exactly once')
            for arm, d in pooled.items():
                assert all(np.isfinite(v).all() for v in d.values())
                # `thr` is a per-row array: every row is thresholded by the validation
                # threshold of the rotation that produced it, as in E15.
                m = metrics(y, d['prob'], d['ms'], d['thr'])
                got.setdefault(arm, []).append(m)
                per_seed.append({'corpus': corpus, 'system': arm, 'seed': seed, **m})
            log(f'A {corpus} seed={seed}: 10 folds pooled, {len(pooled)} arms')
        for arm, per in got.items():
            if arm in EXCLUDED_ARMS:
                continue
            rows.append(summarise(corpus, arm, per, len(frame)))
    save_csv('A_PERCORPUS.csv', rows)
    save_csv('A_PER_SEED.csv', per_seed)
    save_csv('A_PICKS.csv', picks)
    cross_check_e15(rows)


def cross_check_e15(rows):
    """The two arms E15 already produced must come back identical, apart from live timing."""
    path = E15 / 'A_TABLE7.csv'
    if not path.exists():
        log('A cross-check skipped: no E15 A_TABLE7.csv')
        return
    prior = pd.read_csv(path).set_index(['corpus', 'system'])
    now = pd.DataFrame(rows).set_index(['corpus', 'system'])
    checks = []
    for corpus in P.CORPUS_PAIRS:
        for arm, old_name in [('OURS', 'OURS'), ('sel::pool3', 'pool3')]:
            if (corpus, old_name) not in prior.index or (corpus, arm) not in now.index:
                continue
            for k in METRICS:
                diff = abs(float(now.loc[(corpus, arm), k]) - float(prior.loc[(corpus, old_name), k]))
                # OURS ms carries live-measured head/routing/calibration time, which is not
                # bit-reproducible; its difference is reported, not gated.
                gated = not (arm == 'OURS' and k == 'ms')
                checks.append({'corpus': corpus, 'system': arm, 'metric': k, 'abs_diff': diff,
                               'gated': gated, 'pass': bool(diff <= GATE) if gated else None})
    bad = [c for c in checks if c['gated'] and not c['pass']]
    save_json('A_E15_CROSSCHECK.json', {'pass': not bad, 'failures': bad, 'checks': checks})
    assert not bad, 'A cross-check against E15 failed: ' + json.dumps(bad)
    log(f'A cross-check vs E15 passed on {sum(c["gated"] for c in checks)} gated cells')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--only', choices=['A', 'B'], default=None)
    opt = ap.parse_args()
    files = [Path(__file__), Path(P.__file__), Path(S.__file__), Path(V.__file__), Path(core.__file__)]
    save_json('CONTRACT.json', {
        'seeds': P.SEEDS, 'hp': P.HP, 'pool': P.POOL, 'features': P.FEATURES,
        'python': sys.version, 'platform': platform.platform(), 'threads': 8, 'GPU': False,
        'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'B_split': 'original E1/E1b ladder: stratified_group_split at 0.20/0.30/0.40',
        'A_split': 'V.folds_stratified within corpus, V.rotations 8/1/1, new experiment',
        'timing': {'OURS/no_verifier': 'features+head+routing+both calibration stages',
                   'fixed::* (B)': 'verifier call + isotonic apply (original E1)',
                   'fixed::* (A)': 'verifier call only',
                   'sel::* (A,B)': 'selected verifier call only (original E1b)'}})
    train, test, all_frame, verifiers = V.load(with_test_labels=True)
    verifiers = list(verifiers)
    log(f'{len(verifiers)} verifiers: {verifiers}')
    if opt.only != 'A':
        run_b(train, test, verifiers)
    if opt.only != 'B':
        frames = a_frames(all_frame)
        a_preflight(frames)
        run_a(frames, verifiers, opt.smoke)
    if not opt.smoke:
        save_json('DONE.json', {'status': 'complete'})
        log('DONE: per-corpus full A/B complete')


if __name__ == '__main__':
    main()
