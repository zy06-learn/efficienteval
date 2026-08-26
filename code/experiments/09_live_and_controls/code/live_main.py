#!/usr/bin/env python3
"""Protocol B main result, executed LIVE on every test row.

No pre-computed test matrix is consumed on the evaluation side. Training uses the frozen TRAIN
scores (that is what training is for). For every (row, seed) pair the router picks one verifier
from cheap features alone and that verifier is then called for real. Scores are never reused
across seeds: a row selected by two seeds is called twice.

Stages (env LIVE_STAGE): plan -> local -> api -> finish
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
for p in (ROOT, ROOT/"ingest_and_scoring", ROOT/"experiments", ROOT/"experiments"/"08_routing_code"):
    sys.path.insert(0, str(p))
OUT = Path(os.environ["LIVE_OUT"]); (OUT/"logs").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("V3_RUN_DIR", str(OUT))
STAGE = os.environ.get("LIVE_STAGE", "plan")

import v3core as V, core
from verifier_wrappers.unified_summary_verifiers_v1 import build_scorer, API_VERIFIERS

POOL  = list(core.ACTIONS)
SEEDS = list(V.SEEDS)
HPB   = json.loads((ROOT/"experiments"/"cross_stage_contract"/"part1c_main_full_v1"/"00_contract"/
                    "HP_SELECTED.json").read_text())["hyperparameters_B"]
FEATURES = json.loads((ROOT/"experiments"/"cross_stage_contract"/"part1c_main_full_v1"/"00_contract"/
                       "INHERITED_FROZEN_v3.json").read_text())["features"]
TRAIN, TEST, _A, _v = V.load(with_test_labels=True)
EV = TEST.reset_index(drop=True)
N, S = len(EV), len(SEEDS)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def train_seed(seed):
    held = V.stratified_group_split(TRAIN, seed)
    fit  = TRAIN.loc[~held].reset_index(drop=True)
    val  = TRAIN.loc[held].reset_index(drop=True)
    cals = core.platt(fit, actions=POOL)
    c_fit, c_val = core.apply_platt(fit, cals, POOL), core.apply_platt(val, cals, POOL)
    h_val, h_ev = core.fit_heads(fit, [val, EV], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=HPB, target="regret", learner="rf")
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    return dict(fit=fit, val=val, cals=cals, c_val=c_val, h_val=h_val,
                h_ev=h_ev, cvec=cvec, beta=beta)

AVAIL = np.column_stack([EV[f"available__{a}"].astype(bool).to_numpy() for a in POOL])

# ---------------------------------------------------------------- plan
if STAGE == "plan":
    sel  = np.zeros((N, S), np.int8); betas = []
    for si, seed in enumerate(SEEDS):
        t = train_seed(seed)
        util = np.asarray(t["h_ev"], float) * np.exp(-t["beta"] * t["cvec"] / t["cvec"].max())
        sel[:, si] = np.argmax(np.where(AVAIL, util, -np.inf), axis=1)
        betas.append(float(t["beta"]))
        log(f"seed {seed}: beta*={t['beta']}  shares=" +
            str({POOL[k]: round(float((sel[:, si] == k).mean()), 4) for k in range(3)}))
    np.savez(OUT/"plan.npz", sel=sel, betas=np.array(betas), seeds=np.array(SEEDS))
    need = np.array([(sel == k).sum() for k in range(3)])
    log(f"planned live calls: {int(need.sum())} = " + str(dict(zip(POOL, need.tolist()))))
    sys.exit(0)

Z   = np.load(OUT/"plan.npz"); sel = Z["sel"]
store = OUT/"live_raw.npz"
raw = np.load(store)["raw"] if store.exists() else np.full((N, S), np.nan)

# ---------------------------------------------------------------- execute
if STAGE in ("local", "api"):
    want = [a for a in POOL if (a in API_VERIFIERS) == (STAGE == "api")]
    tok  = os.environ.get("TOKENIZER_PATH")
    for k, a in enumerate(POOL):
        if a not in want: continue
        rr, ss = np.where((sel == k) & ~np.isfinite(raw))
        if not len(rr): log(f"{a}: nothing to do"); continue
        kw = {"device": "cuda"}
        if a in API_VERIFIERS:
            kw.update(api_base="http://127.0.0.1:8001/v1",
                      served_model="unified-summary-verifier", tokenizer_path=Path(tok))
        sc = build_scorer(a, **kw); log(f"{a}: {len(rr)} live calls to make")
        t0 = time.perf_counter()
        for n, (i, s) in enumerate(zip(rr, ss), 1):
            o = sc.score_batch([str(EV.loc[i, "source_document"])],
                               [str(EV.loc[i, "candidate_summary"])])[0]
            raw[i, s] = o.get("score", np.nan)
            if n % 500 == 0:
                el = time.perf_counter() - t0
                log(f"  {a} {n}/{len(rr)}  {el:.0f}s  eta {el/n*(len(rr)-n):.0f}s")
                np.savez(store, raw=raw)
        if callable(getattr(sc, "close", None)): sc.close()
        del sc
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
        np.savez(store, raw=raw)
        log(f"{a} done in {time.perf_counter()-t0:.0f}s")
    done = int(np.isfinite(raw).sum()); log(f"stage {STAGE}: {done}/{sel.size} scored")
    sys.exit(0)

# ---------------------------------------------------------------- finish
from sklearn.metrics import roc_auc_score
y = EV["label_supported"].to_numpy(int)
rows = np.arange(N)
au_live, au_mat, mism = [], [], []
for si, seed in enumerate(SEEDS):
    t = train_seed(seed)
    ev_live = EV.copy()
    for k, a in enumerate(POOL):
        idx = np.where(sel[:, si] == k)[0]
        col = ev_live[f"score__{a}"].to_numpy(float).copy(); col[idx] = raw[idx, si]
        ev_live[f"score__{a}"] = col
    c_live = core.apply_platt(ev_live, t["cals"], POOL)
    c_mat  = core.apply_platt(EV,      t["cals"], POOL)
    p1_live = np.column_stack([c_live[a] for a in POOL])[rows, sel[:, si]]
    p1_mat  = np.column_stack([c_mat[a]  for a in POOL])[rows, sel[:, si]]
    _s, p1_val, _m = V.route(t["val"], t["h_val"], t["c_val"], t["beta"], POOL, t["cvec"])
    yv = t["val"]["label_supported"].to_numpy(int)
    q_live = core.isotonic(p1_val, yv, p1_live)
    q_mat  = core.isotonic(p1_val, yv, p1_mat)
    au_live.append(roc_auc_score(y, q_live)); au_mat.append(roc_auc_score(y, q_mat))
    mism.append(float(np.nanmax(np.abs(p1_live - p1_mat))))
    log(f"seed {seed}: live {au_live[-1]:.5f}  matrix {au_mat[-1]:.5f}  "
        f"max|Δp1| {mism[-1]:.3e}")
res = dict(rows=N, seeds=S, live_calls=int(np.isfinite(raw).sum()),
           auroc_live_mean=float(np.mean(au_live)), auroc_live_sd=float(np.std(au_live)),
           auroc_matrix_mean=float(np.mean(au_mat)), auroc_matrix_sd=float(np.std(au_mat)),
           delta=float(np.mean(au_live) - np.mean(au_mat)),
           max_abs_diff_p1=float(np.max(mism)),
           shares={POOL[k]: float((sel == k).mean()) for k in range(3)},
           per_seed_live=au_live, per_seed_matrix=au_mat)
(OUT/"LIVE_MAIN_B.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
