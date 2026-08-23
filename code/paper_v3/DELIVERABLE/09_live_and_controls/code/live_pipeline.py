#!/usr/bin/env python3
"""End-to-end LIVE run of the deployed pipeline: no pre-computed test matrix.

Training uses the frozen TRAIN scores (that is what training is for). At test time the script
computes the routing decision from cheap features alone and then calls EXACTLY ONE verifier per
instance, live. The other two verifiers are never invoked for that row, and their columns in the
test frame are read only afterwards, to check the live result against the matrix-based result.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_v2"))
sys.path.insert(0, str(ROOT / "paper_v3"))
sys.path.insert(0, str(ROOT / "paper_v3" / "DELIVERABLE" / "08_scripts"))
OUT = Path(os.environ["RERUN_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("V3_RUN_DIR", str(OUT))

import v3core as V, core
from afr_v2.unified_summary_verifiers_v1 import build_scorer, API_VERIFIERS

SEED   = int(os.environ.get("LIVE_SEED", "17"))
POOL   = list(core.ACTIONS)
HP     = json.loads((ROOT / "paper_v3" / "artifacts" / "part1c_main_full_v1" /
                     "00_contract" / "HP_SELECTED.json").read_text())
hp_b   = HP["hyperparameters_B"]
print("pool:", POOL, "| hp_B:", hp_b, flush=True)

TRAIN, TEST, _ALL, _v = V.load(with_test_labels=True)
FEATURES = json.loads((ROOT / "paper_v3" / "artifacts" / "part1c_main_full_v1" /
                       "00_contract" / "INHERITED_FROZEN_v3.json").read_text())["features"]
print(f"TRAIN {len(TRAIN)}  TEST {len(TEST)}  features {len(FEATURES)}", flush=True)

# ---------- 1. train exactly as the main experiment does (frozen TRAIN scores) ----------
held = V.stratified_group_split(TRAIN, SEED)
fit, val = TRAIN.loc[~held].reset_index(drop=True), TRAIN.loc[held].reset_index(drop=True)
cals  = core.platt(fit, actions=POOL)
c_fit = core.apply_platt(fit, cals, actions=POOL)
c_val = core.apply_platt(val, cals, actions=POOL)
cvec  = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)

# ---------- 2. the live evaluation cohort ----------
cohort = pd.read_parquet(OUT / "RERUN_COHORT.parquet")
keys   = set(cohort["episode_key"].astype(str))
ev     = TEST[TEST["episode_key"].astype(str).isin(keys)].reset_index(drop=True)
print(f"live eval rows: {len(ev)}", flush=True)

h_val, h_ev = core.fit_heads(fit, [val, ev], SEED, c_fit, actions=POOL,
                             features=FEATURES, hp=hp_b, target="regret", learner="rf")
beta, _ledger = V.choose_beta(val, h_val, c_val, POOL, cvec)
print(f"beta* = {beta}", flush=True)

# ---------- 3. route from cheap features ONLY ----------
avail = np.column_stack([ev[f"available__{a}"].astype(bool).to_numpy() for a in POOL])
util  = np.asarray(h_ev, float) * np.exp(-beta * cvec / cvec.max())
sel   = np.argmax(np.where(avail, util, -np.inf), axis=1)
print("selection shares:", {POOL[k]: round(float((sel == k).mean()), 4) for k in range(3)}, flush=True)

# ---------- 4. call EXACTLY ONE verifier per row, live ----------
# The routing decision above is already final; execution is split by which verifier a row was
# assigned to, because the single GB10 cannot hold the local encoders and the vLLM server at the
# same time. This is the same phase split the original scoring pipeline uses and it cannot change
# any decision: `sel` is computed before this block and never revisited.
PHASE = os.environ.get("LIVE_PHASE", "local")
np.save(OUT / "live_sel.npy", sel)
ev[["episode_key"]].to_parquet(OUT / "live_keys.parquet")
store = OUT / "live_scores.npz"
raw = np.full(len(ev), np.nan); lat = np.full(len(ev), np.nan)
if store.exists():
    z = np.load(store); raw, lat = z["raw"].copy(), z["lat"].copy()

todo_actions = [a for a in POOL if a not in API_VERIFIERS] if PHASE == "local" \
               else [a for a in POOL if a in API_VERIFIERS]
print(f"phase={PHASE}  actions={todo_actions}", flush=True)

tok = os.environ.get("TOKENIZER_PATH")
calls = np.zeros(3, int)
t_all = time.perf_counter()
for k, a in enumerate(POOL):
    if a not in todo_actions:
        continue
    idx = np.where(sel == k)[0]
    idx = np.array([i for i in idx if not np.isfinite(raw[i])], int)
    if not len(idx):
        print(f"  {a}: nothing to do", flush=True); continue
    kw = {"device": "cuda"}
    if a in API_VERIFIERS:
        kw.update(api_base="http://127.0.0.1:8001/v1",
                  served_model="unified-summary-verifier", tokenizer_path=Path(tok))
    sc = build_scorer(a, **kw)
    print(f"  loaded {a}, {len(idx)} rows assigned", flush=True)
    for n, i in enumerate(idx, 1):
        t0 = time.perf_counter()
        out = sc.score_batch([str(ev.loc[i, "source_document"])],
                             [str(ev.loc[i, "candidate_summary"])])[0]
        lat[i] = (time.perf_counter() - t0) * 1000.0
        raw[i] = out.get("score", np.nan)
        calls[k] += 1
        if n % 40 == 0:
            print(f"    {a} {n}/{len(idx)}  {time.perf_counter()-t_all:.0f}s", flush=True)
    if callable(getattr(sc, "close", None)): sc.close()
    del sc
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
np.savez(store, raw=raw, lat=lat)
done = int(np.isfinite(raw).sum())
print(f"phase {PHASE} done: {int(calls.sum())} live calls this phase, "
      f"{done}/{len(ev)} rows scored overall", flush=True)
if done < len(ev):
    print("PHASE_INCOMPLETE — run the other phase before scoring", flush=True)
    sys.exit(0)
calls = np.array([int(((sel == k) & np.isfinite(raw)).sum()) for k in range(3)])
print(f"live calls total = {int(calls.sum())} (= {len(ev)} rows x 1 verifier), "
      f"per verifier {dict(zip(POOL, calls.tolist()))}", flush=True)

# ---------- 5. calibrate the ONE score, decide ----------
# Build a frame carrying the LIVE score in each action's column, then apply the frozen stage-1
# calibrators. Only the [row, sel] entry of the result is ever read, so the other two columns
# never influence anything.
ev_live = ev.copy()
for k, a in enumerate(POOL):
    idx = np.where(sel == k)[0]
    col = ev_live[f"score__{a}"].to_numpy(float).copy()
    col[idx] = raw[idx]
    ev_live[f"score__{a}"] = col
c_live = core.apply_platt(ev_live, cals, actions=POOL)
rows = np.arange(len(ev))
p1_live = np.column_stack([c_live[a] for a in POOL])[rows, sel]
_s, p1_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
y_val = val["label_supported"].to_numpy(int)
q_live = core.isotonic(p1_val, y_val, p1_live)

# ---------- 6. matrix-based counterpart on the SAME rows ----------
c_ev = core.apply_platt(ev, cals, actions=POOL)
_s2, p1_mat, _m2 = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
q_mat = core.isotonic(p1_val, y_val, p1_mat)

from sklearn.metrics import roc_auc_score
y = ev["label_supported"].to_numpy(int)
res = dict(rows=int(len(ev)), live_calls=int(calls.sum()),
           auroc_live_stage1=float(roc_auc_score(y, p1_live)),
           auroc_matrix_stage1=float(roc_auc_score(y, p1_mat)),
           auroc_live_stage2=float(roc_auc_score(y, q_live)),
           auroc_matrix_stage2=float(roc_auc_score(y, q_mat)),
           max_abs_diff_p1=float(np.nanmax(np.abs(p1_live - p1_mat))),
           identical_p1=bool(np.allclose(p1_live, p1_mat, atol=0, rtol=0)),
           mean_live_verifier_ms=float(lat.mean()),
           shares={POOL[k]: float((sel == k).mean()) for k in range(3)})
(OUT / "LIVE_PIPELINE.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
