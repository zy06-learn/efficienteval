#!/usr/bin/env python3
"""Re-run the August-7 pool selection (R6 + R9) with every test-side document removed.

The original selection ran on the 6,850-row day-2 matrix, whose "frank_train" rows are FRANK's
official validation (669) AND test (1,569) splits; five RAGTruth-train rows also share a source
document with the CoGenSumm test split. Those 1,574 rows belong to documents that the final paper
uses as Protocol B TEST (351 of its 645 document groups). This script drops every row whose
content_doc_key appears in paper_v2/data/TEST.parquet -- leaving exactly the 5,276 TRAIN rows --
and repeats the selection unchanged otherwise:

  R6  screen every 2- and 3-member pool (seed 17, 4 rotations, HGB heads, brier target) on the
      validation fold; for each latency budget and pool size keep the highest validation AUROC;
      confirm each pick with core.SEEDS x ten rotations, RF heads, two feature sets.
  R9  rank the confirmed pools together with the fixed verifiers by Balance, anchors recomputed
      on the reduced matrix exactly as r9_run.py computes them.

Added for reference (not part of the original procedure):
  * the paper's pool is always confirmed, so its position is known even if no budget picks it;
  * a validation-only Balance ranking of every screened pool.

Nothing in results/day2_v1 is read or written. Output: results/day2_v1_trainside_20260929/.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "10")
import sys, json, time, itertools, traceback, warnings
from pathlib import Path
ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from afr_v2 import tenfold_v1 as core
from afr_v2 import pool_gate_sweep_v1 as v1
from afr_v2 import summary_router_compact16_direct_v1 as base
import day2_run as D

SMOKE = os.environ.get("SMOKE") == "1"
OUT = ROOT / "results/day2_v1_trainside_20260929" / ("smoke" if SMOKE else "")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
TARGET, SCREEN_LEARNER, FINAL_LEARNER, SCREEN_ROT = "brier", "hgb", "rf", 4
BUDGETS = (100, 150, 200, 250, 300, 400, 600)
PAPER_POOL = ("factcc", "lettuce_v2", "granite_guardian_3_1_2b")


def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)


# ------------------------------------------------------------------ data: drop test-side documents
full = v1.load_matrix(ROOT)
test = pd.read_parquet(ROOT / "paper_v2/data/TEST.parquet", columns=["content_doc_key"])
train = pd.read_parquet(ROOT / "paper_v2/data/TRAIN.parquet", columns=["dataset_key", "episode_id"])
drop = full["content_doc_key"].isin(set(test["content_doc_key"]))
matrix = full.loc[~drop].reset_index(drop=True)
dropped = full.loc[drop, "dataset_key"].value_counts().to_dict()
assert len(matrix) == len(train) == 5276, (len(matrix), len(train))
assert set(matrix["episode_id"]) == set(train["episode_id"]), "reduced matrix is not the TRAIN set"
assert not matrix["content_doc_key"].isin(set(test["content_doc_key"])).any()
log(f"matrix {len(full)} -> {len(matrix)} rows; dropped {dropped}; "
    f"kept {matrix['dataset_key'].value_counts().to_dict()}")

VER = [v for v in v1.ALL_VERIFIERS if f"correct__{v}" in matrix.columns]
VCOST = {v: float(matrix[f"latency_ms__{v}"].mean()) for v in VER}
FEATS = D.FEATS
keys = sorted(matrix["dataset_key"].astype(str).unique())
ONEHOT = []
for k in keys:
    c = f"dsid__{k}"
    matrix[c] = matrix["dataset_key"].astype(str).eq(k).astype(float)
    ONEHOT.append(c)


def screen(pool, seed=17, n_rot=SCREEN_ROT):
    """Identical to r6_run.screen: validation AUROC and validation latency."""
    actions = v1.order_by_cost(matrix, pool)
    costs = v1.pool_costs(matrix, pool)
    folds = core.build_folds(matrix, seed)
    va, vm = [], []
    for test_f, val_f, train_f in core.rotations()[:n_rot]:
        tr = matrix.loc[np.isin(folds, train_f)].reset_index(drop=True)
        val = matrix.loc[folds == val_f].reset_index(drop=True)
        cals = core.fit_calibrators_on_val(val, actions)
        c_tr = core.apply_calibrators(tr, actions, cals)
        c_va = core.apply_calibrators(val, actions, cals)
        hv = D.fit_heads(tr, [val], FEATS, actions, seed, TARGET, c_tr, SCREEN_LEARNER)[0]
        beta = core.select_beta(val, actions, hv, c_va, costs)[0]
        p = v1.router_prediction(val, actions, hv, c_va, costs, beta)
        va.append(v1.scope_auroc(p, "pooled"))
        vm.append(float(p["end_to_end_latency_ms"].mean()))
    return float(np.mean(va)), float(np.mean(vm))


# ------------------------------------------------------------------ R6 screen, k = 2 and 3
rows = []
for k in (2, 3):
    combos = list(itertools.combinations(VER, k))
    if SMOKE:
        combos = combos[:4] + [c for c in combos if set(c) == set(PAPER_POOL)]
    log(f"screening k={k}: {len(combos)} pools")
    for i, pool in enumerate(combos):
        try:
            a, ms = screen(pool)
            cs = sorted(VCOST[v] for v in pool)
            rows.append({"k": k, "pool": "+".join(v1.order_by_cost(matrix, pool)),
                         "val_auroc": a, "val_ms": ms, "cheapest_member_ms": cs[0],
                         "priciest_member_ms": cs[-1], "cost_spread": cs[-1] / max(cs[0], 1e-9)})
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            base.write_csv(OUT / "R6_SCREEN.csv", pd.DataFrame(rows))
            log(f"  k={k} {i+1}/{len(combos)}")
    base.write_csv(OUT / "R6_SCREEN.csv", pd.DataFrame(rows))
sc = pd.DataFrame(rows).dropna(subset=["val_auroc"])
log(f"screened pools available: {sc.k.value_counts().sort_index().to_dict()}")

# ------------------------------------------------------------------ budget picks
picks = []
for B in BUDGETS:
    for kk in (2, 3):
        ok = sc[(sc.val_ms <= B) & (sc.k == kk)]
        if len(ok):
            picks.append({"budget_ms": B, **ok.loc[ok.val_auroc.idxmax()].to_dict()})
pk = pd.DataFrame(picks)
base.write_csv(OUT / "R6_BUDGET_PICKS.csv", pk)
log("best pool per budget (validation):\n" + pk[["budget_ms", "k", "pool", "val_auroc", "val_ms",
                                                  "cost_spread"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ confirm picks (+ the paper pool)
paper = "+".join(v1.order_by_cost(matrix, PAPER_POOL))
todo = list(dict.fromkeys(list(pk.pool) + [paper]))
if SMOKE:
    todo = [paper]
finals = []
for p in todo:
    pool = tuple(p.split("+"))
    r = pk[pk.pool == p]
    for cols, tag in ((FEATS, "compact16"), (ONEHOT, "dataset_id_only")):
        try:
            rs = D.full_eval(matrix, pool, f"TS:{tag}:{p}",
                             seeds=core.SEEDS[:1] if SMOKE else core.SEEDS, cols=cols,
                             target=TARGET, learner=FINAL_LEARNER)
            for x in rs:
                x.update({"budget_ms": int(r.budget_ms.min()) if len(r) else -1,
                          "picked_by_budget": bool(len(r)), "features": tag,
                          "is_paper_pool": p == paper})
            finals += rs
        except Exception:
            log(f"  final {p}/{tag} FAILED\n{traceback.format_exc()[:300]}")
    base.write_csv(OUT / "R6_FINALS.csv", pd.DataFrame(finals))
ff = pd.DataFrame(finals)
agg = ff.groupby(["pool", "features", "picked_by_budget", "is_paper_pool"], as_index=False).agg(
    val=("val_auroc", "mean"), test=("treatment_auroc", "mean"), ms=("ms", "mean"),
    n_seeds=("seed", "nunique"))
base.write_csv(OUT / "R6_FINALS_AGG.csv", agg)

# ------------------------------------------------------------------ R9: Balance, anchors as r9_run.py
fixed = []
for v in VER:
    ok = matrix[f"available__{v}"].astype(bool).to_numpy()
    sub = matrix.loc[ok]
    fixed.append({"name": v, "family": "fixed_verifier", "ms": VCOST[v], "auroc": float(
        roc_auc_score(sub["label_supported"].to_numpy(int), sub[f"score__{v}"].to_numpy(float)))})
fx = pd.DataFrame(fixed)
C_MIN, C_MAX = min(VCOST.values()), max(VCOST.values())
CHEAPEST = min(VER, key=lambda v: VCOST[v])
Q_FLOOR = float(fx.loc[fx.name == CHEAPEST, "auroc"].iloc[0])
BEST = fx.loc[fx.auroc.idxmax()]
Q_CEIL = float(BEST.auroc)
anchors = {"Q_floor": Q_FLOOR, "cheapest": CHEAPEST, "Q_ceil": Q_CEIL, "best_fixed": BEST["name"],
           "C_min": C_MIN, "C_max": C_MAX}
log(f"anchors {anchors}")


def balance(a, ms):
    q = (np.asarray(a, float) - Q_FLOOR) / (Q_CEIL - Q_FLOOR)
    c = (np.log(np.asarray(ms, float)) - np.log(C_MIN)) / (np.log(C_MAX) - np.log(C_MIN))
    return 1.0 - np.sqrt((1.0 - q) ** 2 + c ** 2) / np.sqrt(2.0)


cand = agg.assign(name=agg.pool + " | " + agg.features, family="R6_budget_pool",
                  auroc=agg.test)[["name", "family", "auroc", "ms", "val", "picked_by_budget",
                                   "is_paper_pool"]]
rank = pd.concat([cand, fx[["name", "family", "auroc", "ms"]]], ignore_index=True)
rank["balance"] = balance(rank.auroc, rank.ms)
rank = rank.sort_values("balance", ascending=False).reset_index(drop=True)
rank.insert(0, "rank", np.arange(1, len(rank) + 1))
base.write_csv(OUT / "R9_BALANCE_RANKING.csv", rank)
log("Balance ranking (held-out AUROC over ten rotations, as R9):\n"
    + rank.head(20).round(4).to_string(index=False))

# ------------------------------------------------------------------ reference: validation-only Balance
sv = sc.assign(balance=balance(sc.val_auroc, sc.val_ms)).sort_values("balance", ascending=False)
sv = sv.reset_index(drop=True)
sv.insert(0, "rank", np.arange(1, len(sv) + 1))
base.write_csv(OUT / "SCREEN_BALANCE_VALIDATION.csv", sv)
pos = sv.index[sv.pool == paper]
log("validation-only Balance, top 15:\n" + sv.head(15).round(4).to_string(index=False))
log(f"paper pool {paper}: validation-Balance rank "
    f"{int(pos[0]) + 1 if len(pos) else -1} of {len(sv)}")

base.write_json(OUT / "SUMMARY.json", {
    "status": "COMPLETE", "rows_full": int(len(full)), "rows_used": int(len(matrix)),
    "dropped_by_dataset": {k: int(v) for k, v in dropped.items()}, "anchors": anchors,
    "budget_picks": pk[["budget_ms", "k", "pool", "val_auroc", "val_ms"]].to_dict("records"),
    "balance_rank1": rank.iloc[0][["name", "auroc", "ms", "balance"]].to_dict(),
    "paper_pool_balance_rank": int(rank.index[rank.name.str.startswith(paper + " |")].min()) + 1,
    "paper_pool_validation_balance_rank": int(pos[0]) + 1 if len(pos) else -1,
    "seconds": round(time.time() - T0, 1), "completed_at_utc": base.utc_now()})
log("COMPLETE")
