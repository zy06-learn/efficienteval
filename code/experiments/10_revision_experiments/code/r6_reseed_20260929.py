#!/usr/bin/env python3
"""Screening-seed stability of the test-blind historical selection (descriptive).

The frozen selection (r6r9_trainside_20260929.py + apply_frozen_selection_20260929.py) screens
with seed 17 only, as the August-7 procedure did. This script repeats the whole procedure with
another screening seed: the same screen on the same 5,276 TRAIN rows, the same budget picks, the
same confirmation (core.SEEDS x ten rotations x two feature sets; rows already confirmed by the
frozen run are reused because confirmation does not depend on the screening seed), and the same
Balance ordering with the same anchors. It does not change the frozen selection.

Usage: r6_reseed_20260929.py SEED     output: results/day2_v1_trainside_20260929/reseed_SEED/
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys, json, time, itertools, warnings
from pathlib import Path
ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from afr_v2 import tenfold_v1 as core
from afr_v2 import pool_gate_sweep_v1 as v1
import day2_run as D

SEED = int(sys.argv[1])
FROZEN = ROOT / "results/day2_v1_trainside_20260929"
OUT = FROZEN / f"reseed_{SEED}"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
TARGET, SCREEN_LEARNER, FINAL_LEARNER, SCREEN_ROT = "brier", "hgb", "rf", 4
BUDGETS = (100, 150, 200, 250, 300, 400, 600)
PAPER = "factcc+lettuce_v2+granite_guardian_3_1_2b"


def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)


full = v1.load_matrix(ROOT)
test = pd.read_parquet(ROOT / "paper_v2/data/TEST.parquet", columns=["content_doc_key"])
matrix = full.loc[~full["content_doc_key"].isin(set(test["content_doc_key"]))].reset_index(drop=True)
assert len(matrix) == 5276
VER = [v for v in v1.ALL_VERIFIERS if f"correct__{v}" in matrix.columns]
FEATS = D.FEATS
ONEHOT = []
for k in sorted(matrix["dataset_key"].astype(str).unique()):
    c = f"dsid__{k}"
    matrix[c] = matrix["dataset_key"].astype(str).eq(k).astype(float)
    ONEHOT.append(c)


def screen(pool, seed, n_rot=SCREEN_ROT):
    """Identical to r6r9_trainside_20260929.screen."""
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


# gate: seed 17 on three pools must reproduce the frozen screen exactly
ref = pd.read_csv(FROZEN / "R6_SCREEN.csv").set_index("pool")
for p in [PAPER, "factcc+lettuce_v2+wecheck", "lettuce_v2+granite_guardian_3_1_2b+alignscore"]:
    a, ms = screen(tuple(p.split("+")), 17)
    assert abs(a - ref.loc[p, "val_auroc"]) < 1e-12 and abs(ms - ref.loc[p, "val_ms"]) < 1e-9, p
log("gate: seed-17 screen reproduced exactly on three pools")

rows = []
for k in (2, 3):
    for i, pool in enumerate(itertools.combinations(VER, k)):
        try:
            a, ms = screen(pool, SEED)
            rows.append({"k": k, "pool": "+".join(v1.order_by_cost(matrix, pool)),
                         "val_auroc": a, "val_ms": ms})
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_csv(OUT / "R6_SCREEN.csv", index=False)
            log(f"  k={k} {i+1}")
sc = pd.DataFrame(rows).dropna(subset=["val_auroc"])
sc.to_csv(OUT / "R6_SCREEN.csv", index=False)

picks = []
for B in BUDGETS:
    for kk in (2, 3):
        ok = sc[(sc.val_ms <= B) & (sc.k == kk)].sort_values(
            ["val_auroc", "val_ms", "k", "pool"], ascending=[False, True, True, True])
        if len(ok):
            top = ok.iloc[0]
            picks.append({"budget_ms": B, "k": kk, "pool": top.pool, "val_auroc": top.val_auroc,
                          "val_ms": top.val_ms, "margin_to_runner_up":
                          top.val_auroc - ok.iloc[1].val_auroc if len(ok) > 1 else np.nan})
pk = pd.DataFrame(picks)
pk.to_csv(OUT / "BUDGET_PICKS.csv", index=False)
log("budget picks:\n" + pk.round(4).to_string(index=False))

# confirmation: reuse frozen-run rows (+ the runner-up sensitivity run), confirm anything new
have = pd.read_csv(FROZEN / "R6_FINALS.csv")
sens = FROZEN / "SENS_RUNNERUP_100MS_ROWS.csv"
if sens.exists():
    have = pd.concat([have, pd.read_csv(sens)], ignore_index=True)
new = []
for p in dict.fromkeys(pk.pool):
    for cols, tag in ((FEATS, "compact16"), (ONEHOT, "dataset_id_only")):
        if ((have.pool == p) & (have.features == tag)).any():
            continue
        for x in D.full_eval(matrix, tuple(p.split("+")), f"RS{SEED}:{tag}:{p}", seeds=core.SEEDS,
                             cols=cols, target=TARGET, learner=FINAL_LEARNER):
            x.update({"features": tag})
            new.append(x)
        log(f"  confirmed {p} [{tag}]")
if new:
    pd.DataFrame(new).to_csv(OUT / "R6_FINALS_NEW.csv", index=False)
    have = pd.concat([have, pd.DataFrame(new)], ignore_index=True)
fin = have[have.pool.isin(set(pk.pool))]
agg = fin.groupby(["pool", "features"], as_index=False).agg(
    val=("val_auroc", "mean"), heldout_auroc=("treatment_auroc", "mean"), ms=("ms", "mean"),
    n_seeds=("seed", "nunique"))
assert (agg.n_seeds == 3).all()
A = json.loads((FROZEN / "SUMMARY.json").read_text())["anchors"]
q = (agg.heldout_auroc - A["Q_floor"]) / (A["Q_ceil"] - A["Q_floor"])
c = (np.log(agg.ms) - np.log(A["C_min"])) / (np.log(A["C_max"]) - np.log(A["C_min"]))
agg["balance"] = 1 - np.sqrt((1 - q) ** 2 + c ** 2) / np.sqrt(2)
agg["k"] = agg.pool.str.count(r"\+") + 1
agg = agg.sort_values(["balance", "val", "ms", "k", "pool"],
                      ascending=[False, False, True, True, True]).reset_index(drop=True)
agg.to_csv(OUT / "FINALISTS.csv", index=False)
res = {"screen_seed": SEED, "selected_pool": agg.iloc[0].pool, "features": agg.iloc[0].features,
       "heldout_auroc": float(agg.iloc[0].heldout_auroc), "ms": float(agg.iloc[0].ms),
       "balance": float(agg.iloc[0].balance), "paper_pool_is_budget_pick": PAPER in set(pk.pool),
       "same_as_paper_pool": agg.iloc[0].pool == PAPER,
       "budget_picks": pk.to_dict("records"), "seconds": round(time.time() - T0, 1)}
(OUT / "RESULT.json").write_text(json.dumps(res, indent=2, default=float))
log("finalists:\n" + agg.head(10).round(4).to_string(index=False))
log(f"RESULT {json.dumps({k: res[k] for k in ('selected_pool', 'balance', 'same_as_paper_pool')})}")
log("COMPLETE")
