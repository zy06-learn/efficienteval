#!/usr/bin/env python3
"""Apply FREEZE_testblind_selection.md to the finished selection run. Reads CSVs only.

Outputs (same directory):
  SELECTION_BUDGET_TABLE.csv   one row per budget and pool size: pick, validation AUROC and
                               latency, validation-level Balance
  SELECTION_FINALISTS.csv      eligible finalists with held-out AUROC, latency and Balance
  SELECTION_SEED_STABILITY.csv the Stage-3 winner on each seed separately
  SELECTION_RESULT.json        the frozen choice
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/home/zeyu/projects/adaptive-faithfulness-router-v2/results/day2_v1_trainside_20260929")
BUDGETS = (100, 150, 200, 250, 300, 400, 600)
summ = json.loads((D / "SUMMARY.json").read_text())
A = summ["anchors"]


def balance(a, ms):
    q = (np.asarray(a, float) - A["Q_floor"]) / (A["Q_ceil"] - A["Q_floor"])
    c = (np.log(np.asarray(ms, float)) - np.log(A["C_min"])) / (np.log(A["C_max"]) - np.log(A["C_min"]))
    return 1.0 - np.sqrt((1.0 - q) ** 2 + c ** 2) / np.sqrt(2.0)


# ---- Stage 1: budget picks with the frozen tie-break
sc = pd.read_csv(D / "R6_SCREEN.csv").dropna(subset=["val_auroc"])
rows = []
for B in BUDGETS:
    for k in (2, 3):
        ok = sc[(sc.val_ms <= B) & (sc.k == k)].copy()
        if ok.empty:
            continue
        ok = ok.sort_values(["val_auroc", "val_ms", "k", "pool"],
                            ascending=[False, True, True, True])
        top = ok.iloc[0]
        runner = ok.iloc[1] if len(ok) > 1 else None
        rows.append({"budget_ms": B, "k": k, "pool": top.pool, "val_auroc": top.val_auroc,
                     "val_ms": top.val_ms, "val_balance": float(balance(top.val_auroc, top.val_ms)),
                     "margin_to_runner_up": (top.val_auroc - runner.val_auroc) if runner is not None
                     else np.nan, "tie": bool(runner is not None and runner.val_auroc == top.val_auroc)})
bt = pd.DataFrame(rows)
old = pd.read_csv(D / "R6_BUDGET_PICKS.csv")
same = (bt[["budget_ms", "k", "pool"]].reset_index(drop=True)
        .equals(old[["budget_ms", "k", "pool"]].reset_index(drop=True)))
assert same, "frozen tie-break changes a budget pick"
assert not bt.tie.any(), "an exact tie occurred at a budget pick"
bt.to_csv(D / "SELECTION_BUDGET_TABLE.csv", index=False)

# ---- Stage 2/3: eligible finalists, Balance, frozen ordering
ff = pd.read_csv(D / "R6_FINALS.csv")
elig = ff[ff.picked_by_budget.astype(bool)]
agg = elig.groupby(["pool", "features"], as_index=False).agg(
    val=("val_auroc", "mean"), heldout_auroc=("treatment_auroc", "mean"), ms=("ms", "mean"),
    n_seeds=("seed", "nunique"))
agg["k"] = agg.pool.str.count(r"\+") + 1
agg["balance"] = balance(agg.heldout_auroc, agg.ms)
agg = agg.sort_values(["balance", "val", "ms", "k", "pool"],
                      ascending=[False, False, True, True, True]).reset_index(drop=True)
agg.insert(0, "rank", np.arange(1, len(agg) + 1))
agg.to_csv(D / "SELECTION_FINALISTS.csv", index=False)
winner = agg.iloc[0]

# ---- seed stability (descriptive)
st = []
for s, g in elig.groupby("seed"):
    g = g.assign(k=g.pool.str.count(r"\+") + 1, balance=balance(g.treatment_auroc, g.ms))
    g = g.sort_values(["balance", "val_auroc", "ms", "k", "pool"],
                      ascending=[False, False, True, True, True])
    st.append({"seed": int(s), "winner_pool": g.iloc[0].pool, "features": g.iloc[0].features,
               "balance": float(g.iloc[0].balance), "runner_up_pool": g.iloc[1].pool,
               "runner_up_balance": float(g.iloc[1].balance)})
st = pd.DataFrame(st)
st.to_csv(D / "SELECTION_SEED_STABILITY.csv", index=False)

res = {"selected_pool": winner.pool, "selected_features_config": winner.features,
       "heldout_auroc": float(winner.heldout_auroc), "ms": float(winner.ms),
       "balance": float(winner.balance),
       "runner_up": agg.iloc[1][["pool", "features", "heldout_auroc", "ms", "balance"]].to_dict(),
       "historical_pool": "factcc+lettuce_v2+granite_guardian_3_1_2b",
       "same_as_historical": winner.pool == "factcc+lettuce_v2+granite_guardian_3_1_2b",
       "seed_winner_counts": st.winner_pool.value_counts().to_dict(), "anchors": A,
       "budget_picks_identical_to_run": same}
(D / "SELECTION_RESULT.json").write_text(json.dumps(res, indent=2, default=float))
pd.set_option("display.width", 200)
print("BUDGET PICKS\n", bt.round(4).to_string(index=False))
print("\nFINALISTS (eligible, frozen order)\n", agg.round(4).to_string(index=False))
print("\nSEED STABILITY\n", st.round(4).to_string(index=False))
print("\nRESULT", json.dumps(res, indent=1, default=float))
