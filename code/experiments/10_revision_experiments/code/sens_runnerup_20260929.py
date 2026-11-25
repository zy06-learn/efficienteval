#!/usr/bin/env python3
"""Sensitivity (not part of the frozen selection): confirm the 100 ms runner-up pool
factcc+lettuce_v2+wecheck exactly as the finalists were confirmed, and score it by Balance."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "10")
ROOT = "/home/zeyu/projects/adaptive-faithfulness-router-v2"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from pathlib import Path
import numpy as np, pandas as pd
from afr_v2 import tenfold_v1 as core, pool_gate_sweep_v1 as v1
import day2_run as D
OUT = Path(ROOT) / "results/day2_v1_trainside_20260929"
full = v1.load_matrix(Path(ROOT))
test = pd.read_parquet(Path(ROOT) / "paper_v2/data/TEST.parquet", columns=["content_doc_key"])
m = full.loc[~full.content_doc_key.isin(set(test.content_doc_key))].reset_index(drop=True)
assert len(m) == 5276
keys = sorted(m["dataset_key"].astype(str).unique()); ONEHOT = []
for k in keys:
    c = f"dsid__{k}"; m[c] = m["dataset_key"].astype(str).eq(k).astype(float); ONEHOT.append(c)
A = json.loads((OUT / "SUMMARY.json").read_text())["anchors"]
def bal(a, ms):
    q = (a - A["Q_floor"]) / (A["Q_ceil"] - A["Q_floor"])
    c = (np.log(ms) - np.log(A["C_min"])) / (np.log(A["C_max"]) - np.log(A["C_min"]))
    return 1 - np.sqrt((1 - q) ** 2 + c ** 2) / np.sqrt(2)
rows = []
pool = ("factcc", "lettuce_v2", "wecheck")
for cols, tag in ((D.FEATS, "compact16"), (ONEHOT, "dataset_id_only")):
    for r in D.full_eval(m, pool, f"SENS:{tag}", seeds=core.SEEDS, cols=cols, target="brier", learner="rf"):
        r["features"] = tag; rows.append(r)
d = pd.DataFrame(rows)
g = d.groupby("features", as_index=False).agg(val=("val_auroc", "mean"), heldout_auroc=("treatment_auroc", "mean"), ms=("ms", "mean"))
g["balance"] = bal(g.heldout_auroc, g.ms)
g.insert(0, "pool", "+".join(pool))
g.to_csv(OUT / "SENS_RUNNERUP_100MS.csv", index=False)
print(g.round(4).to_string(index=False))
