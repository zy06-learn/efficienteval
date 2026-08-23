#!/usr/bin/env python3
"""Is the router just a dataset classifier?

Four arms on Protocol B, identical except for what the routing heads may look at:
  none        : no features at all -> the router can only rank actions by their average quality
  dataset     : the corpus one-hot ONLY, no cheap features
  cheap       : the six frozen cheap features (the paper's system)
  cheap+dataset : both

If `dataset` lands near `none`, corpus identity alone buys nothing and the six features are
carrying instance-level signal. If `dataset` lands near `cheap`, the router is a dataset
classifier in disguise.
"""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
for p in (ROOT, ROOT/"paper_v2", ROOT/"paper_v3", ROOT/"paper_v3"/"DELIVERABLE"/"08_scripts"):
    sys.path.insert(0, str(p))
OUT = ROOT/"paper_v3"/"runs"/"dataset_control_v1"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("V3_RUN_DIR", str(OUT))
import v3core as V, core
from sklearn.metrics import roc_auc_score

POOL, SEEDS = list(core.ACTIONS), list(V.SEEDS)
HPB = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                  "HP_SELECTED.json").read_text())["hyperparameters_B"]
CHEAP = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                    "INHERITED_FROZEN_v3.json").read_text())["features"]
TRAIN, TEST, _A, _v = V.load(with_test_labels=True)

# corpus one-hot, matched across the train/test key naming
for k in sorted(TRAIN["dataset_key"].astype(str).unique()):
    pre = k.split("_")[0]
    TRAIN[f"ds__{pre}"] = (TRAIN["dataset_key"].astype(str).str.split("_").str[0] == pre).astype(float)
    TEST[f"ds__{pre}"]  = (TEST["dataset_key"].astype(str).str.split("_").str[0] == pre).astype(float)
DS = sorted(c for c in TRAIN.columns if c.startswith("ds__"))
# a constant column so the "no features" arm has a well-formed design matrix
TRAIN["_const"] = 1.0; TEST["_const"] = 1.0

def run(feats):
    aus = []
    for seed in SEEDS:
        held = V.stratified_group_split(TRAIN, seed)
        fit, val = TRAIN.loc[~held].reset_index(drop=True), TRAIN.loc[held].reset_index(drop=True)
        cals = core.platt(fit, actions=POOL)
        c_fit, c_val = core.apply_platt(fit, cals, POOL), core.apply_platt(val, cals, POOL)
        c_ev = core.apply_platt(TEST, cals, POOL)
        h_val, h_ev = core.fit_heads(fit, [val, TEST], seed, c_fit, actions=POOL,
                                     features=feats, hp=HPB, target="regret", learner="rf")
        cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
        beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
        _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
        _s2, p_ev, _m2 = V.route(TEST, h_ev, c_ev, beta, POOL, cvec)
        q = core.isotonic(p_val, val["label_supported"].to_numpy(int), p_ev)
        aus.append(roc_auc_score(TEST["label_supported"].to_numpy(int), q))
    return float(np.mean(aus)), float(np.std(aus)), aus

arms = {"none (constant only)": ["_const"], "dataset one-hot only": DS,
        "six cheap features": CHEAP, "cheap + dataset": CHEAP + DS}
res = {}
for name, f in arms.items():
    m, s, all_ = run(f)
    res[name] = dict(auroc=m, sd=s, n_features=len(f), per_seed=all_)
    print(f"  {name:24s} {m:.5f} +/- {s:.5f}   ({len(f)} features)", flush=True)
base = res["none (constant only)"]["auroc"]; full = res["six cheap features"]["auroc"]
ds = res["dataset one-hot only"]["auroc"]
res["_reading"] = dict(gain_of_cheap_over_none=full - base,
                       gain_of_dataset_over_none=ds - base,
                       fraction_of_cheap_gain_explained_by_dataset=(ds - base) / (full - base))
print(f"\n  cheap features gain over no-features : {full-base:+.5f}")
print(f"  dataset identity gain over no-features: {ds-base:+.5f}")
print(f"  -> corpus identity explains {100*(ds-base)/(full-base):.1f}% of what the cheap "
      f"features buy")
(OUT/"DATASET_ONLY_ARMS.json").write_text(json.dumps(res, indent=2))
