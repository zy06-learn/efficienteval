#!/usr/bin/env python3
"""The k=0 point the main sweep skipped: pure leave-one-dataset-out."""
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
OUT = ROOT/"paper_v3"/"runs"/"fewshot_v1"; os.environ.setdefault("V3_RUN_DIR", str(OUT))
import v3core as V, core
from sklearn.metrics import roc_auc_score
POOL, SEEDS = list(core.ACTIONS), list(V.SEEDS)
HPB = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                  "HP_SELECTED.json").read_text())["hyperparameters_B"]
FEATURES = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                       "INHERITED_FROZEN_v3.json").read_text())["features"]
TRAIN, TEST, _A, _v = V.load(with_test_labels=True)
PAIRS = {"cogensumm": ("cogensumm_val", "cogensumm_test"),
         "frank": ("frank_valid", "frank_test"),
         "ragtruth": ("ragtruth_train", "ragtruth_test"),
         "unisumeval": ("unisumeval_train", "unisumeval_dev")}
def fit_eval(fit_all, ev, seed):
    held = V.stratified_group_split(fit_all, seed)
    fit, val = fit_all.loc[~held].reset_index(drop=True), fit_all.loc[held].reset_index(drop=True)
    yv = val["label_supported"].to_numpy(int)
    if len(set(yv.tolist())) < 2: return np.nan
    cals = core.platt(fit, actions=POOL)
    c_fit, c_val = core.apply_platt(fit, cals, POOL), core.apply_platt(val, cals, POOL)
    c_ev = core.apply_platt(ev, cals, POOL)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=HPB, target="regret", learner="rf")
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
    _s2, p_ev, _m2 = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
    q = core.isotonic(p_val, yv, p_ev)
    y = ev["label_supported"].to_numpy(int)
    return float(roc_auc_score(y, q)) if len(set(y.tolist())) > 1 else np.nan
rows = []
for corpus, (tr_key, te_key) in PAIRS.items():
    base = TRAIN.loc[TRAIN["dataset_key"].astype(str) != tr_key]
    ev   = TEST[TEST["dataset_key"].astype(str) == te_key].reset_index(drop=True)
    a = np.array([fit_eval(base.reset_index(drop=True), ev, s) for s in SEEDS], float)
    ok = np.isfinite(a)
    rows.append(dict(corpus=corpus, k=0, n_seeds=int(ok.sum()),
                     auroc=float(a[ok].mean()), auroc_sd=float(a[ok].std()),
                     base_rows=int(len(base)), pool_rows=0, eval_rows=int(len(ev))))
    print(f"{corpus:12s} k=0 (LODO)  AUROC {rows[-1]['auroc']:.5f} +/- {rows[-1]['auroc_sd']:.5f}",
          flush=True)
old = pd.read_csv(OUT/"FEWSHOT_CURVE.csv")
pd.concat([pd.DataFrame(rows), old], ignore_index=True).sort_values(
    ["corpus", "k"]).to_csv(OUT/"FEWSHOT_CURVE.csv", index=False)
print("merged into FEWSHOT_CURVE.csv")
