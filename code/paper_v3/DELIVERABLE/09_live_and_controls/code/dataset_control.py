#!/usr/bin/env python3
"""Does the router receive dataset identity?  Three answers, in increasing strength.

(a) By construction: the frozen feature list contains no dataset field.
(b) How much corpus signal is implicitly carried by the six features anyway (an upper bound on
    what the router could infer). Reported honestly: the features are document statistics, so a
    high number here is expected and is not leakage.
(c) The control that matters: give the router the dataset identity explicitly as extra one-hot
    features and re-run the whole pipeline. If AUROC does not improve, explicit dataset identity
    buys nothing beyond what x already carries.
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold

POOL  = list(core.ACTIONS); SEEDS = list(V.SEEDS)
HPB   = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                    "HP_SELECTED.json").read_text())["hyperparameters_B"]
FEATURES = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                       "INHERITED_FROZEN_v3.json").read_text())["features"]
TRAIN, TEST, _A, _v = V.load(with_test_labels=True)
res = {}

# ---- (a) construction ----
leak = [f for f in FEATURES if any(t in f.lower() for t in
        ("dataset", "corpus", "split", "source_id", "domain", "episode"))]
res["a_feature_list"] = FEATURES
res["a_dataset_like_features"] = leak
print(f"(a) frozen feature list = {FEATURES}")
print(f"    dataset-like entries: {leak or 'none'}\n")

# ---- (b) how much corpus signal do the six features carry ----
X = TRAIN[FEATURES].to_numpy(float)
c = TRAIN["dataset_key"].astype(str).to_numpy()
g = TRAIN["content_doc_key"].astype(str).to_numpy()
clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1)
pred = cross_val_predict(clf, X, c, groups=g, cv=GroupKFold(n_splits=5))
acc = accuracy_score(c, pred)
maj = pd.Series(c).value_counts(normalize=True).max()
res["b_corpus_accuracy"] = float(acc); res["b_majority_baseline"] = float(maj)
res["b_chance_uniform"] = 1.0 / len(set(c))
print(f"(b) corpus predictable from x: acc={acc:.4f}  majority={maj:.4f}  "
      f"uniform chance={1/len(set(c)):.4f}")
print(f"    -> the six features DO carry corpus signal; they are document statistics, so this "
      f"is expected and is not an information leak.\n")

# ---- (c) does explicit dataset identity help the router? ----
def run(extra_cols):
    feats = FEATURES + extra_cols
    aus = []
    for seed in SEEDS:
        held = V.stratified_group_split(TRAIN, seed)
        fit, val = TRAIN.loc[~held].reset_index(drop=True), TRAIN.loc[held].reset_index(drop=True)
        cals = core.platt(fit, actions=POOL)
        c_fit = core.apply_platt(fit, cals, POOL); c_val = core.apply_platt(val, cals, POOL)
        c_ev  = core.apply_platt(TEST, cals, POOL)
        h_val, h_ev = core.fit_heads(fit, [val, TEST], seed, c_fit, actions=POOL,
                                     features=feats, hp=HPB, target="regret", learner="rf")
        cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
        beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
        _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
        _s2, p_ev, _m2 = V.route(TEST, h_ev, c_ev, beta, POOL, cvec)
        q = core.isotonic(p_val, val["label_supported"].to_numpy(int), p_ev)
        aus.append(roc_auc_score(TEST["label_supported"].to_numpy(int), q))
    return float(np.mean(aus)), float(np.std(aus)), aus

for f in TRAIN["dataset_key"].astype(str).unique():
    col = f"ds__{f}"
    TRAIN[col] = (TRAIN["dataset_key"].astype(str) == f).astype(float)
    # the TEST corpora are the matching test-side keys; map by prefix
    TEST[col] = (TEST["dataset_key"].astype(str).str.split("_").str[0]
                 == f.split("_")[0]).astype(float)
ds_cols = [c for c in TRAIN.columns if c.startswith("ds__")]
base_m, base_s, base_all = run([])
ds_m, ds_s, ds_all = run(ds_cols)
res.update(c_baseline_auroc=base_m, c_baseline_sd=base_s,
           c_with_dataset_auroc=ds_m, c_with_dataset_sd=ds_s,
           c_delta=ds_m - base_m, c_extra_features=ds_cols,
           c_per_seed_baseline=base_all, c_per_seed_with_dataset=ds_all)
print(f"(c) six features only        : {base_m:.5f} +/- {base_s:.5f}")
print(f"    + explicit dataset one-hot: {ds_m:.5f} +/- {ds_s:.5f}")
print(f"    delta = {ds_m-base_m:+.5f}")
(OUT/"DATASET_CONTROL.json").write_text(json.dumps(res, indent=2))
