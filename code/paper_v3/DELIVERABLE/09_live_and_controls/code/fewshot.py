#!/usr/bin/env python3
"""Few-shot adaptation curve, replacing leave-one-dataset-out.

For each held-out corpus c the router is trained on the other three corpora in full plus k
labelled rows drawn from c, and evaluated on c's own test split. k = 0 is the LODO case; the
curve shows how many in-domain examples are needed before the held-out corpus stops being
out-of-domain, which LODO cannot answer because it only measures the k = 0 point.
"""
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
for p in (ROOT, ROOT/"paper_v2", ROOT/"paper_v3", ROOT/"paper_v3"/"DELIVERABLE"/"08_scripts"):
    sys.path.insert(0, str(p))
OUT = ROOT/"paper_v3"/"runs"/"fewshot_v1"; OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("V3_RUN_DIR", str(OUT))
import v3core as V, core
from sklearn.metrics import roc_auc_score

POOL  = list(core.ACTIONS); SEEDS = list(V.SEEDS)
HPB   = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                    "HP_SELECTED.json").read_text())["hyperparameters_B"]
FEATURES = json.loads((ROOT/"paper_v3"/"artifacts"/"part1c_main_full_v1"/"00_contract"/
                       "INHERITED_FROZEN_v3.json").read_text())["features"]
TRAIN, TEST, _A, _v = V.load(with_test_labels=True)
PAIRS = {"cogensumm": ("cogensumm_val", "cogensumm_test"),
         "frank": ("frank_valid", "frank_test"),
         "ragtruth": ("ragtruth_train", "ragtruth_test"),
         "unisumeval": ("unisumeval_train", "unisumeval_dev")}
KS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, -1]     # -1 = the corpus in full

def fit_eval(fit_all, ev, seed):
    held = V.stratified_group_split(fit_all, seed)
    fit, val = fit_all.loc[~held].reset_index(drop=True), fit_all.loc[held].reset_index(drop=True)
    yv = val["label_supported"].to_numpy(int)
    if len(set(yv.tolist())) < 2 or len(fit) < 30:
        return np.nan
    cals  = core.platt(fit, actions=POOL)
    c_fit = core.apply_platt(fit, cals, POOL)
    c_val = core.apply_platt(val, cals, POOL)
    c_ev  = core.apply_platt(ev,  cals, POOL)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=HPB, target="regret", learner="rf")
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
    _s2, p_ev, _m2 = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
    q = core.isotonic(p_val, yv, p_ev)
    y = ev["label_supported"].to_numpy(int)
    return float(roc_auc_score(y, q)) if len(set(y.tolist())) > 1 else np.nan

rows, t0 = [], time.time()
for corpus, (tr_key, te_key) in PAIRS.items():
    in_c  = TRAIN["dataset_key"].astype(str) == tr_key
    base  = TRAIN.loc[~in_c]
    pool  = TRAIN.loc[in_c]
    ev    = TEST[TEST["dataset_key"].astype(str) == te_key].reset_index(drop=True)
    ks = [k if k > 0 else len(pool) for k in KS]
    ks = sorted({min(k, len(pool)) for k in ks})
    print(f"\n=== {corpus}: base {len(base)} rows | pool {len(pool)} | eval {len(ev)} "
          f"| k grid {ks}", flush=True)
    for k in ks:
        aus = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000 + k)
            take = pool.sample(k, random_state=int(rng.integers(0, 2**31 - 1))) if k else pool.iloc[:0]
            fit_all = pd.concat([base, take], ignore_index=True)
            aus.append(fit_eval(fit_all, ev, seed))
        a = np.array(aus, float); ok = np.isfinite(a)
        rows.append(dict(corpus=corpus, k=int(k), n_seeds=int(ok.sum()),
                         auroc=float(a[ok].mean()) if ok.any() else np.nan,
                         auroc_sd=float(a[ok].std()) if ok.any() else np.nan,
                         base_rows=int(len(base)), pool_rows=int(len(pool)),
                         eval_rows=int(len(ev))))
        print(f"  k={k:5d}  AUROC {rows[-1]['auroc']:.5f} +/- {rows[-1]['auroc_sd']:.5f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT/"FEWSHOT_CURVE.csv", index=False)
df = pd.DataFrame(rows); df.to_csv(OUT/"FEWSHOT_CURVE.csv", index=False)
print("\n=== knee analysis (first k reaching 95% of the k=all gain over k=0) ===")
for corpus in PAIRS:
    d = df[df.corpus == corpus].sort_values("k")
    if d.empty or d.auroc.isna().all(): continue
    a0, aN = d.iloc[0].auroc, d.iloc[-1].auroc
    tgt = a0 + 0.95 * (aN - a0)
    hit = d[d.auroc >= tgt]
    print(f"  {corpus:12s} k=0 {a0:.5f} -> k={int(d.iloc[-1].k)} {aN:.5f} | "
          f"95% of the gain reached at k={int(hit.iloc[0].k) if len(hit) else 'n/a'}")
