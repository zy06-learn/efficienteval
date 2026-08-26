#!/usr/bin/env python3
"""Few-shot adaptation curve on a fraction grid rather than an absolute-count grid.

The absolute grid made the four corpora incomparable: k=512 is 96% of CoGenSumm's pool but 17%
of RAGTruth's, so the same x position meant different things on each curve. It also spent five
of its twelve points on k in {1,2,4,8,16}, where adding a handful of rows to a base of 2,293 to
4,741 cannot move anything -- and indeed did not: FRANK and UniSumEval report identical AUROC to
four decimals at k=0 and k=1.

Fractions of each corpus's own training pool fix both problems. The grid is placed where the
absolute run showed movement.

Everything else follows fewshot.py exactly, including drawing the subset per seed and
concatenating it onto the base in that order, because row order is not neutral here.

Row-level probabilities are kept for the 0% and 100% ends so the endpoint comparison can be
tested with the same paired cluster bootstrap the rest of the paper uses.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
for p in (ROOT, ROOT / "ingest_and_scoring", ROOT / "experiments",
          ROOT / "experiments" / "experiments" / "08_routing_code",
          ROOT / "experiments" / "experiments" / "09_live_and_controls" / "code"):
    sys.path.insert(0, str(p))
OUT = Path(os.environ.get("FS_OUT", ROOT / "experiments" / "runs" / "fewshot_frac_v1"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("V3_RUN_DIR", str(OUT))

import core  # noqa: E402
import v3core as V  # noqa: E402
import _contract  # noqa: E402

POOL, SEEDS = list(core.ACTIONS), list(V.SEEDS)
C = _contract.load()
HPB, FEATURES = dict(C["hp_B"]), list(C["features"])
TRAIN, TEST, _A, _v = V.load(with_test_labels=True)

PAIRS = {"cogensumm": ("cogensumm_val", "cogensumm_test"),
         "frank": ("frank_valid", "frank_test"),
         "ragtruth": ("ragtruth_train", "ragtruth_test"),
         "unisumeval": ("unisumeval_train", "unisumeval_dev")}
FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.85, 1.00]


def fit_eval(fit_all, ev, seed):
    """fewshot.py's fit_eval, with the per-row calibrated probability returned as well."""
    held = V.stratified_group_split(fit_all, seed)
    fit = fit_all.loc[~held].reset_index(drop=True)
    val = fit_all.loc[held].reset_index(drop=True)
    yv = val["label_supported"].to_numpy(int)
    if len(set(yv.tolist())) < 2 or len(fit) < 30:
        return np.nan, None
    cals = core.platt(fit, actions=POOL)
    c_fit = core.apply_platt(fit, cals, POOL)
    c_val = core.apply_platt(val, cals, POOL)
    c_ev = core.apply_platt(ev, cals, POOL)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=HPB, target="regret", learner="rf")
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
    _s2, p_ev, _m2 = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
    q = core.isotonic(p_val, yv, p_ev)
    y = ev["label_supported"].to_numpy(int)
    if len(set(y.tolist())) < 2:
        return np.nan, None
    return float(roc_auc_score(y, q)), q


rows, ends, t0 = [], {}, time.time()
for corpus, (tr_key, te_key) in PAIRS.items():
    in_c = TRAIN["dataset_key"].astype(str) == tr_key
    base = TRAIN.loc[~in_c]
    pool = TRAIN.loc[in_c]
    ev = TEST[TEST["dataset_key"].astype(str) == te_key].reset_index(drop=True)
    print(f"\n=== {corpus}: base {len(base)} | pool {len(pool)} | eval {len(ev)}", flush=True)

    for frac in FRACS:
        k = int(round(frac * len(pool)))
        aus, qs = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000 + k)
            take = (pool.sample(k, random_state=int(rng.integers(0, 2**31 - 1)))
                    if k else pool.iloc[:0])
            fit_all = pd.concat([base, take], ignore_index=True)
            a, q = fit_eval(fit_all, ev, seed)
            aus.append(a)
            qs.append(q)
        a = np.array(aus, float)
        ok = np.isfinite(a)
        rows.append(dict(corpus=corpus, frac=frac, k=k, n_seeds=int(ok.sum()),
                         auroc=float(a[ok].mean()) if ok.any() else np.nan,
                         auroc_sd=float(a[ok].std()) if ok.any() else np.nan,
                         base_rows=len(base), pool_rows=len(pool), eval_rows=len(ev)))
        print(f"  {frac:5.0%}  k={k:5d}  AUROC {rows[-1]['auroc']:.5f} "
              f"+/-{rows[-1]['auroc_sd']:.5f}   ({time.time()-t0:.0f}s)", flush=True)
        if frac in (0.0, 1.00) and all(q is not None for q in qs):
            ends[f"{corpus}|{frac:.2f}"] = np.column_stack(qs)
        pd.DataFrame(rows).to_csv(OUT / "FEWSHOT_FRACTION_CURVE.csv", index=False)

    ends[f"{corpus}|y"] = ev["label_supported"].to_numpy(int)
    ends[f"{corpus}|g"] = ev["content_doc_key"].astype(str).to_numpy()

np.savez(OUT / "fewshot_endpoints.npz", **ends)
pd.DataFrame(rows).to_csv(OUT / "FEWSHOT_FRACTION_CURVE.csv", index=False)
print(f"\nwrote {OUT}/FEWSHOT_FRACTION_CURVE.csv and fewshot_endpoints.npz")
