#!/usr/bin/env python3
"""The deployable per-corpus competitor that E1 was missing.

E1 reports, for each corpus, the best fixed verifier found by taking the maximum test AUROC over
fifteen candidates. That is an oracle: no deployable system knows in advance which verifier will
win on a corpus it has not evaluated. Left as the only comparison, the E1 table reads as "a fixed
verifier beats the router", which is not what it shows.

This adds the deployable version. For each corpus and seed, using exactly the fit/validation
split that E1 used, the single fixed verifier with the highest validation AUROC is selected and
then evaluated on that corpus's test split. Selection never touches test labels. Whatever this
arm achieves is the honest single-verifier baseline for the in-domain setting, and the gap between
it and the oracle is the cost of not knowing which verifier to pick.

Run separately from part3 so the running phase-2 job's recorded code hash stays valid.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3core as V  # noqa: E402
import core  # noqa: E402

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
PART1 = ROOT / "experiments" / "cross_stage_contract" / "part1_main_pooled_v1"
RUN = Path(os.environ["V3_RUN_DIR"]).resolve()
RES = RUN / "results"
RES.mkdir(parents=True, exist_ok=True)
V.RES = RES

CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
POOL = list(CONTRACT["pool"])
SEEDS = list(V.SEEDS)
CORPUS_PAIRS = {"cogensumm": ("cogensumm_val", "cogensumm_test"),
                "frank": ("frank_valid", "frank_test"),
                "ragtruth": ("ragtruth_train", "ragtruth_test"),
                "unisumeval": ("unisumeval_train", "unisumeval_dev")}


def _mcc_threshold(q, y):
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def _stage2(p_val, y_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    return q_ev, _mcc_threshold(q_val, y_val)


def main():
    core.log("===== E1b: validation-selected single verifier, per corpus =====")
    TRAIN, TEST, _ALL, verifiers = V.load(with_test_labels=True)
    # Two candidate sets, because they answer different questions. `all15` asks whether a fixed
    # verifier chosen for this corpus beats in-domain routing when any verifier may be bought;
    # its winners are 133 to 1504 ms models. `pool3` restricts the choice to the three cheap
    # verifiers the router actually routes among, which is the action-space-matched comparison.
    CANDIDATE_SETS = {"all15": list(verifiers), "pool3": list(POOL)}
    rows, picks = [], []
    for cand_name, candidates in CANDIDATE_SETS.items():
        _one_set(cand_name, candidates, TRAIN, TEST, rows, picks)
    D = pd.DataFrame(rows)
    V.save("E1b_PERCORPUS_SELECTED.csv", D)
    V.save("E1b_PERCORPUS_SELECTED_PICKS.csv", pd.DataFrame(picks))
    _join(D)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


def _one_set(cand_name, candidates, TRAIN, TEST, rows, picks):
    for corpus, (tr_key, te_key) in CORPUS_PAIRS.items():
        sub_tr = TRAIN[TRAIN.dataset_key.astype(str) == tr_key].reset_index(drop=True)
        sub_te = TEST[TEST.dataset_key.astype(str) == te_key].reset_index(drop=True)
        y_te = sub_te["label_supported"].to_numpy(int)
        sel_au, sel_ms, chosen, skipped = [], [], [], 0
        for seed in SEEDS:
            # identical split ladder to E1, so the two arms are compared on the same fits
            fit = val = None
            for frac in (0.20, 0.30, 0.40):
                held = V.stratified_group_split(sub_tr, seed, fraction=frac)
                f_, v_ = sub_tr.loc[~held], sub_tr.loc[held]
                yv = v_["label_supported"].to_numpy(int)
                if len(set(yv.tolist())) > 1 and min((yv == 0).sum(), (yv == 1).sum()) >= 5:
                    fit, val = f_.reset_index(drop=True), v_.reset_index(drop=True)
                    break
            if fit is None:
                skipped += 1
                continue
            y_val = val["label_supported"].to_numpy(int)
            best_v, best_au = None, -np.inf
            cache = {}
            for v in candidates:
                cal = core.platt(fit, actions=[v])
                pv = core.apply_platt(val, cal, actions=[v])[v]
                pe = core.apply_platt(sub_te, cal, actions=[v])[v]
                au = float(roc_auc_score(y_val, pv)) if len(set(y_val.tolist())) > 1 else 0.5
                cache[v] = (pv, pe)
                if au > best_au:
                    best_v, best_au = v, au
            pv, pe = cache[best_v]
            q, _thr = _stage2(pv, y_val, pe)
            sel_au.append(float(roc_auc_score(y_te, q)))
            sel_ms.append(float(sub_te[f"latency_ms__{best_v}"].mean()))
            chosen.append(best_v)
            picks.append({"candidates": cand_name, "corpus": corpus, "seed": seed,
                          "selected": best_v, "val_auroc": best_au,
                          "test_auroc": sel_au[-1]})
        counts = pd.Series(chosen).value_counts()
        rows.append({"candidates": cand_name, "corpus": corpus,
                     "seeds_used": len(sel_au), "seeds_skipped": skipped,
                     "test_rows": len(sub_te),
                     "test_groups": int(sub_te.content_doc_key.nunique()),
                     "selected_mode": counts.index[0] if len(counts) else "",
                     "selected_mode_rate": float(counts.iloc[0] / len(chosen)) if chosen else np.nan,
                     "n_distinct_selected": int(len(counts)),
                     "auroc": float(np.mean(sel_au)), "auroc_sd": float(np.std(sel_au)),
                     "ms": float(np.mean(sel_ms))})
        core.log(f"  [{cand_name}] {corpus}: selected {counts.index[0]} in "
                 f"{counts.iloc[0]}/{len(chosen)} seeds ({len(counts)} distinct) -> "
                 f"test AUROC {np.mean(sel_au):.5f} (sd {np.std(sel_au):.5f}) "
                 f"@ {np.mean(sel_ms):.1f} ms")


def _join(D):
    """Put the router, the two validation-selected baselines and the oracle side by side."""
    e1 = RES / "E1_PERCORPUS_SUMMARY.csv"
    if not e1.exists():
        core.log("E1_PERCORPUS_SUMMARY.csv not present; skipping the joined view")
        return
    S = pd.read_csv(e1)
    for cand in ("all15", "pool3"):
        sub = D[D.candidates == cand][["corpus", "auroc", "auroc_sd", "ms", "selected_mode",
                                       "n_distinct_selected"]]
        sub = sub.rename(columns={"auroc": f"sel_{cand}_auroc",
                                  "auroc_sd": f"sel_{cand}_sd",
                                  "ms": f"sel_{cand}_ms",
                                  "selected_mode": f"sel_{cand}_mode",
                                  "n_distinct_selected": f"sel_{cand}_distinct"})
        S = S.merge(sub, on="corpus")
    S["router_minus_sel_pool3"] = S.in_domain_ours - S.sel_pool3_auroc
    S["router_minus_sel_all15"] = S.in_domain_ours - S.sel_all15_auroc
    V.save("E1_PERCORPUS_FOUR_WAY.csv", S)
    cols = ["corpus", "test_groups", "in_domain_ours", "sel_pool3_auroc", "sel_pool3_mode",
            "router_minus_sel_pool3", "sel_all15_auroc", "sel_all15_mode", "sel_all15_ms",
            "router_minus_sel_all15", "in_domain_best_fixed", "in_domain_best_fixed_name"]
    print(S[cols].to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    core.log(f"===== done in {time.time()-t0:.0f}s =====")
