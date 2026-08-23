#!/usr/bin/env python3
"""Re-screen every three-verifier pool on TRAIN only, under the current frozen configuration.

Why this exists. The pool in use was selected in paper_v1 on a 6,850-row matrix that shares
54.4% of Protocol B's TEST document groups. Features and hyperparameters were later reselected
on TRAIN alone, but the pool never was, so under the project's own grouping criterion the pool
choice is not independent of the confirmatory test set. This script closes that gap by asking
where the pool ranks when the selection is redone on data that Protocol B's TEST never touches.

Nothing about the main experiment changes. TEST labels are never loaded: `V.load` is called with
`with_test_labels=False` and the absence of a label column on the test frame is asserted.

Method, matching the conventions of the earlier selection stages so the numbers are comparable:

  * candidates      all C(15,3) = 455 three-verifier subsets
  * data            TRAIN only, grouped fit/validation split by `content_doc_key`, five seeds
  * configuration   the six frozen features, regret target, random forest, part1c's selected
                    Protocol B hyperparameters, the frozen beta grid and tolerance
  * score           pre-stage-2 routed AUROC on the validation part, which is what
                    `fixedpool_select_v1` used
  * eligibility     the rule `freeze_v3.py` declared before consulting any screen: k = 3, not
                    dominated by any single verifier on TRAIN, and every member receiving at
                    least 5% of routed traffic
  * confirmation    the top twenty by B-style AUROC, plus the pool in use, re-evaluated under
                    A-style TRAIN-only rotations

Stages: bstyle | astyle | all
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3core as V  # noqa: E402
import core  # noqa: E402

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
PART1 = ROOT / "paper_v3" / "artifacts" / "part1_main_pooled_v1"
PART1C = ROOT / "paper_v3" / "artifacts" / "part1c_main_full_v1"
RUN = Path(os.environ["V3_RUN_DIR"]).resolve()
RES = RUN / "results"
RES.mkdir(parents=True, exist_ok=True)
V.RES = RES
SMOKE = os.environ.get("V3_SMOKE", "0") == "1"

CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
POOL_IN_USE = list(CONTRACT["pool"])
FEATURES = list(CONTRACT["features"])
TARGET, LEARNER = CONTRACT["target"], CONTRACT["learner"]
HP = dict(json.loads((PART1C / "00_contract" / "HP_SELECTED.json").read_text())
          ["hyperparameters_B"])
SEEDS = list(V.SEEDS[:1] if SMOKE else V.SEEDS[:5])
MIN_SHARE = 0.05
TOP_N = 3 if SMOKE else 20

TRAIN, TEST_NL, _ALL, VERIFIERS = V.load(with_test_labels=False)
assert "label_supported" not in TEST_NL.columns, "test frame must stay label free"
assert not (set(TRAIN.content_doc_key.astype(str))
            & set(TEST_NL.content_doc_key.astype(str))), "TRAIN/TEST document overlap"
core.log(f"TRAIN {len(TRAIN)} rows / {TRAIN.content_doc_key.nunique()} docs | "
         f"{len(VERIFIERS)} verifiers | hp={HP} | features={len(FEATURES)}")


def _splits():
    for seed in SEEDS:
        held = V.stratified_group_split(TRAIN, seed)
        yield seed, TRAIN.loc[~held].reset_index(drop=True), TRAIN.loc[held].reset_index(drop=True)


SPLITS = list(_splits())


def fixed_reference():
    """Per-verifier validation AUROC and latency on TRAIN, for the domination test."""
    rows = []
    for v in VERIFIERS:
        au, ms = [], []
        for _seed, fit, val in SPLITS:
            cal = core.platt(fit, actions=[v])
            p = core.apply_platt(val, cal, actions=[v])[v]
            au.append(float(roc_auc_score(val.label_supported.to_numpy(int), p)))
            ms.append(float(val[f"latency_ms__{v}"].mean()))
        rows.append({"verifier": v, "train_auroc": float(np.mean(au)),
                     "train_ms": float(np.mean(ms))})
    D = pd.DataFrame(rows).sort_values("train_auroc", ascending=False)
    V.save("POOL_FIXED_REFERENCE.csv", D)
    return D


def eval_pool_b(actions):
    au, ms, shares, betas = [], [], [], []
    for seed, fit, val in SPLITS:
        cals = core.platt(fit, actions=actions)
        c_fit = core.apply_platt(fit, cals, actions=actions)
        c_val = core.apply_platt(val, cals, actions=actions)
        (h_val,) = core.fit_heads(fit, [val], seed, c_fit, actions=actions,
                                  features=FEATURES, hp=HP, target=TARGET, learner=LEARNER)
        cvec = np.array([core.fold_costs(fit, actions=actions)[a] for a in actions], float)
        beta, _ = V.choose_beta(val, h_val, c_val, actions, cvec)
        sel, prob, lat = V.route(val, h_val, c_val, beta, actions, cvec)
        au.append(float(roc_auc_score(val.label_supported.to_numpy(int), prob)))
        ms.append(float(lat.mean()))
        shares.append(np.bincount(np.asarray(sel, int), minlength=len(actions)) / len(sel))
        betas.append(float(beta))
    sh = np.mean(shares, axis=0)
    return {"val_auroc": float(np.mean(au)), "sd": float(np.std(au)),
            "val_ms": float(np.mean(ms)), "min_share": float(sh.min()),
            "beta_mode": float(pd.Series(betas).mode().iloc[0]),
            **{f"share__{a}": float(s) for a, s in zip(actions, sh)}}


def eval_pool_a(actions):
    """A-style: TRAIN-only 8/1/1 rotations, pooled out-of-fold AUROC."""
    rots = V.rotations()[:2] if SMOKE else V.rotations()
    au, ms = [], []
    for seed in SEEDS:
        fold = V.folds_stratified(TRAIN, seed)
        probs, labels, times = [], [], []
        for t, vf, trf in rots:
            fit = TRAIN.loc[np.isin(fold, trf)].reset_index(drop=True)
            val = TRAIN.loc[fold == vf].reset_index(drop=True)
            ev = TRAIN.loc[fold == t].reset_index(drop=True)
            cals = core.platt(fit, actions=actions)
            c_fit = core.apply_platt(fit, cals, actions=actions)
            c_val = core.apply_platt(val, cals, actions=actions)
            c_ev = core.apply_platt(ev, cals, actions=actions)
            h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=actions,
                                         features=FEATURES, hp=HP, target=TARGET,
                                         learner=LEARNER)
            cvec = np.array([core.fold_costs(fit, actions=actions)[a] for a in actions], float)
            beta, _ = V.choose_beta(val, h_val, c_val, actions, cvec)
            _s, prob, lat = V.route(ev, h_ev, c_ev, beta, actions, cvec)
            probs.append(prob)
            labels.append(ev.label_supported.to_numpy(int))
            times.append(lat)
        au.append(float(roc_auc_score(np.concatenate(labels), np.concatenate(probs))))
        ms.append(float(np.concatenate(times).mean()))
    return {"a_val_auroc": float(np.mean(au)), "a_sd": float(np.std(au)),
            "a_val_ms": float(np.mean(ms))}


def bstyle():
    core.log("===== B-style screen over every k=3 pool, TRAIN only =====")
    FIX = fixed_reference()
    core.log("fixed-verifier TRAIN reference:")
    print(FIX.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    pools = [list(c) for c in itertools.combinations(sorted(VERIFIERS), 3)]
    if SMOKE:
        pools = pools[:6] + [sorted(POOL_IN_USE)]
    core.log(f"{len(pools)} candidate pools")
    rows, t0 = [], time.time()
    for i, actions in enumerate(pools, 1):
        r = eval_pool_b(actions)
        dominating = FIX[(FIX.train_auroc > r["val_auroc"]) & (FIX.train_ms < r["val_ms"])]
        rows.append({"pool": "+".join(actions), "k": 3,
                     "n_dominating": int(len(dominating)), **r})
        if i % 25 == 0 or i == len(pools):
            el = time.time() - t0
            core.log(f"  {i}/{len(pools)}  elapsed {el/60:.1f} min  "
                     f"eta {el/i*(len(pools)-i)/60:.1f} min")
    D = pd.DataFrame(rows)
    D["rank_by_auroc"] = D["val_auroc"].rank(ascending=False).astype(int)
    D["eligible"] = (D["n_dominating"] == 0) & (D["min_share"] >= MIN_SHARE)
    E = D[D.eligible].copy()
    E["rank_among_eligible"] = E["val_auroc"].rank(ascending=False).astype(int)
    D = D.merge(E[["pool", "rank_among_eligible"]], on="pool", how="left")
    V.save("POOL_RESCREEN_B.csv", D.sort_values("val_auroc", ascending=False))

    key = "+".join(sorted(POOL_IN_USE))
    me = D[D.pool == key].iloc[0]
    core.log("")
    core.log(f"pool in use: {key}")
    core.log(f"  val AUROC {me['val_auroc']:.5f} @ {me['val_ms']:.1f} ms | "
             f"min_share {me['min_share']:.1%} | dominated_by {int(me['n_dominating'])} | "
             f"eligible {bool(me['eligible'])}")
    core.log(f"  rank {int(me['rank_by_auroc'])}/{len(D)} overall | "
             f"rank {int(me['rank_among_eligible'])}/{int(D.eligible.sum())} among eligible")
    core.log("")
    core.log(f"top 10 eligible pools by validation AUROC:")
    print(D[D.eligible].sort_values("val_auroc", ascending=False)
          .head(10)[["pool", "val_auroc", "sd", "val_ms", "min_share", "rank_among_eligible"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))


def astyle():
    core.log("===== A-style confirmation of the top pools, TRAIN only =====")
    D = pd.read_csv(RES / "POOL_RESCREEN_B.csv")
    key = "+".join(sorted(POOL_IN_USE))
    top = D[D.eligible].sort_values("val_auroc", ascending=False).head(TOP_N)["pool"].tolist()
    if key not in top:
        top.append(key)
    rows, t0 = [], time.time()
    for i, p in enumerate(top, 1):
        r = eval_pool_a(p.split("+"))
        b = D[D.pool == p].iloc[0]
        rows.append({"pool": p, "b_val_auroc": float(b["val_auroc"]),
                     "b_val_ms": float(b["val_ms"]), "min_share": float(b["min_share"]), **r})
        core.log(f"  {i}/{len(top)} {p} | A {r['a_val_auroc']:.5f} @ {r['a_val_ms']:.1f} ms "
                 f"| B {b['val_auroc']:.5f}  ({(time.time()-t0)/60:.1f} min)")
    C = pd.DataFrame(rows)
    C["joint_auroc"] = (C["a_val_auroc"] + C["b_val_auroc"]) / 2
    C["rank_a"] = C["a_val_auroc"].rank(ascending=False).astype(int)
    C["rank_joint"] = C["joint_auroc"].rank(ascending=False).astype(int)
    V.save("POOL_RESCREEN_A_CONFIRM.csv", C.sort_values("joint_auroc", ascending=False))
    me = C[C.pool == key].iloc[0]
    core.log("")
    core.log(f"pool in use among the {len(C)} confirmed candidates: "
             f"A rank {int(me['rank_a'])}/{len(C)} | joint rank {int(me['rank_joint'])}/{len(C)}")
    print(C.sort_values("joint_auroc", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    order = ["bstyle", "astyle"] if want == ["all"] else want
    for name in order:
        t0 = time.time()
        {"bstyle": bstyle, "astyle": astyle}[name]()
        core.log(f"===== {name} done in {time.time()-t0:.0f}s =====")
