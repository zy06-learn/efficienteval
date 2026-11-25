#!/usr/bin/env python3
"""Sensitivity for revision experiment 1: the pools that other screening seeds select, run
through the paper's final Protocol B procedure and evaluated once on the test split.

The frozen selection (screening seed 17) returns the paper's pool under both selection-stage
routers, so its final evaluation is part1c's Protocol B result. Screening seeds 29 and 43 return
other pools (historical router: FactCC + Lettuce-v2 + FactCG and FactCC + Lettuce-v2 + Qwen3-30B
fast; paper router: FactCC + Lettuce-v2 + WeCheck). This script gives each of them exactly the
final procedure the paper's pool received:

  hyperparameters  part1c's rule restricted to the Protocol B shape: the same 16 random candidates
                   plus part1's two values, minimum validation head loss over seeds 17/29/43/7/11,
                   TRAIN only
  system           six cheap features, regret target, random forest, cost discount with the beta
                   rule, stage-1 Platt, stage-2 isotonic, MCC threshold on the validation part;
                   ten seeds; end-to-end latency
  test             read once, after selection

Gates, before anything else is computed: with the paper's pool, the B-shape hyperparameter search
must reproduce part1c's HP_SELECTION.csv losses and its hyperparameters_B, and the final system
must reproduce part1c's stored Protocol B probabilities exactly.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2")
RUN = ROOT / "paper_v3" / "runs" / "exp1_altpools_final_20260929"
os.environ["V3_RUN_DIR"] = str(RUN)
sys.path.insert(0, str(ROOT / "paper_v3"))
import part1c_main_full_v1 as P  # noqa: E402
import v3core as V  # noqa: E402
import core  # noqa: E402

RES = RUN / "results"
PART1C = ROOT / "paper_v3" / "artifacts" / "part1c_main_full_v1"
ALT = {"factcc+lettuce_v2+factcg": "historical router, screening seed 29",
       "factcc+lettuce_v2+qwen30_fast": "historical router, screening seed 43",
       "factcc+lettuce_v2+wecheck": "paper router, screening seed 29"}
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)


TRAIN, TEST, _ALL, VERIFIERS = V.load(with_test_labels=True)
y = TEST["label_supported"].to_numpy(int)
PRE = P._pre_call_feature_ms(TEST)


def grid():
    """part1c.hpselect's candidate list, in its order."""
    rng = np.random.default_rng(0)
    out, seen = [], set()
    for _ in range(16):
        h = {"n_estimators": int(rng.choice([200, 400, 800])),
             "min_samples_leaf": int(rng.choice([1, 2, 5, 10])),
             "max_features": str(rng.choice(["sqrt", "log2", "0.5"])),
             "max_depth": int(rng.choice([6, 10, 12, 16, 24]))}
        k = json.dumps(h, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(h)
    for h in (P.LEGACY_HP_A, P.LEGACY_HP_B):
        h = dict(h, max_features=str(h["max_features"]))
        if json.dumps(h, sort_keys=True) not in seen:
            seen.add(json.dumps(h, sort_keys=True))
            out.append(h)
    return out


def _hp(h):
    mf = str(h["max_features"])
    return {"n_estimators": int(h["n_estimators"]), "min_samples_leaf": int(h["min_samples_leaf"]),
            "max_features": 0.5 if mf == "0.5" else mf, "max_depth": int(h["max_depth"])}


def hp_select_b(pool):
    """part1c.hpselect, Protocol B shape only, for an arbitrary pool."""
    P.POOL = list(pool)
    rows = []
    for h in grid():
        loss = []
        for seed in P.HP_SEEDS:
            held = V.stratified_group_split(TRAIN, seed)
            fit = TRAIN.loc[~held].reset_index(drop=True)
            val = TRAIN.loc[held].reset_index(drop=True)
            cals = core.platt(fit, actions=P.POOL)
            c_fit = core.apply_platt(fit, cals, actions=P.POOL)
            c_val = core.apply_platt(val, cals, actions=P.POOL)
            (h_val,) = core.fit_heads(fit, [val], seed, c_fit, actions=P.POOL,
                                      features=P.FEATURES, hp=_hp(h), target=P.TARGET,
                                      learner=P.LEARNER)
            loss.append(P._head_loss(h_val, val, c_val))
        rows.append({**h, "b_val_loss": float(np.mean(loss))})
    D = pd.DataFrame(rows)
    return D, _hp(D.loc[D.b_val_loss.idxmin()].to_dict())


def final_b(pool, hp):
    """part1c.proto_b for the router only, for an arbitrary pool."""
    P.POOL = list(pool)
    probs, ms, thr, sel = [], [], [], []
    for seed in P.SEEDS:
        held = V.stratified_group_split(TRAIN, seed)
        fit = TRAIN.loc[~held].reset_index(drop=True)
        val = TRAIN.loc[held].reset_index(drop=True)
        raw = P.router_raw(fit, val, TEST, seed, hp)
        q, t2, _t1, s2 = P._stage2(raw["p_val"], val["label_supported"].to_numpy(int),
                                   val["content_doc_key"].astype(str).to_numpy(), raw["p_ev"])
        raw["s2_ms"] = s2
        probs.append(q)
        ms.append(P._assemble(raw, PRE, True))
        thr.append(t2[P.PRIMARY_RULE])
        sel.append(raw["sel"])
    return np.column_stack(probs), np.column_stack(ms), np.array(thr), np.column_stack(sel)


def summarize(name, Q, MS, THR, SEL, pool):
    rows = [core.metrics(y, Q[:, s], MS[:, s], threshold=THR[s]) for s in range(Q.shape[1])]
    m = {k: float(np.nanmean([r[k] for r in rows])) for k in
         ("auroc", "auprc_unsup", "bacc", "f1_unsup", "ms")}
    m["auroc_sd"] = float(np.std([r["auroc"] for r in rows]))
    m["hv"] = max(0.0, 2 * m["auroc"] - 1) * max(0.0, 1 - m["ms"] / 1000.0)
    corp = TEST["dataset_key"].astype(str).to_numpy()
    from sklearn.metrics import roc_auc_score
    per = {c: float(np.mean([roc_auc_score(y[corp == c], Q[corp == c, s])
                             for s in range(Q.shape[1])])) for c in sorted(set(corp))}
    m["macro_auroc"] = float(np.mean(list(per.values())))
    for c, v in per.items():
        m[f"auroc__{c}"] = v
    for j, a in enumerate(pool):
        m[f"share__{a}"] = float((SEL == j).mean())
    return {"pool": name, **m}


RES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- gates on the paper's pool
paper = list(P.EXPECTED_CONTRACT["pool"])
D, hp = hp_select_b(paper)
ref = pd.read_csv(PART1C / "05_hyperparameter_selection" / "HP_SELECTION.csv")
keycols = ["n_estimators", "min_samples_leaf", "max_features", "max_depth"]
mrg = D.merge(ref[keycols + ["b_val_loss"]].astype({"max_features": str}), on=keycols,
              suffixes=("", "_ref"))
assert len(mrg) == len(D) == len(ref), (len(mrg), len(D), len(ref))
d_loss = float((mrg.b_val_loss - mrg.b_val_loss_ref).abs().max())
want = json.loads((PART1C / "00_contract" / "HP_SELECTED.json").read_text())["hyperparameters_B"]
assert d_loss < 1e-12 and hp == _hp(want), (d_loss, hp, want)
Z = np.load(PART1C / "10_row_level" / "probs_B.npz")
Q, MS, THR, SEL = final_b(paper, hp)
d_prob = float(np.abs(Q - Z["prob__OURS"]).max())
assert d_prob < 1e-12 and np.array_equal(SEL, Z["sel__OURS"]), d_prob
gates = {"hp_loss_max_abs_diff": d_loss, "hp_b_reproduced": hp,
         "prob_max_abs_diff": d_prob, "sel_identical": True}
log(f"gates pass: {gates}")
rows = [summarize("+".join(sorted(paper)), Q, MS, THR, SEL, paper)]
rows[0]["source"] = "frozen selection (screening seed 17), both routers"
rows[0]["hyperparameters"] = json.dumps(hp)

# ---------------------------------------------------------------- the alternative pools
for name, src in ALT.items():
    pool = name.split("+")
    D, hp = hp_select_b(pool)
    D.to_csv(RES / f"HP_SELECTION_B__{name}.csv", index=False)
    Q, MS, THR, SEL = final_b(pool, hp)
    np.savez_compressed(RES / f"probs_B__{name}.npz", prob=Q, ms=MS, thr_mcc=THR, sel=SEL)
    r = summarize(name, Q, MS, THR, SEL, pool)
    r["source"], r["hyperparameters"] = src, json.dumps(hp)
    rows.append(r)
    log(f"{name}: hp={hp} AUROC {r['auroc']:.4f} @ {r['ms']:.1f} ms, macro {r['macro_auroc']:.4f}")
pd.DataFrame(rows).to_csv(RES / "ALT_POOLS_FINAL_B.csv", index=False)
(RES / "GATES.json").write_text(json.dumps(gates, indent=2, default=str))
print(pd.DataFrame(rows)[["pool", "auroc", "auprc_unsup", "bacc", "f1_unsup", "ms", "hv",
                          "macro_auroc"]].round(4).to_string(index=False))
log("COMPLETE")
