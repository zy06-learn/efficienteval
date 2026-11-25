#!/usr/bin/env python3
"""Revision experiment 2: the pooled-trained Protocol B system evaluated within each corpus.

Nothing is retrained, recalibrated or re-thresholded by corpus. The frozen part1c Protocol B
outputs (10 seeds) are read from 10_row_level/probs_B.npz; the cheap-feature-only arm from
part2_ablation_v1. For the permutation controls the frozen per-seed Platt calibrators and
isotonic maps are needed, so the part1c system is refitted per seed and checked to reproduce the
stored outputs exactly before anything is computed from it.

Outputs (runs/exp2_percorpus_20260929/):
  EXP2_CORPUS_STATS.csv        rows, positives, negatives, documents per corpus
  EXP2_PERCORPUS.csv           pooled, per-corpus and macro AUROC, latency, per system
  EXP2_VALSELECTED_FIXED.csv   per-seed validation AUROC of the 15 fixed verifiers and the pick
  EXP2_BOOTSTRAP.csv           paired source-level bootstrap within corpus, differences vs OURS
  EXP2_ROUTING_CONTROLS.csv    within-corpus / global permutation, corpus-only routing
  EXP2_PERMUTATION_DRAWS.npz   the permutation distributions
  EXP2_CALL_SHARES.csv         verifier call shares per corpus (OURS and corpus-only router)
  EXP2_DECOMPOSITION.csv       pooled AUROC = within-corpus pairs + cross-corpus pairs
  EXP2_CROSS_MATRIX_OURS.csv   A_{c,d} for OURS
  EXP2_GATES.json              reproduction checks
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3")
import v3core as V  # noqa: E402
import core  # noqa: E402

ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2")
PART1 = ROOT / "paper_v3" / "artifacts" / "part1_main_pooled_v1"
PART1C = ROOT / "paper_v3" / "artifacts" / "part1c_main_full_v1"
OUT = ROOT / "paper_v3" / "runs" / "exp2_percorpus_20260929"
OUT.mkdir(parents=True, exist_ok=True)
DRAWS = 2000
N_PERM = 500
RNG_SEED = 20260929
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)


CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
POOL = list(CONTRACT["pool"])
FEATURES = list(CONTRACT["features"])
TARGET, LEARNER = CONTRACT["target"], CONTRACT["learner"]
HP_B = dict(json.loads((PART1C / "00_contract" / "HP_SELECTED.json").read_text())
            ["hyperparameters_B"])

Z = np.load(PART1C / "10_row_level" / "probs_B.npz")
AB = np.load(ROOT / "paper_v3" / "runs" / "part2_ablation_v1" / "results" / "ablation_B.npz")
y = Z["y"].astype(int)
groups = Z["groups"].astype(str)
corpus = Z["corpus"].astype(str)
ekey = Z["episode_key"].astype(str)
SEEDS = [int(s) for s in Z["seeds"]]
assert [str(a) for a in Z["actions"]] == POOL
assert (AB["episode_key"].astype(str) == ekey).all() and (AB["y"].astype(int) == y).all()
n, S = Z["prob__OURS"].shape
CORPORA = sorted(set(corpus.tolist()))
NAME = {"cogensumm_test": "CoGenSumm", "frank_test": "FRANK", "ragtruth_test": "RAGTruth",
        "unisumeval_dev": "UniSumEval"}

TRAIN, TEST, _ALL, VERIFIERS = V.load(with_test_labels=True)
assert (TEST["episode_key"].astype(str).to_numpy() == ekey).all(), "TEST row order differs"
assert (TEST["label_supported"].to_numpy(int) == y).all()
for k in sorted(TRAIN["dataset_key"].astype(str).str.split("_").str[0].unique()):
    for F in (TRAIN, TEST):
        F[f"ds__{k}"] = (F["dataset_key"].astype(str).str.split("_").str[0] == k).astype(float)
ONEHOT = sorted(c for c in TRAIN.columns if c.startswith("ds__"))
gates = {}


# ------------------------------------------------------------------ helpers
def auc_cols(yy, P):
    """AUROC of every column of P (ties averaged), identical to roc_auc_score."""
    P = np.asarray(P, float)
    if P.ndim == 1:
        P = P[:, None]
    r = rankdata(P, axis=0)
    pos = yy == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return (r[pos].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def per_corpus(P):
    """Seed-averaged pooled AUROC, per-corpus AUROC, macro, for P (n, S)."""
    out = {"pooled": float(auc_cols(y, P).mean())}
    for c in CORPORA:
        m = corpus == c
        out[c] = float(auc_cols(y[m], P[m]).mean())
    out["macro"] = float(np.mean([out[c] for c in CORPORA]))
    return out


def fit_router(seed, feats):
    """part1c.router_raw + _stage2 on TEST for one seed; returns frozen pieces."""
    held = V.stratified_group_split(TRAIN, seed)
    fit = TRAIN.loc[~held].reset_index(drop=True)
    val = TRAIN.loc[held].reset_index(drop=True)
    cals = core.platt(fit, actions=POOL)
    c_fit = core.apply_platt(fit, cals, actions=POOL)
    c_val = core.apply_platt(val, cals, actions=POOL)
    t1 = time.perf_counter()
    c_ev = core.apply_platt(TEST, cals, actions=POOL)
    platt_ms = (time.perf_counter() - t1) * 1000.0 / n
    h_val, h_ev = core.fit_heads(fit, [val, TEST], seed, c_fit, actions=POOL, features=feats,
                                 hp=HP_B, target=TARGET, learner=LEARNER)
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
    t2 = time.perf_counter()
    sel, p_ev, _m2 = V.route(TEST, h_ev, c_ev, beta, POOL, cvec)
    route_ms = (time.perf_counter() - t2) * 1000.0 / n
    y_val = val["label_supported"].to_numpy(int)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    t3 = time.perf_counter()
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    iso_ms = (time.perf_counter() - t3) * 1000.0 / n
    C = np.column_stack([np.asarray(c_ev[a], float) for a in POOL])
    return {"sel": np.asarray(sel, int), "p1": np.asarray(p_ev, float), "q": q_ev, "iso": iso,
            "C": C, "beta": float(beta),
            "overhead_no_features": float(h_ev.predict_ms) + route_ms + platt_ms + iso_ms}


def score(iso, C, sel):
    """The frozen conversion: Platt-calibrated score of the assigned member -> frozen isotonic."""
    return np.clip(iso.predict(C[np.arange(len(sel)), sel]), 1e-6, 1 - 1e-6)


LAT = np.column_stack([TEST[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in POOL])
AVAIL = np.column_stack([TEST[f"available__{a}"].to_numpy(bool) for a in POOL])
ROWS = np.arange(n)

# ------------------------------------------------------------------ corpus statistics
cs = []
for c in CORPORA:
    m = corpus == c
    cs.append({"corpus": NAME[c], "split": c, "rows": int(m.sum()), "documents":
               int(len(set(groups[m]))), "supported": int(y[m].sum()),
               "unsupported": int((1 - y[m]).sum()), "supported_rate": float(y[m].mean())})
cs.append({"corpus": "ALL", "split": "TEST", "rows": n, "documents": int(len(set(groups))),
           "supported": int(y.sum()), "unsupported": int((1 - y).sum()),
           "supported_rate": float(y.mean())})
pd.DataFrame(cs).to_csv(OUT / "EXP2_CORPUS_STATS.csv", index=False)
log("corpus stats:\n" + pd.DataFrame(cs).to_string(index=False))

# ------------------------------------------------------------------ refit frozen OURS, check
FROZEN = []
for si, seed in enumerate(SEEDS):
    r = fit_router(seed, FEATURES)
    d1 = float(np.abs(r["p1"] - Z["prob1__OURS"][:, si]).max())
    d2 = float(np.abs(r["q"] - Z["prob__OURS"][:, si]).max())
    ds = int((r["sel"] != Z["sel__OURS"][:, si]).sum())
    q_conv = score(r["iso"], r["C"], Z["sel__OURS"][:, si].astype(int))
    d3 = float(np.abs(q_conv - Z["prob__OURS"][:, si]).max())
    gates[f"ours_seed{seed}"] = {"max_abs_stage1": d1, "max_abs_stage2": d2, "sel_mismatch": ds,
                                 "max_abs_conversion": d3}
    assert d1 < 1e-12 and d2 < 1e-12 and ds == 0 and d3 < 1e-12, gates[f"ours_seed{seed}"]
    r["overhead"] = Z["ms_full__OURS"][:, si] - LAT[ROWS, r["sel"]]
    FROZEN.append(r)
log("frozen OURS refit reproduces stored outputs exactly on all seeds")

# ------------------------------------------------------------------ corpus-only router
ref_ds = json.loads((ROOT / "paper_v3" / "runs" / "dataset_control_v1" /
                     "DATASET_ONLY_ARMS.json").read_text())["dataset one-hot only"]["per_seed"]
CORP = []
for si, seed in enumerate(SEEDS):
    r = fit_router(seed, ONEHOT)
    a = float(roc_auc_score(y, r["q"]))
    gates[f"corpus_only_seed{seed}"] = {"auroc": a, "reference": ref_ds[si],
                                        "abs_diff": abs(a - ref_ds[si])}
    assert abs(a - ref_ds[si]) < 1e-12, gates[f"corpus_only_seed{seed}"]
    assert np.allclose(r["C"], FROZEN[si]["C"], atol=0, rtol=0), "Platt differs between routers"
    CORP.append(r)
log("corpus-only router reproduces DATASET_ONLY_ARMS per-seed AUROC exactly")

# ------------------------------------------------------------------ validation-selected fixed
vs_rows, VSF, VS_PICK = [], np.zeros((n, S)), []
for si, seed in enumerate(SEEDS):
    held = V.stratified_group_split(TRAIN, seed)
    fit = TRAIN.loc[~held].reset_index(drop=True)
    val = TRAIN.loc[held].reset_index(drop=True)
    yv = val["label_supported"].to_numpy(int)
    best, best_a = None, -1.0
    for v in VERIFIERS:
        cal = core.platt(fit, actions=[v])
        a = float(roc_auc_score(yv, core.apply_platt(val, cal, actions=[v])[v]))
        vs_rows.append({"seed": seed, "verifier": v, "val_auroc": a,
                        "val_ms": float(val[f"latency_ms__{v}"].mean())})
        if a > best_a:
            best, best_a = v, a
    for row in vs_rows[-len(VERIFIERS):]:
        row["selected"] = row["verifier"] == best
    VSF[:, si] = Z[f"prob__fixed__{best}"][:, si]
    VS_PICK.append(best)
VS = pd.DataFrame(vs_rows)
VS.to_csv(OUT / "EXP2_VALSELECTED_FIXED.csv", index=False)
picked = VS[VS.selected].verifier.value_counts().to_dict()
log(f"validation-selected fixed verifier per seed: {picked}")
# One verifier for all seeds (user decision, 2026-09-29): the one the per-seed validation rule
# selects most often. Its per-seed picks and the mean-validation-AUROC ranking are both recorded,
# because the two rules disagree narrowly here (see EXP2_VALSELECTED_CHOICE.json).
VSF_NAME = max(picked, key=lambda v: (picked[v], v))
_mean = VS.groupby("verifier").val_auroc.mean().sort_values(ascending=False)
(OUT / "EXP2_VALSELECTED_CHOICE.json").write_text(json.dumps({
    "rule": "the verifier the per-seed validation-AUROC rule selects most often across the seeds",
    "selected": VSF_NAME, "per_seed_pick_counts": picked,
    "mean_validation_auroc_top3": {k: float(v) for k, v in _mean.head(3).items()}}, indent=2))
VSF = Z[f"prob__fixed__{VSF_NAME}"]
VSF_MS = float(Z[f"ms_full__fixed__{VSF_NAME}"].mean())
log(f"validation-selected fixed verifier (most frequent pick): {VSF_NAME}")

# ------------------------------------------------------------------ systems
SYS = {
    "EfficientEval": (Z["prob__OURS"], Z["ms_full__OURS"].mean()),
    "Cheap features": (AB["prob__no_verifier"], AB["ms__no_verifier"].mean()),
    "FactCC": (Z["prob__fixed__factcc"], Z["ms_full__fixed__factcc"].mean()),
    "Lettuce-v2": (Z["prob__fixed__lettuce_v2"], Z["ms_full__fixed__lettuce_v2"].mean()),
    "Granite 3.1": (Z["prob__fixed__granite_guardian_3_1_2b"],
                    Z["ms_full__fixed__granite_guardian_3_1_2b"].mean()),
    "Qwen3 30B": (Z["prob__fixed__qwen30_fast"], Z["ms_full__fixed__qwen30_fast"].mean()),
    "Validation-selected fixed": (VSF, VSF_MS),
    "Corpus-only router": (np.column_stack([r["q"] for r in CORP]),
                           float(np.mean([(LAT[ROWS, r["sel"]] + r["overhead_no_features"]).mean()
                                          for r in CORP]))),
}
rows = []
for name, (P, ms) in SYS.items():
    pc = per_corpus(P)
    rows.append({"system": name, "pooled": pc["pooled"],
                 **{NAME[c]: pc[c] for c in CORPORA}, "macro": pc["macro"], "ms": float(ms),
                 "pooled_sd_seeds": float(auc_cols(y, P).std())})
# every fixed verifier, for the appendix
for s in [x for x in Z["systems"].astype(str) if x.startswith("fixed::")]:
    k = s.replace("::", "__")
    pc = per_corpus(Z[f"prob__{k}"])
    rows.append({"system": s, "pooled": pc["pooled"], **{NAME[c]: pc[c] for c in CORPORA},
                 "macro": pc["macro"], "ms": float(Z[f"ms_full__{k}"].mean()),
                 "pooled_sd_seeds": float(auc_cols(y, Z[f"prob__{k}"]).std())})
PC = pd.DataFrame(rows)
PC.to_csv(OUT / "EXP2_PERCORPUS.csv", index=False)
log("per-corpus AUROC:\n" + PC.round(4).to_string(index=False))

# ------------------------------------------------------------------ paired source-level bootstrap
rng = np.random.default_rng(RNG_SEED)
by_corpus = {}
for c in CORPORA:
    m = np.where(corpus == c)[0]
    g = groups[m]
    uniq, inv = np.unique(g, return_inverse=True)
    by_corpus[c] = [m[inv == k] for k in range(len(uniq))]
redraws = {c: 0 for c in CORPORA}
names = list(SYS)
stat = {nm: {"pooled": np.empty(DRAWS), "macro": np.empty(DRAWS),
             **{c: np.empty(DRAWS) for c in CORPORA}} for nm in names}
for b in range(DRAWS):
    parts = []
    for c in CORPORA:
        docs = by_corpus[c]
        while True:  # one-class rule: redraw this corpus's documents until both labels occur
            idx = np.concatenate([docs[k] for k in rng.integers(0, len(docs), len(docs))])
            if 0 < y[idx].sum() < len(idx):
                break
            redraws[c] += 1
        parts.append((c, idx))
    allidx = np.concatenate([p[1] for p in parts])
    for nm in names:
        P = SYS[nm][0]
        vals = []
        for c, idx in parts:
            a = float(auc_cols(y[idx], P[idx]).mean())
            stat[nm][c][b] = a
            vals.append(a)
        stat[nm]["macro"][b] = float(np.mean(vals))
        stat[nm]["pooled"][b] = float(auc_cols(y[allidx], P[allidx]).mean())
    if (b + 1) % 500 == 0:
        log(f"  bootstrap {b+1}/{DRAWS}")
obs = {nm: per_corpus(SYS[nm][0]) for nm in names}
brows = []
for nm in names:
    if nm == "EfficientEval":
        continue
    for metric in ["macro", "pooled"] + CORPORA:
        d = stat["EfficientEval"][metric] - stat[nm][metric]
        lo, hi = np.percentile(d, [2.5, 97.5])
        brows.append({"baseline": nm, "metric": NAME.get(metric, metric),
                      "ours": obs["EfficientEval"][metric], "baseline_value": obs[nm][metric],
                      "d_ours_minus_baseline": obs["EfficientEval"][metric] - obs[nm][metric],
                      "ci95_lo": float(lo), "ci95_hi": float(hi),
                      "share_draws_ours_better": float((d > 0).mean()),
                      "excludes_zero": bool(lo > 0 or hi < 0)})
BT = pd.DataFrame(brows)
BT.to_csv(OUT / "EXP2_BOOTSTRAP.csv", index=False)
gates["bootstrap"] = {"draws": DRAWS, "rng_seed": RNG_SEED, "one_class_redraws": redraws}
log("bootstrap (EfficientEval minus baseline):\n" + BT.round(4).to_string(index=False))

# ------------------------------------------------------------------ permutation controls
def strata(keys):
    out = {}
    for i, kk in enumerate(zip(*keys)):
        out.setdefault(kk, []).append(i)
    return [np.array(v) for v in out.values()]


pat = [AVAIL[:, j] for j in range(len(POOL))]
ST_WITHIN = strata([corpus] + pat)
ST_GLOBAL = strata(pat)
perm = {k: {"pooled": np.zeros((N_PERM, S)), "macro": np.zeros((N_PERM, S)),
            "ms": np.zeros((N_PERM, S)), **{c: np.zeros((N_PERM, S)) for c in CORPORA}}
        for k in ("within", "global")}
for si, seed in enumerate(SEEDS):
    fz = FROZEN[si]
    sel0 = fz["sel"]
    prng = np.random.default_rng(RNG_SEED + seed)
    for kind, st in (("within", ST_WITHIN), ("global", ST_GLOBAL)):
        Q = np.empty((n, N_PERM))
        MS = np.empty(N_PERM)
        for r in range(N_PERM):
            ps = sel0.copy()
            for idx in st:
                ps[idx] = sel0[idx][prng.permutation(len(idx))]
            assert AVAIL[ROWS, ps].all()
            Q[:, r] = score(fz["iso"], fz["C"], ps)
            MS[r] = (fz["overhead"] + LAT[ROWS, ps]).mean()
            if kind == "within" and r == 0:
                for c in CORPORA:
                    m = corpus == c
                    assert np.array_equal(np.bincount(ps[m], minlength=3),
                                          np.bincount(sel0[m], minlength=3))
        perm[kind]["pooled"][:, si] = auc_cols(y, Q)
        vals = []
        for c in CORPORA:
            m = corpus == c
            perm[kind][c][:, si] = auc_cols(y[m], Q[m])
            vals.append(perm[kind][c][:, si])
        perm[kind]["macro"][:, si] = np.mean(vals, axis=0)
        perm[kind]["ms"][:, si] = MS
    log(f"  permutations seed {seed} done")
np.savez_compressed(OUT / "EXP2_PERMUTATION_DRAWS.npz",
                    **{f"{k}__{m}": v for k, d in perm.items() for m, v in d.items()})

# corpus-only assignments scored through the frozen EfficientEval conversion
CO_FROZEN = np.column_stack([score(FROZEN[si]["iso"], FROZEN[si]["C"], CORP[si]["sel"])
                             for si in range(S)])
co_ms_frozen = float(np.mean([(FROZEN[si]["overhead"] + LAT[ROWS, CORP[si]["sel"]]).mean()
                              for si in range(S)]))
o_ours = per_corpus(Z["prob__OURS"])
rc = [{"condition": "EfficientEval", "pooled": o_ours["pooled"], "macro": o_ours["macro"],
       **{NAME[c]: o_ours[c] for c in CORPORA}, "ms": float(Z["ms_full__OURS"].mean()),
       "pooled_p2.5": np.nan, "pooled_p97.5": np.nan, "macro_p2.5": np.nan,
       "macro_p97.5": np.nan, "p_perm_pooled": np.nan, "p_perm_macro": np.nan}]
for kind, label in (("within", "Within-corpus permutation"), ("global", "Global permutation")):
    d = {m: perm[kind][m].mean(1) for m in perm[kind]}  # seed-averaged, one value per draw
    rc.append({"condition": label, "pooled": float(d["pooled"].mean()),
               "macro": float(d["macro"].mean()),
               **{NAME[c]: float(d[c].mean()) for c in CORPORA}, "ms": float(d["ms"].mean()),
               "pooled_p2.5": float(np.percentile(d["pooled"], 2.5)),
               "pooled_p97.5": float(np.percentile(d["pooled"], 97.5)),
               "macro_p2.5": float(np.percentile(d["macro"], 2.5)),
               "macro_p97.5": float(np.percentile(d["macro"], 97.5)),
               "p_perm_pooled": float((1 + (d["pooled"] >= o_ours["pooled"]).sum()) / (N_PERM + 1)),
               "p_perm_macro": float((1 + (d["macro"] >= o_ours["macro"]).sum()) / (N_PERM + 1))})
o_cf = per_corpus(CO_FROZEN)
rc.append({"condition": "Corpus-only routing", "pooled": o_cf["pooled"], "macro": o_cf["macro"],
           **{NAME[c]: o_cf[c] for c in CORPORA}, "ms": co_ms_frozen})
o_ct = per_corpus(SYS["Corpus-only router"][0])
rc.append({"condition": "Corpus-only router, own calibration", "pooled": o_ct["pooled"],
           "macro": o_ct["macro"], **{NAME[c]: o_ct[c] for c in CORPORA},
           "ms": SYS["Corpus-only router"][1]})
RC = pd.DataFrame(rc)
RC.to_csv(OUT / "EXP2_ROUTING_CONTROLS.csv", index=False)
log("routing controls:\n" + RC.round(4).to_string(index=False))

# ------------------------------------------------------------------ call shares
cr = []
for label, sels, lat_rows in (
        ("EfficientEval", [r["sel"] for r in FROZEN], [Z["ms_full__OURS"][:, si] for si in range(S)]),
        ("Corpus-only router", [r["sel"] for r in CORP],
         [LAT[ROWS, r["sel"]] + r["overhead_no_features"] for r in CORP])):
    for c in ["ALL"] + CORPORA:
        m = np.ones(n, bool) if c == "ALL" else corpus == c
        sh = np.array([[np.mean(s[m] == j) for j in range(3)] for s in sels])  # (S, 3)
        cr.append({"router": label, "corpus": NAME.get(c, c), "rows": int(m.sum()),
                   **{f"share__{a}": float(sh[:, j].mean()) for j, a in enumerate(POOL)},
                   **{f"sd__{a}": float(sh[:, j].std()) for j, a in enumerate(POOL)},
                   "mean_ms": float(np.mean([lr[m].mean() for lr in lat_rows]))})
CR = pd.DataFrame(cr)
CR.to_csv(OUT / "EXP2_CALL_SHARES.csv", index=False)
log("call shares:\n" + CR.round(3).to_string(index=False))


# ------------------------------------------------------------------ decomposition
def cross_auc(p_pos, p_neg):
    s = np.sort(p_neg)
    lo = np.searchsorted(s, p_pos, "left")
    hi = np.searchsorted(s, p_pos, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(p_pos) * len(p_neg)))


def decompose(p):
    pos, neg = y == 1, y == 0
    Np, Nn = pos.sum(), neg.sum()
    within_w = within_s = cross_w = cross_s = 0.0
    M = np.zeros((len(CORPORA), len(CORPORA)))
    for i, c in enumerate(CORPORA):
        for j, d in enumerate(CORPORA):
            pp = p[pos & (corpus == c)]
            pn = p[neg & (corpus == d)]
            a = cross_auc(pp, pn)
            M[i, j] = a
            w = len(pp) * len(pn)
            if i == j:
                within_w += w
                within_s += w * a
            else:
                cross_w += w
                cross_s += w * a
    pooled = (within_s + cross_s) / (Np * Nn)
    return {"pooled": pooled, "within_pair_share": within_w / (Np * Nn),
            "within_auroc_pairweighted": within_s / within_w, "cross_auroc": cross_s / cross_w,
            "within_contribution": within_s / (Np * Nn), "cross_contribution": cross_s / (Np * Nn)}, M


dec, M_ours = [], []
for name, (P, _ms) in list(SYS.items()) + [("Corpus-only routing (frozen scoring)", (CO_FROZEN, 0))]:
    per = [decompose(P[:, si]) for si in range(S)]
    for si in range(S):
        assert abs(per[si][0]["pooled"] - roc_auc_score(y, P[:, si])) < 1e-12
    agg = {k: float(np.mean([d[0][k] for d in per])) for k in per[0][0]}
    dec.append({"system": name, **agg})
    if name == "EfficientEval":
        M_ours = np.mean([d[1] for d in per], axis=0)
DEC = pd.DataFrame(dec)
DEC.to_csv(OUT / "EXP2_DECOMPOSITION.csv", index=False)
pd.DataFrame(M_ours, index=[f"pos_{NAME[c]}" for c in CORPORA],
             columns=[f"neg_{NAME[c]}" for c in CORPORA]).to_csv(OUT / "EXP2_CROSS_MATRIX_OURS.csv")
log("decomposition:\n" + DEC.round(4).to_string(index=False))

(OUT / "EXP2_GATES.json").write_text(json.dumps(gates, indent=2))
(OUT / "SUMMARY.json").write_text(json.dumps({
    "status": "COMPLETE", "seeds": SEEDS, "draws": DRAWS, "n_perm_per_seed": N_PERM,
    "rng_seed": RNG_SEED, "validation_selected_fixed": picked,
    "one_class_redraws": redraws, "seconds": round(time.time() - T0, 1)}, indent=2))
log("COMPLETE")
