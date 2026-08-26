#!/usr/bin/env python3
"""Significance for the three control experiments, under the same test as the main tables.

The three controls reported point estimates and across-seed standard deviations but no interval,
so their claims rested on comparing a difference against a standard deviation by eye. This runs
the identical procedure used for the main tables -- paired cluster bootstrap over
content_doc_key, one shared index set per family, the per-seed AUROC difference averaged inside
each draw, Bonferroni over the family -- on each of them.

Family sizes differ because the questions differ.

  live       1 comparison   live execution against the frozen matrix
  arms       3 comparisons  each dataset-identity arm against the six-feature arm
  fewshot    4 comparisons  full-corpus against k=0, one per held-out corpus

Row-level calibrated probabilities were never saved by the original runs, only the summaries, so
each arm is refitted here and its per-row output retained. The refits reuse the frozen
configuration exactly; the reference arm is checked against FROZEN_B_AUROC before anything else
runs.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
for p in (ROOT, ROOT / "ingest_and_scoring", ROOT / "experiments",
          ROOT / "experiments" / "experiments" / "08_routing_code",
          ROOT / "experiments" / "experiments" / "09_live_and_controls" / "code"):
    sys.path.insert(0, str(p))

OUT = Path(os.environ["SIG_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("V3_RUN_DIR", str(OUT))
LIVE_DIR = Path(os.environ.get("LIVE_DIR", ROOT / "experiments" / "runs" / "live_main_v1"))
B = int(os.environ.get("SIG_B", 2000))
SEED = int(os.environ.get("SIG_SEED", 20260821))

import core  # noqa: E402
import v3core as V  # noqa: E402
import _contract  # noqa: E402

POOL, SEEDS = list(core.ACTIONS), list(V.SEEDS)
C = _contract.load()
HPB, CHEAP = dict(C["hp_B"]), list(C["features"])
TRAIN, TEST, _A, _v = V.load(with_test_labels=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def auc(y_bool, score, n_pos, n_neg):
    """AUROC by the rank identity; averaged ranks, because isotonic creates ties."""
    r = rankdata(score)
    return (r[y_bool].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap(y, groups, qa, qb, family, label):
    """Paired cluster bootstrap of mean-over-seeds AUROC(qa) - AUROC(qb)."""
    yb = y.astype(bool)
    npos, nneg = int(yb.sum()), int((~yb).sum())
    S = qa.shape[1]
    d_obs = float(np.mean([auc(yb, qa[:, j], npos, nneg) - auc(yb, qb[:, j], npos, nneg)
                           for j in range(S)]))
    uniq, ginv = np.unique(groups, return_inverse=True)
    by_g = [np.where(ginv == k)[0] for k in range(len(uniq))]
    G = len(uniq)
    rng = np.random.default_rng(SEED)
    ds = []
    for _ in range(B):
        idx = np.concatenate([by_g[k] for k in rng.integers(0, G, G)])
        yy = y[idx].astype(bool)
        p, q = int(yy.sum()), int((~yy).sum())
        if p == 0 or q == 0:
            continue
        ds.append(np.mean([auc(yy, qa[idx, j], p, q) - auc(yy, qb[idx, j], p, q)
                           for j in range(S)]))
    ds = np.asarray(ds)
    lo95, hi95 = np.percentile(ds, [2.5, 97.5])
    a = 0.05 / family
    lob, hib = np.percentile(ds, [100 * a / 2, 100 * (1 - a / 2)])
    below, above = int((ds <= 0).sum()), int((ds >= 0).sum())
    out = dict(comparison=label, d_auroc=d_obs, ci95_lo=float(lo95), ci95_hi=float(hi95),
               sig95=bool(lo95 > 0 or hi95 < 0), bonf_lo=float(lob), bonf_hi=float(hib),
               sig_bonf=bool(lob > 0 or hib < 0),
               p_boot=2.0 * min(below, above) / len(ds), family=family,
               n_draws=int(len(ds)), n_groups=G, n_rows=int(len(y)))
    log(f"  {label:44s} d={d_obs:+.5f} 95%[{lo95:+.5f},{hi95:+.5f}] "
        f"bonf[{lob:+.5f},{hib:+.5f}] p={out['p_boot']:.5f} "
        f"{'SEPARABLE' if out['sig_bonf'] else 'not separable'}")
    return out


# ---------------------------------------------------------------- shared refit
def train_seed(seed, train=None, feats=None, ev=None):
    train = TRAIN if train is None else train
    feats = CHEAP if feats is None else feats
    ev = TEST if ev is None else ev
    held = V.stratified_group_split(train, seed)
    fit = train.loc[~held].reset_index(drop=True)
    val = train.loc[held].reset_index(drop=True)
    yv = val["label_supported"].to_numpy(int)
    if len(set(yv.tolist())) < 2 or len(fit) < 30:
        return None
    cals = core.platt(fit, actions=POOL)
    c_fit, c_val = core.apply_platt(fit, cals, POOL), core.apply_platt(val, cals, POOL)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=POOL,
                                 features=feats, hp=HPB, target="regret", learner="rf")
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
    return dict(fit=fit, val=val, yv=yv, cals=cals, c_val=c_val, h_val=h_val, h_ev=h_ev,
                cvec=cvec, beta=beta, p_val=p_val)


def arm_probs(feats, train=None, ev=None, train_fn=None):
    """Per-row calibrated probability for every seed, for one configuration.

    train_fn(seed) lets the training frame depend on the seed. The few-shot arms need this:
    fewshot.py draws its k in-corpus rows per seed and concatenates them onto the base, and the
    resulting row order is not neutral -- reordering the same rows moves the fit by about as
    much as changing the seed does.
    """
    ev = TEST if ev is None else ev
    cols = []
    for seed in SEEDS:
        tr = train_fn(seed) if train_fn is not None else train
        t = train_seed(seed, train=tr, feats=feats, ev=ev)
        if t is None:
            return None
        c_ev = core.apply_platt(ev, t["cals"], POOL)
        _s, p_ev, _m = V.route(ev, t["h_ev"], c_ev, t["beta"], POOL, t["cvec"])
        cols.append(core.isotonic(t["p_val"], t["yv"], p_ev))
    return np.column_stack(cols)


results = {}
y_test = TEST["label_supported"].to_numpy(int)
g_test = TEST["content_doc_key"].astype(str).to_numpy()

# Gate: the rank AUC must agree with the reference implementation.
_yb = y_test.astype(bool)
_p, _q = int(_yb.sum()), int((~_yb).sum())
_probe = TEST["score__factcc"].to_numpy(float)
if abs(auc(_yb, _probe, _p, _q) - roc_auc_score(y_test, _probe)) > 1e-12:
    sys.exit("rank AUC disagrees with roc_auc_score")
log("rank AUC matches roc_auc_score")

# ---------------------------------------------------------------- 2. dataset-identity arms
log("dataset-identity arms: refitting four configurations")
for k in sorted(TRAIN["dataset_key"].astype(str).unique()):
    pre = k.split("_")[0]
    TRAIN[f"ds__{pre}"] = (TRAIN["dataset_key"].astype(str).str.split("_").str[0] == pre).astype(float)
    TEST[f"ds__{pre}"] = (TEST["dataset_key"].astype(str).str.split("_").str[0] == pre).astype(float)
DS = sorted(c for c in TRAIN.columns if c.startswith("ds__"))
TRAIN["_const"] = 1.0
TEST["_const"] = 1.0

ARMS = {"none (constant only)": ["_const"], "dataset one-hot only": DS,
        "six cheap features": CHEAP, "cheap + dataset": CHEAP + DS}
Q = {}
for name, feats in ARMS.items():
    Q[name] = arm_probs(feats)
    a = float(np.mean([roc_auc_score(y_test, Q[name][:, j]) for j in range(len(SEEDS))]))
    log(f"  {name:24s} AUROC {a:.5f}  ({len(feats)} features)")

ref = float(np.mean([roc_auc_score(y_test, Q["six cheap features"][:, j])
                     for j in range(len(SEEDS))]))
_contract.check_reference(ref, what="dataset-arms reference (six cheap features)")

results["arms"] = [
    bootstrap(y_test, g_test, Q["six cheap features"], Q[n], 3, f"six cheap vs {n}")
    for n in ("none (constant only)", "dataset one-hot only", "cheap + dataset")]

# ---------------------------------------------------------------- 1. live vs frozen matrix
if (LIVE_DIR / "live_raw.npz").exists() and (LIVE_DIR / "plan.npz").exists():
    log("live execution: rebuilding per-row probabilities")
    raw = np.load(LIVE_DIR / "live_raw.npz")["raw"]
    sel = np.load(LIVE_DIR / "plan.npz")["sel"]
    EV = TEST.reset_index(drop=True)
    rows = np.arange(len(EV))
    q_live, q_mat = [], []
    for si, seed in enumerate(SEEDS):
        t = train_seed(seed, ev=EV)
        ev_live = EV.copy()
        for k, a in enumerate(POOL):
            idx = np.where(sel[:, si] == k)[0]
            col = ev_live[f"score__{a}"].to_numpy(float).copy()
            col[idx] = raw[idx, si]
            ev_live[f"score__{a}"] = col
        c_live = core.apply_platt(ev_live, t["cals"], POOL)
        c_mat = core.apply_platt(EV, t["cals"], POOL)
        p1_live = np.column_stack([c_live[a] for a in POOL])[rows, sel[:, si]]
        p1_mat = np.column_stack([c_mat[a] for a in POOL])[rows, sel[:, si]]
        q_live.append(core.isotonic(t["p_val"], t["yv"], p1_live))
        q_mat.append(core.isotonic(t["p_val"], t["yv"], p1_mat))
    q_live, q_mat = np.column_stack(q_live), np.column_stack(q_mat)
    log(f"  live   AUROC {np.mean([roc_auc_score(y_test, q_live[:, j]) for j in range(10)]):.5f}")
    log(f"  matrix AUROC {np.mean([roc_auc_score(y_test, q_mat[:, j]) for j in range(10)]):.5f}")
    results["live"] = [bootstrap(y_test, g_test, q_live, q_mat, 1, "live vs frozen matrix")]
else:
    log(f"live artifacts absent under {LIVE_DIR}; skipping the live comparison")

# ---------------------------------------------------------------- 3. few-shot, k=0 vs full
log("few-shot: refitting k=0 and full-corpus per held-out corpus")
PAIRS = {"cogensumm": ("cogensumm_val", "cogensumm_test"),
         "frank": ("frank_valid", "frank_test"),
         "ragtruth": ("ragtruth_train", "ragtruth_test"),
         "unisumeval": ("unisumeval_train", "unisumeval_dev")}
fs = []
for corpus, (tr_key, te_key) in PAIRS.items():
    in_c = TRAIN["dataset_key"].astype(str) == tr_key
    base = TRAIN.loc[~in_c]
    pool = TRAIN.loc[in_c]
    ev = TEST[TEST["dataset_key"].astype(str) == te_key].reset_index(drop=True)

    def build(k):
        def f(seed):
            rng = np.random.default_rng(seed * 1000 + k)
            take = (pool.sample(k, random_state=int(rng.integers(0, 2**31 - 1)))
                    if k else pool.iloc[:0])
            return pd.concat([base, take], ignore_index=True)
        return f

    q0 = arm_probs(CHEAP, ev=ev, train_fn=build(0))
    qf = arm_probs(CHEAP, ev=ev, train_fn=build(len(pool)))
    if q0 is None or qf is None:
        log(f"  {corpus}: a configuration was degenerate; skipped")
        continue
    yc = ev["label_supported"].to_numpy(int)
    gc = ev["content_doc_key"].astype(str).to_numpy()
    log(f"  {corpus:12s} k=0 {np.mean([roc_auc_score(yc, q0[:, j]) for j in range(10)]):.5f}"
        f" -> full {np.mean([roc_auc_score(yc, qf[:, j]) for j in range(10)]):.5f}"
        f"  ({len(ev)} rows, {len(set(gc))} groups)")
    fs.append(bootstrap(yc, gc, qf, q0, 4, f"{corpus}: full-corpus vs k=0"))
results["fewshot"] = fs

(OUT / "CONTROLS_SIGNIFICANCE.json").write_text(json.dumps(results, indent=2))
flat = [r for v in results.values() for r in v]
pd.DataFrame(flat).to_csv(OUT / "CONTROLS_SIGNIFICANCE.csv", index=False)
log(f"wrote {OUT}/CONTROLS_SIGNIFICANCE.{{json,csv}}  ({len(flat)} comparisons)")
