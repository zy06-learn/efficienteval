#!/usr/bin/env python3
"""Part 3 — extended ablation: per-corpus training, subset lattices, convergence, feature sets.

Reference system is `part1c_main_full_v1`. The subset lattices contain the reference twice, once
as the full feature subset and once as the full pool subset, and both must reproduce part1c bit
for bit; that is the attribution gate. Pool, features, target, learner, hyperparameters, beta
grid and tolerance, seeds, split construction, calibration and the threshold rule are inherited
and asserted.

Stages
  prep              per-corpus prior-shift and feasibility record; nothing is fitted
  percorpus         train and evaluate inside one corpus, four corpora, official train -> test
  lattice{B,A}      all 64 feature subsets and all 8 pool subsets, exact Shapley attribution
  converge{B,A}     loss and routed AUROC against training-set size, and against tree count
  featuresets{B,A}  the superseded nine- and five-feature sets against the unified six
  controls{B,A}     call-proportion-matched random routing, and the label oracle
  report{B,A}       gate, Shapley, best-k curves, paired intervals for the declared contrasts

Why Shapley rather than 64 significance tests. The six features are redundant: Part 2 showed
only one to three survive a twenty-five-comparison correction, because removing a feature whose
information another also carries produces no measurable loss. Testing 64 subsets would need a
Bonferroni threshold of alpha = 0.00078 and would return "not separable" almost everywhere,
which is a property of the design rather than a finding. The average marginal contribution over
all subsets containing a feature is the quantity that answers "what is this feature worth" under
redundancy, and 64 subsets is exactly the material for computing it exactly at n = 6.

Paired intervals are therefore reported only for the small pre-declared families -- the feature
sets and the controls -- whose arms store row-level predictions for that purpose. The lattice
stores aggregates only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
PART1C = ROOT / "experiments" / "cross_stage_contract" / "part1c_main_full_v1"
RUN = Path(os.environ["V3_RUN_DIR"]).resolve()
RES = RUN / "results"
RES.mkdir(parents=True, exist_ok=True)
V.RES = RES
SMOKE = os.environ.get("V3_SMOKE", "0") == "1"

CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
POOL = list(CONTRACT["pool"])
FEATURES = list(CONTRACT["features"])
TARGET, LEARNER = CONTRACT["target"], CONTRACT["learner"]
_HPSEL = json.loads((PART1C / "00_contract" / "HP_SELECTED.json").read_text())
HP = {"A": dict(_HPSEL["hyperparameters_A"]), "B": dict(_HPSEL["hyperparameters_B"])}
SEEDS = list(V.SEEDS[:2] if SMOKE else V.SEEDS)
DRAWS = 200 if SMOKE else int(V.C.BOOTSTRAP_DRAWS)

# Smoke exercises the whole lattice code path on three features, so Shapley, best-k and the
# empty-set convention all run, without paying for 64 subsets.
LATTICE_FEATURES = FEATURES[:3] if SMOKE else FEATURES

FEATURES_A9 = ["claim_token_count", "claim_source_length_ratio", "fact_mention_density",
               "tfidf_top1", "entity_coverage", "rougeL_top1",
               "structured_source_line_ratio", "evidence_span_normalized",
               "summary_sentence_count"]
FEATURES_B5 = ["structured_source_line_ratio", "entity_coverage", "sentence_count",
               "idf_weighted_coverage_mean", "number_coverage"]

CORPUS_PAIRS = {"cogensumm": ("cogensumm_val", "cogensumm_test"),
                "frank": ("frank_valid", "frank_test"),
                "ragtruth": ("ragtruth_train", "ragtruth_test"),
                "unisumeval": ("unisumeval_train", "unisumeval_dev")}

SIZE_FRACTIONS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
TREE_GRID = (1, 2, 5, 10, 25, 50, 100, 200, 400, 800)
CONST_COL = "_const_zero"

REFERENCE = {
    proto: {"auroc": float(r["auroc"]), "ms_det": float(r["ms_part1_basis"])}
    for proto in ("A", "B")
    for _i, r in pd.read_csv(PART1C / "01_main_tables" / "publication"
                             / f"{proto}_MAIN.csv").iterrows()
    if r["system"] == "OURS"
}
TOL_AUROC, TOL_MS = 1e-9, 1e-6


# ------------------------------------------------------------------ shared pipeline
def _mcc_threshold(q, y):
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def _stage2(p_val, y_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    t0 = time.perf_counter()
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    apply_ms = (time.perf_counter() - t0) * 1000.0 / max(len(p_ev), 1)
    return q_ev, _mcc_threshold(q_val, y_val), float(apply_ms)


def _pre_ms(frame):
    return (frame["feature_latency_ms"].to_numpy(float)
            + frame["compact16_feature_latency_ms"].to_numpy(float))


def _fit_heads_models(fit, seed, cal_fit, actions, features, hp):
    """Mirror of `core.fit_heads` that also returns the fitted forests.

    The tree-count stage needs `estimators_`. The fitting rule is copied exactly: one forest per
    action, trained only on rows where that action is available, predictions clipped to [0, 1].
    """
    Y, is_clf = core.targets(fit, cal_fit, actions, TARGET)
    assert not is_clf, "regression target expected"
    xt = fit[features].to_numpy(float)
    models = []
    for j, a in enumerate(actions):
        av = fit[f"available__{a}"].astype(bool).to_numpy()
        m = RandomForestRegressor(criterion=core.C.CRITERION, n_jobs=8, random_state=seed,
                                  **dict(hp))
        m.fit(xt[av], np.asarray(Y[av, j], float))
        models.append(m)
    return models


def _predict_heads(models, frame, features):
    x = frame[features].to_numpy(float)
    t0 = time.perf_counter()
    out = np.column_stack([np.clip(m.predict(x), 0.0, 1.0) for m in models])
    ms = (time.perf_counter() - t0) * 1000.0 / max(len(frame), 1)
    return core.HeadPredictions(out, ms), float(ms)


def route_and_score(fit, val, evalf, seed, proto, *, actions=None, features=None,
                    beta=None, oracle=False, shares=None):
    """One fitted router, end to end.

    `oracle` replaces head predictions with the target computed from the labels of the frame
    being predicted, which is the upper bound on action selection and is not deployable.
    `shares` replaces the argmax with sampling from a fixed action distribution, which is the
    call-proportion-matched random control: the budget is preserved and only the per-instance
    assignment is destroyed.
    """
    actions = list(actions or POOL)
    features = list(features or FEATURES)
    hp = HP[proto]
    y_val = val["label_supported"].to_numpy(int)
    pre = _pre_ms(evalf)

    cals = core.platt(fit, actions=actions)
    c_fit = core.apply_platt(fit, cals, actions=actions)
    c_val = core.apply_platt(val, cals, actions=actions)
    t0 = time.perf_counter()
    c_ev = core.apply_platt(evalf, cals, actions=actions)
    platt_ms = (time.perf_counter() - t0) * 1000.0 / max(len(evalf), 1)

    if oracle:
        h_val = core.HeadPredictions(core.targets(val, c_val, actions, TARGET)[0], 0.0)
        h_ev = core.HeadPredictions(core.targets(evalf, c_ev, actions, TARGET)[0], 0.0)
        head_ms = 0.0
        models = None
    else:
        models = _fit_heads_models(fit, seed, c_fit, actions, features, hp)
        h_val, _ = _predict_heads(models, val, features)
        h_ev, head_ms = _predict_heads(models, evalf, features)

    cvec = np.array([core.fold_costs(fit, actions=actions)[a] for a in actions], float)
    b = V.choose_beta(val, h_val, c_val, actions, cvec)[0] if beta is None else float(beta)
    _s, p_val, _m = V.route(val, h_val, c_val, b, actions, cvec)

    t0 = time.perf_counter()
    sel, p_ev, ms_route = V.route(evalf, h_ev, c_ev, b, actions, cvec)
    route_ms = (time.perf_counter() - t0) * 1000.0 / max(len(evalf), 1)

    if shares is not None:
        rng = np.random.default_rng(core.base.stable_seed(seed, "matched", "random"))
        sel = rng.choice(len(actions), size=len(evalf), p=np.asarray(shares, float))
        rows = np.arange(len(evalf))
        avail = np.column_stack([evalf[f"available__{a}"].to_numpy(bool) for a in actions])
        # keep the control legal: an unavailable draw falls back to the first available action
        bad = ~avail[rows, sel]
        if bad.any():
            sel[bad] = np.argmax(avail[bad], axis=1)
        p_ev = np.column_stack([c_ev[a] for a in actions])[rows, sel]
        ms_route = (np.column_stack([evalf[f"latency_ms__{a}"].fillna(0).to_numpy(float)
                                     for a in actions])[rows, sel]
                    + evalf["feature_latency_ms"].to_numpy(float))

    ver_ms = ms_route - evalf["feature_latency_ms"].to_numpy(float)
    q_ev, thr, s2_ms = _stage2(p_val, y_val, p_ev)
    return {"prob": q_ev, "thr": thr, "sel": np.asarray(sel, int), "beta": b,
            "ms": ver_ms + pre + head_ms + route_ms + platt_ms + s2_ms,
            "ms_det": ms_route, "models": models, "c_val": c_val, "h_val": h_val,
            "c_ev": c_ev, "cvec": cvec}


def no_verifier_arm(fit, val, evalf, seed, proto, features=None):
    """The cheap features predict the label directly; no verifier is called."""
    features = list(features or FEATURES)
    m = RandomForestRegressor(criterion=core.C.CRITERION, n_jobs=8, random_state=seed,
                              **dict(HP[proto]))
    m.fit(fit[features].to_numpy(float), fit["label_supported"].to_numpy(float))
    p_val = np.clip(m.predict(val[features].to_numpy(float)), 0.0, 1.0)
    t0 = time.perf_counter()
    p_ev = np.clip(m.predict(evalf[features].to_numpy(float)), 0.0, 1.0)
    head_ms = (time.perf_counter() - t0) * 1000.0 / max(len(evalf), 1)
    pre = _pre_ms(evalf)
    q_ev, thr, s2_ms = _stage2(p_val, val["label_supported"].to_numpy(int), p_ev)
    return {"prob": q_ev, "thr": thr, "sel": np.zeros(len(evalf), int), "beta": float("nan"),
            "ms": pre + head_ms + s2_ms, "ms_det": pre}


def _with_const(frame):
    if CONST_COL in frame.columns:
        return frame
    out = frame.copy()
    out[CONST_COL] = 0.0
    return out


def _frames(proto, seed, TRAIN, TEST, ALL):
    if proto == "B":
        held = V.stratified_group_split(TRAIN, seed)
        yield (TRAIN.loc[~held].reset_index(drop=True),
               TRAIN.loc[held].reset_index(drop=True), TEST)
    else:
        fold = V.folds_stratified(ALL, seed)
        for t, vf, trf in V.rotations():
            yield (ALL.loc[np.isin(fold, trf)].reset_index(drop=True),
                   ALL.loc[fold == vf].reset_index(drop=True),
                   ALL.loc[fold == t].reset_index(drop=True))


# ------------------------------------------------------------------ provenance
def _sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def _write_provenance():
    prov = RUN / "04_provenance"
    prov.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__).resolve(), ROOT / "experiments" / "v3core.py",
            ROOT / "shared" / "core.py", ROOT / "shared" / "config.py",
            ROOT / "ingest_and_scoring" / "config_v2.py", ROOT / "verifiers" / "tenfold_v1.py"]
    (prov / "CODE_SNAPSHOT.sha256").write_text(
        "".join(f"{_sha256(p)}  {p}\n" for p in code if p.exists()))
    (prov / "GIT_STATE.txt").write_text(
        f"HEAD: {_git('rev-parse', 'HEAD')}\n\nstatus --porcelain:\n"
        f"{_git('status', '--porcelain')}\n")
    (prov / "RUN_METADATA.txt").write_text(
        f"command={' '.join(sys.argv)}\nstarted={time.strftime('%Y-%m-%d %H:%M:%S%z')}\n"
        f"python={sys.version}\nplatform={platform.platform()}\n"
        f"nproc={os.cpu_count()}\nsmoke={SMOKE}\nseeds={SEEDS}\n"
        f"hp_A={HP['A']}\nhp_B={HP['B']}\n")
    (prov / "PID").write_text(f"{os.getpid()}\n")


# ------------------------------------------------------------------ E1
def prep():
    """Per-corpus prior shift and feasibility. Nothing is fitted.

    The shift is the finding, not a caveat: a corpus whose own training and evaluation splits
    disagree on the label prior by twenty-nine percentage points cannot calibrate a threshold
    against itself, which is the argument for pooling the four corpora.
    """
    core.log("===== prep: per-corpus prior shift and feasibility =====")
    TRAIN, TEST, _ALL, _v = V.load(with_test_labels=True)
    rows = []
    for corpus, (tr_key, te_key) in CORPUS_PAIRS.items():
        a = TRAIN[TRAIN.dataset_key.astype(str) == tr_key]
        b = TEST[TEST.dataset_key.astype(str) == te_key]
        rows.append({
            "corpus": corpus, "train_split": tr_key, "test_split": te_key,
            "train_rows": len(a), "train_groups": int(a.content_doc_key.nunique()),
            "train_pos_rate": float(a.label_supported.mean()),
            "train_minority_count": int(min(a.label_supported.sum(),
                                            (1 - a.label_supported).sum())),
            "test_rows": len(b), "test_groups": int(b.content_doc_key.nunique()),
            "test_pos_rate": float(b.label_supported.mean()),
            "prior_shift_pp": 100 * (float(b.label_supported.mean())
                                     - float(a.label_supported.mean()))})
    D = pd.DataFrame(rows)
    D = D.reindex(D.prior_shift_pp.abs().sort_values(ascending=False).index)
    V.save("E1_PRIOR_SHIFT.csv", D)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ptr, pte = float(TRAIN.label_supported.mean()), float(TEST.label_supported.mean())
    core.log(f"pooled TRAIN pos rate {ptr:.4f} | pooled TEST {pte:.4f} | "
             f"shift {100*(pte-ptr):+.2f} pp")
    core.log(f"worst single-corpus shift {D.prior_shift_pp.abs().max():.2f} pp; pooling reduces "
             f"the worst-case shift one fitted model must absorb to {abs(100*(pte-ptr)):.2f} pp")
    _write_provenance()


def _pooled_reference(corpus):
    """part1c's pooled-trained Protocol B AUROC restricted to this corpus's evaluation split."""
    col = f"auroc__{CORPUS_PAIRS[corpus][1]}"
    D = pd.read_csv(PART1C / "01_main_tables" / "publication" / "B_MAIN.csv")
    row = D[D.system == "OURS"]
    return float(row[col].iloc[0]) if col in D.columns and len(row) else float("nan")


def percorpus():
    core.log("===== E1: per-corpus training and evaluation =====")
    TRAIN, TEST, _ALL, verifiers = V.load(with_test_labels=True)
    rows, summary = [], []
    for corpus, (tr_key, te_key) in CORPUS_PAIRS.items():
        sub_tr = TRAIN[TRAIN.dataset_key.astype(str) == tr_key].reset_index(drop=True)
        sub_te = TEST[TEST.dataset_key.astype(str) == te_key].reset_index(drop=True)
        y = sub_te["label_supported"].to_numpy(int)
        core.log(f"  {corpus}: train {len(sub_tr)}/{sub_tr.content_doc_key.nunique()} -> "
                 f"test {len(sub_te)}/{sub_te.content_doc_key.nunique()}")
        got = {"OURS": [], "no_verifier": []}
        got.update({f"fixed::{v}": [] for v in verifiers})
        skipped, used_frac = 0, []
        for seed in SEEDS:
            # small, heavily imbalanced corpora need a larger validation share before Platt,
            # beta, isotonic and the threshold can all be fitted on it; the fraction is raised
            # until both classes and at least five of the minority are present
            fit = val = None
            for frac in (0.20, 0.30, 0.40):
                held = V.stratified_group_split(sub_tr, seed, fraction=frac)
                f_, v_ = sub_tr.loc[~held], sub_tr.loc[held]
                yv = v_["label_supported"].to_numpy(int)
                if len(set(yv.tolist())) > 1 and min((yv == 0).sum(), (yv == 1).sum()) >= 5:
                    fit, val, = f_.reset_index(drop=True), v_.reset_index(drop=True)
                    used_frac.append(frac)
                    break
            if fit is None:
                skipped += 1
                continue
            got["OURS"].append(route_and_score(fit, val, sub_te, seed, "B"))
            got["no_verifier"].append(no_verifier_arm(fit, val, sub_te, seed, "B"))
            for v in verifiers:
                cal = core.platt(fit, actions=[v])
                p_val = core.apply_platt(val, cal, actions=[v])[v]
                p_ev = core.apply_platt(sub_te, cal, actions=[v])[v]
                q, thr, s2 = _stage2(p_val, val["label_supported"].to_numpy(int), p_ev)
                got[f"fixed::{v}"].append(
                    {"prob": q, "thr": thr,
                     "ms": sub_te[f"latency_ms__{v}"].to_numpy(float) + s2})
        if skipped:
            core.log(f"    {skipped}/{len(SEEDS)} seeds skipped: no validation split reached "
                     f"five of each class at any fraction in the ladder")
        for name, preds in got.items():
            if not preds:
                continue
            au = [float(roc_auc_score(y, p["prob"])) for p in preds]
            mets = [core.metrics(y, p["prob"], p["ms"], threshold=p["thr"]) for p in preds]
            rows.append({"corpus": corpus, "system": name, "seeds_used": len(preds),
                         "test_rows": len(sub_te),
                         "test_groups": int(sub_te.content_doc_key.nunique()),
                         "auroc": float(np.mean(au)), "auroc_sd": float(np.std(au)),
                         **{k: float(np.nanmean([m[k] for m in mets]))
                            for k in ("bacc", "mcc", "ece", "brier", "ms", "p95_ms", "aurc")}})
        fx = [r for r in rows if r["corpus"] == corpus and r["system"].startswith("fixed")]
        ours = next(r for r in rows if r["corpus"] == corpus and r["system"] == "OURS")
        best = max(fx, key=lambda r: r["auroc"]) if fx else None
        pooled = _pooled_reference(corpus)
        summary.append({"corpus": corpus, "test_groups": ours["test_groups"],
                        "seeds_used": ours["seeds_used"],
                        "val_fraction_used": float(np.mean(used_frac)) if used_frac else np.nan,
                        "in_domain_ours": ours["auroc"], "in_domain_ours_sd": ours["auroc_sd"],
                        "in_domain_best_fixed": best["auroc"] if best else np.nan,
                        "in_domain_best_fixed_name": best["system"] if best else "",
                        "pooled_trained_ours": pooled,
                        "in_domain_minus_pooled": ours["auroc"] - pooled})
        core.log(f"    OURS {ours['auroc']:.5f} (sd {ours['auroc_sd']:.5f}) | best fixed "
                 f"{best['auroc']:.5f} ({best['system']}) | pooled-trained on this corpus "
                 f"{pooled:.5f} | in-domain minus pooled {ours['auroc']-pooled:+.5f}")
    V.save("E1_PERCORPUS.csv", pd.DataFrame(rows))
    D = pd.DataFrame(summary)
    V.save("E1_PERCORPUS_SUMMARY.csv", D)
    print(D.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


# ------------------------------------------------------------------ E2 lattices
def _subsets(items):
    return [list(c) for k in range(len(items) + 1) for c in itertools.combinations(items, k)]


def _lattice_arms():
    arms = [("feat", "|".join(s) if s else "(empty)", s) for s in _subsets(LATTICE_FEATURES)]
    arms += [("pool", "+".join(s) if s else "(empty)", s) for s in _subsets(POOL)]
    return arms


def lattice(proto):
    core.log(f"===== E2: subset lattices, Protocol {proto} =====")
    TRAIN, TEST, ALL, _v = V.load(with_test_labels=True)
    arms = _lattice_arms()
    core.log(f"{len(arms)} arms: {2**len(LATTICE_FEATURES)} feature subsets "
             f"+ {2**len(POOL)} pool subsets | hp={HP[proto]}")
    rows, t0 = [], time.time()

    for i, (kind, key, members) in enumerate(arms, 1):
        au, ms, msd = [], [], []
        for seed in SEEDS:
            probs, ys, mss, msds = [], [], [], []
            for fit, val, evalf in _frames(proto, seed, TRAIN, TEST, ALL):
                fit, val, evalf = _with_const(fit), _with_const(val), _with_const(evalf)
                if kind == "feat":
                    r = route_and_score(fit, val, evalf, seed, proto,
                                        features=members or [CONST_COL])
                elif members:
                    r = route_and_score(fit, val, evalf, seed, proto, actions=members)
                else:
                    r = no_verifier_arm(fit, val, evalf, seed, proto)
                probs.append(r["prob"])
                ys.append(evalf["label_supported"].to_numpy(int))
                n = len(r["prob"])
                mss.append(np.broadcast_to(r["ms"], (n,)))
                msds.append(np.broadcast_to(r["ms_det"], (n,)))
            au.append(float(roc_auc_score(np.concatenate(ys), np.concatenate(probs))))
            ms.append(float(np.concatenate(mss).mean()))
            msd.append(float(np.concatenate(msds).mean()))
        rows.append({"kind": kind, "subset": key, "size": len(members),
                     "auroc": float(np.mean(au)), "auroc_sd": float(np.std(au)),
                     "ms": float(np.mean(ms)), "ms_det": float(np.mean(msd))})
        if i % (5 if proto == "A" else 10) == 0 or i == len(arms):
            el = time.time() - t0
            core.log(f"  {i}/{len(arms)}  {el/60:.1f} min  eta {el/i*(len(arms)-i)/60:.1f} min")

    D = pd.DataFrame(rows)
    V.save(f"E2_LATTICE_{proto}.csv",
           D.sort_values(["kind", "auroc"], ascending=[True, False]))
    _shapley_and_bestk(D, proto)


def _shapley_and_bestk(D, proto):
    """Exact Shapley value over the full lattice, plus the best subset at each size."""
    out = []
    for kind, items, sep in (("feat", LATTICE_FEATURES, "|"), ("pool", POOL, "+")):
        sub = D[D.kind == kind]
        val = {(frozenset() if s == "(empty)" else frozenset(s.split(sep))): float(a)
               for s, a in zip(sub.subset, sub.auroc)}
        n = len(items)
        for it in items:
            others = [x for x in items if x != it]
            phi = 0.0
            for k in range(n):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                for c in itertools.combinations(others, k):
                    s = frozenset(c)
                    phi += w * (val[s | {it}] - val[s])
            out.append({"kind": kind, "item": it, "shapley_auroc": phi,
                        "solo_auroc": val[frozenset([it])],
                        "leave_one_out_auroc": val[frozenset(items) - {it}]})
        total = val[frozenset(items)] - val[frozenset()]
        got = sum(o["shapley_auroc"] for o in out if o["kind"] == kind)
        core.log(f"  Shapley over {kind}: sum {got:.6f} vs full-minus-empty {total:.6f} "
                 f"(efficiency residual {got-total:+.2e})")
        if abs(got - total) > 1e-9:
            raise AssertionError(f"Shapley efficiency violated for {kind}")
    S = pd.DataFrame(out)
    S["shapley_share"] = S.groupby("kind")["shapley_auroc"].transform(
        lambda c: c / c.sum() if c.sum() else np.nan)
    S = S.sort_values(["kind", "shapley_auroc"], ascending=[True, False])
    V.save(f"E2_SHAPLEY_{proto}.csv", S)
    print(S.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    rows = []
    for kind in ("feat", "pool"):
        for k, grp in D[D.kind == kind].groupby("size"):
            best = grp.loc[grp.auroc.idxmax()]
            rows.append({"kind": kind, "size": int(k), "n_subsets": int(len(grp)),
                         "best_subset": best.subset, "best_auroc": float(best.auroc),
                         "best_ms": float(best.ms), "mean_auroc": float(grp.auroc.mean()),
                         "worst_auroc": float(grp.auroc.min())})
    B = pd.DataFrame(rows)
    V.save(f"E2_BEST_K_{proto}.csv", B)
    print(B.to_string(index=False, float_format=lambda v: f"{v:.5f}"))


# ------------------------------------------------------------------ E3 convergence
def converge(proto):
    core.log(f"===== E3: convergence, Protocol {proto} =====")
    TRAIN, TEST, ALL, _v = V.load(with_test_labels=True)
    hp = HP[proto]

    fracs = SIZE_FRACTIONS[:2] if SMOKE else SIZE_FRACTIONS
    rows = []
    for frac in fracs:
        losses, aurocs = [], []
        for seed in SEEDS:
            probs, ys, ls = [], [], []
            for fit, val, evalf in _frames(proto, seed, TRAIN, TEST, ALL):
                rng = np.random.default_rng(core.base.stable_seed(seed, "size", f"{frac}"))
                grp = fit.content_doc_key.astype(str).to_numpy()
                uniq = np.unique(grp)
                keep = rng.choice(uniq, max(int(round(len(uniq) * frac)), 2), replace=False)
                sub = fit.loc[np.isin(grp, keep)].reset_index(drop=True)
                r = route_and_score(sub, val, evalf, seed, proto)
                Y, _ = core.targets(val, r["c_val"], POOL, TARGET)
                ls.append(float(np.mean((np.asarray(r["h_val"], float) - Y) ** 2)))
                probs.append(r["prob"])
                ys.append(evalf["label_supported"].to_numpy(int))
            losses.append(float(np.mean(ls)))
            aurocs.append(float(roc_auc_score(np.concatenate(ys), np.concatenate(probs))))
        rows.append({"fit_group_fraction": frac, "val_head_loss": float(np.mean(losses)),
                     "eval_auroc": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs))})
        core.log(f"  size {frac:.0%}: val loss {rows[-1]['val_head_loss']:.6f} | "
                 f"eval auroc {rows[-1]['eval_auroc']:.5f}")
    V.save(f"E3_LEARNING_CURVE_{proto}.csv", pd.DataFrame(rows))

    # --- tree count, from a growing prefix of the same fitted forests.
    # Protocol A uses the first rotation only: this is a diagnostic about how many trees the
    # forest needs, not a headline number, and one rotation already answers it at a tenth of
    # the cost.
    grid = sorted({k for k in TREE_GRID if k <= hp["n_estimators"]} | {hp["n_estimators"]})
    rows = []
    for seed in SEEDS:
        for fit, val, evalf in _frames(proto, seed, TRAIN, TEST, ALL):
            cals = core.platt(fit, actions=POOL)
            c_fit = core.apply_platt(fit, cals, actions=POOL)
            c_val = core.apply_platt(val, cals, actions=POOL)
            c_ev = core.apply_platt(evalf, cals, actions=POOL)
            models = _fit_heads_models(fit, seed, c_fit, POOL, FEATURES, hp)
            xv, xe = val[FEATURES].to_numpy(float), evalf[FEATURES].to_numpy(float)
            cum_v = [np.cumsum([t.predict(xv) for t in m.estimators_], axis=0) for m in models]
            cum_e = [np.cumsum([t.predict(xe) for t in m.estimators_], axis=0) for m in models]
            Y, _ = core.targets(val, c_val, POOL, TARGET)
            cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
            y_ev = evalf["label_supported"].to_numpy(int)
            for k in grid:
                hv = np.clip(np.column_stack([c[k - 1] / k for c in cum_v]), 0, 1)
                he = np.clip(np.column_stack([c[k - 1] / k for c in cum_e]), 0, 1)
                b = V.choose_beta(val, core.HeadPredictions(hv), c_val, POOL, cvec)[0]
                _s, p, _m = V.route(evalf, core.HeadPredictions(he), c_ev, b, POOL, cvec)
                rows.append({"seed": seed, "n_trees": k,
                             "val_head_loss": float(np.mean((hv - Y) ** 2)),
                             "eval_auroc": float(roc_auc_score(y_ev, p)), "beta": float(b)})
            break
    T = (pd.DataFrame(rows).groupby("n_trees")
         .agg(val_head_loss=("val_head_loss", "mean"), eval_auroc=("eval_auroc", "mean"),
              auroc_sd=("eval_auroc", "std"), n_fits=("eval_auroc", "size")).reset_index())
    V.save(f"E3_TREE_CURVE_{proto}.csv", T)
    print(T.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


# ------------------------------------------------------------------ E4 / E5 declared families
def _declared_arms(proto):
    """Small pre-declared families. These store row-level predictions so paired intervals are
    meaningful; unlike the lattice, the family size here is five, not sixty-four."""
    return [("reference", {}),
            ("featureset::legacy_A9", {"features": FEATURES_A9}),
            ("featureset::legacy_B5", {"features": FEATURES_B5}),
            ("control::matched_random", {"matched": True}),
            ("control::oracle_heads", {"oracle": True}),
            ("control::oracle_heads_beta0", {"oracle": True, "beta": 0.0})]


def declared(proto):
    core.log(f"===== E4/E5: declared contrasts, Protocol {proto} =====")
    TRAIN, TEST, ALL, _v = V.load(with_test_labels=True)
    arms = _declared_arms(proto)
    n = len(TEST) if proto == "B" else len(ALL)
    S = len(SEEDS)
    acc = {a: {"prob": np.zeros((n, S)), "ms": np.zeros((n, S)),
               "ms_det": np.zeros((n, S)), "thr": np.zeros((n, S))} for a, _ in arms}
    y_ref = g_ref = ek_ref = None

    for si, seed in enumerate(SEEDS):
        t0 = time.time()
        buf = {a: {k: [] for k in ("prob", "ms", "ms_det", "thr")} for a, _ in arms}
        ys, gs, eks = [], [], []
        for fit, val, evalf in _frames(proto, seed, TRAIN, TEST, ALL):
            base = route_and_score(fit, val, evalf, seed, proto)
            shares = np.bincount(base["sel"], minlength=len(POOL)) / len(base["sel"])
            for name, spec in arms:
                if name == "reference":
                    r = base
                else:
                    r = route_and_score(fit, val, evalf, seed, proto,
                                        features=spec.get("features"),
                                        oracle=spec.get("oracle", False),
                                        beta=spec.get("beta"),
                                        shares=shares if spec.get("matched") else None)
                m = len(evalf)
                buf[name]["prob"].append(np.asarray(r["prob"], float))
                buf[name]["ms"].append(np.broadcast_to(r["ms"], (m,)).astype(float))
                buf[name]["ms_det"].append(np.broadcast_to(r["ms_det"], (m,)).astype(float))
                buf[name]["thr"].append(np.full(m, r["thr"], float))
            ys.append(evalf["label_supported"].to_numpy(int))
            gs.append(evalf["content_doc_key"].astype(str).to_numpy())
            eks.append(evalf["episode_key"].astype(str).to_numpy())
        order = np.argsort(np.concatenate(eks), kind="stable")
        for name, _ in arms:
            for k in ("prob", "ms", "ms_det", "thr"):
                acc[name][k][:, si] = np.concatenate(buf[name][k])[order]
        y_now = np.concatenate(ys)[order]
        if y_ref is None:
            y_ref, g_ref, ek_ref = y_now, np.concatenate(gs)[order], np.concatenate(eks)[order]
        elif not np.array_equal(y_ref, y_now):
            raise AssertionError("canonical row order differs between seeds")
        core.log(f"  seed {seed}: reference "
                 f"{roc_auc_score(y_ref, acc['reference']['prob'][:, si]):.7f} @ "
                 f"{acc['reference']['ms'][:, si].mean():.4f} ms  ({time.time()-t0:.0f}s)")

    store = {"y": y_ref.astype(np.int8), "groups": np.asarray(g_ref, dtype=np.str_),
             "episode_key": np.asarray(ek_ref, dtype=np.str_),
             "arms": np.asarray([a for a, _ in arms], dtype=np.str_),
             "seeds": np.asarray(SEEDS)}
    for name, _ in arms:
        for k, arr in acc[name].items():
            store[f"{k}__{name.replace('::', '__')}"] = arr
    np.savez_compressed(RES / f"declared_{proto}.npz", **store)
    core.log(f"stored {len(arms)} declared arms x {S} seeds x {n} rows")


# ------------------------------------------------------------------ report
def report(proto):
    core.log(f"===== report {proto} =====")
    lat = pd.read_csv(RES / f"E2_LATTICE_{proto}.csv")
    ref = REFERENCE[proto]

    # gate: the lattice contains the reference twice
    gate = {}
    for kind, items, sep in (("feat", LATTICE_FEATURES, "|"), ("pool", POOL, "+")):
        row = lat[(lat.kind == kind) & (lat.subset == sep.join(items))]
        if not len(row):
            raise AssertionError(f"full {kind} subset missing from the lattice")
        got_a = float(row.auroc.iloc[0])
        got_m = float(row.ms_det.iloc[0])
        d_a, d_m = abs(got_a - ref["auroc"]), abs(got_m - ref["ms_det"])
        ok = d_a <= TOL_AUROC and d_m <= TOL_MS
        gate[kind] = {"auroc": got_a, "delta_auroc": d_a, "ms_det": got_m,
                      "delta_ms": d_m, "pass": bool(ok)}
        core.log(f"  gate {proto}/{kind}: auroc {got_a:.10f} vs {ref['auroc']:.10f} "
                 f"(d={d_a:.2e}) | det ms {got_m:.6f} vs {ref['ms_det']:.6f} "
                 f"(d={d_m:.2e}) -> {'PASS' if ok else 'FAIL'}")
    (RES / f"GATE_{proto}.json").write_text(json.dumps(
        {"protocol": proto, "part1c_auroc": ref["auroc"],
         "part1c_ms_deterministic": ref["ms_det"], "checks": gate,
         "note": "the lattice's full feature subset and full pool subset are both the reference "
                 "system; only the deterministic latency is gated because head, routing and "
                 "calibration timings are wall clock",
         "smoke": SMOKE}, indent=2))

    # declared families: paired intervals at 95% and Bonferroni over the family
    z = np.load(RES / f"declared_{proto}.npz", allow_pickle=False)
    arms = [str(a) for a in z["arms"]]
    y, g = z["y"].astype(int), z["groups"].astype(str)
    P = {a: z[f"prob__{a.replace('::', '__')}"] for a in arms}
    MS = {a: z[f"ms__{a.replace('::', '__')}"] for a in arms}
    THR = {a: z[f"thr__{a.replace('::', '__')}"] for a in arms}

    rows = []
    for a in arms:
        m = [core.metrics(y, P[a][:, s], MS[a][:, s], threshold=THR[a][:, s])
             for s in range(P[a].shape[1])]
        agg = {k: float(np.nanmean([r[k] for r in m])) for k in m[0]}
        agg["auroc_sd"] = float(np.std([r["auroc"] for r in m]))
        rows.append({"arm": a, **agg,
                     "d_auroc_vs_reference": agg["auroc"] - float(np.nanmean(
                         [core.metrics(y, P["reference"][:, s], MS["reference"][:, s],
                                       threshold=THR["reference"][:, s])["auroc"]
                          for s in range(P["reference"].shape[1])]))})
    T = pd.DataFrame(rows)
    front = ["arm", "auroc", "auroc_sd", "d_auroc_vs_reference", "ece", "brier", "aurc",
             "bacc", "mcc", "ms", "p95_ms"]
    V.save(f"E4E5_DECLARED_{proto}.csv", T[front + [c for c in T.columns if c not in front]])

    fam = [a for a in arms if a != "reference"]
    alpha = 0.05 / len(fam)
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for a in fam:
        pt, q = _paired(y, P["reference"], P[a], g, 17, [2.5, 97.5, lo_p, hi_p])
        rows.append({"arm": a, "d_auroc_reference_minus": pt,
                     "ci95_lo": q[2.5], "ci95_hi": q[97.5],
                     "significant_95": bool(q[2.5] > 0 or q[97.5] < 0),
                     "bonf_lo": q[lo_p], "bonf_hi": q[hi_p],
                     "significant_bonferroni": bool(q[lo_p] > 0 or q[hi_p] < 0),
                     "d_ms": float(MS["reference"].mean() - MS[a].mean())})
    Pd = pd.DataFrame(rows).sort_values("d_auroc_reference_minus", ascending=False)
    Pd["n_comparisons"] = len(fam)
    Pd["bonferroni_alpha"] = alpha
    V.save(f"E4E5_DECLARED_PAIRED_{proto}.csv", Pd)
    print(T[front].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(Pd.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    if not all(v["pass"] for v in gate.values()) and not SMOKE:
        raise AssertionError(f"attribution gate failed for Protocol {proto}; see GATE_{proto}.json")


def _paired(y, A, B, groups, seed, pcts):
    rng = np.random.default_rng(core.base.stable_seed(seed, "paired_group", "boot"))
    uniq = np.unique(groups)
    by = {gg: np.flatnonzero(groups == gg) for gg in uniq}

    def delta(idx):
        yy = y[idx]
        return float(np.mean([roc_auc_score(yy, A[idx, s]) - roc_auc_score(yy, B[idx, s])
                              for s in range(A.shape[1])]))

    point = delta(np.arange(len(y)))
    vals = []
    for _ in range(DRAWS):
        idx = np.concatenate([by[gg] for gg in rng.choice(uniq, len(uniq), replace=True)])
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(delta(idx))
    return point, {p: float(np.percentile(vals, p)) for p in pcts}


# ------------------------------------------------------------------ entry point
STAGES = {
    "prep": prep,
    "percorpus": percorpus,
    "latticeB": lambda: lattice("B"),
    "latticeA": lambda: lattice("A"),
    "convergeB": lambda: converge("B"),
    "convergeA": lambda: converge("A"),
    "declaredB": lambda: declared("B"),
    "declaredA": lambda: declared("A"),
    "reportB": lambda: report("B"),
    "reportA": lambda: report("A"),
}
PHASE1 = ["prep", "percorpus", "latticeB", "convergeB", "declaredB", "reportB"]
PHASE2 = ["latticeA", "convergeA", "declaredA", "reportA"]

if __name__ == "__main__":
    want = sys.argv[1:] or ["phase1"]
    if want == ["phase1"]:
        order = PHASE1
    elif want == ["phase2"]:
        order = PHASE2
    elif want == ["all"]:
        order = PHASE1 + PHASE2
    else:
        order = want
    for name in order:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name}")
        t0 = time.time()
        STAGES[name]()
        core.log(f"===== {name} done in {time.time()-t0:.0f}s =====")
    (RUN / "04_provenance").mkdir(parents=True, exist_ok=True)
    (RUN / "04_provenance" / f"DONE_{'_'.join(order)[:60]}.json").write_text(json.dumps(
        {"status": "complete", "stages": order, "smoke": SMOKE,
         "finished": time.strftime("%Y-%m-%d %H:%M:%S%z")}, indent=2))
