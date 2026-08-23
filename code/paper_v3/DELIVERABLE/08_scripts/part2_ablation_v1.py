#!/usr/bin/env python3
"""Part 2: formal ablation of the part1c system, both protocols.

Everything is inherited from part1c and asserted: pool, the six features, regret target, random
forest, the selected hyperparameters, the beta grid and tolerance, the ten seeds, and the split
construction. Each arm changes exactly one thing, and `OURS` here must reproduce part1c exactly,
which is the gate.

Latency is end-to-end for every arm, on the same basis as part1c: both feature extractors,
head inference, the routing arithmetic, both calibration stages, and the verifier call. Arms
that use fewer features still pay the full extraction cost, because the extractors are not
per-feature; arms that call no verifier pay no verifier cost.

Arms, grouped by what they remove:

  cost        no_cost                     beta pinned to 0
  calibration no_stage1                   no per-verifier Platt; route on raw scores
              no_stage2                   computed from part1c's stored stage-1 probabilities
  routing     random_routing_serving      head outputs replaced by noise at serving only
              random_routing_refit        noise at validation too, so beta/isotonic/threshold
                                          are fitted on random routing
  verifiers   no_verifier                 cheap features predict the label; nothing is called
              drop_retrain::X             X removed, heads/beta/calibration refitted
              drop_serving::X             X masked at serving only; the heads still know it
  features    drop_feature::F             one feature removed, heads refitted, no substitute
                                          feature is searched for
              keep_base3_only             only the three base-extractor features
              keep_compact16_3_only       only the three Compact16 features
  target      target::T                   only the supervision target changes

Stages: protoB | protoA | report | all
"""
from __future__ import annotations

import hashlib
import itertools
import json
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
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

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
(RUN / "logs").mkdir(parents=True, exist_ok=True)
V.RES = RES
SMOKE = os.environ.get("V3_SMOKE", "0") == "1"

CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
HPSEL = json.loads((PART1C / "00_contract" / "HP_SELECTED.json").read_text())
POOL = list(CONTRACT["pool"])
FEATURES = list(CONTRACT["features"])
TARGET, LEARNER = CONTRACT["target"], CONTRACT["learner"]
HP = {"A": dict(HPSEL["hyperparameters_A"]), "B": dict(HPSEL["hyperparameters_B"])}
SEEDS = list(V.SEEDS[:2] if SMOKE else V.SEEDS)
DRAWS = 200 if SMOKE else int(V.C.BOOTSTRAP_DRAWS)
# Never reduced, not even for smoke: Protocol A's out-of-fold set is only complete when all ten
# rotations run, so a shortened smoke would exercise a code path the real run never takes.
N_ROT = len(V.rotations())

BASE3 = ["bm25_mean3", "entity_coverage", "year_count"]
COMPACT3 = ["structured_source_line_ratio", "entity_value_colocation", "conflicting_value_rate"]
TARGETS = ["prob_quality", "brier", "pairwise_rank", "is_best", "correctness"]
CONFORMAL_DELTA, PRIMARY_RULE = 0.10, "mcc"
RULES = ("fixed05", "mcc", "youden", "conformal")

# part1c's OURS, at full precision. The gate compares AUROC and the deterministic part of the
# latency. It cannot compare the end-to-end total, because head inference, the routing
# arithmetic and the calibration application are wall-clock measurements that differ by
# microseconds between runs; `ms_det` is the reproducible part, the selected verifier's recorded
# latency plus the recorded feature-extraction cost.
REFERENCE = {
    proto: {"auroc": float(r["auroc"]), "ms_det": float(r["ms_part1_basis"])}
    for proto in ("A", "B")
    for _i, r in pd.read_csv(PART1C / "01_main_tables" / "publication"
                             / f"{proto}_MAIN.csv").iterrows()
    if r["system"] == "OURS"
}
TOL_AUROC, TOL_MS = 1e-9, 1e-6


def arm_list():
    arms = [("OURS", {}), ("no_cost", {"beta": 0.0}), ("no_stage1", {"platt": False}),
            ("random_routing_serving", {"random": "serving"}),
            ("random_routing_refit", {"random": "refit"}),
            ("no_verifier", {"no_verifier": True})]
    for f in FEATURES:
        arms.append((f"drop_feature::{f}", {"features": [x for x in FEATURES if x != f]}))
    arms += [("keep_base3_only", {"features": list(BASE3)}),
             ("keep_compact16_3_only", {"features": list(COMPACT3)})]
    for a in POOL:
        arms.append((f"drop_retrain::{a}", {"actions": [x for x in POOL if x != a]}))
        arms.append((f"drop_serving::{a}", {"mask": [x == a for x in POOL]}))
    for t in TARGETS:
        arms.append((f"target::{t}", {"target": t}))
    return arms


ARMS = arm_list()


# ------------------------------------------------------------------ shared machinery
def _rule_fixed05(q, y, g):
    return 0.5


def _rule_mcc(q, y, g):
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def _rule_youden(q, y, g):
    if len(set(np.asarray(y, int).tolist())) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, q)
    return float(min(max(float(thr[int(np.argmax(tpr - fpr))]), 0.0), 1.0))


def _rule_conformal(q, y, g):
    tau, _n = core.group_conformal_tau(y, q, g, CONFORMAL_DELTA)
    return float(tau) if np.isfinite(tau) else 1.0


RULE_FN = {"fixed05": _rule_fixed05, "mcc": _rule_mcc,
           "youden": _rule_youden, "conformal": _rule_conformal}


def add_correctness(frame, actions):
    """`correctness` needs per-verifier hard-decision correctness, which the frames do not carry.
    Derived here from the decision column the loader already builds, so the target becomes
    runnable without touching any stored data."""
    y = frame["label_supported"].to_numpy(int)
    for a in actions:
        frame[f"correct__{a}"] = (frame[f"decision__{a}"].to_numpy(int) == y).astype(int)
    return frame


def pre_call_ms(frame):
    return (frame["feature_latency_ms"].to_numpy(float)
            + frame["compact16_feature_latency_ms"].to_numpy(float))


def stage2(p_val, y_val, g_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    t0 = time.perf_counter()
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    ms = (time.perf_counter() - t0) * 1000.0 / max(len(p_ev), 1)
    return q_ev, {r: float(RULE_FN[r](q_val, y_val, g_val)) for r in RULES}, float(ms)


def run_arm(name, spec, fit, val, evalf, seed, proto):
    """One arm on one (fit, val, eval) triple. Returns probabilities, latency and thresholds."""
    hp = dict(HP[proto])
    actions = list(spec.get("actions", POOL))
    feats = list(spec.get("features", FEATURES))
    target = spec.get("target", TARGET)
    platt_on = spec.get("platt", True)
    y_val = val["label_supported"].to_numpy(int)
    g_val = val["content_doc_key"].astype(str).to_numpy()
    pre = pre_call_ms(evalf)

    if spec.get("no_verifier"):
        # the cheap features predict the label directly; the frozen head hyperparameters are
        # reused so only the presence of a verifier changes
        t0 = time.perf_counter()
        m = RandomForestRegressor(n_jobs=8, random_state=seed, **hp)
        m.fit(fit[feats].to_numpy(float), fit["label_supported"].to_numpy(float))
        p_val = np.clip(m.predict(val[feats].to_numpy(float)), 1e-6, 1 - 1e-6)
        t1 = time.perf_counter()
        p_ev = np.clip(m.predict(evalf[feats].to_numpy(float)), 1e-6, 1 - 1e-6)
        head_ms = (time.perf_counter() - t1) * 1000.0 / len(evalf)
        q_ev, thr, s2_ms = stage2(p_val, y_val, g_val, p_ev)
        return {"prob": q_ev, "ms": pre + head_ms + s2_ms,
                "ms_det": pre, "thr": thr, "calls": 0.0, "sel": None, "beta": np.nan}

    if target == "correctness":
        fit, val = add_correctness(fit.copy(), actions), add_correctness(val.copy(), actions)

    if platt_on:
        cals = core.platt(fit, actions=actions)
        c_fit = core.apply_platt(fit, cals, actions=actions)
        c_val = core.apply_platt(val, cals, actions=actions)
        t0 = time.perf_counter()
        c_ev = core.apply_platt(evalf, cals, actions=actions)
        platt_ms = (time.perf_counter() - t0) * 1000.0 / len(evalf)
    else:
        raw = lambda f: {a: f[f"score__{a}"].fillna(0.5).to_numpy(float) for a in actions}
        c_fit, c_val, c_ev = raw(fit), raw(val), raw(evalf)
        platt_ms = 0.0

    h_val, h_ev = core.fit_heads(fit, [val, evalf], seed, c_fit, actions=actions,
                                 features=feats, hp=hp, target=target, learner=LEARNER)
    head_ms = float(h_ev.predict_ms)
    rnd = spec.get("random")
    if rnd:
        rng = np.random.default_rng(seed)
        h_ev = core.HeadPredictions(rng.random(np.asarray(h_ev).shape), head_ms)
        if rnd == "refit":
            h_val = core.HeadPredictions(rng.random(np.asarray(h_val).shape), 0.0)

    cvec = np.array([core.fold_costs(fit, actions=actions)[a] for a in actions], float)
    beta = spec.get("beta")
    if beta is None:
        beta, _ledger = V.choose_beta(val, h_val, c_val, actions, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, actions, cvec)
    mask = spec.get("mask")
    t0 = time.perf_counter()
    sel, p_ev, ms_route = V.route(evalf, h_ev, c_ev, beta, actions, cvec, mask=mask)
    route_ms = (time.perf_counter() - t0) * 1000.0 / len(evalf)
    ver_ms = ms_route - evalf["feature_latency_ms"].to_numpy(float)
    q_ev, thr, s2_ms = stage2(p_val, y_val, g_val, p_ev)
    return {"prob": q_ev, "ms": ver_ms + pre + head_ms + route_ms + platt_ms + s2_ms,
            "ms_det": ms_route, "thr": thr, "calls": 1.0,
            "sel": np.asarray(sel, int), "beta": float(beta)}


def _key(s):
    return s.replace("::", "__")


def _str(v):
    return np.asarray(v, dtype=np.str_)


# ------------------------------------------------------------------ protocols
def proto_b():
    core.log(f"===== Protocol B: {len(ARMS)} arms x {len(SEEDS)} seeds =====")
    TRAIN, TEST, _A, _v = V.load(with_test_labels=True)
    y = TEST["label_supported"].to_numpy(int)
    n, S = len(TEST), len(SEEDS)
    store = {"y": y.astype(np.int8), "arms": _str([a for a, _ in ARMS]),
             "groups": _str(TEST["content_doc_key"].astype(str).to_numpy()),
             "corpus": _str(TEST["dataset_key"].astype(str).to_numpy()),
             "episode_key": _str(TEST["episode_key"].astype(str).to_numpy())}
    acc = {a: {"prob": np.zeros((n, S)), "ms": np.zeros((n, S)),
               "ms_det": np.zeros((n, S)), "calls": np.zeros(S),
               **{f"thr_{r}": np.zeros((n, S)) for r in RULES}} for a, _ in ARMS}
    for si, seed in enumerate(SEEDS):
        held = V.stratified_group_split(TRAIN, seed)
        fit = TRAIN.loc[~held].reset_index(drop=True)
        val = TRAIN.loc[held].reset_index(drop=True)
        t0 = time.time()
        for name, spec in ARMS:
            r = run_arm(name, spec, fit, val, TEST, seed, "B")
            acc[name]["prob"][:, si] = r["prob"]
            acc[name]["ms"][:, si] = r["ms"]
            acc[name]["ms_det"][:, si] = r["ms_det"]
            acc[name]["calls"][si] = r["calls"]
            for rule in RULES:
                acc[name][f"thr_{rule}"][:, si] = r["thr"][rule]
        core.log(f"  seed {seed}: OURS {roc_auc_score(y, acc['OURS']['prob'][:, si]):.7f} "
                 f"@ {acc['OURS']['ms'][:, si].mean():.4f} ms  ({time.time()-t0:.0f}s)")
    for a, _ in ARMS:
        for k, arr in acc[a].items():
            store[f"{k}__{_key(a)}"] = arr
    np.savez_compressed(RES / "ablation_B.npz", **store)
    core.log("Protocol B stored")


def proto_a():
    core.log(f"===== Protocol A: {len(ARMS)} arms x {len(SEEDS)} seeds x {N_ROT} rotations =====")
    _t, _e, ALL, _v = V.load(with_test_labels=True)
    n, S = len(ALL), len(SEEDS)
    rots = V.rotations()[:N_ROT]
    slots = ["prob", "ms", "ms_det"] + [f"thr_{r}" for r in RULES]
    acc = {a: {k: np.zeros((n, S)) for k in slots} for a, _ in ARMS}
    calls = {a: np.zeros(S) for a, _ in ARMS}
    y_ref = g_ref = ds_ref = ek_ref = None
    for si, seed in enumerate(SEEDS):
        fold = V.folds_stratified(ALL, seed)
        buf = {a: {k: [] for k in slots} for a, _ in ARMS}
        ys, gs, dss, eks = [], [], [], []
        t0 = time.time()
        for tf, vf, trf in rots:
            fit = ALL.loc[np.isin(fold, trf)].reset_index(drop=True)
            val = ALL.loc[fold == vf].reset_index(drop=True)
            ev = ALL.loc[fold == tf].reset_index(drop=True)
            for name, spec in ARMS:
                r = run_arm(name, spec, fit, val, ev, seed, "A")
                buf[name]["prob"].append(r["prob"])
                buf[name]["ms"].append(np.broadcast_to(r["ms"], (len(ev),)).copy())
                buf[name]["ms_det"].append(np.broadcast_to(r["ms_det"], (len(ev),)).copy())
                for rule in RULES:
                    buf[name][f"thr_{rule}"].append(np.full(len(ev), r["thr"][rule], float))
                calls[name][si] = r["calls"]
            ys.append(ev["label_supported"].to_numpy(int))
            gs.append(ev["content_doc_key"].astype(str).to_numpy())
            dss.append(ev["dataset_key"].astype(str).to_numpy())
            eks.append(ev["episode_key"].astype(str).to_numpy())
        order = np.argsort(np.concatenate(eks), kind="stable")
        for a, _ in ARMS:
            for k in slots:
                acc[a][k][:, si] = np.concatenate(buf[a][k])[order]
        y_now, ek_now = np.concatenate(ys)[order], np.concatenate(eks)[order]
        if y_ref is None:
            y_ref, ek_ref = y_now, ek_now
            g_ref, ds_ref = np.concatenate(gs)[order], np.concatenate(dss)[order]
        elif not (np.array_equal(y_ref, y_now) and np.array_equal(ek_ref, ek_now)):
            raise AssertionError("canonical row order differs between seeds")
        core.log(f"  seed {seed}: OURS {roc_auc_score(y_ref, acc['OURS']['prob'][:, si]):.7f} "
                 f"@ {acc['OURS']['ms'][:, si].mean():.4f} ms  ({time.time()-t0:.0f}s)")
    store = {"y": y_ref.astype(np.int8), "arms": _str([a for a, _ in ARMS]),
             "groups": _str(g_ref), "corpus": _str(ds_ref), "episode_key": _str(ek_ref)}
    for a, _ in ARMS:
        for k, arr in acc[a].items():
            store[f"{k}__{_key(a)}"] = arr
        store[f"calls__{_key(a)}"] = calls[a]
    np.savez_compressed(RES / "ablation_A.npz", **store)
    core.log("Protocol A stored")


# ------------------------------------------------------------------ report
def _paired(y, A, B, groups, seed, pcts):
    """One cluster bootstrap pass, several percentile pairs read off it.

    Same estimator and same resampling seed as `core.paired_cluster_bootstrap`; written out here
    only so the nominal 95% interval and the family-wise-corrected interval come from a single
    pass instead of two.
    """
    rng = np.random.default_rng(core.base.stable_seed(seed, "paired_group", "boot"))
    uniq = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in uniq}

    def delta(idx):
        yy = y[idx]
        return float(np.mean([roc_auc_score(yy, A[idx, s]) - roc_auc_score(yy, B[idx, s])
                              for s in range(A.shape[1])]))

    point = delta(np.arange(len(y)))
    vals = []
    for _ in range(DRAWS):
        idx = np.concatenate([by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(delta(idx))
    return point, {p: float(np.percentile(vals, p)) for p in pcts}


def report(proto):
    core.log(f"===== report {proto} =====")
    z = np.load(RES / f"ablation_{proto}.npz", allow_pickle=False)
    arms = [str(a) for a in z["arms"]]
    y, g, ds = z["y"].astype(int), z["groups"].astype(str), z["corpus"].astype(str)

    # the free arm: no_stage2 comes straight from part1c's stored stage-1 probabilities
    zc = np.load(PART1C / "10_row_level" / f"probs_{proto}.npz", allow_pickle=False)
    assert np.array_equal(zc["episode_key"].astype(str), z["episode_key"].astype(str)), \
        "part1c row order does not match this run"
    S = z["prob__OURS"].shape[1]
    # part1c stores ten seed columns in the same seed order, so a shortened run compares against
    # the matching prefix rather than against a differently shaped matrix
    extra = {"no_stage2": (zc["prob1__OURS"][:, :S], zc["ms_full__OURS"][:, :S],
                           zc["ms_part1__OURS"][:, :S],
                           zc[f"thr1_{PRIMARY_RULE}__OURS"][:, :S])}

    def agg(P, MS, TH):
        rows = [core.metrics(y, P[:, s], MS[:, s], threshold=TH[:, s])
                for s in range(P.shape[1])]
        out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
        out["auroc_sd"] = float(np.std([r["auroc"] for r in rows]))
        return out

    series = {a: (z[f"prob__{_key(a)}"], z[f"ms__{_key(a)}"], z[f"ms_det__{_key(a)}"],
                  z[f"thr_{PRIMARY_RULE}__{_key(a)}"]) for a in arms}
    series.update(extra)
    base = agg(series["OURS"][0], series["OURS"][1], series["OURS"][3])

    rows = []
    for a, (P, MS, MSD, TH) in series.items():
        m = agg(P, MS, TH)
        calls = float(np.mean(z[f"calls__{_key(a)}"])) if f"calls__{_key(a)}" in z else 1.0
        rows.append({"arm": a, **m, "ms_deterministic": float(MSD.mean()),
                     "mean_calls": 0.0 if a == "no_verifier" else calls,
                     "d_auroc_vs_ours": m["auroc"] - base["auroc"],
                     "d_ms_vs_ours": m["ms"] - base["ms"],
                     "d_ece_vs_ours": m["ece"] - base["ece"]})
    T = pd.DataFrame(rows)
    front = ["arm", "auroc", "auroc_sd", "d_auroc_vs_ours", "ece", "d_ece_vs_ours", "brier",
             "aurc", "bacc", "mcc", "ms", "d_ms_vs_ours", "p95_ms", "ms_deterministic",
             "mean_calls", "auprc_unsup"]
    T = T[front + [c for c in T.columns if c not in front]]
    V.save(f"{proto}_ABLATION.csv", T.sort_values("d_auroc_vs_ours", ascending=False))

    ref = REFERENCE[proto]
    got = T[T.arm == "OURS"].iloc[0]
    d_a = abs(got["auroc"] - ref["auroc"])
    d_m = abs(got["ms_deterministic"] - ref["ms_det"])
    ok = d_a <= TOL_AUROC and d_m <= TOL_MS
    core.log(f"  gate {proto}: OURS auroc {got['auroc']:.10f} vs part1c {ref['auroc']:.10f} "
             f"(d={d_a:.2e}) | deterministic ms {got['ms_deterministic']:.6f} vs "
             f"{ref['ms_det']:.6f} (d={d_m:.2e}) -> {'PASS' if ok else 'FAIL'}")
    core.log(f"    end-to-end ms {got['ms']:.4f} (includes re-measured head, routing and "
             f"calibration overhead, which is wall clock and not expected to be bit-identical)")
    (RES / f"GATE_{proto}.json").write_text(json.dumps(
        {"protocol": proto, "observed_auroc": float(got["auroc"]),
         "part1c_auroc": ref["auroc"], "delta_auroc": d_a,
         "observed_ms_deterministic": float(got["ms_deterministic"]),
         "part1c_ms_deterministic": ref["ms_det"], "delta_ms": d_m,
         "observed_ms_end_to_end": float(got["ms"]),
         "note": "the end-to-end total is not gated: head inference, routing arithmetic and "
                 "calibration application are wall-clock measurements",
         "pass": bool(ok), "smoke": SMOKE}, indent=2))

    A = series["OURS"][0]
    m_tests = len(series) - 1
    alpha = 0.05 / m_tests
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rows = []
    for a, (P, _MS, _MSD, _TH) in series.items():
        if a == "OURS":
            continue
        pt, q = _paired(y, A, P, g, 17, [2.5, 97.5, lo_p, hi_p])
        rows.append({"arm": a, "d_auroc_ours_minus": pt,
                     "ci95_lo": q[2.5], "ci95_hi": q[97.5],
                     "significant_95": bool(q[2.5] > 0 or q[97.5] < 0),
                     "bonf_lo": q[lo_p], "bonf_hi": q[hi_p],
                     "significant_bonferroni": bool(q[lo_p] > 0 or q[hi_p] < 0)})
    Pd = pd.DataFrame(rows).sort_values("d_auroc_ours_minus", ascending=False)
    Pd["n_comparisons"] = m_tests
    Pd["bonferroni_alpha"] = alpha
    V.save(f"{proto}_ABLATION_PAIRED.csv", Pd)
    core.log(f"  separable at 95%: {int(Pd.significant_95.sum())}/{m_tests} | "
             f"under Bonferroni (alpha={alpha:.5f}): "
             f"{int(Pd.significant_bonferroni.sum())}/{m_tests}")

    print(T.sort_values("d_auroc_vs_ours", ascending=False)[
        ["arm", "auroc", "d_auroc_vs_ours", "ece", "ms", "mean_calls"]]
        .to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if not ok and not SMOKE:
        raise AssertionError(f"gate failed for Protocol {proto}: OURS did not reproduce part1c")


if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    order = ["protoB", "protoA", "report"] if want == ["all"] else want
    prov = RUN / "04_provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "RUN_METADATA.txt").write_text(
        f"command={' '.join(sys.argv)}\nstarted={time.strftime('%Y-%m-%d %H:%M:%S%z')}\n"
        f"python={sys.version}\nplatform={platform.platform()}\nnproc={os.cpu_count()}\n"
        f"arms={len(ARMS)}\nseeds={SEEDS}\nsmoke={SMOKE}\n")
    (prov / "PID").write_text(f"{os.getpid()}\n")
    for name in order:
        t0 = time.time()
        if name == "protoB":
            proto_b()
        elif name == "protoA":
            proto_a()
        elif name == "report":
            report("B")
            report("A")
        else:
            raise SystemExit(f"unknown stage {name}")
        core.log(f"===== {name} done in {time.time()-t0:.0f}s =====")
    (prov / "DONE.json").write_text(json.dumps(
        {"status": "complete", "stages": order, "arms": len(ARMS), "smoke": SMOKE,
         "finished": time.strftime("%Y-%m-%d %H:%M:%S%z")}, indent=2))
