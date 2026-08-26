#!/usr/bin/env python3
"""Part 4 — the competitors the ablation did not yet answer: cascades, second calls, learners.

Kept in its own run directory and its own archive. It shares nothing with Part 2 or Part 3 except
the reference system and the shared layer, and it writes nowhere they write.

Why these three families. The obvious question about a router that chooses before paying is why
not start with the cheapest verifier and escalate when unsure -- that is the cascade family, and
it is the comparison a reviewer reaches for first. The second question is whether one call is
actually the right budget or merely the convenient one -- that is the second-call family, which
keeps the router's choice and adds a conditional second verifier. The third asks whether any of
the conclusions depend on the forest: that is the learner sweep.

    reference                      the frozen system, which must reproduce part1c exactly
    cascade::confidence            run the cheapest verifier; escalate when it is unsure
    cascade::disagreement          run the two cheapest; escalate when they disagree
    cascade::learned_deferral      run the cheapest; escalate when a learned model predicts it errs
    second::verifier_confidence    route as usual, then add a call when the verifier is unsure
    second::raw_margin             add a call when the head margin is small
    second::discounted_margin      add a call when the cost-discounted margin is small
    learner::<name>                seven alternatives to the random forest

Every escalating arm pays for every call it makes: latency is the sum of the verifiers actually
invoked plus feature extraction, head inference, routing arithmetic and both calibration stages,
and `mean_calls` reports the realised average. Escalation thresholds are chosen on the validation
part under a latency ceiling of 1.5x the reference arm's own validation latency, so no arm buys
quality with unbounded budget.

Stages: protoB | protoA | reportB | reportA
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

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
TARGET = CONTRACT["target"]
_HP = json.loads((PART1C / "00_contract" / "HP_SELECTED.json").read_text())
HP = {"A": dict(_HP["hyperparameters_A"]), "B": dict(_HP["hyperparameters_B"])}
SEEDS = list(V.SEEDS[:2] if SMOKE else V.SEEDS)
DRAWS = 200 if SMOKE else int(V.C.BOOTSTRAP_DRAWS)
LATENCY_CEILING = 1.5

REFERENCE = {
    proto: {"auroc": float(r["auroc"]), "ms_det": float(r["ms_part1_basis"])}
    for proto in ("A", "B")
    for _i, r in pd.read_csv(PART1C / "01_main_tables" / "publication"
                             / f"{proto}_MAIN.csv").iterrows()
    if r["system"] == "OURS"
}
TOL_AUROC, TOL_MS = 1e-9, 1e-6

LEARNERS = {
    "extra_trees": lambda s, hp: ExtraTreesRegressor(n_jobs=8, random_state=s, **hp),
    "hgb": lambda s, hp: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                       max_leaf_nodes=31, l2_regularization=1.0,
                                                       early_stopping=False, random_state=s),
    "gbr": lambda s, hp: GradientBoostingRegressor(n_estimators=300, learning_rate=0.06,
                                                   max_depth=4, random_state=s),
    "ridge": lambda s, hp: Ridge(alpha=1.0),
    "tree": lambda s, hp: DecisionTreeRegressor(max_depth=10, min_samples_leaf=5,
                                                random_state=s),
    "knn": lambda s, hp: KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=8),
    "mlp": lambda s, hp: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400,
                                      early_stopping=True, random_state=s),
}

ARMS = (["reference"]
        + ["cascade::confidence", "cascade::disagreement", "cascade::learned_deferral"]
        + ["second::verifier_confidence", "second::raw_margin", "second::discounted_margin"]
        + [f"learner::{k}" for k in LEARNERS])


# ------------------------------------------------------------------ shared pieces
def _mcc_threshold(q, y):
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def _stage2(p_val, y_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    t0 = time.perf_counter()
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    return q_ev, _mcc_threshold(q_val, y_val), (time.perf_counter() - t0) * 1000.0 / max(len(p_ev), 1)


def _pre_ms(frame):
    return (frame["feature_latency_ms"].to_numpy(float)
            + frame["compact16_feature_latency_ms"].to_numpy(float))


def _lat_matrix(frame, actions):
    return np.column_stack([frame[f"latency_ms__{a}"].fillna(0).to_numpy(float)
                            for a in actions])


def _fit_heads(fit, seed, cal_fit, features, hp, learner="rf"):
    Y, _ = core.targets(fit, cal_fit, POOL, TARGET)
    xt = fit[features].to_numpy(float)
    models = []
    for j, a in enumerate(POOL):
        av = fit[f"available__{a}"].astype(bool).to_numpy()
        if learner == "rf":
            m = RandomForestRegressor(criterion=core.C.CRITERION, n_jobs=8,
                                      random_state=seed, **dict(hp))
        else:
            m = LEARNERS[learner](seed, dict(hp))
        m.fit(xt[av], np.asarray(Y[av, j], float))
        models.append(m)
    return models


def _heads(models, frame, features):
    x = frame[features].to_numpy(float)
    t0 = time.perf_counter()
    out = np.column_stack([np.clip(m.predict(x), 0.0, 1.0) for m in models])
    return out, (time.perf_counter() - t0) * 1000.0 / max(len(frame), 1)


class Fit:
    """One fitted stack: Platt, heads, cost vector, beta. Every arm is built on top of one."""

    def __init__(self, fit, val, evalf, seed, proto, learner="rf"):
        self.fit, self.val, self.ev, self.seed, self.proto = fit, val, evalf, seed, proto
        hp = HP[proto]
        self.cals = core.platt(fit, actions=POOL)
        self.c_fit = core.apply_platt(fit, self.cals, actions=POOL)
        self.c_val = core.apply_platt(val, self.cals, actions=POOL)
        t0 = time.perf_counter()
        self.c_ev = core.apply_platt(evalf, self.cals, actions=POOL)
        self.platt_ms = (time.perf_counter() - t0) * 1000.0 / max(len(evalf), 1)
        self.models = _fit_heads(fit, seed, self.c_fit, FEATURES, hp, learner)
        self.h_val, _ = _heads(self.models, val, FEATURES)
        self.h_ev, self.head_ms = _heads(self.models, evalf, FEATURES)
        self.cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
        self.beta = V.choose_beta(val, core.HeadPredictions(self.h_val), self.c_val,
                                  POOL, self.cvec)[0]
        self.y_val = val["label_supported"].to_numpy(int)
        self.pre_val, self.pre_ev = _pre_ms(val), _pre_ms(evalf)
        self.lat_val, self.lat_ev = _lat_matrix(val, POOL), _lat_matrix(evalf, POOL)
        # the reference routing, needed as the escalation baseline and the latency ceiling
        t0 = time.perf_counter()
        self.sel_ev, self.p_ev, _m = V.route(evalf, core.HeadPredictions(self.h_ev),
                                             self.c_ev, self.beta, POOL, self.cvec)
        self.route_ms = (time.perf_counter() - t0) * 1000.0 / max(len(evalf), 1)
        self.sel_val, self.p_val, _m = V.route(val, core.HeadPredictions(self.h_val),
                                               self.c_val, self.beta, POOL, self.cvec)
        self.ref_ms_val = float((self.lat_val[np.arange(len(val)), self.sel_val]
                                 + self.pre_val).mean())
        self.overhead = self.platt_ms + self.head_ms + self.route_ms

    def cheapest_order(self):
        return list(np.argsort(self.cvec))

    def finish(self, p_val_arm, p_ev_arm, ver_val, ver_ev, calls_ev):
        q, thr, s2 = _stage2(p_val_arm, self.y_val, p_ev_arm)
        return {"prob": q, "thr": thr,
                "ms": ver_ev + self.pre_ev + self.overhead + s2,
                "ms_det": ver_ev + self.ev["feature_latency_ms"].to_numpy(float),
                "calls": float(np.mean(calls_ev))}


def _pick_tau(conf_val, y_val, p_first_val, p_esc_val, ms_first, ms_esc, ceiling):
    """Highest validation AUROC among thresholds whose validation latency respects the ceiling."""
    grid = np.unique(np.quantile(conf_val, np.linspace(0.0, 0.6, 25)))
    best, best_au = 0.0, -np.inf
    for tau in grid:
        esc = conf_val < tau
        p = np.where(esc, p_esc_val, p_first_val)
        ms = float((ms_first + esc * ms_esc).mean())
        if ms > ceiling:
            continue
        au = float(roc_auc_score(y_val, p)) if len(set(y_val.tolist())) > 1 else 0.5
        if au > best_au:
            best, best_au = float(tau), au
    return best


# ------------------------------------------------------------------ arms
def arm_reference(F):
    rows = np.arange(len(F.ev))
    return F.finish(F.p_val, F.p_ev,
                    F.lat_val[np.arange(len(F.val)), F.sel_val],
                    F.lat_ev[rows, F.sel_ev], np.ones(len(F.ev)))


def arm_cascade_confidence(F):
    """Run the cheapest verifier; if it is not confident, call the highest-head-value alternative."""
    order = F.cheapest_order()
    first = order[0]
    nv, ne = len(F.val), len(F.ev)
    Cv = np.column_stack([F.c_val[a] for a in POOL])
    Ce = np.column_stack([F.c_ev[a] for a in POOL])
    # escalation target: the best remaining action by predicted quality, per row
    hv, he = F.h_val.copy(), F.h_ev.copy()
    hv[:, first] = -np.inf
    he[:, first] = -np.inf
    tv, te = hv.argmax(1), he.argmax(1)
    conf_v = np.abs(Cv[:, first] - 0.5)
    conf_e = np.abs(Ce[:, first] - 0.5)
    tau = _pick_tau(conf_v, F.y_val, Cv[:, first], Cv[np.arange(nv), tv],
                    F.lat_val[:, first], F.lat_val[np.arange(nv), tv],
                    LATENCY_CEILING * F.ref_ms_val - F.pre_val.mean())
    ev_v, ev_e = conf_v < tau, conf_e < tau
    p_val = np.where(ev_v, Cv[np.arange(nv), tv], Cv[:, first])
    p_ev = np.where(ev_e, Ce[np.arange(ne), te], Ce[:, first])
    ver_v = F.lat_val[:, first] + ev_v * F.lat_val[np.arange(nv), tv]
    ver_e = F.lat_ev[:, first] + ev_e * F.lat_ev[np.arange(ne), te]
    return F.finish(p_val, p_ev, ver_v, ver_e, 1.0 + ev_e)


def arm_cascade_disagreement(F):
    """Run the two cheapest; if their hard decisions disagree, add the most expensive."""
    order = F.cheapest_order()
    a, b, c = order[0], order[1], order[-1]
    nv, ne = len(F.val), len(F.ev)
    Cv = np.column_stack([F.c_val[x] for x in POOL])
    Ce = np.column_stack([F.c_ev[x] for x in POOL])
    dv = (Cv[:, a] >= 0.5) != (Cv[:, b] >= 0.5)
    de = (Ce[:, a] >= 0.5) != (Ce[:, b] >= 0.5)
    # when they agree, average the two cheap calibrated scores; when not, use the escalation
    p_val = np.where(dv, Cv[:, c], 0.5 * (Cv[:, a] + Cv[:, b]))
    p_ev = np.where(de, Ce[:, c], 0.5 * (Ce[:, a] + Ce[:, b]))
    ver_v = F.lat_val[:, a] + F.lat_val[:, b] + dv * F.lat_val[:, c]
    ver_e = F.lat_ev[:, a] + F.lat_ev[:, b] + de * F.lat_ev[:, c]
    return F.finish(p_val, p_ev, ver_v, ver_e, 2.0 + de)


def arm_cascade_learned_deferral(F):
    """Run the cheapest; escalate when a model trained on the features predicts it will err."""
    order = F.cheapest_order()
    first = order[0]
    nv, ne = len(F.val), len(F.ev)
    Cv = np.column_stack([F.c_val[x] for x in POOL])
    Ce = np.column_stack([F.c_ev[x] for x in POOL])
    Cf = np.column_stack([F.c_fit[x] for x in POOL])
    wrong = ((Cf[:, first] >= 0.5).astype(int)
             != F.fit["label_supported"].to_numpy(int)).astype(int)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=8,
                                 random_state=F.seed)
    if len(set(wrong.tolist())) < 2:
        return arm_reference(F)
    clf.fit(F.fit[FEATURES].to_numpy(float), wrong)
    rv = clf.predict_proba(F.val[FEATURES].to_numpy(float))[:, 1]
    re_ = clf.predict_proba(F.ev[FEATURES].to_numpy(float))[:, 1]
    hv, he = F.h_val.copy(), F.h_ev.copy()
    hv[:, first] = -np.inf
    he[:, first] = -np.inf
    tv, te = hv.argmax(1), he.argmax(1)
    # high predicted error means escalate, so the confidence signal is negated
    tau = _pick_tau(-rv, F.y_val, Cv[:, first], Cv[np.arange(nv), tv],
                    F.lat_val[:, first], F.lat_val[np.arange(nv), tv],
                    LATENCY_CEILING * F.ref_ms_val - F.pre_val.mean())
    ev_v, ev_e = (-rv) < tau, (-re_) < tau
    p_val = np.where(ev_v, Cv[np.arange(nv), tv], Cv[:, first])
    p_ev = np.where(ev_e, Ce[np.arange(ne), te], Ce[:, first])
    ver_v = F.lat_val[:, first] + ev_v * F.lat_val[np.arange(nv), tv]
    ver_e = F.lat_ev[:, first] + ev_e * F.lat_ev[np.arange(ne), te]
    return F.finish(p_val, p_ev, ver_v, ver_e, 1.0 + ev_e)


def _second_call(F, signal_val, signal_ev):
    """Keep the router's choice, then add the best remaining action when the signal is small."""
    nv, ne = len(F.val), len(F.ev)
    Cv = np.column_stack([F.c_val[x] for x in POOL])
    Ce = np.column_stack([F.c_ev[x] for x in POOL])
    hv, he = F.h_val.copy(), F.h_ev.copy()
    hv[np.arange(nv), F.sel_val] = -np.inf
    he[np.arange(ne), F.sel_ev] = -np.inf
    tv, te = hv.argmax(1), he.argmax(1)
    p1v, p1e = F.p_val, F.p_ev
    p2v, p2e = Cv[np.arange(nv), tv], Ce[np.arange(ne), te]
    l1v = F.lat_val[np.arange(nv), F.sel_val]
    l1e = F.lat_ev[np.arange(ne), F.sel_ev]
    tau = _pick_tau(signal_val, F.y_val, p1v, 0.5 * (p1v + p2v),
                    l1v, F.lat_val[np.arange(nv), tv],
                    LATENCY_CEILING * F.ref_ms_val - F.pre_val.mean())
    ev_v, ev_e = signal_val < tau, signal_ev < tau
    p_val = np.where(ev_v, 0.5 * (p1v + p2v), p1v)
    p_ev = np.where(ev_e, 0.5 * (p1e + p2e), p1e)
    ver_v = l1v + ev_v * F.lat_val[np.arange(nv), tv]
    ver_e = l1e + ev_e * F.lat_ev[np.arange(ne), te]
    return F.finish(p_val, p_ev, ver_v, ver_e, 1.0 + ev_e)


def arm_second_verifier_confidence(F):
    return _second_call(F, np.abs(F.p_val - 0.5), np.abs(F.p_ev - 0.5))


def _margin(h):
    s = np.sort(h, axis=1)
    return s[:, -1] - s[:, -2]


def arm_second_raw_margin(F):
    return _second_call(F, _margin(F.h_val), _margin(F.h_ev))


def arm_second_discounted_margin(F):
    d = np.exp(-F.beta * F.cvec / F.cvec.max())
    return _second_call(F, _margin(F.h_val * d), _margin(F.h_ev * d))


ARM_FN = {
    "reference": arm_reference,
    "cascade::confidence": arm_cascade_confidence,
    "cascade::disagreement": arm_cascade_disagreement,
    "cascade::learned_deferral": arm_cascade_learned_deferral,
    "second::verifier_confidence": arm_second_verifier_confidence,
    "second::raw_margin": arm_second_raw_margin,
    "second::discounted_margin": arm_second_discounted_margin,
}


def run_arm(name, fit, val, evalf, seed, proto, base):
    if name.startswith("learner::"):
        F = Fit(fit, val, evalf, seed, proto, learner=name.split("::", 1)[1])
        return arm_reference(F)
    return ARM_FN[name](base)


# ------------------------------------------------------------------ protocol drivers
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


def _key(n):
    return n.replace("::", "__")


def protocol(proto):
    core.log(f"===== Part 4, Protocol {proto}: {len(ARMS)} arms =====")
    TRAIN, TEST, ALL, _v = V.load(with_test_labels=True)
    n = len(TEST) if proto == "B" else len(ALL)
    S = len(SEEDS)
    slots = ("prob", "ms", "ms_det", "thr")
    acc = {a: {k: np.zeros((n, S)) for k in slots} for a in ARMS}
    calls = {a: np.zeros(S) for a in ARMS}
    y_ref = g_ref = ek_ref = None

    for si, seed in enumerate(SEEDS):
        t0 = time.time()
        buf = {a: {k: [] for k in slots} for a in ARMS}
        cbuf = {a: [] for a in ARMS}
        ys, gs, eks = [], [], []
        for fit, val, evalf in _frames(proto, seed, TRAIN, TEST, ALL):
            base = Fit(fit, val, evalf, seed, proto)
            m = len(evalf)
            for a in ARMS:
                r = run_arm(a, fit, val, evalf, seed, proto, base)
                buf[a]["prob"].append(np.asarray(r["prob"], float))
                buf[a]["ms"].append(np.broadcast_to(r["ms"], (m,)).astype(float))
                buf[a]["ms_det"].append(np.broadcast_to(r["ms_det"], (m,)).astype(float))
                buf[a]["thr"].append(np.full(m, r["thr"], float))
                cbuf[a].append(r["calls"] * m)
            ys.append(evalf["label_supported"].to_numpy(int))
            gs.append(evalf["content_doc_key"].astype(str).to_numpy())
            eks.append(evalf["episode_key"].astype(str).to_numpy())
        order = np.argsort(np.concatenate(eks), kind="stable")
        for a in ARMS:
            for k in slots:
                acc[a][k][:, si] = np.concatenate(buf[a][k])[order]
            calls[a][si] = float(np.sum(cbuf[a]) / n)
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
             "arms": np.asarray(ARMS, dtype=np.str_), "seeds": np.asarray(SEEDS)}
    for a in ARMS:
        for k in slots:
            store[f"{k}__{_key(a)}"] = acc[a][k]
        store[f"calls__{_key(a)}"] = calls[a]
    np.savez_compressed(RES / f"part4_{proto}.npz", **store)
    core.log(f"stored {len(ARMS)} arms x {S} seeds x {n} rows")


def _paired(y, A, B, groups, seed, pcts):
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
    z = np.load(RES / f"part4_{proto}.npz", allow_pickle=False)
    arms = [str(a) for a in z["arms"]]
    y, g = z["y"].astype(int), z["groups"].astype(str)
    P = {a: z[f"prob__{_key(a)}"] for a in arms}
    MS = {a: z[f"ms__{_key(a)}"] for a in arms}
    MSD = {a: z[f"ms_det__{_key(a)}"] for a in arms}
    TH = {a: z[f"thr__{_key(a)}"] for a in arms}
    CA = {a: z[f"calls__{_key(a)}"] for a in arms}

    def agg(a):
        m = [core.metrics(y, P[a][:, s], MS[a][:, s], threshold=TH[a][:, s])
             for s in range(P[a].shape[1])]
        out = {k: float(np.nanmean([r[k] for r in m])) for k in m[0]}
        out["auroc_sd"] = float(np.std([r["auroc"] for r in m]))
        out["mean_calls"] = float(np.mean(CA[a]))
        out["ms_deterministic"] = float(MSD[a].mean())
        return out

    base = agg("reference")
    rows = [{"arm": a, **agg(a)} for a in arms]
    T = pd.DataFrame(rows)
    T["d_auroc_vs_reference"] = T.auroc - base["auroc"]
    T["d_ms_vs_reference"] = T.ms - base["ms"]
    front = ["arm", "auroc", "auroc_sd", "d_auroc_vs_reference", "mean_calls", "ms",
             "d_ms_vs_reference", "ece", "brier", "aurc", "bacc", "mcc", "p95_ms"]
    V.save(f"PART4_{proto}.csv",
           T[front + [c for c in T.columns if c not in front]]
           .sort_values("d_auroc_vs_reference", ascending=False))

    ref = REFERENCE[proto]
    d_a = abs(base["auroc"] - ref["auroc"])
    d_m = abs(base["ms_deterministic"] - ref["ms_det"])
    ok = d_a <= TOL_AUROC and d_m <= TOL_MS
    core.log(f"  gate {proto}: auroc {base['auroc']:.10f} vs {ref['auroc']:.10f} (d={d_a:.2e}) | "
             f"det ms {base['ms_deterministic']:.6f} vs {ref['ms_det']:.6f} (d={d_m:.2e}) "
             f"-> {'PASS' if ok else 'FAIL'}")
    (RES / f"GATE_{proto}.json").write_text(json.dumps(
        {"protocol": proto, "observed_auroc": base["auroc"], "part1c_auroc": ref["auroc"],
         "delta_auroc": d_a, "observed_ms_deterministic": base["ms_deterministic"],
         "part1c_ms_deterministic": ref["ms_det"], "delta_ms": d_m,
         "pass": bool(ok), "smoke": SMOKE}, indent=2))

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
                     "d_ms": float(MS["reference"].mean() - MS[a].mean()),
                     "mean_calls": float(np.mean(CA[a]))})
    Pd = pd.DataFrame(rows).sort_values("d_auroc_reference_minus", ascending=False)
    Pd["n_comparisons"] = len(fam)
    Pd["bonferroni_alpha"] = alpha
    V.save(f"PART4_PAIRED_{proto}.csv", Pd)
    print(T[front].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(Pd.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if not ok and not SMOKE:
        raise AssertionError(f"gate failed for Protocol {proto}; see GATE_{proto}.json")


def _prov():
    p = RUN / "04_provenance"
    p.mkdir(parents=True, exist_ok=True)

    def sha(f):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    code = [Path(__file__).resolve(), ROOT / "experiments" / "v3core.py",
            ROOT / "shared" / "core.py"]
    (p / "CODE_SNAPSHOT.sha256").write_text(
        "".join(f"{sha(f)}  {f}\n" for f in code if f.exists()))
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception:
        head = "<unavailable>"
    (p / "RUN_METADATA.txt").write_text(
        f"command={' '.join(sys.argv)}\nstarted={time.strftime('%Y-%m-%d %H:%M:%S%z')}\n"
        f"git={head}\npython={sys.version}\nplatform={platform.platform()}\n"
        f"nproc={os.cpu_count()}\nsmoke={SMOKE}\nseeds={SEEDS}\narms={ARMS}\n"
        f"latency_ceiling={LATENCY_CEILING}\n")
    (p / "PID").write_text(f"{os.getpid()}\n")


STAGES = {"protoB": lambda: protocol("B"), "protoA": lambda: protocol("A"),
          "reportB": lambda: report("B"), "reportA": lambda: report("A")}

if __name__ == "__main__":
    want = sys.argv[1:] or ["protoB", "reportB", "protoA", "reportA"]
    _prov()
    for name in want:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name}")
        t0 = time.time()
        STAGES[name]()
        core.log(f"===== {name} done in {time.time()-t0:.0f}s =====")
    (RUN / "04_provenance" / f"DONE_{'_'.join(want)[:50]}.json").write_text(json.dumps(
        {"status": "complete", "stages": want, "smoke": SMOKE,
         "finished": time.strftime("%Y-%m-%d %H:%M:%S%z")}, indent=2))
