#!/usr/bin/env python3
"""Protocol A replication of the reviewer-driven experiments.

Protocol B is the confirmatory protocol and carries these arms in p0_exp_v1.py / p0_cascade_v1.py.
Protocol A is the higher-powered rotation, so every conclusion drawn from B is repeated here.
One seed produces one full-length out-of-fold prediction vector, assembled from the ten 8/1/1
rotations, exactly as part1c does.

Subcommands: poolsize | rules | mixture | cascade
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as V           # noqa: E402
import core                  # noqa: E402

ART = HERE / "artifacts"
C0 = json.loads((ART / "part1_main_pooled_v1" / "00_contract" / "FROZEN_v3.json").read_text())
HPS = json.loads((ART / "part1c_main_full_v1" / "00_contract" / "HP_SELECTED.json").read_text())
POOL, FEATURES = list(C0["pool"]), list(C0["features"])
HP_A = dict(HPS["hyperparameters_A"])
GRID, SEEDS, EPS = tuple(C0["beta_grid"]), list(V.SEEDS), 0.005
CORPORA = ("cogensumm", "frank", "ragtruth", "unisumeval")
NICE = {"factcc": "FactCC", "lettuce_v2": "Lettuce-v2", "granite_guardian_3_1_2b": "Granite-3.1-2B"}
OUT = HERE / "results_p0exp"; OUT.mkdir(exist_ok=True)

ALL = None


def au(y, p):
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def stage2(p_val, y_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    return np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)


def pre_ms(f):
    return (f["feature_latency_ms"].to_numpy(float) + f["compact16_feature_latency_ms"].to_numpy(float))


def lat_matrix(f, actions):
    return np.column_stack([f[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in actions])


def load_all():
    global ALL
    if ALL is None:
        _tr, _te, ALL_, _v = V.load(with_test_labels=True)
        ALL = ALL_.reset_index(drop=True)
    return ALL


def rotations_for(seed):
    """Yields (fit, val, ev, ev_positions) for the ten 8/1/1 rotations of one seed."""
    frame = load_all()
    fold = V.folds_stratified(frame, seed)
    for test_fold, val_fold, train_folds in V.rotations():
        m_fit = np.isin(fold, train_folds)
        m_val = fold == val_fold
        m_ev = fold == test_fold
        yield (frame.loc[m_fit].reset_index(drop=True),
               frame.loc[m_val].reset_index(drop=True),
               frame.loc[m_ev].reset_index(drop=True),
               np.flatnonzero(m_ev))


# ------------------------------------------------------------------ one fitted router
def route_arm(fit, val, ev, seed, actions, *, rule="mult", target="regret"):
    cals = core.platt(fit, actions=actions)
    c_fit = core.apply_platt(fit, cals, actions=actions)
    c_val = core.apply_platt(val, cals, actions=actions)
    c_ev = core.apply_platt(ev, cals, actions=actions)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=actions,
                                 features=FEATURES, hp=HP_A, target=target, learner="rf")
    cost = core.fold_costs(fit, actions=actions)
    cn = np.array([cost[a] for a in actions], float)
    cn = cn / cn.max()

    def go(frame, heads, cal, mult):
        avail = np.column_stack([frame[f"available__{a}"].astype(bool).to_numpy() for a in actions])
        H = np.asarray(heads, float)
        util = H * np.exp(-mult * cn) if rule == "mult" else H - mult * cn
        sel = np.argmax(np.where(avail, util, -np.inf), axis=1)
        r = np.arange(len(frame))
        return (sel, np.column_stack([cal[a] for a in actions])[r, sel],
                lat_matrix(frame, actions)[r, sel] + pre_ms(frame))

    yv = val["label_supported"].to_numpy(int)
    scored = [(m,) + (lambda t: (au(yv, t[1]), float(t[2].mean())))(go(val, h_val, c_val, m))
              for m in GRID]
    best = max(r[1] for r in scored)
    mult = min([r for r in scored if r[1] >= best - EPS], key=lambda r: r[2])[0]
    _s, p_val, _m = go(val, h_val, c_val, mult)
    sel, p_ev, ms = go(ev, h_ev, c_ev, mult)
    return stage2(p_val, yv, p_ev), ms, mult


def assemble(fn):
    """fn(fit,val,ev,seed) -> (prob_ev, ms_ev, extra). Returns (P, MS, extras) over all seeds."""
    frame = load_all()
    n = len(frame)
    P = np.zeros((n, len(SEEDS)))
    MS = np.zeros((n, len(SEEDS)))
    extras = []
    for si, seed in enumerate(SEEDS):
        for fit, val, ev, pos in rotations_for(seed):
            q, ms, extra = fn(fit, val, ev, seed)
            P[pos, si] = q
            MS[pos, si] = ms
            extras.append(extra)
    return P, MS, extras


def score(P):
    frame = load_all()
    y = frame["label_supported"].to_numpy(int)
    f = np.array([str(c).split("_")[0] for c in frame["dataset_key"].astype(str)])
    pooled = float(np.nanmean([au(y, P[:, s]) for s in range(P.shape[1])]))
    per = {c: float(np.nanmean([au(y[f == c], P[f == c, s]) for s in range(P.shape[1])]))
           for c in CORPORA if (f == c).any()}
    return pooled, per, float(np.nanmean(list(per.values())))


def run_arm(actions, *, rule="mult", target="regret", label=""):
    P, MS, ex = assemble(lambda fit, val, ev, s: route_arm(fit, val, ev, s, actions,
                                                           rule=rule, target=target))
    pooled, per, macro = score(P)
    return {"arm": label, "k": len(actions), "actions": "+".join(actions), "pooled": pooled,
            "macro": macro, **{c: per.get(c, np.nan) for c in CORPORA},
            "ms": float(MS.mean()), "multiplier": float(np.mean(ex))}


# ------------------------------------------------------------------ poolsize
def cmd_poolsize():
    path = pd.read_csv(OUT / "E1_greedy_path.csv")   # TRAIN-only path, reused verbatim
    chosen = list(path["added"])
    rows = []
    for k in (1, 2, 3, 5, 8, 11, 15):
        rows.append(run_arm(chosen[:k], label=f"greedy k={k}"))
        print("  " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b)
                                 for a, b in rows[-1].items()}), flush=True)
    rows.append(run_arm(POOL, label="frozen pool (paper)"))
    df = pd.DataFrame(rows); df.to_csv(OUT / "E1_poolsize_A.csv", index=False)
    print(df[["arm", "k", "pooled", "macro", "ms", "multiplier"]].to_string(index=False))


# ------------------------------------------------------------------ rules
def cmd_rules():
    rows = [run_arm(POOL, rule="mult", target="regret", label="multiplicative exp (paper)"),
            run_arm(POOL, rule="add", target="regret", label="additive cost penalty"),
            run_arm(POOL, rule="add", target="is_best", label="cost-sensitive L2D (one-hot + additive)"),
            run_arm(POOL, rule="mult", target="is_best", label="one-hot + multiplicative")]

    def pred_label(fit, val, ev, seed):
        cals = core.platt(fit, actions=POOL)
        c_val = core.apply_platt(val, cals, actions=POOL)
        c_ev = core.apply_platt(ev, cals, actions=POOL)
        clf = RandomForestClassifier(n_jobs=8, random_state=seed, **HP_A)
        clf.fit(fit[FEATURES].to_numpy(float), fit["label_supported"].to_numpy(int))
        yh_v = clf.predict(val[FEATURES].to_numpy(float))
        yh_e = clf.predict(ev[FEATURES].to_numpy(float))
        yv = val["label_supported"].to_numpy(int)
        Qv = np.column_stack([c_val[a] for a in POOL])
        Qv = np.where(yv[:, None] > 0.5, Qv, 1 - Qv)
        best = {c: int(np.argmax(Qv[yh_v == c].mean(0))) if (yh_v == c).any() else 0 for c in (0, 1)}
        sv = np.array([best[int(c)] for c in yh_v]); se = np.array([best[int(c)] for c in yh_e])
        p_val = np.column_stack([c_val[a] for a in POOL])[np.arange(len(val)), sv]
        p_ev = np.column_stack([c_ev[a] for a in POOL])[np.arange(len(ev)), se]
        ms = lat_matrix(ev, POOL)[np.arange(len(ev)), se] + pre_ms(ev)
        return stage2(p_val, yv, p_ev), ms, np.nan

    P, MS, _ = assemble(pred_label)
    pooled, per, macro = score(P)
    rows.append({"arm": "predicted-label routing", "k": len(POOL), "actions": "+".join(POOL),
                 "pooled": pooled, "macro": macro, **{c: per.get(c, np.nan) for c in CORPORA},
                 "ms": float(MS.mean()), "multiplier": np.nan})
    df = pd.DataFrame(rows); df.to_csv(OUT / "E2_rules_A.csv", index=False)
    print(df[["arm", "pooled", "macro", "ms", "multiplier"]].to_string(index=False))


# ------------------------------------------------------------------ mixture
def cmd_mixture():
    d = np.load(ART / "part1c_main_full_v1" / "10_row_level" / "probs_A.npz", allow_pickle=True)
    y = d["y"]; f = np.array([str(c).split("_")[0] for c in d["corpus"]])
    names, ms = {"EfficientEval": d["prob__OURS"]}, {"EfficientEval": d["ms_full__OURS"]}
    for s in d["systems"]:
        if str(s).startswith("fixed::"):
            k = str(s).split("::", 1)[1]
            names[k] = d[f"prob__fixed__{k}"]; ms[k] = d[f"ms_full__fixed__{k}"]
    idx = {c: np.flatnonzero(f == c) for c in CORPORA}
    base = np.array([len(idx[c]) for c in CORPORA], float); base /= base.sum()

    def resample(w, r, n=8512):
        return np.concatenate([r.choice(idx[c], size=max(int(round(n * wi)), 2), replace=True)
                               for c, wi in zip(CORPORA, w)])

    rows = []
    for label, w in (("natural", base), ("equal", np.full(4, 0.25))):
        r = np.random.default_rng(1); rec = {"mixture": label}
        for name, P in names.items():
            vals = [np.nanmean([au(y[ii], P[ii, s]) for s in range(P.shape[1])])
                    for ii in (resample(w, r) for _ in range(50))]
            rec[name] = float(np.mean(vals))
        rows.append(rec)
    rng = np.random.default_rng(0); r = np.random.default_rng(2)
    hits, wins = 0, {k: 0 for k in names}
    draws = [rng.dirichlet(np.ones(4)) for _ in range(200)]
    for w in draws:
        ii = resample(w, r)
        a = {k: float(np.nanmean([au(y[ii], P[ii, s]) for s in range(P.shape[1])]))
             for k, P in names.items()}
        om = ms["EfficientEval"].mean()
        hits += not any(a[k] > a["EfficientEval"] and ms[k].mean() < om
                        for k in names if k != "EfficientEval")
        wins[max(a, key=a.get)] += 1
    rows.append({"mixture": "dirichlet: on frontier", "EfficientEval": hits / len(draws)})
    rows.append({"mixture": "dirichlet: best-AUROC share",
                 **{k: v / len(draws) for k, v in wins.items() if v}})
    df = pd.DataFrame(rows); df.to_csv(OUT / "E7_mixture_A.csv", index=False)
    pd.set_option("display.width", 260, "display.max_columns", 40)
    print(df.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    t0 = time.time()
    {"poolsize": cmd_poolsize, "rules": cmd_rules, "mixture": cmd_mixture}[cmd]()
    print(f"\n[{cmd}-A] {time.time()-t0:.0f}s")
