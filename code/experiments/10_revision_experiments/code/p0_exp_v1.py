#!/usr/bin/env python3
"""Reviewer-driven experiments on the frozen score matrix. No verifier is called.

Subcommands
  poolsize  how routing quality and cost move with the number of routable actions, k = 1..15
  rules     the selection rule itself: multiplicative vs additive cost, a cost-sensitive
            learning-to-defer baseline, and a predicted-label routing control
  mixture   sensitivity of the pooled result to the corpus mixture weights

Every arm reuses the frozen primitives (Platt on fit, regret target, RF heads, the
"cheapest within eps of best validation AUROC" rule, isotonic on validation) so that only the
named thing differs from the paper's configuration.
"""
from __future__ import annotations
import itertools, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as V           # noqa: E402
import core                  # noqa: E402

ART = HERE / "artifacts"
CONTRACT = json.loads((ART / "part1_main_pooled_v1" / "00_contract" / "FROZEN_v3.json").read_text())
POOL = list(CONTRACT["pool"])
FEATURES = list(CONTRACT["features"])
HP_SEL = json.loads((ART / "part1c_main_full_v1" / "00_contract" / "HP_SELECTED.json").read_text())
HP_B = dict(HP_SEL["hyperparameters_B"])   # the values part1c selected on TRAIN, not part1 legacy
GRID = tuple(CONTRACT["beta_grid"])
EPS = 0.005
SEEDS = list(V.SEEDS)
OUT = HERE / "results_p0exp"; OUT.mkdir(exist_ok=True)
CORPORA = ("cogensumm", "frank", "ragtruth", "unisumeval")


def fam(frame):
    return np.array([str(c).split("_")[0] for c in frame["dataset_key"].astype(str)])


def au(y, p):
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def stage2(p_val, y_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    return np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)


def pre_ms(frame):
    return (frame["feature_latency_ms"].to_numpy(float)
            + frame["compact16_feature_latency_ms"].to_numpy(float))


def select_from_grid(scored):
    """scored: list of (value, val_auroc, val_ms). The frozen rule, reused verbatim."""
    best = max(r[1] for r in scored)
    return min([r for r in scored if r[1] >= best - EPS], key=lambda r: r[2])[0]


# ------------------------------------------------------------------ one fitted router
def fit_router(fit, val, ev, seed, actions, *, rule="mult", target="regret", hp=None):
    """Returns (prob_ev_stage2, sel_ev, latency_ev, chosen_multiplier)."""
    hp = hp or HP_B
    cals = core.platt(fit, actions=actions)
    c_fit = core.apply_platt(fit, cals, actions=actions)
    c_val = core.apply_platt(val, cals, actions=actions)
    c_ev = core.apply_platt(ev, cals, actions=actions)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=actions,
                                 features=FEATURES, hp=hp, target=target, learner="rf")
    cost = core.fold_costs(fit, actions=actions)
    cv = np.array([cost[a] for a in actions], float)
    cn = cv / cv.max()

    def route(frame, heads, cal, mult):
        avail = np.column_stack([frame[f"available__{a}"].astype(bool).to_numpy() for a in actions])
        H = np.asarray(heads, float)
        util = H * np.exp(-mult * cn) if rule == "mult" else H - mult * cn
        sel = np.argmax(np.where(avail, util, -np.inf), axis=1)
        rows = np.arange(len(frame))
        prob = np.column_stack([cal[a] for a in actions])[rows, sel]
        lat = np.column_stack([frame[f"latency_ms__{a}"].fillna(0).to_numpy(float)
                               for a in actions])[rows, sel]
        return sel, prob, lat + pre_ms(frame)

    y_val = val["label_supported"].to_numpy(int)
    scored = []
    for m in GRID:
        _s, p, ms = route(val, h_val, c_val, m)
        scored.append((m, au(y_val, p), float(ms.mean())))
    mult = select_from_grid(scored)
    _sv, p_val, _mv = route(val, h_val, c_val, mult)
    sel, p_ev, ms_ev = route(ev, h_ev, c_ev, mult)
    return stage2(p_val, y_val, p_ev), sel, ms_ev, mult


def protob_splits():
    TRAIN, TEST, _ALL, verifiers = V.load(with_test_labels=True)
    for seed in SEEDS:
        held = V.stratified_group_split(TRAIN, seed)
        yield seed, TRAIN.loc[~held].reset_index(drop=True), TRAIN.loc[held].reset_index(drop=True), TEST
    return


def score(ev, P):
    """P is (rows, seeds). Returns pooled, per-corpus and macro AUROC."""
    y, f = ev["label_supported"].to_numpy(int), fam(ev)
    pooled = float(np.nanmean([au(y, P[:, s]) for s in range(P.shape[1])]))
    per = {c: float(np.nanmean([au(y[f == c], P[f == c, s]) for s in range(P.shape[1])]))
           for c in CORPORA if (f == c).any()}
    return pooled, per, float(np.nanmean(list(per.values())))


def run_arm(actions, *, rule="mult", target="regret", label=""):
    rows_P, rows_MS, mults = [], [], []
    ev_ref = None
    for seed, fit, val, ev in protob_splits():
        p, _sel, ms, mult = fit_router(fit, val, ev, seed, actions, rule=rule, target=target)
        rows_P.append(p); rows_MS.append(ms); mults.append(mult); ev_ref = ev
    P = np.column_stack(rows_P)
    pooled, per, macro = score(ev_ref, P)
    return {"arm": label, "k": len(actions), "actions": "+".join(actions),
            "pooled": pooled, "macro": macro, **{c: per.get(c, np.nan) for c in CORPORA},
            "ms": float(np.mean([m.mean() for m in rows_MS])),
            "multiplier": float(np.mean(mults))}


# ------------------------------------------------------------------ E1 pool size
def cmd_poolsize():
    TRAIN, TEST, _A, verifiers = V.load(with_test_labels=True)
    cand = [v for v in verifiers]
    print(f"{len(cand)} candidate verifiers")
    # greedy forward selection, scored on TRAIN validation only, averaged over three seeds
    greedy_seeds = SEEDS[:3]
    chosen, path = [], []
    remaining = list(cand)
    while remaining:
        best, best_score = None, -np.inf
        for v in remaining:
            trial = chosen + [v]
            vals = []
            for seed in greedy_seeds:
                held = V.stratified_group_split(TRAIN, seed)
                fit = TRAIN.loc[~held].reset_index(drop=True)
                val = TRAIN.loc[held].reset_index(drop=True)
                p, _s, _m, _b = fit_router(fit, val, val, seed, trial)
                vals.append(au(val["label_supported"].to_numpy(int), p))
            s = float(np.mean(vals))
            if s > best_score:
                best, best_score = v, s
        chosen = chosen + [best]; remaining.remove(best)
        path.append({"k": len(chosen), "added": best, "train_val_auroc": best_score})
        print(f"  k={len(chosen):2d}  +{best:42s} val {best_score:.4f}", flush=True)
    pd.DataFrame(path).to_csv(OUT / "E1_greedy_path.csv", index=False)

    rows = []
    for k in (1, 2, 3, 5, 8, 11, 15):
        if k > len(chosen):
            continue
        rows.append(run_arm(chosen[:k], label=f"greedy k={k}"))
        print("  " + json.dumps({a: (round(b, 4) if isinstance(b, float) else b)
                                 for a, b in rows[-1].items()}), flush=True)
    rows.append(run_arm(POOL, label="frozen pool (paper)"))
    df = pd.DataFrame(rows); df.to_csv(OUT / "E1_poolsize.csv", index=False)
    print(df.to_string(index=False))


# ------------------------------------------------------------------ E2/E3/E4 selection rules
def cmd_rules():
    rows = [run_arm(POOL, rule="mult", target="regret", label="multiplicative exp (paper)"),
            run_arm(POOL, rule="add", target="regret", label="additive cost penalty"),
            run_arm(POOL, rule="add", target="is_best", label="cost-sensitive L2D (one-hot + additive)"),
            run_arm(POOL, rule="mult", target="is_best", label="one-hot + multiplicative")]

    # predicted-label routing: predict y from the same features, then use one fixed action
    # per predicted class, chosen on validation.
    from sklearn.ensemble import RandomForestClassifier
    P_cols, MS_cols, ev_ref = [], [], None
    for seed, fit, val, ev in protob_splits():
        cals = core.platt(fit, actions=POOL)
        c_val = core.apply_platt(val, cals, actions=POOL)
        c_ev = core.apply_platt(ev, cals, actions=POOL)
        clf = RandomForestClassifier(n_jobs=8, random_state=seed, **HP_B)
        clf.fit(fit[FEATURES].to_numpy(float), fit["label_supported"].to_numpy(int))
        yhat_val = clf.predict(val[FEATURES].to_numpy(float))
        yhat_ev = clf.predict(ev[FEATURES].to_numpy(float))
        yv = val["label_supported"].to_numpy(int)
        Qv = np.column_stack([c_val[a] for a in POOL])
        Qv = np.where(yv[:, None] > 0.5, Qv, 1 - Qv)
        best_for = {c: int(np.argmax(Qv[yhat_val == c].mean(0))) if (yhat_val == c).any() else 0
                    for c in (0, 1)}
        sel = np.array([best_for[int(c)] for c in yhat_ev])
        rows_i = np.arange(len(ev))
        p_ev = np.column_stack([c_ev[a] for a in POOL])[rows_i, sel]
        selv = np.array([best_for[int(c)] for c in yhat_val])
        p_val = np.column_stack([c_val[a] for a in POOL])[np.arange(len(val)), selv]
        lat = np.column_stack([ev[f"latency_ms__{a}"].fillna(0).to_numpy(float)
                               for a in POOL])[rows_i, sel]
        P_cols.append(stage2(p_val, yv, p_ev)); MS_cols.append(lat + pre_ms(ev)); ev_ref = ev
    P = np.column_stack(P_cols)
    pooled, per, macro = score(ev_ref, P)
    rows.append({"arm": "predicted-label routing", "k": len(POOL), "actions": "+".join(POOL),
                 "pooled": pooled, "macro": macro, **{c: per.get(c, np.nan) for c in CORPORA},
                 "ms": float(np.mean([m.mean() for m in MS_cols])), "multiplier": np.nan})
    df = pd.DataFrame(rows); df.to_csv(OUT / "E2_rules.csv", index=False)
    print(df.to_string(index=False))


# ------------------------------------------------------------------ E7 mixture sensitivity
def cmd_mixture():
    d = np.load(ART / "part1c_main_full_v1" / "10_row_level" / "probs_B.npz", allow_pickle=True)
    y = d["y"]; f = np.array([str(c).split("_")[0] for c in d["corpus"]])
    names = {"EfficientEval": d["prob__OURS"]}
    for s in d["systems"]:
        if str(s).startswith("fixed::"):
            k = str(s).split("::", 1)[1]
            names[k] = d[f"prob__fixed__{k}"]
    ms = {"EfficientEval": d["ms_full__OURS"]}
    for s in d["systems"]:
        if str(s).startswith("fixed::"):
            k = str(s).split("::", 1)[1]
            ms[k] = d[f"ms_full__fixed__{k}"]

    idx = {c: np.flatnonzero(f == c) for c in CORPORA}
    base = np.array([len(idx[c]) for c in CORPORA], float); base /= base.sum()
    rng = np.random.default_rng(0)
    weights = {"natural": base, "equal": np.full(4, 0.25)}
    draws = [rng.dirichlet(np.ones(4)) for _ in range(200)]

    def resample(w, r):
        n = 3236
        take = []
        for c, wi in zip(CORPORA, w):
            pool = idx[c]
            take.append(r.choice(pool, size=max(int(round(n * wi)), 2), replace=True))
        return np.concatenate(take)

    rows = []
    for label, w in weights.items():
        r = np.random.default_rng(1)
        rec = {"mixture": label}
        for name, P in names.items():
            vals = []
            for _ in range(50):
                ii = resample(w, r)
                vals.append(np.nanmean([au(y[ii], P[ii, s]) for s in range(P.shape[1])]))
            rec[name] = float(np.mean(vals))
            rec[name + "__ms"] = float(ms[name].mean())
        rows.append(rec)

    # how often does EfficientEval stay non-dominated under a random mixture?
    r = np.random.default_rng(2)
    frontier_hits, wins = 0, {k: 0 for k in names}
    for w in draws:
        ii = resample(w, r)
        a = {name: float(np.nanmean([au(y[ii], P[ii, s]) for s in range(P.shape[1])]))
             for name, P in names.items()}
        ours_a, ours_ms = a["EfficientEval"], ms["EfficientEval"].mean()
        dominated = any(a[k] > ours_a and ms[k].mean() < ours_ms for k in names if k != "EfficientEval")
        frontier_hits += (not dominated)
        wins[max(a, key=a.get)] += 1
    rows.append({"mixture": f"dirichlet(1,1,1,1) x{len(draws)}: on frontier",
                 "EfficientEval": frontier_hits / len(draws)})
    rows.append({"mixture": "dirichlet: best-AUROC system counts",
                 **{k: v / len(draws) for k, v in wins.items() if v}})
    df = pd.DataFrame(rows); df.to_csv(OUT / "E7_mixture.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(df.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "rules"
    t0 = time.time()
    {"poolsize": cmd_poolsize, "rules": cmd_rules, "mixture": cmd_mixture}[cmd]()
    print(f"\n[{cmd}] {time.time() - t0:.0f}s -> {OUT}")
