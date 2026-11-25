#!/usr/bin/env python3
"""Day-2 chain R1..R5, in the order the decisions actually depend on each other.

  R1  train ALL 455 three-verifier pools -- last night's ranking proxy (oracle headroom) turned
      out to anti-correlate with trained performance, so the pool question is genuinely open.
  R2  re-run the learners on whatever R1 picks.
  R3  sweep six supervision targets. Last night changing the target was worth +.0255, more than
      every learner swap combined, and only two targets were ever tried.
  R4  redo the feature analysis under the winning target, and ADDITIVELY (one family alone),
      because drop-one-out is insensitive when families are correlated.
  R5  final: four router paradigms, budget curve, 10 seeds, paired dataset-ID control.

Pooled 8/1/1 rotation throughout. Zero verifier inference. GPU only for tabular learners.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")

import sys, json, time, itertools, traceback, warnings
from pathlib import Path
ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)

from afr_v2 import tenfold_v1 as core
from afr_v2 import pool_gate_sweep_v1 as v1
from afr_v2 import pool_gate_sweep_v2 as v2
from afr_v2 import summary_router_compact16_direct_v1 as base
import night_run as N

OUT = ROOT / "results/day2_v1"
ST = OUT / "status"
for p in (OUT, ST):
    p.mkdir(parents=True, exist_ok=True)
T0 = time.time()
FEATS = list(base.FEATURE_COLUMNS)
SEEDS3 = tuple(core.SEEDS)
SEEDS10 = N.SEEDS10
SCREEN_ROTATIONS = 4          # enough to rank 455 pools; the finalists get all ten


def log(m): print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)
def done(s): return (ST / f"{s}.done").exists()
def mark(s, p=None): (ST / f"{s}.done").write_text(json.dumps(p or {"ok": True}, default=str))
def save(n, f): base.write_csv(OUT / n, f); return f


# --------------------------------------------------------------------- supervision targets
def build_targets(frame, actions, cal, kind):
    """Return (Y, is_classification). Y[:, j] is what head j should learn for action j."""
    y = frame["label_supported"].to_numpy(float)
    C = np.column_stack([np.asarray(cal[a], float) for a in actions])          # calibrated probs
    Q = np.where(y[:, None] > 0.5, C, 1.0 - C)                                 # prob of the truth
    k = len(actions)
    if kind == "correctness":                       # the original: did its hard decision match?
        return np.column_stack([frame[f"correct__{a}"].fillna(0).to_numpy(int) for a in actions]), True
    if kind == "prob_quality":                      # last night's winner
        return Q, False
    if kind == "brier":                             # same ordering, different scale
        return 1.0 - (C - y[:, None]) ** 2, False
    if kind == "pairwise_rank":                     # relative: how many actions does it beat here
        return (Q[:, :, None] > Q[:, None, :]).sum(2) / max(k - 1, 1), False
    if kind == "regret":                            # relative: how far below the best action
        return 1.0 + (Q - Q.max(1, keepdims=True)), False
    if kind == "is_best":                           # binary: is this the best action on this row
        return (Q >= Q.max(1, keepdims=True) - 1e-12).astype(int), True
    raise ValueError(kind)


def make_model(learner, seed, is_clf, hp=None):
    hp = hp or {}
    if learner == "rf":
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        return cls(n_estimators=hp.get("n_estimators", 400), min_samples_leaf=hp.get("min_samples_leaf", 5),
                   n_jobs=8, random_state=seed)
    cls = HistGradientBoostingClassifier if is_clf else HistGradientBoostingRegressor
    return cls(max_iter=hp.get("max_iter", 200), learning_rate=hp.get("learning_rate", 0.06),
               max_leaf_nodes=hp.get("max_leaf_nodes", 31), l2_regularization=hp.get("l2_regularization", 1.0),
               early_stopping=False, random_state=seed)


def fit_heads(train, out_frames, cols, actions, seed, target, cal_train, learner, hp=None):
    """Fit one head per action on `train`, predict on every frame in out_frames."""
    Y, is_clf = build_targets(train, actions, cal_train, target)
    xt = train[cols].to_numpy(float)
    xs = [f[cols].to_numpy(float) for f in out_frames]
    preds = [np.zeros((len(f), len(actions))) for f in out_frames]
    for j, a in enumerate(actions):
        av = train[f"available__{a}"].astype(bool).to_numpy()
        yv = Y[av, j]
        if is_clf and len(set(np.asarray(yv).astype(int).tolist())) < 2:
            for i in range(len(preds)):
                preds[i][:, j] = float(np.mean(yv)) if len(yv) else 0.5
            continue
        m = make_model(learner, seed, is_clf, hp)
        m.fit(xt[av], np.asarray(yv).astype(int) if is_clf else np.asarray(yv, float))
        for i in range(len(preds)):
            p = m.predict_proba(xs[i])[:, 1] if is_clf else np.clip(m.predict(xs[i]), 0.0, 1.0)
            preds[i][:, j] = p
    return preds


def pass_once(matrix, pool, seed, *, cols=None, target="correctness", learner="hgb", hp=None,
              n_rot=None, want_test=True, tag="day2"):
    """One rotation sweep. Returns (val_auroc_mean, pred_or_None, ctrls_or_None)."""
    cols = cols or FEATS
    actions = v1.order_by_cost(matrix, pool)
    costs = v1.pool_costs(matrix, pool)
    folds = core.build_folds(matrix, seed)
    rots = core.rotations()[:n_rot] if n_rot else core.rotations()
    vals, preds, banks = [], [], []
    for test_f, val_f, train_f in rots:
        train = matrix.loc[np.isin(folds, train_f)].reset_index(drop=True)
        val = matrix.loc[folds == val_f].reset_index(drop=True)
        test = matrix.loc[folds == test_f].reset_index(drop=True)
        cals = core.fit_calibrators_on_val(val, actions)
        c_tr = core.apply_calibrators(train, actions, cals)
        c_va = core.apply_calibrators(val, actions, cals)
        c_te = core.apply_calibrators(test, actions, cals)
        hv, ht = fit_heads(train, [val, test], cols, actions, seed, target, c_tr, learner, hp)
        beta = core.select_beta(val, actions, hv, c_va, costs)[0]
        vals.append(v1.scope_auroc(v1.router_prediction(val, actions, hv, c_va, costs, beta), "pooled"))
        if want_test:
            p = v1.router_prediction(test, actions, ht, c_te, costs, beta)
            preds.append(p)
            banks.append(v2.control_bank_single(test, actions, p, c_te, (seed, tag, str(test_f), f"b{beta}")))
    if not want_test:
        return float(np.mean(vals)), None, None
    pred = pd.concat(preds, ignore_index=True)
    ctrls = [pd.concat([b[j] for b in banks], ignore_index=True) for j in range(core.CONTROL_DRAWS)]
    return float(np.mean(vals)), pred, ctrls


def full_eval(matrix, pool, label, seeds=SEEDS3, **kw):
    rows = []
    for s in seeds:
        va, pred, ctrls = pass_once(matrix, pool, s, tag=label, **kw)
        r = N.summarize(pred, ctrls, s, label, f"day2|{label}|{s}")
        r["val_auroc"] = va
        r["pool"] = "+".join(v1.order_by_cost(matrix, pool))
        rows.append(r)
        log(f"    {label} seed {s}: val={va:.4f} test={r['treatment_auroc']:.4f} "
            f"d={r['delta_point']:+.4f} CI[{r['delta_ci_low']:+.4f},{r['delta_ci_high']:+.4f}] {r['ms']:.0f}ms")
    return rows


# ==================================================================== R1  all 455 triples
def stage_R1(matrix):
    VER = [v for v in v1.ALL_VERIFIERS if f"correct__{v}" in matrix.columns]
    triples = list(itertools.combinations(VER, 3))
    log(f"R1 screening {len(triples)} triples with {SCREEN_ROTATIONS} rotations, seed 17")
    rows = []
    for i, pool in enumerate(triples):
        try:
            va, _, _ = pass_once(matrix, pool, 17, n_rot=SCREEN_ROTATIONS, want_test=False, tag="r1")
            ms = float(np.mean([v1.pool_costs(matrix, pool)[a] for a in pool]))
            rows.append({"pool": "+".join(v1.order_by_cost(matrix, pool)), "screen_val_auroc": va,
                         "mean_member_cost_ms": ms})
        except Exception:
            rows.append({"pool": "+".join(pool), "screen_val_auroc": np.nan, "mean_member_cost_ms": np.nan})
        if (i + 1) % 50 == 0:
            log(f"  R1 {i+1}/{len(triples)} ({time.time()-T0:.0f}s) best so far "
                f"{max(r['screen_val_auroc'] for r in rows if r['screen_val_auroc']==r['screen_val_auroc']):.4f}")
            save("R1_SCREEN_ALL_455.csv", pd.DataFrame(rows))
    sc = save("R1_SCREEN_ALL_455.csv", pd.DataFrame(rows)).dropna(subset=["screen_val_auroc"])
    sc = sc.sort_values("screen_val_auroc", ascending=False).reset_index(drop=True)
    cur = "+".join(v1.order_by_cost(matrix, core.POOL))
    pos = sc.index[sc.pool == cur]
    log(f"R1 你手选的池子 {cur} 排名 {int(pos[0])+1 if len(pos) else -1} / {len(sc)}")
    log("R1 筛选 top-15:\n" + sc.head(15).round(4).to_string(index=False))

    finalists = sc.head(20).pool.tolist()
    if cur not in finalists:
        finalists.append(cur)
    log(f"R1 confirming {len(finalists)} finalists with 10 rotations x 3 seeds")
    fin = []
    for p in finalists:
        try:
            rs = full_eval(matrix, tuple(p.split("+")), f"R1:{p}", seeds=SEEDS3)
            for r in rs:
                r["is_current"] = (p == cur)
            fin += rs
        except Exception:
            log(f"  R1 finalist {p} FAILED")
    fr = save("R1_FINALISTS.csv", pd.DataFrame(fin))
    agg = fr.groupby(["pool", "is_current"], as_index=False).agg(
        val=("val_auroc", "mean"), test=("treatment_auroc", "mean"), delta=("delta_point", "mean"),
        ci_low=("delta_ci_low", "min"), ms=("ms", "mean")).sort_values("val", ascending=False)
    save("R1_FINALISTS_AGG.csv", agg)
    log("R1 决赛（按 val 排序，val 是唯一合法依据）:\n" + agg.round(4).to_string(index=False))
    champ = agg.iloc[0]
    (OUT / "CHAMPION_R1.json").write_text(json.dumps({
        "pool": champ.pool.split("+"), "val": float(champ.val), "test": float(champ.test),
        "ms": float(champ.ms), "current_pool": cur.split("+")}, indent=2))
    log(f"R1 冠军三元组 = {champ.pool}  val {champ.val:.4f} test {champ.test:.4f} @ {champ.ms:.0f}ms")


def pool_r1():
    p = OUT / "CHAMPION_R1.json"
    return tuple(json.loads(p.read_text())["pool"]) if p.exists() else tuple(core.POOL)


# ==================================================================== R2  learners on that pool
def stage_R2(matrix):
    pool = pool_r1()
    log(f"R2 on {pool}")
    F = {**N.learner_factories(), **N.torch_learners()}
    F.pop("tabpfn", None)                     # needs an interactive licence the user must accept
    rows = []
    for name, (factory, space) in F.items():
        try:
            t = time.time()
            best_hp, best_val = {}, -1.0
            if space:
                rng = np.random.default_rng(base.stable_seed(17, "day2", name))
                actions = v1.order_by_cost(matrix, pool)
                costs = v1.pool_costs(matrix, pool)
                folds = core.build_folds(matrix, 17)
                for _ in range(N.N_TRIALS):
                    hp = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
                    va = []
                    for test_f, val_f, train_f in core.rotations()[:4]:
                        train = matrix.loc[np.isin(folds, train_f)].reset_index(drop=True)
                        val = matrix.loc[folds == val_f].reset_index(drop=True)
                        cals = core.fit_calibrators_on_val(val, actions)
                        cv = core.apply_calibrators(val, actions, cals)
                        hv = N.heads(train, val, FEATS, actions, 17, factory, hp)
                        b = core.select_beta(val, actions, hv, cv, costs)[0]
                        va.append(v1.scope_auroc(v1.router_prediction(val, actions, hv, cv, costs, b), "pooled"))
                    if np.mean(va) > best_val:
                        best_val, best_hp = float(np.mean(va)), hp
            rs = N.evaluate(matrix, pool, name, seeds=SEEDS3, factory=factory, hp=best_hp)
            for r in rs:
                r["learner"] = name
                r["hp"] = json.dumps(best_hp, default=str)
                r["search_val_auroc"] = best_val
                r["fit_seconds"] = round(time.time() - t, 1)
            rows += rs
            save("R2_LEARNERS.csv", pd.DataFrame(rows))
        except Exception:
            log(f"  R2 learner {name} FAILED\n{traceback.format_exc()[:300]}")
    fr = pd.DataFrame(rows)
    agg = fr.groupby("learner", as_index=False).agg(
        val=("val_auroc", "mean"), test=("treatment_auroc", "mean"), delta=("delta_point", "mean"),
        ci_low=("delta_ci_low", "min"), ci_high=("delta_ci_high", "max"), ms=("ms", "mean"),
        fit_s=("fit_seconds", "mean")).sort_values("val", ascending=False)
    save("R2_LEARNERS_AGG.csv", agg)
    log("R2 学习器（新池子上）:\n" + agg.round(4).to_string(index=False))
    best = agg.iloc[0].learner
    (OUT / "CHAMPION_R2.json").write_text(json.dumps({"learner": best}, indent=2))
    log(f"R2 冠军学习器 = {best}")


def learner_r2():
    p = OUT / "CHAMPION_R2.json"
    if not p.exists():
        return "hgb"
    n = json.loads(p.read_text())["learner"]
    return n if n in ("hgb", "rf", "random_forest") else "hgb"


# ==================================================================== R3  six supervision targets
TARGETS = ("correctness", "prob_quality", "brier", "pairwise_rank", "regret", "is_best")


def stage_R3(matrix):
    pool = pool_r1()
    lr = "rf" if learner_r2() in ("rf", "random_forest") else "hgb"
    log(f"R3 on {pool} with learner={lr}")
    rows = []
    for tg in TARGETS:
        try:
            rows += [dict(r, target=tg) for r in
                     full_eval(matrix, pool, f"tgt_{tg}", seeds=SEEDS3, target=tg, learner=lr)]
            save("R3_TARGETS.csv", pd.DataFrame(rows))
        except Exception:
            log(f"  R3 target {tg} FAILED\n{traceback.format_exc()[:300]}")
    fr = pd.DataFrame(rows)
    agg = fr.groupby("target", as_index=False).agg(
        val=("val_auroc", "mean"), test=("treatment_auroc", "mean"), delta=("delta_point", "mean"),
        ci_low=("delta_ci_low", "min"), ms=("ms", "mean")).sort_values("val", ascending=False)
    save("R3_TARGETS_AGG.csv", agg)
    log("R3 监督目标:\n" + agg.round(4).to_string(index=False))
    (OUT / "CHAMPION_R3.json").write_text(json.dumps({"target": agg.iloc[0].target}, indent=2))
    log(f"R3 冠军监督目标 = {agg.iloc[0].target}")


def target_r3():
    p = OUT / "CHAMPION_R3.json"
    return json.loads(p.read_text())["target"] if p.exists() else "correctness"


# ==================================================================== R4  feature analysis, additive
def stage_R4(matrix):
    pool, lr, tg = pool_r1(), ("rf" if learner_r2() in ("rf", "random_forest") else "hgb"), target_r3()
    log(f"R4 on {pool} learner={lr} target={tg}")
    fam = {}
    for c in FEATS:
        fam.setdefault(c.split("_")[0], []).append(c)
    keys = sorted(matrix["dataset_key"].astype(str).unique())
    ONEHOT = []
    for k in keys:
        c = f"dsid__{k}"
        matrix[c] = matrix["dataset_key"].astype(str).eq(k).astype(float)
        ONEHOT.append(c)
    log(f"R4 families: { {k: len(v) for k, v in fam.items()} }")

    rows = []
    for label, cols, mode in ([("ALL", FEATS, "reference"), ("DATASET_ID_ONLY", ONEHOT, "reference")]
                              + [(f"only_{f}", c, "additive") for f, c in fam.items()]
                              + [(f"drop_{f}", [x for x in FEATS if x not in c], "subtractive")
                                 for f, c in fam.items()]):
        if not cols:
            continue
        try:
            rows += [dict(r, mode=mode, n_cols=len(cols)) for r in
                     full_eval(matrix, pool, label, seeds=SEEDS3, cols=cols, target=tg, learner=lr)]
            save("R4_FEATURES.csv", pd.DataFrame(rows))
        except Exception:
            log(f"  R4 {label} FAILED")
    fr = pd.DataFrame(rows)
    agg = fr.groupby(["label", "mode", "n_cols"], as_index=False).agg(
        val=("val_auroc", "mean"), test=("treatment_auroc", "mean"), delta=("delta_point", "mean"),
        ci_low=("delta_ci_low", "min"))
    save("R4_FEATURES_AGG.csv", agg)
    for m in ("reference", "additive", "subtractive"):
        sub = agg[agg["mode"] == m].sort_values("test", ascending=False)
        log(f"R4 [{m}]:\n" + sub.round(4).to_string(index=False))


# ==================================================================== R5  paradigms + final
def stage_R5(matrix):
    pool, tg = pool_r1(), target_r3()
    lr = "rf" if learner_r2() in ("rf", "random_forest") else "hgb"
    actions = v1.order_by_cost(matrix, pool)
    costs = v1.pool_costs(matrix, pool)
    cmax = max(costs.values())
    log(f"R5 final on {pool} learner={lr} target={tg}")
    rows = []

    # -- decomposed + exponential discount (the incumbent), on the full stack
    rows += [dict(r, paradigm="decomposed_exp") for r in
             full_eval(matrix, pool, "decomposed_exp", seeds=SEEDS3, target=tg, learner=lr)]

    # -- the other three paradigms
    WIDTHS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
    LAMS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
    for name in ("decomposed_linear", "cascade", "oracle_imitation"):
        for seed in SEEDS3:
            try:
                folds = core.build_folds(matrix, seed)
                preds, banks = [], []
                for test_f, val_f, train_f in core.rotations():
                    train = matrix.loc[np.isin(folds, train_f)].reset_index(drop=True)
                    val = matrix.loc[folds == val_f].reset_index(drop=True)
                    test = matrix.loc[folds == test_f].reset_index(drop=True)
                    cals = core.fit_calibrators_on_val(val, actions)
                    c_tr = core.apply_calibrators(train, actions, cals)
                    c_va = core.apply_calibrators(val, actions, cals)
                    c_te = core.apply_calibrators(test, actions, cals)
                    hv, ht = fit_heads(train, [val, test], FEATS, actions, seed, tg, c_tr, lr)

                    def bl(fr, cal, h): return v1.router_prediction(fr, actions, h, cal, costs, 0.0).copy()

                    def ow(p, k, cal):
                        p = p.copy()
                        p["selected_action"] = [actions[i] for i in k]
                        p["probability_supported"] = np.array(
                            [np.asarray(cal[actions[i]], float)[r] for r, i in enumerate(k)])
                        p["end_to_end_latency_ms"] = np.array([costs[actions[i]] for i in k], float)
                        return p

                    if name == "decomposed_linear":
                        pen = np.array([costs[a] / cmax for a in actions])[None, :]
                        mk = lambda fr, cal, h, l: ow(bl(fr, cal, h), (h - l * pen).argmax(1), cal)
                        lam = max(LAMS, key=lambda l: v1.scope_auroc(mk(val, c_va, hv, l), "pooled"))
                        p = mk(test, c_te, ht, lam)
                    elif name == "cascade":
                        def mk(fr, cal, h, w):
                            n = len(fr); k = np.zeros(n, int); ms = np.full(n, costs[actions[0]])
                            prob = np.asarray(cal[actions[0]], float).copy(); act = np.ones(n, bool)
                            for j in range(1, len(actions)):
                                esc = act & (np.abs(prob - 0.5) <= w)
                                if not esc.any(): break
                                k[esc] = j; prob[esc] = np.asarray(cal[actions[j]], float)[esc]
                                ms[esc] += costs[actions[j]]; act = esc
                            q = ow(bl(fr, cal, h), k, cal); q["end_to_end_latency_ms"] = ms; return q
                        w = max(WIDTHS, key=lambda w: v1.scope_auroc(mk(val, c_va, hv, w), "pooled"))
                        p = mk(test, c_te, ht, w)
                    else:
                        C = np.column_stack([train[f"correct__{a}"].fillna(0).to_numpy(int) for a in actions])
                        m = make_model(lr, seed, True)
                        m.fit(train[FEATS].to_numpy(float), np.where(C.any(1), C.argmax(1), 0))
                        cls = list(m.classes_)
                        k = np.array([cls[i] for i in m.predict_proba(test[FEATS].to_numpy(float)).argmax(1)])
                        p = ow(bl(test, c_te, ht), k, c_te)

                    preds.append(p)
                    banks.append(v2.control_bank_single(test, actions, p, c_te, (seed, name, str(test_f), "r5")))
                pred = pd.concat(preds, ignore_index=True)
                ctrls = [pd.concat([b[j] for b in banks], ignore_index=True) for j in range(core.CONTROL_DRAWS)]
                r = N.summarize(pred, ctrls, seed, name, f"r5|{name}|{seed}")
                r["paradigm"] = name
                rows.append(r)
                log(f"  R5 {name} seed {seed}: {r['treatment_auroc']:.4f} d={r['delta_point']:+.4f} {r['ms']:.0f}ms")
            except Exception:
                log(f"  R5 {name} seed {seed} FAILED\n{traceback.format_exc()[:250]}")
    save("R5_PARADIGMS.csv", pd.DataFrame(rows))
    agg = pd.DataFrame(rows).groupby("paradigm", as_index=False).agg(
        test=("treatment_auroc", "mean"), delta=("delta_point", "mean"),
        ci_low=("delta_ci_low", "min"), ms=("ms", "mean")).sort_values("test", ascending=False)
    save("R5_PARADIGMS_AGG.csv", agg)
    log("R5 router 范式:\n" + agg.round(4).to_string(index=False))

    # -- budget curve on the winning stack (Figure 1)
    curve = []
    for seed in SEEDS3:
        folds = core.build_folds(matrix, seed)
        for test_f, val_f, train_f in core.rotations():
            train = matrix.loc[np.isin(folds, train_f)].reset_index(drop=True)
            val = matrix.loc[folds == val_f].reset_index(drop=True)
            test = matrix.loc[folds == test_f].reset_index(drop=True)
            cals = core.fit_calibrators_on_val(val, actions)
            c_tr = core.apply_calibrators(train, actions, cals)
            c_va = core.apply_calibrators(val, actions, cals)
            c_te = core.apply_calibrators(test, actions, cals)
            hv, ht = fit_heads(train, [val, test], FEATS, actions, seed, tg, c_tr, lr)
            for b in core.BETA_GRID:
                pv = v1.router_prediction(val, actions, hv, c_va, costs, b)
                pt = v1.router_prediction(test, actions, ht, c_te, costs, b)
                curve.append({"seed": seed, "fold": test_f, "beta": b,
                              "val_auroc": v1.scope_auroc(pv, "pooled"),
                              "val_ms": float(pv.end_to_end_latency_ms.mean()),
                              "test_auroc": v1.scope_auroc(pt, "pooled"),
                              "test_ms": float(pt.end_to_end_latency_ms.mean())})
    cv = save("R5_BUDGET_CURVE.csv", pd.DataFrame(curve))
    cc = cv.groupby("beta", as_index=False).agg(val=("val_auroc", "mean"), val_ms=("val_ms", "mean"),
                                                test=("test_auroc", "mean"), test_ms=("test_ms", "mean"))
    log("R5 预算曲线 (Figure 1):\n" + cc.round(4).to_string(index=False))

    # -- 10 seeds + paired dataset-ID control
    keys = sorted(matrix["dataset_key"].astype(str).unique())
    ONEHOT = []
    for k in keys:
        c = f"dsid__{k}"; matrix[c] = matrix["dataset_key"].astype(str).eq(k).astype(float); ONEHOT.append(c)
    fin, pair = [], []
    for seed in SEEDS10:
        try:
            va, pf, ct = pass_once(matrix, pool, seed, target=tg, learner=lr, tag="final")
            r = N.summarize(pf, ct, seed, "FINAL", f"final|{seed}"); r["val_auroc"] = va
            fin.append(r)
            _, pdd, _ = pass_once(matrix, pool, seed, cols=ONEHOT, target=tg, learner=lr, tag="finaldsid")
            c = v2.Comparison(pf, [pdd], "pooled")
            pair.append({"seed": seed, "full": c.point()["treatment_auroc"],
                         "dsid": c.point()["control_auroc_mean"],
                         "delta": c.point()["delta_point"], **c.bootstrap(f"finalpair|{seed}"),
                         "agreement": float((pf.selected_action.to_numpy() == pdd.selected_action.to_numpy()).mean())})
            log(f"  R5 final seed {seed}: {r['treatment_auroc']:.4f} @ {r['ms']:.0f}ms | "
                f"vs dsid {pair[-1]['delta']:+.4f} CI[{pair[-1]['delta_ci_low']:+.4f},"
                f"{pair[-1]['delta_ci_high']:+.4f}] agree={pair[-1]['agreement']:.3f}")
        except Exception:
            log(f"  R5 final seed {seed} FAILED")
    if fin:
        ff = save("R5_FINAL_10SEED.csv", pd.DataFrame(fin))
        log(f"R5 终版: test {ff.treatment_auroc.mean():.4f} (sd {ff.treatment_auroc.std():.4f}) "
            f"@ {ff.ms.mean():.1f}ms | delta {ff.delta_point.mean():+.4f} | "
            f"显著 {int((ff.delta_ci_low>0).sum())}/{len(ff)}")
    if pair:
        pp = pd.DataFrame(pair); pp["significant"] = pp.delta_ci_low > 0
        save("R5_PAIRED_VS_DSID.csv", pp)
        log(f"R5 配对 vs 查表: 均值 {pp.delta.mean():+.4f} | 显著 {int(pp.significant.sum())}/{len(pp)} "
            f"| 一致率 {pp.agreement.mean():.3f}")


# ==================================================================== driver
STAGES = [("R1", stage_R1), ("R2", stage_R2), ("R3", stage_R3), ("R4", stage_R4), ("R5", stage_R5)]


def main():
    log("day2 start")
    matrix = v1.load_matrix(ROOT)
    ok = all(abs(v1.fixed_verifier_macro_auroc(matrix, v) - e) <= v1.ANCHOR_TOLERANCE
             for (v, _), e in v1.FIXED_BASELINE_ANCHORS.items())
    log(f"anchor gate {'PASS' if ok else 'FAIL'}  matrix {matrix.shape}")
    if not ok:
        return 3
    failed = []
    for name, fn in STAGES:
        if done(name):
            log(f"=== {name} already done, skipping"); continue
        log(f"=== {name} start"); t = time.time()
        try:
            fn(matrix); mark(name, {"seconds": round(time.time() - t, 1)})
            log(f"=== {name} COMPLETE in {time.time()-t:.0f}s")
        except Exception:
            tb = traceback.format_exc(); (ST / f"{name}.failed").write_text(tb)
            failed.append(name); log(f"=== {name} FAILED -- continuing\n{tb[:1000]}")
    base.write_json(ST / "RUN_SUMMARY.json", {
        "status": "COMPLETE_WITH_FAILURES" if failed else "COMPLETE", "failed_stages": failed,
        "protocol": "pooled 8/1/1 ten-fold rotation", "verifier_inference_calls": 0,
        "storysumm_rows_read": 0, "official_or_sealed_test_rows_read": 0,
        "wall_clock_seconds": round(time.time() - T0, 1), "completed_at_utc": base.utc_now()})
    log(f"ALL DONE in {time.time()-T0:.0f}s | failed: {failed or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
