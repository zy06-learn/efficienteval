#!/usr/bin/env python3
"""Test-blind exhaustive pool selection under the paper's final router (Protocol B).

Companion of r6r9_trainside_20260929.py, which re-runs the historical selection-stage router.
The rule is frozen in paper/revision_20260929/FREEZE_paperconfig_selection.md. TRAIN only; the
test frame is loaded label-free and never used.

Stages:
  gate                 reproduce pool_rescreen_v1's A-style numbers for the paper pool
  screen SEED I N      every 2/3-member pool, one seed, ten inner rotations, shard I of N
  fixed                the fifteen single verifiers under the same protocol (Balance anchors)
  finals               budget picks of the seed-17 screen + the paper pool, seeds 17/29/43,
                       six features and corpus one-hot
  apply                frozen selection rule -> tables
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3")
import v3core as V  # noqa: E402
import core  # noqa: E402

ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2")
PART1 = ROOT / "paper_v3" / "artifacts" / "part1_main_pooled_v1"
PART1C = ROOT / "paper_v3" / "artifacts" / "part1c_main_full_v1"
RUN = Path(os.environ.get("EXP1P_RUN_DIR", ROOT / "paper_v3" / "runs" / "exp1_paperconfig_20260929"))
RES = RUN / "results"
RES.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("SMOKE", "0") == "1"

CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
PAPER_POOL = sorted(CONTRACT["pool"])
FEATURES = list(CONTRACT["features"])
TARGET, LEARNER = CONTRACT["target"], CONTRACT["learner"]
HP = dict(json.loads((PART1C / "00_contract" / "HP_SELECTED.json").read_text())["hyperparameters_B"])
BUDGETS = (100, 150, 200, 250, 300, 400, 600)
FINAL_SEEDS = [17, 29, 43]
ROTS = V.rotations()[:2] if SMOKE else V.rotations()

TRAIN, TEST_NL, _ALL, VERIFIERS = V.load(with_test_labels=False)
assert "label_supported" not in TEST_NL.columns, "test frame must stay label free"
del TEST_NL, _ALL
assert len(TRAIN) == 5276 and TRAIN.content_doc_key.nunique() == 890
for k in sorted(TRAIN["dataset_key"].astype(str).str.split("_").str[0].unique()):
    TRAIN[f"ds__{k}"] = (TRAIN["dataset_key"].astype(str).str.split("_").str[0] == k).astype(float)
ONEHOT = sorted(c for c in TRAIN.columns if c.startswith("ds__"))
CONFIGS = {"six_features": FEATURES, "corpus_onehot": ONEHOT}
PRE_MS = (TRAIN["feature_latency_ms"].to_numpy(float)
          + TRAIN["compact16_feature_latency_ms"].to_numpy(float))
Y = TRAIN["label_supported"].to_numpy(int)


def key(actions):
    return "+".join(sorted(actions))


def evaluable(actions):
    av = np.column_stack([TRAIN[f"available__{a}"].to_numpy(bool) for a in actions])
    return bool(av.any(1).all())


def _iso(p_val, y_val, p_ev):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    t0 = time.perf_counter()
    q = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    return q, (time.perf_counter() - t0) * 1000.0 / max(len(p_ev), 1)


def router_rotation(fit, val, ev, pre_ev, actions, feats, seed):
    """part1c.router_raw + _stage2 + _assemble, for an arbitrary pool and feature list."""
    cals = core.platt(fit, actions=actions)
    c_fit = core.apply_platt(fit, cals, actions=actions)
    c_val = core.apply_platt(val, cals, actions=actions)
    t1 = time.perf_counter()
    c_ev = core.apply_platt(ev, cals, actions=actions)
    platt_ms = (time.perf_counter() - t1) * 1000.0 / max(len(ev), 1)
    h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=actions, features=feats,
                                 hp=HP, target=TARGET, learner=LEARNER)
    cvec = np.array([core.fold_costs(fit, actions=actions)[a] for a in actions], float)
    beta, _ = V.choose_beta(val, h_val, c_val, actions, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, actions, cvec)
    t2 = time.perf_counter()
    sel, p_ev, ms_part1 = V.route(ev, h_ev, c_ev, beta, actions, cvec)
    route_ms = (time.perf_counter() - t2) * 1000.0 / max(len(ev), 1)
    lat = np.column_stack([ev[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in actions])
    ver_ms = lat[np.arange(len(ev)), np.asarray(sel, int)]
    y_val = val["label_supported"].to_numpy(int)
    q_ev, iso_ms = _iso(p_val, y_val, p_ev)
    ms_full = ver_ms + pre_ev + float(h_ev.predict_ms) + route_ms + platt_ms + iso_ms
    return {"q": q_ev, "p1": np.asarray(p_ev, float), "sel": np.asarray(sel, int),
            "ms_full": ms_full, "ms_part1": np.asarray(ms_part1, float), "beta": float(beta),
            "val_auroc": float(roc_auc_score(y_val, p_val))}


def fixed_rotation(fit, val, ev, v):
    """part1c.fixed_raw + _stage2 + _assemble."""
    cal = core.platt(fit, actions=[v])
    p_val = core.apply_platt(val, cal, actions=[v])[v]
    t0 = time.perf_counter()
    p_ev = core.apply_platt(ev, cal, actions=[v])[v]
    platt_ms = (time.perf_counter() - t0) * 1000.0 / max(len(ev), 1)
    q_ev, iso_ms = _iso(p_val, val["label_supported"].to_numpy(int), p_ev)
    ver = ev[f"latency_ms__{v}"].to_numpy(float)
    assert np.isfinite(ver).all()
    return {"q": q_ev, "p1": np.asarray(p_ev, float), "ms_full": ver + platt_ms + iso_ms}


def over_rotations(seed, fn):
    """Run fn(fit, val, ev, idx_ev) over the ten rotations; return outputs in TRAIN row order."""
    fold = V.folds_stratified(TRAIN, seed)
    parts = []
    for t, vf, trf in ROTS:
        i_fit, i_val, i_ev = (np.where(np.isin(fold, trf))[0], np.where(fold == vf)[0],
                              np.where(fold == t)[0])
        fit, val, ev = (TRAIN.iloc[i].reset_index(drop=True) for i in (i_fit, i_val, i_ev))
        parts.append((i_ev, fn(fit, val, ev, i_ev)))
    idx = np.concatenate([p[0] for p in parts])
    out = {}
    for k in parts[0][1]:
        v = parts[0][1][k]
        if np.ndim(v) == 0:
            out[k] = [p[1][k] for p in parts]
        else:
            arr = np.concatenate([p[1][k] for p in parts])
            full = np.full(len(TRAIN), np.nan)
            full[idx] = arr
            out[k] = full
    out["_idx"] = np.sort(idx)
    return out


def eval_pool(actions, seed, feats):
    actions = list(actions)
    o = over_rotations(seed, lambda fit, val, ev, i: router_rotation(
        fit, val, ev, PRE_MS[i], actions, feats, seed))
    m = o["_idx"]
    sel = o["sel"][m].astype(int)
    sh = np.bincount(sel, minlength=len(actions)) / len(sel)
    return {"auroc": float(roc_auc_score(Y[m], o["q"][m])),
            "auroc_stage1": float(roc_auc_score(Y[m], o["p1"][m])),
            "ms": float(o["ms_full"][m].mean()), "ms_part1": float(o["ms_part1"][m].mean()),
            "val_auroc_mean": float(np.mean(o["val_auroc"])),
            "beta_mode": float(pd.Series(o["beta"]).mode().iloc[0]),
            "min_share": float(sh.min()),
            **{f"share__{a}": float(s) for a, s in zip(actions, sh)}}


def log(msg):
    core.log(msg)


# ------------------------------------------------------------------ stages
def gate():
    """Drift check: the pre-stage-2 pooled AUROC and part1-basis latency of the paper pool must
    reproduce pool_rescreen_v1's A-style confirmation (same seeds, HP, features, rotations)."""
    ref = pd.read_csv(PART1C / "11_pool_rescreen" / "POOL_RESCREEN_A_CONFIRM.csv")
    ref = ref[ref.pool == key(PAPER_POOL)].iloc[0]
    au, ms = [], []
    for seed in V.SEEDS[:5]:
        r = eval_pool(PAPER_POOL, seed, FEATURES)
        au.append(r["auroc_stage1"])
        ms.append(r["ms_part1"])
    got = {"a_val_auroc": float(np.mean(au)), "a_val_ms": float(np.mean(ms))}
    d = {k: abs(got[k] - float(ref[k])) for k in got}
    ok = d["a_val_auroc"] < 1e-9 and d["a_val_ms"] < 1e-6
    (RES / "GATE.json").write_text(json.dumps({"reference": {k: float(ref[k]) for k in got},
                                               "reproduced": got, "abs_diff": d, "pass": ok},
                                              indent=2))
    log(f"gate: {got} vs ref a_val_auroc={ref['a_val_auroc']:.10f} a_val_ms={ref['a_val_ms']:.6f}"
        f" -> {'PASS' if ok else 'FAIL'}")
    if not ok and not SMOKE:
        raise SystemExit("gate failed")


def all_pools():
    pools = [list(c) for k in (2, 3) for c in itertools.combinations(sorted(VERIFIERS), k)]
    if SMOKE:
        pools = pools[:3] + pools[105:108] + [PAPER_POOL]
    return pools


def screen(seed, shard, n_shards):
    pools = all_pools()[shard::n_shards]
    out = RES / f"SCREEN_seed{seed}_shard{shard}of{n_shards}.csv"
    rows, t0 = [], time.time()
    for i, actions in enumerate(pools, 1):
        row = {"pool": key(actions), "k": len(actions), "seed": seed, "evaluable": evaluable(actions)}
        if row["evaluable"]:
            row.update(eval_pool(actions, seed, FEATURES))
        rows.append(row)
        if i % 20 == 0 or i == len(pools):
            pd.DataFrame(rows).to_csv(out, index=False)
            el = time.time() - t0
            log(f"  seed {seed} shard {shard}/{n_shards}: {i}/{len(pools)}  "
                f"{el/60:.1f} min, eta {el/i*(len(pools)-i)/60:.1f} min")
    pd.DataFrame(rows).to_csv(out, index=False)


def load_screen(seed):
    parts = sorted(RES.glob(f"SCREEN_seed{seed}_shard*.csv"))
    D = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    n_expected = len(all_pools())
    assert len(D) == n_expected and D.pool.is_unique, (len(D), n_expected)
    return D


def budget_picks(D):
    E = D[D.evaluable.astype(bool)]
    rows = []
    for B in BUDGETS:
        for k in (2, 3):
            ok = E[(E.ms <= B) & (E.k == k)].sort_values(
                ["auroc", "ms", "k", "pool"], ascending=[False, True, True, True])
            if ok.empty:
                continue
            top = ok.iloc[0]
            nxt = ok.iloc[1] if len(ok) > 1 else None
            rows.append({"budget_ms": B, "k": k, "pool": top.pool, "auroc": top.auroc,
                         "ms": top.ms, "runner_up": nxt.pool if nxt is not None else "",
                         "margin_to_runner_up": (top.auroc - nxt.auroc) if nxt is not None
                         else np.nan,
                         "tie": bool(nxt is not None and nxt.auroc == top.auroc)})
    return pd.DataFrame(rows)


def fixed():
    rows = []
    for v in VERIFIERS:
        for seed in FINAL_SEEDS:
            o = over_rotations(seed, lambda fit, val, ev, i: fixed_rotation(fit, val, ev, v))
            m = o["_idx"]
            rows.append({"verifier": v, "seed": seed,
                         "auroc": float(roc_auc_score(Y[m], o["q"][m])),
                         "auroc_stage1": float(roc_auc_score(Y[m], o["p1"][m])),
                         "ms": float(o["ms_full"][m].mean())})
        log(f"  fixed {v}: {np.mean([r['auroc'] for r in rows[-3:]]):.5f} @ "
            f"{np.mean([r['ms'] for r in rows[-3:]]):.1f} ms")
    pd.DataFrame(rows).to_csv(RES / "FIXED_VERIFIERS.csv", index=False)


def finals():
    pk = budget_picks(load_screen(17))
    pk.to_csv(RES / "BUDGET_PICKS_seed17.csv", index=False)
    todo = list(dict.fromkeys(list(pk.pool) + [key(PAPER_POOL)]))
    scr = load_screen(17).set_index("pool")
    rows = []
    for p in todo:
        for cfg, feats in CONFIGS.items():
            for seed in FINAL_SEEDS:
                r = eval_pool(p.split("+"), seed, feats)
                rows.append({"pool": p, "features": cfg, "seed": seed,
                             "picked_by_budget": p in set(pk.pool),
                             "is_paper_pool": p == key(PAPER_POOL), **r})
                if cfg == "six_features" and seed == 17:
                    d = abs(r["auroc"] - scr.loc[p, "auroc"])
                    assert d < 1e-12, f"seed-17 final differs from screen for {p}: {d}"
            log(f"  final {p} [{cfg}]: {np.mean([x['auroc'] for x in rows[-3:]]):.5f} @ "
                f"{np.mean([x['ms'] for x in rows[-3:]]):.1f} ms")
            pd.DataFrame(rows).to_csv(RES / "FINALS.csv", index=False)


def apply():
    fx = pd.read_csv(RES / "FIXED_VERIFIERS.csv").groupby("verifier", as_index=False).agg(
        auroc=("auroc", "mean"), ms=("ms", "mean"))
    cheapest = fx.loc[fx.ms.idxmin()]
    best = fx.loc[fx.auroc.idxmax()]
    A = {"Q_floor": float(cheapest.auroc), "cheapest": cheapest.verifier,
         "Q_ceil": float(best.auroc), "best_fixed": best.verifier,
         "C_min": float(fx.ms.min()), "C_max": float(fx.ms.max()),
         "priciest": fx.loc[fx.ms.idxmax(), "verifier"]}

    def balance(a, ms):
        q = (np.asarray(a, float) - A["Q_floor"]) / (A["Q_ceil"] - A["Q_floor"])
        c = (np.log(np.asarray(ms, float)) - np.log(A["C_min"])) / (np.log(A["C_max"])
                                                                   - np.log(A["C_min"]))
        return 1.0 - np.sqrt((1.0 - q) ** 2 + c ** 2) / np.sqrt(2.0)

    order = (["balance", "auroc", "ms", "k", "pool"], [False, False, True, True, True])

    # stage 1
    D17 = load_screen(17)
    pk = budget_picks(D17)
    pk["balance"] = balance(pk.auroc, pk.ms)
    pk.to_csv(RES / "SELECTION_BUDGET_TABLE.csv", index=False)
    assert not pk.tie.any(), "exact tie at a budget pick"

    # stage 2/3
    ff = pd.read_csv(RES / "FINALS.csv")
    agg = ff.groupby(["pool", "features", "picked_by_budget", "is_paper_pool"],
                     as_index=False).agg(auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
                                         ms=("ms", "mean"), n_seeds=("seed", "nunique"))
    agg["k"] = agg.pool.str.count(r"\+") + 1
    agg["balance"] = balance(agg.auroc, agg.ms)
    elig = agg[agg.picked_by_budget].sort_values(order[0], ascending=order[1]).reset_index(drop=True)
    elig.insert(0, "rank", np.arange(1, len(elig) + 1))
    elig.to_csv(RES / "SELECTION_FINALISTS.csv", index=False)
    agg.sort_values(order[0], ascending=order[1]).to_csv(RES / "FINALS_ALL_WITH_REFERENCE.csv", index=False)
    win = elig.iloc[0]
    nxt = elig[elig.pool != win.pool].iloc[0]

    # stability: stage 3 per seed
    st = []
    for s, g in ff[ff.picked_by_budget].groupby("seed"):
        g = g.assign(k=g.pool.str.count(r"\+") + 1, balance=balance(g.auroc, g.ms)).sort_values(
            order[0], ascending=order[1])
        st.append({"seed": int(s), "winner_pool": g.iloc[0].pool, "features": g.iloc[0].features,
                   "balance": float(g.iloc[0].balance),
                   "runner_up_pool": g[g.pool != g.iloc[0].pool].iloc[0].pool,
                   "runner_up_balance": float(g[g.pool != g.iloc[0].pool].iloc[0].balance)})
    st = pd.DataFrame(st)
    st.to_csv(RES / "SELECTION_SEED_STABILITY.csv", index=False)

    # stability: stage 1 per screening seed (descriptive)
    freq = []
    for s in (17, 29, 43):
        if not list(RES.glob(f"SCREEN_seed{s}_shard*.csv")):
            continue
        p = budget_picks(load_screen(s))
        freq += [{"screen_seed": s, **r} for r in p[["budget_ms", "k", "pool", "auroc", "ms",
                                                    "margin_to_runner_up"]].to_dict("records")]
    fq = pd.DataFrame(freq)
    fq.to_csv(RES / "BUDGET_PICKS_BY_SCREEN_SEED.csv", index=False)
    pick_freq = (fq.groupby(["budget_ms", "k", "pool"]).size().rename("n_seeds").reset_index()
                 if len(fq) else pd.DataFrame())
    pick_freq.to_csv(RES / "BUDGET_PICK_FREQUENCY.csv", index=False)

    # screened pools ranked by Balance (reference only)
    sv = D17[D17.evaluable.astype(bool)].copy()
    sv["balance"] = balance(sv.auroc, sv.ms)
    sv = sv.sort_values(order[0], ascending=order[1]).reset_index(drop=True)
    sv.insert(0, "rank", np.arange(1, len(sv) + 1))
    sv.to_csv(RES / "SCREEN_BALANCE_RANKING.csv", index=False)
    pos = int(sv.index[sv.pool == key(PAPER_POOL)][0]) + 1

    res = {"selected_pool": win.pool, "selected_features_config": win.features,
           "heldout_auroc": float(win.auroc), "ms": float(win.ms), "balance": float(win.balance),
           "next_different_pool": nxt[["pool", "features", "auroc", "ms", "balance"]].to_dict(),
           "paper_pool": key(PAPER_POOL), "same_as_paper_pool": win.pool == key(PAPER_POOL),
           "seed_winner_counts": st.winner_pool.value_counts().to_dict(),
           "paper_pool_screen_balance_rank": pos, "n_screened_evaluable": int(len(sv)),
           "n_screened": {int(k): int(v) for k, v in D17.groupby("k").size().items()},
           "n_evaluable": {int(k): int(v) for k, v in
                           D17[D17.evaluable.astype(bool)].groupby("k").size().items()},
           "anchors": A, "hyperparameters": HP}
    (RES / "SELECTION_RESULT.json").write_text(json.dumps(res, indent=2, default=float))
    pd.set_option("display.width", 220)
    print("BUDGET PICKS\n", pk.round(4).to_string(index=False))
    print("\nFINALISTS\n", elig.round(4).to_string(index=False))
    print("\nSEED STABILITY\n", st.round(4).to_string(index=False))
    if len(pick_freq):
        print("\nBUDGET PICK FREQUENCY\n", pick_freq.to_string(index=False))
    print("\nRESULT", json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    log(f"===== {stage} {sys.argv[2:]} | TRAIN {len(TRAIN)} rows | hp={HP} | smoke={SMOKE} =====")
    if stage == "gate":
        gate()
    elif stage == "screen":
        screen(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif stage == "fixed":
        fixed()
    elif stage == "finals":
        finals()
    elif stage == "apply":
        apply()
    else:
        raise SystemExit(f"unknown stage {stage}")
    log(f"===== {stage} done in {time.time()-t0:.0f}s =====")
