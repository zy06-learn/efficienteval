#!/usr/bin/env python3
"""p12 -- HalluScan's Adaptive Detection Routing (Cherif 2026, arXiv:2605.02443, Sec. 3.4,
Algorithm 1) reimplemented as a stand-alone baseline on the frozen score matrix.

Kept from the paper
  * a gradient-boosted tree predicts one detector-agnostic hallucination risk from cheap
    input features;
  * detectors are ordered by strength and the risk is cut by two thresholds into three tiers
    (risk > theta_high -> strongest, theta_med < risk <= theta_high -> middle, else cheapest);
  * exactly one detector runs and its own raw score is returned -- no calibration, no fusion;
  * thresholds are chosen by cross-validation on the training portion, maximising AUROC
    under a cost constraint.

Changed because the setting differs (paper/ADR_BASELINE_DESIGN_20260930.md, Sec. 2.2)
  * detectors -> our pool, ordered weakest -> strongest by raw-score AUROC on the training
    portion (factcc < lettuce_v2 < granite_guardian_3_1_2b on TRAIN);
  * features re-implemented on (source, summary): summary dependency depth and entity count,
    source and summary length, summary number count; the domain indicator is dropped;
  * cost is measured latency; the budget bounds mean latency during threshold selection;
  * the paper estimated ADR's AUROC from routing shares; here ADR is executed and scored.

Nothing from EfficientEval is used: not its six features, relative-quality heads, cost
discount, beta/eps rule, or Platt + isotonic calibration. Only the evaluation protocol is
shared with Table 1 (splits, seeds, core.metrics, validation-MCC decision threshold).

Stages, run in order:  features | hpselect | gates | B | A | report
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score  # noqa: E402

V3 = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3")
sys.path.insert(0, str(V3))
import v3core as V  # noqa: E402
import core  # noqa: E402

PART1 = V3 / "artifacts" / "part1_main_pooled_v1"
PART1C = V3 / "artifacts" / "part1c_main_full_v1"
ABL = V3 / "runs" / "part2_ablation_v1" / "results"
OUT = V3 / "runs" / "p12_adr_20260930"
for _d in ("features", "hp", "gates", "B", "A", "report"):
    (OUT / _d).mkdir(parents=True, exist_ok=True)

CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
POOL = list(CONTRACT["pool"])
SEEDS = list(V.SEEDS)
HP_SEEDS = SEEDS[:5]                      # same seeds as part1c's head search
N_HP = 16                                 # same candidate count as part1c's head search
INNER_FOLDS = 5
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
PAIRS = np.array([(m, h) for i, m in enumerate(GRID) for h in GRID[i:]])  # med <= high
FRONTIER = tuple(float(b) for b in range(50, 210, 10))
ADR_FEATURES = ["summary_dep_depth", "summary_entity_count", "source_len_tokens",
                "summary_len_tokens", "summary_number_count"]
HP_SPACE = {"max_iter": [100, 200, 400], "learning_rate": [0.03, 0.1],
            "max_depth": [2, 3, 4], "min_samples_leaf": [10, 20, 50]}
N_JOBS = 18
T0 = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-T0:6.0f}s] {m}", flush=True)


def hv(auroc, ms):
    return max(0.0, 2 * auroc - 1) * max(0.0, 1 - ms / 1000.0)


def auc_cols(y, P):
    """AUROC of every column of P (ties averaged), identical to roc_auc_score."""
    P = np.asarray(P, float)
    P = P[:, None] if P.ndim == 1 else P
    r = rankdata(P, axis=0)
    pos = np.asarray(y) == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return (r[pos].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def rule_mcc(q, y):
    """part1c._rule_mcc, verbatim."""
    grid = np.unique(np.quantile(np.asarray(q, float), np.linspace(0.02, 0.98, 97)))
    return float(max(grid, key=lambda t: matthews_corrcoef(y, (np.asarray(q) >= t).astype(int))))


def agg(y, P, MS, THR):
    """part1c._agg: core.metrics per seed, then the seed mean."""
    rows = [core.metrics(y, P[:, s], MS[:, s], threshold=THR[:, s]) for s in range(P.shape[1])]
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out["auroc_sd"] = float(np.std([r["auroc"] for r in rows]))
    out["hv"] = hv(out["auroc"], out["ms"])
    return out


# ------------------------------------------------------------------ stage: features
def _depth(tok):
    d = 0
    while tok.head.i != tok.i:          # spaCy builds a new Token per .head access,
        tok, d = tok.head, d + 1        # so identity (`is`) never terminates at the root
    return d


def stage_features():
    """HalluScan's feature types on (source, summary), timed per row at batch size 1."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp("Warm-up: Paris had 3 visitors in 2020.")
    _tr, _te, ALL, _v = V.load(with_test_labels=False)       # label-free
    rows = []
    for ek, src, summ in zip(ALL["episode_key"].astype(str), ALL["source_document"].astype(str),
                             ALL["candidate_summary"].astype(str)):
        t0 = time.perf_counter()
        doc = nlp(summ)
        depths = [max(_depth(t) for t in s) for s in doc.sents if len(s)]
        f = {"summary_dep_depth": float(np.mean(depths)) if depths else 0.0,
             "summary_entity_count": float(len(doc.ents)),
             "source_len_tokens": float(len(src.split())),
             "summary_len_tokens": float(len(summ.split())),
             "summary_number_count": float(sum(t.like_num for t in doc))}
        rows.append({"episode_key": ek, **f, "adr_feature_ms": (time.perf_counter() - t0) * 1e3})
    F = pd.DataFrame(rows)
    assert F["episode_key"].is_unique and len(F) == len(ALL)
    F.to_parquet(OUT / "features" / "adr_features.parquet", index=False)
    ms = F["adr_feature_ms"]
    log(f"features: {len(F)} rows | ms mean {ms.mean():.3f} p50 {ms.median():.3f} "
        f"p95 {ms.quantile(.95):.3f}")
    log(F[ADR_FEATURES].describe().T.to_string())


def attach(frame):
    """Slim frame with only what ADR may use, plus the ADR features."""
    F = pd.read_parquet(OUT / "features" / "adr_features.parquet")
    keep = (["episode_key", "dataset_key", "content_doc_key"]
            + (["label_supported"] if "label_supported" in frame.columns else [])
            + [f"{k}__{a}" for a in POOL for k in ("score", "available", "latency_ms")])
    out = frame[keep].merge(F, on="episode_key", how="left", validate="one_to_one")
    assert out[ADR_FEATURES + ["adr_feature_ms"]].notna().all().all()
    assert (out["episode_key"].to_numpy() == frame["episode_key"].to_numpy()).all()
    return out


# ------------------------------------------------------------------ the ADR model
def gbdt(hp, seed):
    return HistGradientBoostingClassifier(**hp, early_stopping=False, random_state=seed)


def oof_risk(train, hp, seed):
    """Out-of-fold hallucination risk on the training portion (5 group-disjoint folds)."""
    X = train[ADR_FEATURES].to_numpy(float)
    r = 1 - train["label_supported"].to_numpy(int)
    fold = V.folds_stratified(train, seed, INNER_FOLDS)
    out = np.full(len(train), np.nan)
    for k in range(INNER_FOLDS):
        m = fold == k
        out[m] = gbdt(hp, seed).fit(X[~m], r[~m]).predict_proba(X[m])[:, 1]
    assert np.isfinite(out).all()
    return out


def tier_order(train):
    y = train["label_supported"].to_numpy(int)
    auc = {a: float(roc_auc_score(y, train[f"score__{a}"].to_numpy(float))) for a in POOL}
    return sorted(POOL, key=lambda a: auc[a]), auc


def mats(frame, order):
    S = np.column_stack([frame[f"score__{a}"].to_numpy(float) for a in order])
    L = np.column_stack([frame[f"latency_ms__{a}"].to_numpy(float) for a in order])
    A = np.column_stack([frame[f"available__{a}"].to_numpy(bool) for a in order])
    assert A[:, 2].all() and np.isfinite(L).all() and np.isfinite(S).all()
    return S, L, A


def assign(risk, med, high, A):
    """Algorithm 1, lines 4-14; an unavailable tier falls back to the next stronger one."""
    T = (risk[:, None] > med).astype(int) + (risk[:, None] > high).astype(int)
    rows = np.arange(len(risk))[:, None]
    for _ in range(2):
        T = np.where(~A[rows, T] & (T < 2), T + 1, T)
    return T


def grid_eval(risk, y, S, L, A, over_ms, chunk=500):
    """Training-portion AUROC and mean latency of every (theta_med, theta_high) pair."""
    auc = np.empty(len(PAIRS))
    ms = np.empty(len(PAIRS))
    rows = np.arange(len(risk))[:, None]
    for i in range(0, len(PAIRS), chunk):
        P = PAIRS[i:i + chunk]
        T = assign(risk, P[None, :, 0], P[None, :, 1], A)
        auc[i:i + len(P)] = auc_cols(y, S[rows, T])
        ms[i:i + len(P)] = (L[rows, T] + over_ms[:, None]).mean(0)
    return auc, ms


def choose(auc, ms, budget):
    """Highest AUROC among pairs within budget; exact ties -> lower latency -> grid order."""
    idx = np.flatnonzero(ms <= budget + 1e-9)
    if not len(idx):
        return None
    cand = idx[auc[idx] >= auc[idx].max() - 1e-12]
    return int(cand[np.argmin(ms[cand])])


def adr_partition(tag, train, ev, seed, hp, budget_main):
    """One training portion -> one evaluation portion. `ev` carries no labels (gate G4)."""
    assert "label_supported" not in ev.columns
    assert not set(train["content_doc_key"]) & set(ev["content_doc_key"])
    order, auc_train = tier_order(train)
    y = train["label_supported"].to_numpy(int)
    S_tr, L_tr, A_tr = mats(train, order)
    S_ev, L_ev, A_ev = mats(ev, order)
    pool_idx = np.array([POOL.index(a) for a in order])

    risk_tr = oof_risk(train, hp, seed)
    model = gbdt(hp, seed).fit(train[ADR_FEATURES].to_numpy(float), 1 - y)
    X_ev = ev[ADR_FEATURES].to_numpy(float)
    t0 = time.perf_counter()
    risk_ev = model.predict_proba(X_ev)[:, 1]
    tree_ms = (time.perf_counter() - t0) * 1e3 / len(ev)

    over_tr = train["adr_feature_ms"].to_numpy(float) + tree_ms
    g_auc, g_ms = grid_eval(risk_tr, y, S_tr, L_tr, A_tr, over_tr)

    budgets = {"main": budget_main, "nobudget": np.inf,
               **{f"b{int(b):03d}": b for b in FRONTIER}}
    rows_tr = np.arange(len(train))
    rows_ev = np.arange(len(ev))
    out = {"tag": tag, "seed": seed, "order": order, "auc_train": auc_train,
           "risk_ev": risk_ev, "episode_key": ev["episode_key"].to_numpy(str),
           "tree_ms": tree_ms, "variants": {}, "records": []}
    for name, b in budgets.items():
        k = choose(g_auc, g_ms, b)
        if k is None:
            out["records"].append({"tag": tag, "seed": seed, "variant": name, "budget": b,
                                   "feasible": False})
            continue
        med, high = PAIRS[k]
        T_tr = assign(risk_tr, med, high, A_tr)[:, 0]
        thr = rule_mcc(S_tr[rows_tr, T_tr], y)
        t1 = time.perf_counter()
        T_ev = assign(risk_ev, med, high, A_ev)[:, 0]
        route_ms = (time.perf_counter() - t1) * 1e3 / len(ev)
        feat = ev["adr_feature_ms"].to_numpy(float)
        ver = L_ev[rows_ev, T_ev]
        ms = ver + feat + tree_ms + route_ms
        assert np.allclose(ms - ver - feat, tree_ms + route_ms)          # gate G5
        raw_tier = (risk_ev[:, None] > med).astype(int)[:, 0] + (risk_ev > high).astype(int)
        out["variants"][name] = {"prob": S_ev[rows_ev, T_ev], "ms": ms, "thr": thr,
                                 "sel": pool_idx[T_ev],
                                 "components": {"verifier": float(ver.mean()),
                                                "features": float(feat.mean()),
                                                "tree": tree_ms, "routing": route_ms}}
        out["records"].append({"tag": tag, "seed": seed, "variant": name, "budget": b,
                               "feasible": True, "theta_med": float(med),
                               "theta_high": float(high), "train_auroc": float(g_auc[k]),
                               "train_ms": float(g_ms[k]), "mcc_threshold": thr,
                               "fallback_rows_eval": int((raw_tier != T_ev).sum()),
                               **{f"train_share__{order[t]}": float((T_tr == t).mean())
                                  for t in range(3)}})
    # gate G2: a forced single tier reproduces that verifier's own score
    out["corners"] = {order[t]: S_ev[rows_ev, assign(risk_ev, -1.0 if t else 2.0,
                                                     -1.0 if t == 2 else 2.0, A_ev)[:, 0]]
                      for t in range(3)}
    return out


def load_hp():
    return json.loads((OUT / "hp" / "HP_SELECTED.json").read_text())["hp"]


def budgets_from_ledger(protocol):
    sel = pd.read_csv(PART1C / "04_beta_evidence" / f"{protocol}_BETA_SELECTED.csv")
    led = pd.read_csv(PART1C / "04_beta_evidence" / f"{protocol}_BETA_LEDGER.csv")
    keys = ["seed"] + (["rotation"] if protocol == "A" else [])
    m = sel[keys + ["beta"]].merge(led, on=keys + ["beta"], how="left", validate="one_to_one")
    assert m["val_ms"].notna().all()
    return {(tuple(int(r[k]) for k in keys) if protocol == "A" else int(r["seed"])):
            float(r["val_ms"]) for _, r in m.iterrows()}


# ------------------------------------------------------------------ stage: hpselect
def stage_hpselect():
    """16 GBDT candidates, TRAIN only, lowest 5-fold log-loss over 5 seeds (parity with the
    head search in part1c: 16 candidates, TRAIN only, 5 seeds, lowest validation loss)."""
    TRAIN, _te, _all, _v = V.load(with_test_labels=False)
    TRAIN = attach(TRAIN)
    rng = np.random.default_rng(0)
    cands, seen = [], set()
    while len(cands) < N_HP:
        c = {k: v[int(rng.integers(len(v)))] for k, v in HP_SPACE.items()}
        c = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in c.items()}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            cands.append(c)
    r = 1 - TRAIN["label_supported"].to_numpy(int)

    def one(ci, seed):
        return ci, seed, float(log_loss(r, oof_risk(TRAIN, cands[ci], seed), labels=[0, 1]))

    res = Parallel(n_jobs=N_JOBS)(delayed(one)(ci, s) for ci in range(N_HP) for s in HP_SEEDS)
    D = pd.DataFrame(res, columns=["candidate", "seed", "cv_log_loss"])
    M = D.groupby("candidate")["cv_log_loss"].mean().reset_index()
    M = M.assign(**{k: [cands[i][k] for i in M["candidate"]] for k in HP_SPACE})
    M.sort_values("cv_log_loss").to_csv(OUT / "hp" / "HP_SELECTION.csv", index=False)
    best = cands[int(M.loc[M["cv_log_loss"].idxmin(), "candidate"])]
    (OUT / "hp" / "HP_SELECTED.json").write_text(json.dumps(
        {"rule": "lowest mean 5-fold CV log-loss, TRAIN only, 16 candidates, seeds "
                 f"{HP_SEEDS}", "space": HP_SPACE, "hp": best}, indent=2))
    log(f"hpselect: {best}\n{M.sort_values('cv_log_loss').head(5).to_string(index=False)}")


# ------------------------------------------------------------------ stage: gates
def stage_gates():
    """G-metric: this script's metric path reproduces the published OURS and pool rows.
    G1(B): row order equals probs_B.npz.  G3: raw-score direction on TRAIN."""
    gates = {}
    for p in ("B", "A"):
        Z = np.load(PART1C / "10_row_level" / f"probs_{p}.npz")
        pub = pd.read_csv(PART1C / "01_main_tables" / "with_sentinel" /
                          f"{p}_MAIN_WITH_SENTINEL.csv").set_index("system")
        y = Z["y"].astype(int)
        for s in ["OURS"] + [f"fixed::{a}" for a in POOL]:
            k = s.replace("::", "__")
            m = agg(y, Z[f"prob__{k}"], Z[f"ms_full__{k}"], Z[f"thr_mcc__{k}"])
            d = {c: abs(m[c] - float(pub.loc[s, c]))
                 for c in ("auroc", "auprc_unsup", "bacc", "f1_unsup", "ms")}
            gates[f"{p}:{s}"] = {**d, "hv": m["hv"]}
            assert max(d.values()) < 1e-9, (p, s, d)
        log(f"G-metric {p}: OURS and pool rows reproduced; OURS HV {gates[f'{p}:OURS']['hv']:.4f}")
    TRAIN, TEST, _all, _v = V.load(with_test_labels=True)
    ZB = np.load(PART1C / "10_row_level" / "probs_B.npz")
    assert (TEST["episode_key"].astype(str).to_numpy() == ZB["episode_key"].astype(str)).all()
    order, auc = tier_order(TRAIN)
    assert all(v > 0.5 for v in auc.values()), auc
    gates["G1_B_row_order"] = True
    gates["G3_train_direction"] = {"order_weak_to_strong": order, "auroc": auc}
    (OUT / "gates" / "GATES_PRE.json").write_text(json.dumps(gates, indent=2))
    log(f"G1(B) row order ok | G3 TRAIN AUROC {auc} -> order {order}")


# ------------------------------------------------------------------ stages: B and A
def _assemble(n, variants):
    S = len(SEEDS)
    acc = {v: {"prob": np.full((n, S), np.nan), "ms": np.full((n, S), np.nan),
               "thr": np.full((n, S), np.nan), "sel": np.full((n, S), -1, np.int8)}
           for v in variants}
    risk = np.full((n, S), np.nan)
    corners = {a: np.full((n, S), np.nan) for a in POOL}
    return acc, risk, corners


def _store(protocol, y, ek, corpus, acc, risk, corners, records, comps, order_rows):
    feasible = [v for v in acc if np.isfinite(acc[v]["prob"]).all()]
    store = {"y": y.astype(np.int8), "episode_key": np.asarray(ek, dtype=str),
             "corpus": np.asarray(corpus, dtype=str),
             "seeds": np.array(SEEDS), "actions": np.array(POOL), "risk": risk,
             **{f"corner__{a}": corners[a] for a in POOL}}
    for v in feasible:
        for k, arr in acc[v].items():
            store[f"{k}__{v}"] = arr
    np.savez_compressed(OUT / protocol / f"adr_rows_{protocol}.npz", **store)
    pd.DataFrame(records).to_csv(OUT / protocol / f"ADR_{protocol}_RECORDS.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT / protocol / f"ADR_{protocol}_LATENCY.csv", index=False)
    pd.DataFrame(order_rows).to_csv(OUT / protocol / f"ADR_{protocol}_ORDER.csv", index=False)
    rows = []
    for v in feasible:
        m = agg(y, acc[v]["prob"], acc[v]["ms"], acc[v]["thr"])
        rows.append({"variant": v, **{k: m[k] for k in ("auroc", "auroc_sd", "auprc_unsup",
                                                        "bacc", "f1_unsup", "ms", "hv")},
                     **{f"share__{a}": float((acc[v]["sel"] == i).mean())
                        for i, a in enumerate(POOL)}})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / protocol / f"ADR_{protocol}_SUMMARY.csv", index=False)
    log(f"Protocol {protocol} summary:\n{T.to_string(index=False)}")


def _collect(protocol, results_by_seed, y, ek_ref, corpus):
    """results_by_seed[si] = list of partition outputs whose eval rows tile the protocol rows."""
    variants = ["main", "nobudget"] + [f"b{int(b):03d}" for b in FRONTIER]
    n = len(y)
    acc, risk, corners = _assemble(n, variants)
    records, comps, order_rows = [], [], []
    for si, parts in enumerate(results_by_seed):
        ek = np.concatenate([p["episode_key"] for p in parts])
        o = np.argsort(ek, kind="stable") if protocol == "A" else np.arange(n)
        assert (ek[o] == ek_ref).all(), f"{protocol} seed {SEEDS[si]}: row order differs"
        risk[:, si] = np.concatenate([p["risk_ev"] for p in parts])[o]
        for a in POOL:
            corners[a][:, si] = np.concatenate([p["corners"][a] for p in parts])[o]
        for v in variants:
            if all(v in p["variants"] for p in parts):
                for k in ("prob", "ms", "sel"):
                    acc[v][k][:, si] = np.concatenate([p["variants"][v][k] for p in parts])[o]
                acc[v]["thr"][:, si] = np.concatenate(
                    [np.full(len(p["episode_key"]), p["variants"][v]["thr"]) for p in parts])[o]
        for p in parts:
            records += p["records"]
            order_rows.append({"tag": p["tag"], "seed": p["seed"], "order": ">".join(p["order"]),
                               **{f"train_auroc__{a}": p["auc_train"][a] for a in POOL}})
            if "main" in p["variants"]:
                comps.append({"tag": p["tag"], "seed": p["seed"],
                              **p["variants"]["main"]["components"]})
    return acc, risk, corners, records, comps, order_rows


def stage_B():
    hp = load_hp()
    TRAIN, TEST, _all, _v = V.load(with_test_labels=True)
    Z = np.load(PART1C / "10_row_level" / "probs_B.npz")
    ek_ref = Z["episode_key"].astype(str)
    assert (TEST["episode_key"].astype(str).to_numpy() == ek_ref).all()
    y = TEST["label_supported"].to_numpy(int)
    TRAIN = attach(TRAIN)
    EV = attach(TEST).drop(columns=["label_supported"])
    bud = budgets_from_ledger("B")
    log(f"B budgets (OURS validation ms at its selected beta): {bud}")
    res = Parallel(n_jobs=min(N_JOBS, len(SEEDS)))(
        delayed(adr_partition)(f"B:{s}", TRAIN, EV, s, hp, bud[s]) for s in SEEDS)
    acc, risk, corners, records, comps, order_rows = _collect(
        "B", [[r] for r in res], y, ek_ref, TEST["dataset_key"].astype(str).to_numpy())
    _store("B", y, ek_ref, TEST["dataset_key"].astype(str).to_numpy(), acc, risk, corners,
           records, comps, order_rows)


def stage_A():
    hp = load_hp()
    _tr, _te, ALL, _v = V.load(with_test_labels=True)
    Z = np.load(PART1C / "10_row_level" / "probs_A.npz")
    ek_ref = Z["episode_key"].astype(str)
    SLIM = attach(ALL)
    bud = budgets_from_ledger("A")
    jobs = []
    for seed in SEEDS:
        fold = V.folds_stratified(ALL, seed)
        for test_fold, _val_fold, _train_folds in V.rotations():
            tr = SLIM.loc[fold != test_fold].reset_index(drop=True)
            ev = SLIM.loc[fold == test_fold].reset_index(drop=True).drop(
                columns=["label_supported"])
            jobs.append((f"A:{seed}:{test_fold}", tr, ev, seed, bud[(seed, test_fold)]))
    log(f"A: {len(jobs)} partitions")
    res = Parallel(n_jobs=N_JOBS)(delayed(adr_partition)(t, tr, ev, s, hp, b)
                                  for t, tr, ev, s, b in jobs)
    by_seed = [[r for r in res if r["seed"] == s] for s in SEEDS]
    key = pd.Series(np.arange(len(ALL)), index=ALL["episode_key"].astype(str))
    y = ALL["label_supported"].to_numpy(int)[key[ek_ref].to_numpy()]
    corpus = ALL["dataset_key"].astype(str).to_numpy()[key[ek_ref].to_numpy()]
    assert (y == Z["y"].astype(int)).all()
    acc, risk, corners, records, comps, order_rows = _collect("A", by_seed, y, ek_ref, corpus)
    _store("A", y, ek_ref, corpus, acc, risk, corners, records, comps, order_rows)


# ------------------------------------------------------------------ stage: report
def stage_report():
    """Main rows next to published ones, selection accuracy, call shares, corner gate."""
    lines = ["# p12 ADR baseline -- internal report", ""]
    gates = json.loads((OUT / "gates" / "GATES_PRE.json").read_text())
    for p, frame_of in (("B", "TEST"), ("A", "ALL")):
        R = np.load(OUT / p / f"adr_rows_{p}.npz", allow_pickle=True)   # our own file
        Z = np.load(PART1C / "10_row_level" / f"probs_{p}.npz")
        assert (R["episode_key"] == Z["episode_key"].astype(str)).all()
        y, corpus = R["y"].astype(int), R["corpus"].astype(str)
        tr, te, al, _v = V.load(with_test_labels=True)
        F = (te if p == "B" else al).set_index("episode_key").loc[R["episode_key"]]
        RAW = np.column_stack([F[f"score__{a}"].to_numpy(float) for a in POOL])
        OK = ((RAW >= 0.5).astype(int) == y[:, None])                 # native decisions
        S = len(SEEDS)

        # G2: a forced single tier reproduces that verifier's raw-score AUROC. (Not the
        # published stage-one column: under A each rotation fits its own Platt map, so the
        # pooled stage-one scores are not rank-equivalent to the raw ones.) FactCC differs
        # only on rows where it is unavailable and ADR falls back to Lettuce-v2.
        for i, a in enumerate(POOL):
            d = max(abs(auc_cols(y, R[f"corner__{a}"][:, s])[0] - auc_cols(y, RAW[:, i])[0])
                    for s in range(S))
            gates[f"G2_{p}_{a}"] = float(d)

        rows = []
        for v in [x.split("__", 1)[1] for x in R.files if x.startswith("prob__")]:
            m = agg(y, R[f"prob__{v}"], R[f"ms__{v}"], R[f"thr__{v}"])
            sel = R[f"sel__{v}"].astype(int)
            rows.append({"system": f"ADR:{v}", **{c: m[c] for c in
                         ("auroc", "auroc_sd", "auprc_unsup", "bacc", "f1_unsup", "ms", "hv")},
                         "select_acc": float(np.mean([OK[np.arange(len(y)), sel[:, s]].mean()
                                                      for s in range(S)])),
                         **{f"share__{a}": float((sel == i).mean())
                            for i, a in enumerate(POOL)}})
        for s_name in ["OURS"] + [f"fixed::{a}" for a in POOL]:
            k = s_name.replace("::", "__")
            m = agg(y, Z[f"prob__{k}"], Z[f"ms_full__{k}"], Z[f"thr_mcc__{k}"])
            row = {"system": s_name, **{c: m[c] for c in ("auroc", "auroc_sd", "auprc_unsup",
                                                        "bacc", "f1_unsup", "ms", "hv")}}
            if s_name == "OURS":
                sel = Z["sel__OURS"].astype(int)
                row["select_acc"] = float(np.mean([OK[np.arange(len(y)), sel[:, s]].mean()
                                                   for s in range(S)]))
                row.update({f"share__{a}": float((sel == i).mean())
                            for i, a in enumerate(POOL)})
            else:
                row["select_acc"] = float(OK[:, POOL.index(s_name.split("::")[1])].mean())
            rows.append(row)
        rows.append({"system": "oracle(any native decision correct)",
                     "select_acc": float(OK.any(1).mean())})
        rows.append({"system": "ADR risk used directly as the score",
                     "auroc": float(np.mean(auc_cols(y, 1.0 - R["risk"])))})
        abl = ABL / f"ablation_{p}.npz"
        if abl.exists():
            AB = np.load(abl)
            if ("prob__no_verifier" in AB.files and "episode_key" in AB.files
                    and (AB["episode_key"].astype(str) == R["episode_key"]).all()):
                rows.append({"system": "EfficientEval cheap features only (published ablation)",
                             "auroc": float(np.mean(auc_cols(y, AB["prob__no_verifier"])))})
        T = pd.DataFrame(rows)
        T.to_csv(OUT / "report" / f"MAIN_{p}.csv", index=False)

        # call shares by corpus: ADR main vs OURS
        cs = []
        for c in sorted(set(corpus)):
            mk = corpus == c
            for name, sel in (("ADR:main", R["sel__main"].astype(int)),
                              ("OURS", Z["sel__OURS"].astype(int))):
                cs.append({"corpus": c, "system": name,
                           **{a: float((sel[mk] == i).mean()) for i, a in enumerate(POOL)}})
        CS = pd.DataFrame(cs)
        CS.to_csv(OUT / "report" / f"CALLS_BY_CORPUS_{p}.csv", index=False)
        th = pd.read_csv(OUT / p / f"ADR_{p}_RECORDS.csv")
        TH = (th[th["variant"].isin(["main", "nobudget"])]
              .groupby("variant")[["theta_med", "theta_high", "train_auroc", "train_ms",
                                   "fallback_rows_eval"]].describe().round(4))
        TH.to_csv(OUT / "report" / f"THRESHOLDS_{p}.csv")
        lines += [f"## Protocol {p}", "", "```", T.round(4).to_string(index=False), "```", "",
                  "Calls by corpus", "", "```", CS.round(4).to_string(index=False), "```", ""]
    (OUT / "gates" / "GATES_ALL.json").write_text(json.dumps(gates, indent=2))
    g2 = {k: v for k, v in gates.items() if k.startswith("G2_")}
    lines += ["## Gates", "", "```", json.dumps(g2, indent=2), "```"]
    (OUT / "report" / "REPORT.md").write_text("\n".join(lines))
    log("report written; G2 max |dAUROC| per corner: " + json.dumps(g2))


STAGES = {"features": stage_features, "hpselect": stage_hpselect, "gates": stage_gates,
          "B": stage_B, "A": stage_A, "report": stage_report}

if __name__ == "__main__":
    for st in sys.argv[1:] or ["features", "hpselect", "gates", "B", "A", "report"]:
        log(f"===== stage {st} =====")
        STAGES[st]()
