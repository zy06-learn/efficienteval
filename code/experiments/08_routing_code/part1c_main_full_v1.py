#!/usr/bin/env python3
"""part1c: the main experiment with hyperparameters reselected and end-to-end latency charged.

Inherited verbatim from the part1 contract, asserted field by field before anything is fitted,
never re-searched here:

    pool        factcc, lettuce_v2, granite_guardian_3_1_2b
    features    the six unified cheap features
    target      regret
    learner     rf
    beta        grid and the "cheapest within 0.005 of best validation AUROC" rule
    seeds       the ten declared seeds
    splits      grouped by content_doc_key, identical construction to part1

Selected by this run, on TRAIN only:

    hyperparameters   sixteen randomly initialised candidates, one per protocol chosen by
                      minimum validation head loss under that protocol's own CV shape

Changed relative to part1b:

  1. End-to-end latency. Every millisecond the deployed system spends is charged: base feature
     extraction, Compact16 feature extraction, random-forest head inference, the routing
     arithmetic, both calibration stages, and the verifier call. part1 and part1b charged only
     the verifier call plus base feature extraction. Fixed verifiers compute no features and run
     no heads, so their end-to-end cost is the verifier call plus calibration -- that is correct
     accounting, not a handicap, and it is what the tables report.
  2. Hyperparameters are selected here rather than inherited. part1 carried `hyperparameters_A`
     from a first-round hard-coded default and `hyperparameters_B` from a second-round minimum-loss
     search over a different feature set; neither had been reselected for the current six
     features.
  3. `OURS_current_pool` is gone. It compared a pool that an earlier automatic search had
     chosen, evaluated with features selected for a different pool, and is not part of this
     experiment.
  4. `OURS_legacy_hp` replaces it as a drift sentinel: the same router under part1's frozen
     hyperparameters, which must reproduce part1's numbers exactly. Since the router's
     hyperparameters are the only thing that changed, this is the check that the primitives --
     splits, Platt, the regret target, head fitting, routing, beta selection, isotonic -- have
     not moved.

Stages: preflight | hpselect | protoB | protoA | report | all
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3core as V  # noqa: E402
import core  # noqa: E402

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
PART1 = ROOT / "experiments" / "cross_stage_contract" / "part1_main_pooled_v1"
RUN = Path(os.environ["V3_RUN_DIR"]).resolve()
RES = RUN / "results"
RES.mkdir(parents=True, exist_ok=True)
(RUN / "logs").mkdir(parents=True, exist_ok=True)
V.RES = RES
SMOKE = os.environ.get("V3_SMOKE", "0") == "1"

# ------------------------------------------------------------------ inherited contract
CONTRACT = json.loads((PART1 / "00_contract" / "FROZEN_v3.json").read_text())
POOL = list(CONTRACT["pool"])
FEATURES = list(CONTRACT["features"])
TARGET = CONTRACT["target"]
LEARNER = CONTRACT["learner"]
LEGACY_HP_A = dict(CONTRACT["hyperparameters_A"])
LEGACY_HP_B = dict(CONTRACT["hyperparameters_B"])
SEEDS = list(V.SEEDS[:2] if SMOKE else V.SEEDS)
HP_SEEDS = list(V.SEEDS[:1] if SMOKE else V.SEEDS[:5])
DRAWS = 200 if SMOKE else int(V.C.BOOTSTRAP_DRAWS)
N_ROTATIONS = len(V.rotations())

EXPECTED_CONTRACT = {
    "pool": ["factcc", "lettuce_v2", "granite_guardian_3_1_2b"],
    "features": ["structured_source_line_ratio", "bm25_mean3", "entity_coverage",
                 "entity_value_colocation", "year_count", "conflicting_value_rate"],
    "target": "regret",
    "learner": "rf",
    "beta_grid": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2],
}

# part1's frozen router values, read at full precision. `OURS_legacy_hp` must land on these.
LEGACY_REFERENCE = {
    proto: {"auroc": float(r["auroc"]), "ms": float(r["ms"])}
    for proto in ("A", "B")
    for _i, r in pd.read_csv(PART1 / "01_main_tables" / "publication"
                             / f"{proto}_MAIN.csv").iterrows()
    if r["system"] == "OURS"
}
G1_TOL_AUROC, G1_TOL_MS = 1e-9, 1e-6

CONFORMAL_DELTA = 0.10
PRIMARY_RULE = "mcc"
RULES = ("fixed05", "mcc", "youden", "conformal")
HP_FILE = RES / "HP_SELECTED.json"


# ------------------------------------------------------------------ threshold rules
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


# ------------------------------------------------------------------ latency accounting
def _pre_call_feature_ms(frame):
    """Everything the router must compute before it may choose an action.

    The six frozen features span two extractors and both are charged in full:
    `feature_latency_ms` is the base extractor (itself an exact sum of
    `feature_query_latency_ms` and `feature_document_setup_ms`) and
    `compact16_feature_latency_ms` is the second one.
    """
    return (frame["feature_latency_ms"].to_numpy(float)
            + frame["compact16_feature_latency_ms"].to_numpy(float))


def _verifier_ms(frame, actions, sel):
    lat = np.column_stack([frame[f"latency_ms__{a}"].fillna(0).to_numpy(float) for a in actions])
    return lat[np.arange(len(frame)), np.asarray(sel, int)]


# ------------------------------------------------------------------ one fitted system
def _stage2(p_val, y_val, g_val, p_ev):
    """Shared post-processing. Returns stage-2 probabilities, both threshold sets, and the
    measured per-row cost of applying the calibration chain."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
    q_val = np.clip(iso.predict(p_val), 1e-6, 1 - 1e-6)
    t0 = time.perf_counter()
    q_ev = np.clip(iso.predict(p_ev), 1e-6, 1 - 1e-6)
    apply_ms = (time.perf_counter() - t0) * 1000.0 / max(len(p_ev), 1)
    thr1 = {r: float(RULE_FN[r](p_val, y_val, g_val)) for r in RULES}
    thr2 = {r: float(RULE_FN[r](q_val, y_val, g_val)) for r in RULES}
    return q_ev, thr2, thr1, float(apply_ms)


def router_raw(fit, val, evalf, seed, hp):
    """Pre-stage-2 router outputs plus every measured cost component."""
    t0 = time.perf_counter()
    cals = core.platt(fit, actions=POOL)
    c_fit = core.apply_platt(fit, cals, actions=POOL)
    c_val = core.apply_platt(val, cals, actions=POOL)
    t1 = time.perf_counter()
    c_ev = core.apply_platt(evalf, cals, actions=POOL)
    platt_ms = (time.perf_counter() - t1) * 1000.0 / max(len(evalf), 1)

    h_val, h_ev = core.fit_heads(fit, [val, evalf], seed, c_fit, actions=POOL,
                                 features=FEATURES, hp=hp, target=TARGET, learner=LEARNER)
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    beta, ledger = V.choose_beta(val, h_val, c_val, POOL, cvec)
    _s, p_val, _m = V.route(val, h_val, c_val, beta, POOL, cvec)

    t2 = time.perf_counter()
    sel, p_ev, ms_route_out = V.route(evalf, h_ev, c_ev, beta, POOL, cvec)
    route_ms = (time.perf_counter() - t2) * 1000.0 / max(len(evalf), 1)

    ver_ms = _verifier_ms(evalf, POOL, sel)
    if not np.isfinite(ver_ms).all():
        raise AssertionError("router selected an action with a non-finite latency")
    # part1's definition, kept only so the legacy-hp arm can be checked against it
    ms_part1 = ver_ms + evalf["feature_latency_ms"].to_numpy(float)
    if not np.allclose(ms_part1, ms_route_out, rtol=0, atol=1e-9):
        raise AssertionError("part1 latency reconstruction disagrees with v3core.route")
    return {"p_val": p_val, "p_ev": p_ev, "sel": np.asarray(sel, int), "beta": float(beta),
            "ledger": ledger, "ver_ms": ver_ms, "ms_part1": ms_part1,
            "head_ms": float(h_ev.predict_ms), "route_ms": float(route_ms),
            "platt_ms": float(platt_ms), "fit_ms": (t1 - t0) * 1000.0}


def fixed_raw(fit, val, evalf, v):
    cal = core.platt(fit, actions=[v])
    p_val = core.apply_platt(val, cal, actions=[v])[v]
    t0 = time.perf_counter()
    p_ev = core.apply_platt(evalf, cal, actions=[v])[v]
    platt_ms = (time.perf_counter() - t0) * 1000.0 / max(len(evalf), 1)
    ver_ms = evalf[f"latency_ms__{v}"].to_numpy(float)
    if not np.isfinite(ver_ms).all():
        raise AssertionError(f"fixed::{v} has a non-finite latency on the evaluation frame")
    return {"p_val": p_val, "p_ev": p_ev, "ver_ms": ver_ms, "ms_part1": ver_ms,
            "platt_ms": float(platt_ms)}


def _key(system):
    return system.replace("::", "__")


def _str(values):
    return np.asarray(values, dtype=np.str_)


def systems_for(verifiers):
    return ["OURS", "OURS_legacy_hp"] + [f"fixed::{v}" for v in verifiers]


# ------------------------------------------------------------------ preflight
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:                                        # pragma: no cover
        return f"<unavailable: {exc}>"


def preflight():
    core.log("===== preflight =====")
    for key, want in EXPECTED_CONTRACT.items():
        got = list(CONTRACT[key]) if isinstance(want, list) else CONTRACT[key]
        if got != want:
            raise AssertionError(f"contract drift on {key}: frozen={got} expected={want}")
    core.log(f"inherited contract ok: pool={POOL} features={len(FEATURES)} "
             f"target={TARGET} learner={LEARNER}")

    # --- stale-content audit: nothing from the earlier rounds may leak in through a default
    if list(core.ACTIONS) != POOL:
        raise AssertionError(f"shared default pool {core.ACTIONS} != current pool {POOL}")
    if core.C.TARGET != TARGET or core.C.LEARNER != LEARNER:
        raise AssertionError("shared default target/learner disagree with the contract")
    stale_features = [f for f in core.C.FEATURES if f not in FEATURES]
    stale_hp = dict(core.C.HP)
    core.log(f"stale-default audit: shared carries {len(core.C.FEATURES)} features "
             f"({len(stale_features)} not in ours) and hp {stale_hp}; every call site in this "
             f"file passes features/hp/target/actions explicitly, so none of it is reachable")
    if "comparison_pool" in CONTRACT:
        core.log(f"note: inherited contract still lists comparison_pool "
                 f"{CONTRACT['comparison_pool']} and secondary_pool "
                 f"{CONTRACT.get('secondary_pool')}; both are ignored by this run")

    TRAIN, TEST, ALL, verifiers = V.load(with_test_labels=True)
    if list(verifiers) != list(V.C.VERIFIERS):
        raise AssertionError(f"verifier set drift: {verifiers} != {list(V.C.VERIFIERS)}")
    if not all(a in verifiers for a in POOL):
        raise AssertionError("a pool member is missing from the verifier set")
    core.log(f"verifier set ok: {len(verifiers)} verifiers, pool members all present")
    core.log(f"declared contamination flags (out of pool, reported in table notes): "
             f"{sorted(V.C.CONFIRMED_TRAIN)}")

    exp = V.C.EXPECTED
    for name, got, want in [("train rows", len(TRAIN), exp["train_rows"]),
                            ("train groups", TRAIN.content_doc_key.nunique(), exp["train_groups"]),
                            ("test rows", len(TEST), exp["test_rows"]),
                            ("test groups", TEST.content_doc_key.nunique(), exp["test_groups"]),
                            ("all rows", len(ALL), exp["train_rows"] + exp["test_rows"]),
                            ("all groups", ALL.content_doc_key.nunique(),
                             exp["train_groups"] + exp["test_groups"])]:
        if int(got) != int(want):
            raise AssertionError(f"{name}: got {got}, expected {want}")
        core.log(f"  {name}: {got}")

    if set(TRAIN.content_doc_key.astype(str)) & set(TEST.content_doc_key.astype(str)):
        raise AssertionError("documents straddle TRAIN/TEST")

    lat_cols = ["feature_latency_ms", "feature_query_latency_ms",
                "feature_document_setup_ms", "compact16_feature_latency_ms"]
    for name, frame in (("TRAIN", TRAIN), ("TEST", TEST), ("ALL", ALL)):
        missing = [f for f in FEATURES if f not in frame.columns]
        if missing:
            raise AssertionError(f"{name} is missing features {missing}")
        if not np.isfinite(frame[FEATURES].to_numpy(float)).all():
            raise AssertionError(f"{name} has non-finite feature values")
        for c in lat_cols:
            if c not in frame.columns:
                raise AssertionError(f"{name} is missing latency column {c}")
            if not np.isfinite(frame[c].to_numpy(float)).all():
                raise AssertionError(f"{name}/{c} has non-finite values")
        resid = (frame["feature_latency_ms"] - frame["feature_query_latency_ms"]
                 - frame["feature_document_setup_ms"]).abs().max()
        if resid > 1e-6:
            raise AssertionError(f"{name}: base feature latency is not query + setup "
                                 f"(max residual {resid})")
        avail = np.column_stack([frame[f"available__{a}"].to_numpy(bool) for a in POOL])
        if int((~avail.any(1)).sum()):
            raise AssertionError(f"{name} has rows with no available pool action")
        for a in POOL:
            m = frame[f"available__{a}"].to_numpy(bool)
            if np.isnan(frame[f"latency_ms__{a}"].to_numpy(float)[m]).any():
                raise AssertionError(f"{name}/{a}: available row with NaN latency")
        if not frame.episode_key.is_unique:
            raise AssertionError(f"{name} has duplicate episode_key")
    core.log("  features, latency columns, additivity, availability and key uniqueness: ok")

    prov = RUN / "04_provenance"
    prov.mkdir(parents=True, exist_ok=True)
    data_inputs = [ROOT / "ingest_and_scoring" / "data" / "TRAIN.parquet",
                   ROOT / "ingest_and_scoring" / "data" / "TEST.parquet",
                   ROOT / "ingest_and_scoring" / "data" / "TEST_SCORING.parquet"]
    data_inputs += sorted((ROOT / "ingest_and_scoring" / "results" / "p1_scoring").glob("*.parquet"))
    (prov / "DATA_INPUTS.sha256").write_text(
        "".join(f"{_sha256(p)}  {p.relative_to(ROOT)}\n" for p in data_inputs if p.exists()))
    code = [Path(__file__).resolve(), ROOT / "experiments" / "v3core.py",
            ROOT / "shared" / "core.py", ROOT / "shared" / "config.py",
            ROOT / "ingest_and_scoring" / "config_v2.py", ROOT / "verifiers" / "tenfold_v1.py",
            ROOT / "verifiers" / "pool_gate_sweep_v1.py",
            ROOT / "verifiers" / "summary_router_compact16_direct_v1.py"]
    (prov / "CODE_SNAPSHOT.sha256").write_text(
        "".join(f"{_sha256(p)}  {p}\n" for p in code if p.exists()))
    (prov / "GIT_STATE.txt").write_text(
        f"HEAD: {_git('rev-parse', 'HEAD')}\n\nstatus --porcelain:\n"
        f"{_git('status', '--porcelain')}\n")
    (prov / "RUN_METADATA.txt").write_text(
        f"cwd={Path.cwd()}\ncommand={' '.join(sys.argv)}\n"
        f"started={time.strftime('%Y-%m-%d %H:%M:%S%z')}\npython={sys.version}\n"
        f"platform={platform.platform()}\nnproc={os.cpu_count()}\nsmoke={SMOKE}\n")
    (prov / "PID").write_text(f"{os.getpid()}\n")
    core.log("preflight ok")


# ------------------------------------------------------------------ hyperparameter selection
def _head_loss(h_val, val, c_val):
    """Mean squared error of the head predictions against the regret target on validation.

    Identical in definition to the loss the earlier selection stages minimised, so the numbers
    here are directly comparable to `ingest_and_scoring/results/P5_S4_HYPERPARAMS.csv`.
    """
    Y, _clf = core.targets(val, c_val, POOL, TARGET)
    return float(np.mean((np.asarray(h_val, float) - np.asarray(Y, float)) ** 2))


def hpselect():
    """Sixteen randomly initialised candidates, scored on TRAIN only, one chosen per protocol
    by minimum validation head loss under that protocol's own cross-validation shape.

    Selecting on head loss keeps the selection quantity disjoint from the metric the paper
    reports. The trade-off is recorded rather than hidden: `freeze_v3.py` observed that head
    loss and routing AUROC are positively rank-correlated here, so the minimum-loss region is
    the shallow-forest region, and both the loss and the AUROC of every candidate are written
    to `HP_SELECTION.csv` so the cost of the rule is visible.
    """
    core.log("===== hyperparameter selection (TRAIN only, minimum validation head loss) =====")
    TRAIN, TEST_NL, _ALL, _v = V.load(with_test_labels=False)
    assert "label_supported" not in TEST_NL.columns, "TEST frame must stay label free"
    rots = V.rotations()[:2] if SMOKE else V.rotations()

    rng = np.random.default_rng(0)
    grid, seen = [], set()
    for _ in range(16):
        h = {"n_estimators": int(rng.choice([200, 400, 800])),
             "min_samples_leaf": int(rng.choice([1, 2, 5, 10])),
             "max_features": str(rng.choice(["sqrt", "log2", "0.5"])),
             "max_depth": int(rng.choice([6, 10, 12, 16, 24]))}
        k = json.dumps(h, sort_keys=True)
        if k not in seen:
            seen.add(k)
            grid.append(h)
    legacy = {"legacy_A": dict(LEGACY_HP_A, max_features=str(LEGACY_HP_A["max_features"])),
              "legacy_B": dict(LEGACY_HP_B, max_features=str(LEGACY_HP_B["max_features"]))}
    for _tag, h in legacy.items():
        if json.dumps(h, sort_keys=True) not in seen:
            seen.add(json.dumps(h, sort_keys=True))
            grid.append(dict(h))
    core.log(f"{len(grid)} candidates (16 random draws deduplicated, plus part1's two values "
             f"as reference points)")

    def _hp(h):
        mf = str(h["max_features"])
        return dict(h, max_features=(0.5 if mf == "0.5" else mf))

    rows = []
    for i, h in enumerate(grid, 1):
        t0 = time.time()
        hp = _hp(h)
        b_loss, b_auroc = [], []
        for seed in HP_SEEDS:
            held = V.stratified_group_split(TRAIN, seed)
            fit = TRAIN.loc[~held].reset_index(drop=True)
            val = TRAIN.loc[held].reset_index(drop=True)
            cals = core.platt(fit, actions=POOL)
            c_fit = core.apply_platt(fit, cals, actions=POOL)
            c_val = core.apply_platt(val, cals, actions=POOL)
            # one fit serves both the loss and the routing measurement
            (h_val,) = core.fit_heads(fit, [val], seed, c_fit, actions=POOL, features=FEATURES,
                                      hp=hp, target=TARGET, learner=LEARNER)
            b_loss.append(_head_loss(h_val, val, c_val))
            cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
            beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
            _s, p, _m = V.route(val, h_val, c_val, beta, POOL, cvec)
            b_auroc.append(float(roc_auc_score(val.label_supported.to_numpy(int), p)))

        a_loss, a_auroc = [], []
        for seed in HP_SEEDS:
            fold = V.folds_stratified(TRAIN, seed)
            losses, probs, labels = [], [], []
            for t, vf, trf in rots:
                fit = TRAIN.loc[np.isin(fold, trf)].reset_index(drop=True)
                val = TRAIN.loc[fold == vf].reset_index(drop=True)
                ev = TRAIN.loc[fold == t].reset_index(drop=True)
                cals = core.platt(fit, actions=POOL)
                c_fit = core.apply_platt(fit, cals, actions=POOL)
                c_val = core.apply_platt(val, cals, actions=POOL)
                c_ev = core.apply_platt(ev, cals, actions=POOL)
                h_val, h_ev = core.fit_heads(fit, [val, ev], seed, c_fit, actions=POOL,
                                             features=FEATURES, hp=hp, target=TARGET,
                                             learner=LEARNER)
                losses.append(_head_loss(h_val, val, c_val))
                cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
                beta, _ = V.choose_beta(val, h_val, c_val, POOL, cvec)
                _s, p, _m = V.route(ev, h_ev, c_ev, beta, POOL, cvec)
                probs.append(p)
                labels.append(ev.label_supported.to_numpy(int))
            a_loss.append(float(np.mean(losses)))
            a_auroc.append(float(roc_auc_score(np.concatenate(labels), np.concatenate(probs))))

        tag = next((t for t, f in legacy.items()
                    if all(str(f[k]) == str(h[k]) for k in h)), "")
        rows.append({**h, "part1_value": tag,
                     "a_val_loss": float(np.mean(a_loss)), "a_val_auroc": float(np.mean(a_auroc)),
                     "a_sd": float(np.std(a_auroc)),
                     "b_val_loss": float(np.mean(b_loss)), "b_val_auroc": float(np.mean(b_auroc)),
                     "b_sd": float(np.std(b_auroc))})
        core.log(f"  {i}/{len(grid)} {h} | A loss={rows[-1]['a_val_loss']:.6f} "
                 f"auroc={rows[-1]['a_val_auroc']:.5f} | B loss={rows[-1]['b_val_loss']:.6f} "
                 f"auroc={rows[-1]['b_val_auroc']:.5f}  ({time.time()-t0:.0f}s)")

    D = pd.DataFrame(rows)
    D["rank_a_by_loss"] = D["a_val_loss"].rank().astype(int)
    D["rank_a_by_auroc"] = D["a_val_auroc"].rank(ascending=False).astype(int)
    D["rank_b_by_loss"] = D["b_val_loss"].rank().astype(int)
    D["rank_b_by_auroc"] = D["b_val_auroc"].rank(ascending=False).astype(int)
    V.save("HP_SELECTION.csv", D.sort_values("b_val_loss").reset_index(drop=True))

    pick_a = _hp(D.loc[D["a_val_loss"].idxmin()].to_dict())
    pick_b = _hp(D.loc[D["b_val_loss"].idxmin()].to_dict())
    pick_a = {k: pick_a[k] for k in ("n_estimators", "min_samples_leaf",
                                     "max_features", "max_depth")}
    pick_b = {k: pick_b[k] for k in ("n_estimators", "min_samples_leaf",
                                     "max_features", "max_depth")}
    for p in (pick_a, pick_b):
        p["n_estimators"] = int(p["n_estimators"])
        p["min_samples_leaf"] = int(p["min_samples_leaf"])
        p["max_depth"] = int(p["max_depth"])
    rho_a = float(D["a_val_loss"].corr(D["a_val_auroc"], method="spearman"))
    rho_b = float(D["b_val_loss"].corr(D["b_val_auroc"], method="spearman"))
    cost_a = float(D["a_val_auroc"].max() - D.loc[D["a_val_loss"].idxmin(), "a_val_auroc"])
    cost_b = float(D["b_val_auroc"].max() - D.loc[D["b_val_loss"].idxmin(), "b_val_auroc"])

    payload = {"rule": "minimum validation head loss (MSE against the regret target), "
                       "TRAIN only, 16 randomly initialised candidates",
               "candidate_draw": "numpy default_rng(0) over n_estimators {200,400,800}, "
                                 "min_samples_leaf {1,2,5,10}, max_features {sqrt,log2,0.5}, "
                                 "max_depth {6,10,12,16,24}, deduplicated",
               "seeds": HP_SEEDS, "hyperparameters_A": pick_a, "hyperparameters_B": pick_b,
               "spearman_loss_vs_auroc_A": rho_a, "spearman_loss_vs_auroc_B": rho_b,
               "auroc_forgone_by_loss_rule_A": cost_a,
               "auroc_forgone_by_loss_rule_B": cost_b,
               "part1_values": {"A": LEGACY_HP_A, "B": LEGACY_HP_B}}
    HP_FILE.write_text(json.dumps(payload, indent=2))
    core.log(f"selected HP_A = {pick_a}")
    core.log(f"selected HP_B = {pick_b}")
    core.log(f"Spearman(loss, auroc): A {rho_a:+.3f}  B {rho_b:+.3f}")
    core.log(f"AUROC forgone by selecting on loss instead of AUROC: "
             f"A {cost_a:+.5f}  B {cost_b:+.5f}")
    print(D.sort_values("b_val_loss").to_string(index=False,
                                                float_format=lambda v: f"{v:.6f}"))


def _selected_hp():
    if not HP_FILE.exists():
        raise SystemExit("run the hpselect stage first: HP_SELECTED.json is missing")
    d = json.loads(HP_FILE.read_text())
    return dict(d["hyperparameters_A"]), dict(d["hyperparameters_B"])


# ------------------------------------------------------------------ protocols
def _assemble(raw, pre_ms, is_router):
    """End-to-end per-row latency. The router pays for features, heads, routing arithmetic and
    both calibration stages; a fixed verifier computes no features and runs no heads, so it pays
    for its call and the calibration applied to it."""
    overhead = raw["platt_ms"] + raw.get("s2_ms", 0.0)
    if is_router:
        return raw["ver_ms"] + pre_ms + raw["head_ms"] + raw["route_ms"] + overhead
    return raw["ver_ms"] + overhead


def _pack(store, name, key, arr):
    store[f"{key}__{_key(name)}"] = arr


def _run_seed(fit, val, evalf, seed, hp_primary, hp_legacy, verifiers, pre_ms):
    """One (fit, val, eval) triple through every system. Returns per-system dicts."""
    y_val = val["label_supported"].to_numpy(int)
    g_val = val["content_doc_key"].astype(str).to_numpy()
    out = {}
    for label, hp in (("OURS", hp_primary), ("OURS_legacy_hp", hp_legacy)):
        raw = router_raw(fit, val, evalf, seed, hp)
        q_ev, t2, t1, s2_ms = _stage2(raw["p_val"], y_val, g_val, raw["p_ev"])
        raw["s2_ms"] = s2_ms
        out[label] = {"prob": q_ev, "prob1": raw["p_ev"], "thr2": t2, "thr1": t1,
                      "ms_full": _assemble(raw, pre_ms, True), "ms_part1": raw["ms_part1"],
                      "sel": raw["sel"], "beta": raw["beta"], "ledger": raw["ledger"],
                      "components": {"n": len(evalf), "verifier": float(raw["ver_ms"].mean()),
                                     "features": float(pre_ms.mean()),
                                     "heads": raw["head_ms"], "routing": raw["route_ms"],
                                     "platt": raw["platt_ms"], "isotonic": s2_ms}}
    for v in verifiers:
        raw = fixed_raw(fit, val, evalf, v)
        q_ev, t2, t1, s2_ms = _stage2(raw["p_val"], y_val, g_val, raw["p_ev"])
        raw["s2_ms"] = s2_ms
        out[f"fixed::{v}"] = {"prob": q_ev, "prob1": raw["p_ev"], "thr2": t2, "thr1": t1,
                              "ms_full": _assemble(raw, pre_ms, False),
                              "ms_part1": raw["ms_part1"], "sel": None, "beta": None,
                              "ledger": None,
                              "components": {"n": len(evalf),
                                             "verifier": float(raw["ver_ms"].mean()),
                                             "features": 0.0, "heads": 0.0, "routing": 0.0,
                                             "platt": raw["platt_ms"], "isotonic": s2_ms}}
    return out


def proto_b():
    core.log("===== Protocol B =====")
    hp_a, hp_b = _selected_hp()
    core.log(f"primary HP_B = {hp_b} | legacy HP_B = {LEGACY_HP_B}")
    TRAIN, TEST, _ALL, verifiers = V.load(with_test_labels=True)
    y = TEST["label_supported"].to_numpy(int)
    n, S = len(TEST), len(SEEDS)
    names = systems_for(verifiers)
    pre_ms = _pre_call_feature_ms(TEST)

    acc = {s: {"prob": np.zeros((n, S)), "prob1": np.zeros((n, S)),
               "ms_full": np.zeros((n, S)), "ms_part1": np.zeros((n, S)),
               **{f"thr_{r}": np.zeros((n, S)) for r in RULES},
               **{f"thr1_{r}": np.zeros((n, S)) for r in RULES}} for s in names}
    sel = np.zeros((n, S), np.int8)
    ledger_rows, beta_rows, comp_rows = [], [], []

    for si, seed in enumerate(SEEDS):
        t0 = time.time()
        held = V.stratified_group_split(TRAIN, seed)
        fit = TRAIN.loc[~held].reset_index(drop=True)
        val = TRAIN.loc[held].reset_index(drop=True)
        if set(fit.content_doc_key.astype(str)) & set(val.content_doc_key.astype(str)):
            raise AssertionError("fit and validation share a document")
        res = _run_seed(fit, val, TEST, seed, hp_b, LEGACY_HP_B, verifiers, pre_ms)
        for s in names:
            r = res[s]
            for k in ("prob", "prob1", "ms_full", "ms_part1"):
                acc[s][k][:, si] = r[k]
            for rule in RULES:
                acc[s][f"thr_{rule}"][:, si] = r["thr2"][rule]
                acc[s][f"thr1_{rule}"][:, si] = r["thr1"][rule]
            comp_rows.append({"protocol": "B", "seed": seed, "system": s, **r["components"]})
        sel[:, si] = res["OURS"]["sel"]
        beta_rows.append({"seed": seed, "beta": res["OURS"]["beta"],
                          "beta_legacy_hp": res["OURS_legacy_hp"]["beta"]})
        for b, auroc, mms in res["OURS"]["ledger"]:
            ledger_rows.append({"protocol": "B", "seed": seed, "beta": b,
                                "val_auroc": auroc, "val_ms": mms})
        core.log(f"  seed {seed}: OURS {roc_auc_score(y, acc['OURS']['prob'][:, si]):.7f} "
                 f"@ {acc['OURS']['ms_full'][:, si].mean():.4f} ms end-to-end | legacy_hp "
                 f"{roc_auc_score(y, acc['OURS_legacy_hp']['prob'][:, si]):.7f} "
                 f"@ {acc['OURS_legacy_hp']['ms_part1'][:, si].mean():.4f} ms part1-basis "
                 f"({time.time()-t0:.0f}s)")

    store = {"y": y.astype(np.int8),
             "groups": _str(TEST["content_doc_key"].astype(str).to_numpy()),
             "corpus": _str(TEST["dataset_key"].astype(str).to_numpy()),
             "episode_key": _str(TEST["episode_key"].astype(str).to_numpy()),
             "systems": _str(names), "seeds": np.array(SEEDS), "actions": _str(POOL),
             "sel__OURS": sel}
    for s in names:
        for k, arr in acc[s].items():
            _pack(store, s, k, arr)
    np.savez_compressed(RES / "probs_B.npz", **store)
    V.save("B_BETA_LEDGER.csv", pd.DataFrame(ledger_rows))
    V.save("B_BETA_SELECTED.csv", pd.DataFrame(beta_rows))
    V.save("B_LATENCY_COMPONENTS.csv", pd.DataFrame(comp_rows))
    core.log(f"Protocol B stored: {n} rows x {S} seeds x {len(names)} systems")


def proto_a():
    core.log("===== Protocol A =====")
    hp_a, hp_b = _selected_hp()
    core.log(f"primary HP_A = {hp_a} | legacy HP_A = {LEGACY_HP_A}")
    _TRAIN, _TEST, ALL, verifiers = V.load(with_test_labels=True)
    n, S = len(ALL), len(SEEDS)
    names = systems_for(verifiers)
    rotations = V.rotations()[:N_ROTATIONS]
    slots = (["prob", "prob1", "ms_full", "ms_part1"]
             + [f"thr_{r}" for r in RULES] + [f"thr1_{r}" for r in RULES])

    acc = {s: {k: np.zeros((n, S)) for k in slots} for s in names}
    sel = np.zeros((n, S), np.int8)
    y_ref = g_ref = ds_ref = ek_ref = None
    ledger_rows, beta_rows, comp_rows = [], [], []

    for si, seed in enumerate(SEEDS):
        t0 = time.time()
        fold = V.folds_stratified(ALL, seed)
        buf = {s: {k: [] for k in slots} for s in names}
        sel_buf, ys, gs, dss, eks = [], [], [], [], []
        for test_fold, val_fold, train_folds in rotations:
            fit = ALL.loc[np.isin(fold, train_folds)].reset_index(drop=True)
            val = ALL.loc[fold == val_fold].reset_index(drop=True)
            ev = ALL.loc[fold == test_fold].reset_index(drop=True)
            for a, b in ((fit, val), (fit, ev), (val, ev)):
                if set(a.content_doc_key.astype(str)) & set(b.content_doc_key.astype(str)):
                    raise AssertionError("rotation partitions share a document")
            res = _run_seed(fit, val, ev, seed, hp_a, LEGACY_HP_A, verifiers,
                            _pre_call_feature_ms(ev))
            m = len(ev)
            for s in names:
                r = res[s]
                for k in ("prob", "prob1", "ms_full", "ms_part1"):
                    buf[s][k].append(np.asarray(r[k], float))
                for rule in RULES:
                    buf[s][f"thr_{rule}"].append(np.full(m, r["thr2"][rule], float))
                    buf[s][f"thr1_{rule}"].append(np.full(m, r["thr1"][rule], float))
                comp_rows.append({"protocol": "A", "seed": seed, "rotation": test_fold,
                                  "system": s, **r["components"]})
            sel_buf.append(res["OURS"]["sel"])
            beta_rows.append({"seed": seed, "rotation": test_fold,
                              "beta": res["OURS"]["beta"],
                              "beta_legacy_hp": res["OURS_legacy_hp"]["beta"]})
            for b, auroc, mms in res["OURS"]["ledger"]:
                ledger_rows.append({"protocol": "A", "seed": seed, "rotation": test_fold,
                                    "beta": b, "val_auroc": auroc, "val_ms": mms})
            ys.append(ev["label_supported"].to_numpy(int))
            gs.append(ev["content_doc_key"].astype(str).to_numpy())
            dss.append(ev["dataset_key"].astype(str).to_numpy())
            eks.append(ev["episode_key"].astype(str).to_numpy())

        order = np.argsort(np.concatenate(eks), kind="stable")
        for s in names:
            for k in slots:
                acc[s][k][:, si] = np.concatenate(buf[s][k])[order]
        sel[:, si] = np.concatenate(sel_buf)[order]
        y_now, ek_now = np.concatenate(ys)[order], np.concatenate(eks)[order]
        if y_ref is None:
            y_ref, ek_ref = y_now, ek_now
            g_ref, ds_ref = np.concatenate(gs)[order], np.concatenate(dss)[order]
        elif not (np.array_equal(y_ref, y_now) and np.array_equal(ek_ref, ek_now)):
            raise AssertionError("canonical row order differs between seeds")
        core.log(f"  seed {seed}: OURS {roc_auc_score(y_ref, acc['OURS']['prob'][:, si]):.7f} "
                 f"@ {acc['OURS']['ms_full'][:, si].mean():.4f} ms end-to-end | legacy_hp "
                 f"{roc_auc_score(y_ref, acc['OURS_legacy_hp']['prob'][:, si]):.7f} "
                 f"@ {acc['OURS_legacy_hp']['ms_part1'][:, si].mean():.4f} ms part1-basis "
                 f"({time.time()-t0:.0f}s)")

    store = {"y": y_ref.astype(np.int8), "groups": _str(g_ref), "corpus": _str(ds_ref),
             "episode_key": _str(ek_ref), "systems": _str(names),
             "seeds": np.array(SEEDS), "actions": _str(POOL), "sel__OURS": sel}
    for s in names:
        for k, arr in acc[s].items():
            _pack(store, s, k, arr)
    np.savez_compressed(RES / "probs_A.npz", **store)
    V.save("A_BETA_LEDGER.csv", pd.DataFrame(ledger_rows))
    V.save("A_BETA_SELECTED.csv", pd.DataFrame(beta_rows))
    V.save("A_LATENCY_COMPONENTS.csv", pd.DataFrame(comp_rows))
    core.log(f"Protocol A stored: {n} rows x {S} seeds x {len(names)} systems")


# ------------------------------------------------------------------ reporting
def _agg(y, P, MS, THR):
    rows = [core.metrics(y, P[:, s], MS[:, s], threshold=THR[:, s]) for s in range(P.shape[1])]
    out = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    out["auroc_sd"] = float(np.std([r["auroc"] for r in rows]))
    return out


def _per_corpus(y, P, ds):
    out, vals, sizes = {}, [], []
    for d in sorted(set(ds.tolist())):
        m = ds == d
        if len(set(y[m].tolist())) < 2:
            continue
        a = float(np.mean([roc_auc_score(y[m], P[m, s]) for s in range(P.shape[1])]))
        out[f"auroc__{d}"] = a
        vals.append(a)
        sizes.append(int(m.sum()))
    out["macro_auroc"] = float(np.mean(vals))
    out["macro_auroc_sample_weighted"] = float(np.average(vals, weights=sizes))
    out["worst_corpus_auroc"] = float(np.min(vals))
    return out


def _risk_coverage_curve(y, P, THR):
    grid = np.linspace(0.02, 1.0, 50)
    curves = []
    for s in range(P.shape[1]):
        p = P[:, s]
        ok = ((p >= THR[:, s]).astype(int) == y).astype(int)
        ok = ok[np.argsort(-np.abs(p - 0.5), kind="stable")]
        run = np.cumsum(ok) / np.arange(1, len(ok) + 1)
        idx = np.maximum((grid * len(ok)).round().astype(int), 1) - 1
        curves.append(1.0 - run[idx])
    return grid, np.mean(curves, axis=0)


def report(protocol):
    core.log(f"===== report {protocol} =====")
    z = np.load(RES / f"probs_{protocol}.npz", allow_pickle=False)
    names = [str(x) for x in z["systems"]]
    y = z["y"].astype(int)
    g = z["groups"].astype(str)
    ds = z["corpus"].astype(str)
    actions = [str(a) for a in z["actions"]]
    P = {s: z[f"prob__{_key(s)}"] for s in names}
    MS = {s: z[f"ms_full__{_key(s)}"] for s in names}
    MS1 = {s: z[f"ms_part1__{_key(s)}"] for s in names}
    TH = {(r, s): z[f"thr_{r}__{_key(s)}"] for r in RULES for s in names}

    rows = []
    for s in names:
        m = _agg(y, P[s], MS[s], TH[(PRIMARY_RULE, s)])
        legacy = _agg(y, P[s], MS1[s], TH[(PRIMARY_RULE, s)])
        m.update({"system": s, "mean_calls": 1.0,
                  "ms_part1_basis": legacy["ms"], "p95_ms_part1_basis": legacy["p95_ms"]},
                 **_per_corpus(y, P[s], ds))
        rows.append(m)
    T = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    front = ["system", "auroc", "auroc_sd", "auprc_unsup", "acc", "bacc", "mcc", "f1_macro",
             "f1_unsup", "prec_unsup", "rec_unsup", "ece", "mce", "brier", "aurc",
             "ms", "p50_ms", "p95_ms", "p99_ms", "qps", "ms_part1_basis",
             "p95_ms_part1_basis", "mean_calls", "macro_auroc",
             "macro_auroc_sample_weighted", "worst_corpus_auroc"]
    T = T[front + [c for c in T.columns if c not in front]]
    V.save(f"{protocol}_MAIN_WITH_SENTINEL.csv", T)
    V.save(f"{protocol}_MAIN.csv",
           T[T.system != "OURS_legacy_hp"].reset_index(drop=True))

    # -------- drift sentinel: the legacy-hp router must reproduce part1 on part1's basis
    got = T[T.system == "OURS_legacy_hp"].iloc[0]
    want = LEGACY_REFERENCE[protocol]
    d_auroc = abs(float(got["auroc"]) - want["auroc"])
    d_ms = abs(float(got["ms_part1_basis"]) - want["ms"])
    ok = d_auroc <= G1_TOL_AUROC and d_ms <= G1_TOL_MS
    core.log(f"  sentinel {protocol}: auroc {got['auroc']:.10f} vs {want['auroc']:.10f} "
             f"(d={d_auroc:.2e}) | part1-basis ms {got['ms_part1_basis']:.6f} vs "
             f"{want['ms']:.6f} (d={d_ms:.2e}) -> {'PASS' if ok else 'FAIL'}")
    (RES / f"SENTINEL_{protocol}.json").write_text(json.dumps(
        {"protocol": protocol, "system": "OURS_legacy_hp",
         "observed_auroc": float(got["auroc"]), "part1_auroc": want["auroc"],
         "delta_auroc": d_auroc, "observed_ms_part1_basis": float(got["ms_part1_basis"]),
         "part1_ms": want["ms"], "delta_ms": d_ms, "tolerance_auroc": G1_TOL_AUROC,
         "tolerance_ms": G1_TOL_MS, "pass": bool(ok), "smoke": SMOKE}, indent=2))

    rows = []
    for s in names:
        if s == "OURS":
            continue
        pt, lo, hi = core.paired_cluster_bootstrap(y, P["OURS"], P[s], g, 17, draws=DRAWS)
        rows.append({"system": s, "d_auroc_ours_minus": pt, "ci_lo": lo, "ci_hi": hi,
                     "significant": bool(lo > 0 or hi < 0),
                     "d_ms": float(MS["OURS"].mean() - MS[s].mean())})
    B = pd.DataFrame(rows).sort_values("d_auroc_ours_minus", ascending=False)
    V.save(f"{protocol}_PAIRED_WITH_SENTINEL.csv", B)
    V.save(f"{protocol}_PAIRED.csv",
           B[B.system != "OURS_legacy_hp"].reset_index(drop=True))

    rows = []
    for s in names:
        for r in RULES:
            m = _agg(y, P[s], MS[s], TH[(r, s)])
            rows.append({"system": s, "rule": r, "primary": r == PRIMARY_RULE,
                         "mean_threshold": float(TH[(r, s)].mean()),
                         **{k: m[k] for k in ("acc", "bacc", "mcc", "f1_macro", "f1_unsup",
                                              "prec_unsup", "rec_unsup", "auroc", "ece",
                                              "brier", "aurc")}})
    V.save(f"{protocol}_THRESHOLD_SENSITIVITY.csv", pd.DataFrame(rows))

    rows = []
    for s in names:
        cov, err = _risk_coverage_curve(y, P[s], TH[(PRIMARY_RULE, s)])
        rows += [{"system": s, "coverage": float(c), "selective_error": float(e)}
                 for c, e in zip(cov, err)]
    V.save(f"{protocol}_RISK_COVERAGE.csv", pd.DataFrame(rows))

    rows = []
    for s in names:
        one = _agg(y, z[f"prob1__{_key(s)}"], MS[s], z[f"thr1_{PRIMARY_RULE}__{_key(s)}"])
        two = _agg(y, P[s], MS[s], TH[(PRIMARY_RULE, s)])
        rows.append({"system": s, "auroc_stage1": one["auroc"], "auroc_stage2": two["auroc"],
                     "d_auroc": two["auroc"] - one["auroc"],
                     "ece_stage1": one["ece"], "ece_stage2": two["ece"],
                     "d_ece": two["ece"] - one["ece"],
                     "brier_stage1": one["brier"], "brier_stage2": two["brier"],
                     "d_brier": two["brier"] - one["brier"]})
    V.save(f"{protocol}_STAGE2_EFFECT.csv",
           pd.DataFrame(rows).sort_values("d_auroc", ascending=False))

    # Row-weighted, not per-fit: Protocol A's ten evaluation folds differ in size, so an
    # unweighted mean over fits does not reconstruct the per-row mean the main table reports.
    comp = pd.read_csv(RES / f"{protocol}_LATENCY_COMPONENTS.csv")
    parts = ["verifier", "features", "heads", "routing", "platt", "isotonic"]
    agg = (comp.groupby("system")
           .apply(lambda d: pd.Series({c: float(np.average(d[c], weights=d["n"]))
                                       for c in parts}), include_groups=False)
           .reset_index())
    agg["end_to_end"] = agg[parts].sum(axis=1)
    agg["part1_basis"] = [float(MS1[s].mean()) for s in agg["system"]]
    agg["uncharged_by_part1"] = agg["end_to_end"] - agg["part1_basis"]
    agg["measured_mean_ms"] = [float(MS[s].mean()) for s in agg["system"]]
    resid = float((agg["end_to_end"] - agg["measured_mean_ms"]).abs().max())
    core.log(f"  latency breakdown reconciles with the reported mean to {resid:.2e} ms")
    if resid > 1e-6:
        raise AssertionError(f"latency breakdown does not reconstruct the reported mean "
                             f"(max residual {resid} ms)")
    V.save(f"{protocol}_LATENCY_BREAKDOWN.csv",
           agg.sort_values("end_to_end", ascending=False))

    sel = z["sel__OURS"]
    rows = []
    for corpus in ["ALL"] + sorted(set(ds.tolist())):
        m = np.ones(len(y), bool) if corpus == "ALL" else (ds == corpus)
        rows.append({"corpus": corpus, "n": int(m.sum()),
                     "mean_ms": float(MS["OURS"][m].mean()),
                     **{f"share__{a}": float((sel[m] == i).mean())
                        for i, a in enumerate(actions)}})
    V.save(f"{protocol}_CALLS.csv", pd.DataFrame(rows))

    print(T[["system", "auroc", "auroc_sd", "ms", "ms_part1_basis", "ece", "bacc", "mcc",
             "macro_auroc"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if not ok and not SMOKE:
        raise AssertionError(
            f"drift sentinel failed for Protocol {protocol}: the legacy-hp router did not "
            f"reproduce part1. Tables were written; see results/SENTINEL_{protocol}.json")
    return bool(ok)


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    order = (["preflight", "hpselect", "protoB", "protoA", "report"]
             if want == ["all"] else want)
    for name in order:
        t0 = time.time()
        if name == "preflight":
            preflight()
        elif name == "hpselect":
            hpselect()
        elif name == "protoB":
            proto_b()
        elif name == "protoA":
            proto_a()
        elif name == "report":
            report("B")
            report("A")
        else:
            raise SystemExit(f"unknown stage {name}")
        core.log(f"===== {name} done in {time.time()-t0:.0f}s =====")
    (RUN / "04_provenance").mkdir(parents=True, exist_ok=True)
    (RUN / "04_provenance" / "DONE.json").write_text(json.dumps(
        {"status": "complete", "stages": order, "smoke": SMOKE,
         "finished": time.strftime("%Y-%m-%d %H:%M:%S%z")}, indent=2))
