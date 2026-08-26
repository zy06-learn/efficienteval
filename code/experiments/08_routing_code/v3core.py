#!/usr/bin/env python3
"""v3 shared layer. One data boundary, one routing implementation, used by every stage.

Data boundary, decided 2026-08-13 and enforced here rather than by convention:

    SELECT = TRAIN only (5,276 rows / 890 documents).
             Every choice -- features, supervision target, learner, hyperparameters, pool,
             beta rule, calibration -- is made on this and nothing else.

    Protocol A = pooled 8/1/1 rotation over ALL rows (TRAIN + TEST, 8,512 / 1,535).
             Its configuration was selected on the TRAIN subset of its own pool, so A is a
             cross-validated estimate, not a held-out confirmation. Declared, not hidden.

    Protocol B = fit/validation split inside TRAIN, then TEST read once.
             TEST never participates in any choice, so B is the confirmatory result.

The previous round had this backwards: the configuration was selected on A, and A contained
54.4% of B's test documents, so B inherited a configuration that had already seen half of B's
test set. Selecting on TRAIN alone removes that path.
"""
from __future__ import annotations
import os
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
# AFR_INPUTS points at the text-free reproduction bundle shipped in 00_inputs; without it
# the original ingest_and_scoring tree is used, so behaviour on the author's machine is unchanged.
_INPUTS = os.environ.get("AFR_INPUTS")
V2 = ROOT / "ingest_and_scoring"
_DATA = Path(_INPUTS) if _INPUTS else V2 / "data"
_SCORES = (Path(_INPUTS) / "p1_scoring") if _INPUTS else V2 / "results" / "p1_scoring"
V3 = ROOT / "experiments"
RES = V3 / "results"
RES.mkdir(parents=True, exist_ok=True)
(V3 / "logs").mkdir(exist_ok=True)

sys.path.insert(0, str(V2))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "shared"))
import config_v2 as C  # noqa: E402
import core  # noqa: E402

SEEDS = list(C.SEEDS)
VAL_FRACTION = 0.20
INNER_FOLDS_A = 10
BETA_GRID = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2)
EPS = float(C.QUALITY_TOLERANCE)

SKIP = ("score__", "decision__", "available__", "latency_ms__", "threshold__", "correct__",
        "semantic_tokens__", "model_input_tokens__", "output_tokens__", "model_calls__",
        "model_forward_calls__", "forward_items__", "source_window_count__",
        "source_selected__", "source_sentence_coverage__", "context_overflow__")
META = {"episode_key", "episode_id", "dataset_key", "role", "official_split", "group_id",
        "doc_group_key", "content_doc_key", "label_supported", "source_document",
        "candidate_summary", "fold", "summary_model", "source_id", "num_sentences",
        "candidate_sentence", "feature_latency_ms", "feature_query_latency_ms",
        "feature_document_setup_ms", "compact16_feature_latency_ms"}


def _merge_scores(frame):
    """Attach the P1 scoring outputs to the label-free test frame."""
    keyed = frame["episode_key"].to_numpy()
    for v in C.VERIFIERS:
        path = _SCORES / f"{v}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"P1 output missing for {v}")
        part = pd.read_parquet(path).drop_duplicates("episode_key").set_index("episode_key")
        for src, dst in (("score", f"score__{v}"), ("available", f"available__{v}"),
                         ("latency_total_ms", f"latency_ms__{v}")):
            vals = part[src].reindex(keyed)
            take = vals.notna().to_numpy()
            if dst not in frame.columns:
                frame[dst] = np.nan
            frame.loc[take, dst] = vals.to_numpy()[take]
    return frame


def _finish(frame, verifiers):
    for v in verifiers:
        av = f"available__{v}"
        if av not in frame.columns:
            frame[av] = frame[f"score__{v}"].notna()
        frame[av] = frame[av].fillna(False).astype(bool) & frame[f"score__{v}"].notna()
        frame[f"latency_ms__{v}"] = frame[f"latency_ms__{v}"].astype(float)
        frame[f"decision__{v}"] = (frame[f"score__{v}"].fillna(0.5) >= 0.5).astype(int)
    return frame


def load(with_test_labels: bool):
    """TRAIN, and ALL = TRAIN + TEST. `with_test_labels` gates the confirmatory read."""
    train = pd.read_parquet(_DATA / "TRAIN.parquet")
    test = _merge_scores(pd.read_parquet(_DATA / "TEST_SCORING.parquet"))
    assert not [c for c in test.columns if "label" in c.lower()], "test frame must be label-free"
    if with_test_labels:
        gold = pd.read_parquet(_DATA / "TEST.parquet",
                               columns=["episode_key", "label_supported"])
        joined = test[["episode_key"]].merge(gold, on="episode_key", how="left")
        assert joined["label_supported"].notna().all(), "label join left holes"
        test["label_supported"] = joined["label_supported"].to_numpy(int)

    verifiers = [v for v in C.VERIFIERS if f"score__{v}" in train.columns]
    train = _finish(train.copy(), verifiers)
    test = _finish(test.copy(), verifiers)

    shared = [c for c in train.columns if c in test.columns]
    both = pd.concat([train[shared].assign(_side="TRAIN"),
                      test[shared].assign(_side="TEST")], ignore_index=True)

    # group isolation: no document may straddle the TRAIN/TEST boundary
    g_tr = set(train["content_doc_key"].astype(str))
    g_te = set(test["content_doc_key"].astype(str))
    overlap = g_tr & g_te
    if overlap:
        raise AssertionError(f"{len(overlap)} documents straddle TRAIN/TEST: "
                             f"{sorted(overlap)[:3]}")
    return train, test, both, verifiers


def features_of(frame):
    return [c for c in frame.columns
            if not c.startswith(SKIP) and c not in META
            and pd.api.types.is_numeric_dtype(frame[c])
            and np.isfinite(frame[c].to_numpy(float)).all()]


def stratified_group_split(frame, seed, fraction=VAL_FRACTION):
    """True = held-out part. Group-disjoint, stratified by corpus and majority label."""
    g = frame["content_doc_key"].astype(str).to_numpy()
    info = (pd.DataFrame({"g": g, "d": frame["dataset_key"].astype(str).to_numpy(),
                          "y": frame["label_supported"].to_numpy(int)})
            .groupby("g", sort=True)
            .agg(d=("d", "first"), y=("y", lambda s: int(round(float(s.mean())))))
            .reset_index())
    rng = np.random.default_rng(seed)
    held = set()
    for _, stratum in info.groupby(["d", "y"], sort=True):
        gg = stratum["g"].to_numpy()
        rng.shuffle(gg)
        held.update(gg[:max(int(round(len(gg) * fraction)), 1)].tolist())
    return np.isin(g, list(held))


def folds_stratified(frame, seed, n_folds=INNER_FOLDS_A):
    """Group-disjoint folds, stratified by corpus and majority label (Protocol A)."""
    g = frame["content_doc_key"].astype(str).to_numpy()
    info = (pd.DataFrame({"g": g, "d": frame["dataset_key"].astype(str).to_numpy(),
                          "y": frame["label_supported"].to_numpy(int)})
            .groupby("g", sort=True)
            .agg(d=("d", "first"), y=("y", lambda s: int(round(float(s.mean())))))
            .reset_index())
    rng = np.random.default_rng(seed)
    assign = {}
    for _, stratum in info.groupby(["d", "y"], sort=True):
        gg = stratum["g"].to_numpy()
        rng.shuffle(gg)
        for i, name in enumerate(gg):
            assign[name] = i % n_folds
    return np.array([assign[x] for x in g])


def rotations(n_folds=INNER_FOLDS_A):
    """(test fold, validation fold, training folds) -- the 8/1/1 contract."""
    return [(t, (t + 1) % n_folds, [f for f in range(n_folds) if f not in (t, (t + 1) % n_folds)])
            for t in range(n_folds)]


def route(frame, heads, cal, beta, actions, cost_vec, mask=None):
    avail = np.column_stack([frame[f"available__{a}"].astype(bool).to_numpy() for a in actions])
    if mask is not None:
        avail = avail & (~np.asarray(mask, bool))[None, :]
    cv = np.asarray(cost_vec, float)
    util = np.asarray(heads, float) * np.exp(-beta * cv / cv.max())
    sel = np.argmax(np.where(avail, util, -np.inf), axis=1)
    rows = np.arange(len(frame))
    prob = np.column_stack([cal[a] for a in actions])[rows, sel]
    lat = np.column_stack([frame[f"latency_ms__{a}"].fillna(0).to_numpy(float)
                           for a in actions])[rows, sel]
    return sel, prob, lat + frame["feature_latency_ms"].to_numpy(float)


def choose_beta(val, heads, cal, actions, cost_vec, eps=EPS):
    """Cheapest beta whose validation AUROC is within eps of the best."""
    y = val["label_supported"].to_numpy(int)
    ledger = []
    for b in BETA_GRID:
        _s, p, ms = route(val, heads, cal, b, actions, cost_vec)
        ledger.append((b, float(roc_auc_score(y, p)) if len(set(y)) > 1 else .5, float(ms.mean())))
    best = max(r[1] for r in ledger)
    return min([r for r in ledger if r[1] >= best - eps], key=lambda r: r[2])[0], ledger


def save(name, frame):
    frame.to_csv(RES / name, index=False)
    return frame
