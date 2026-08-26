from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from rouge_score import tokenizers as rouge_tokenizers
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
NUMBER_RE = re.compile(r"(?<!\w)\d[\d,]*(?:\.\d+)?(?!\w)")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9'-]+\b")
NEGATIONS = frozenset({"no", "not", "never", "none", "neither", "nor", "without"})
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ROUGE_TOKENIZER = rouge_tokenizers.DefaultTokenizer(use_stemmer=True)

FEATURE_COLUMNS = [
    "claim_token_count",
    "source_token_count",
    "sentence_count",
    "claim_source_length_ratio",
    "word_coverage",
    "bigram_coverage",
    "number_count",
    "number_coverage",
    "year_count",
    "year_coverage",
    "entity_count",
    "entity_coverage",
    "claim_has_negation",
    "negation_match",
    "bm25_top1",
    "bm25_mean3",
    "bm25_gap12",
    "tfidf_top1",
    "tfidf_mean3",
    "tfidf_gap12",
    "rougeL_top1",
    "rougeL_mean3",
    "rougeL_gap12",
    "retrieval_top_agreement",
    "bm25_top_index_normalized",
    "tfidf_top_index_normalized",
    "rougeL_top_index_normalized",
]


def _tokens(text: Any) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(str(text or ""))]


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return int(previous[-1])


def _rouge_l_fmeasure_tokens(
    target_tokens: Sequence[str], prediction_tokens: Sequence[str]
) -> float:
    if not target_tokens or not prediction_tokens:
        return 0.0
    lcs = _lcs_length(target_tokens, prediction_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(target_tokens)
    return float(2.0 * precision * recall / (precision + recall))


def rouge_l_fmeasure(target: str, prediction: str) -> float:
    return _rouge_l_fmeasure_tokens(
        ROUGE_TOKENIZER.tokenize(str(target)),
        ROUGE_TOKENIZER.tokenize(str(prediction)),
    )


def _sentences(text: Any) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return [""]
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(value) if part.strip()]
    return sentences or [value]


def _coverage(items: Sequence[str], source_items: set[str]) -> float:
    if not items:
        return 1.0
    return float(sum(item in source_items for item in items) / len(items))


def _top_stats(values: np.ndarray) -> tuple[float, float, float, int]:
    if not len(values):
        return 0.0, 0.0, 0.0, 0
    order = np.argsort(values)[::-1]
    top = float(values[order[0]])
    second = float(values[order[1]]) if len(order) > 1 else 0.0
    mean3 = float(np.mean(values[order[: min(3, len(order))]]))
    return top, mean3, top - second, int(order[0])


@dataclass
class _DocumentIndex:
    sentences: list[str]
    sentence_tokens: list[list[str]]
    sentence_counts: list[Counter[str]]
    avg_sentence_length: float
    document_frequency: Counter[str]
    tfidf_vectorizer: TfidfVectorizer
    tfidf_sentences: Any
    source_tokens: list[str]
    source_token_set: set[str]
    source_bigrams: set[str]
    source_numbers: set[str]
    source_years: set[str]
    source_entities: set[str]
    source_has_negation: bool
    rouge_sentence_tokens: list[list[str]]
    setup_ms: float


def _build_document_index(source: str) -> _DocumentIndex:
    started = time.perf_counter()
    sentences = _sentences(source)
    sentence_tokens = [_tokens(sentence) for sentence in sentences]
    sentence_counts = [Counter(tokens) for tokens in sentence_tokens]
    document_frequency: Counter[str] = Counter()
    for tokens in sentence_tokens:
        document_frequency.update(set(tokens))
    avg_sentence_length = float(
        np.mean([len(tokens) for tokens in sentence_tokens])
    )
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        norm="l2",
    )
    try:
        tfidf_sentences = vectorizer.fit_transform(sentences)
    except ValueError:
        tfidf_sentences = vectorizer.fit_transform(["empty"] * len(sentences))
    source_tokens = _tokens(source)
    return _DocumentIndex(
        sentences=sentences,
        sentence_tokens=sentence_tokens,
        sentence_counts=sentence_counts,
        avg_sentence_length=max(avg_sentence_length, 1.0),
        document_frequency=document_frequency,
        tfidf_vectorizer=vectorizer,
        tfidf_sentences=tfidf_sentences,
        source_tokens=source_tokens,
        source_token_set=set(source_tokens),
        source_bigrams={
            f"{left}\u001f{right}"
            for left, right in zip(source_tokens, source_tokens[1:])
        },
        source_numbers={
            value.replace(",", "") for value in NUMBER_RE.findall(source)
        },
        source_years=set(YEAR_RE.findall(source)),
        source_entities={
            value.casefold() for value in ENTITY_RE.findall(source)
        },
        source_has_negation=any(token in NEGATIONS for token in source_tokens),
        rouge_sentence_tokens=[
            ROUGE_TOKENIZER.tokenize(sentence) for sentence in sentences
        ],
        setup_ms=(time.perf_counter() - started) * 1000.0,
    )


def _bm25_scores(
    query_tokens: Sequence[str],
    index: _DocumentIndex,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    scores = np.zeros(len(index.sentences), dtype=np.float64)
    if not query_tokens:
        return scores
    query_counts = Counter(query_tokens)
    document_count = len(index.sentences)
    for sentence_index, counts in enumerate(index.sentence_counts):
        sentence_length = len(index.sentence_tokens[sentence_index])
        score = 0.0
        for term, query_frequency in query_counts.items():
            term_frequency = counts.get(term, 0)
            if term_frequency <= 0:
                continue
            frequency = index.document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            denominator = term_frequency + k1 * (
                1.0 - b + b * sentence_length / index.avg_sentence_length
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * term_frequency
                * (k1 + 1.0)
                / max(denominator, 1e-12)
            )
        scores[sentence_index] = score
    return scores


def _retrieval_features(claim: str, index: _DocumentIndex) -> dict[str, float]:
    query_tokens = _tokens(claim)
    bm25 = _bm25_scores(query_tokens, index)
    tfidf_query = index.tfidf_vectorizer.transform([claim])
    tfidf = (tfidf_query @ index.tfidf_sentences.T).toarray().ravel()
    rouge_query_tokens = ROUGE_TOKENIZER.tokenize(claim)
    rouge = np.asarray(
        [
            _rouge_l_fmeasure_tokens(rouge_query_tokens, sentence_tokens)
            for sentence_tokens in index.rouge_sentence_tokens
        ],
        dtype=np.float64,
    )
    bm25_top, bm25_mean, bm25_gap, bm25_index = _top_stats(bm25)
    tfidf_top, tfidf_mean, tfidf_gap, tfidf_index = _top_stats(tfidf)
    rouge_top, rouge_mean, rouge_gap, rouge_index = _top_stats(rouge)
    pair_agreement = sum(
        left == right
        for left, right in (
            (bm25_index, tfidf_index),
            (bm25_index, rouge_index),
            (tfidf_index, rouge_index),
        )
    )
    denominator = max(len(index.sentences) - 1, 1)
    return {
        "bm25_top1": bm25_top,
        "bm25_mean3": bm25_mean,
        "bm25_gap12": bm25_gap,
        "tfidf_top1": tfidf_top,
        "tfidf_mean3": tfidf_mean,
        "tfidf_gap12": tfidf_gap,
        "rougeL_top1": rouge_top,
        "rougeL_mean3": rouge_mean,
        "rougeL_gap12": rouge_gap,
        "retrieval_top_agreement": float(pair_agreement / 3.0),
        "bm25_top_index_normalized": float(bm25_index / denominator),
        "tfidf_top_index_normalized": float(tfidf_index / denominator),
        "rougeL_top_index_normalized": float(rouge_index / denominator),
    }


def build_cheap_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"doc_group_key", "source_document", "candidate_sentence"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature frame missing columns: {missing}")
    indices: dict[str, _DocumentIndex] = {}
    rows: list[dict[str, float]] = []
    for episode in frame.itertuples(index=False):
        key = str(episode.doc_group_key)
        setup_ms = 0.0
        if key not in indices:
            indices[key] = _build_document_index(str(episode.source_document))
            setup_ms = indices[key].setup_ms
        index = indices[key]
        started = time.perf_counter()
        claim = str(episode.candidate_sentence)
        claim_tokens = _tokens(claim)
        claim_bigrams = [
            f"{left}\u001f{right}"
            for left, right in zip(claim_tokens, claim_tokens[1:])
        ]
        claim_numbers = [
            value.replace(",", "") for value in NUMBER_RE.findall(claim)
        ]
        claim_years = YEAR_RE.findall(claim)
        claim_entities = [value.casefold() for value in ENTITY_RE.findall(claim)]
        claim_negation = any(token in NEGATIONS for token in claim_tokens)
        features = {
            "claim_token_count": float(len(claim_tokens)),
            "source_token_count": float(len(index.source_tokens)),
            "sentence_count": float(len(index.sentences)),
            "claim_source_length_ratio": float(
                len(claim_tokens) / max(len(index.source_tokens), 1)
            ),
            "word_coverage": _coverage(claim_tokens, index.source_token_set),
            "bigram_coverage": _coverage(claim_bigrams, index.source_bigrams),
            "number_count": float(len(claim_numbers)),
            "number_coverage": _coverage(claim_numbers, index.source_numbers),
            "year_count": float(len(claim_years)),
            "year_coverage": _coverage(claim_years, index.source_years),
            "entity_count": float(len(claim_entities)),
            "entity_coverage": _coverage(claim_entities, index.source_entities),
            "claim_has_negation": float(claim_negation),
            "negation_match": float(claim_negation == index.source_has_negation),
            **_retrieval_features(claim, index),
        }
        features["feature_query_latency_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        features["feature_document_setup_ms"] = setup_ms
        rows.append(features)
    result = pd.DataFrame(rows, index=frame.index)
    values = result[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("cheap features contain non-finite values")
    return result


def _decision(score: float, threshold: Mapping[str, Any]) -> int | None:
    if threshold.get("low_enabled") and float(score) <= float(threshold["tau_low"]):
        return 0
    if threshold.get("high_enabled") and float(score) >= float(threshold["tau_high"]):
        return 1
    return None


def build_selective_targets(
    frame: pd.DataFrame,
    *,
    actions: Sequence[str],
    thresholds: Mapping[str, Mapping[str, Any]],
    action_cost_ms: Mapping[str, float],
) -> pd.DataFrame:
    actions = tuple(actions)
    if set(actions) != set(thresholds) or set(actions) != set(action_cost_ms):
        raise ValueError("actions, thresholds, and costs must match")
    cost_reference = float(sum(float(action_cost_ms[action]) for action in actions))
    if cost_reference <= 0:
        raise ValueError("total action cost must be positive")
    rows = []
    for episode in frame.itertuples(index=False):
        label = int(episode.label_supported)
        candidates = []
        row: dict[str, Any] = {"abstain__cost_to_go": 1.0}
        for rank, action in enumerate(actions):
            decision = _decision(
                getattr(episode, f"{action}__score"), thresholds[action]
            )
            correct = decision is not None and decision == label
            wrong = decision is not None and decision != label
            normalized_cost = float(action_cost_ms[action]) / cost_reference
            row[f"{action}__decision"] = decision
            row[f"{action}__correct_stop"] = bool(correct)
            row[f"{action}__wrong_stop"] = bool(wrong)
            row[f"{action}__cost_to_go"] = normalized_cost + (0.0 if correct else 1.0)
            if correct:
                candidates.append((float(action_cost_ms[action]), rank, action))
        row["oracle_action"] = min(candidates)[2] if candidates else "ABSTAIN"
        row["oracle_available"] = bool(candidates)
        rows.append(row)
    return pd.DataFrame(rows, index=frame.index)


def evaluate_direct_policy(
    frame: pd.DataFrame,
    *,
    selected_actions: Sequence[str],
    thresholds: Mapping[str, Mapping[str, Any]],
    action_cost_ms: Mapping[str, float],
    router_overhead_ms: float,
) -> pd.DataFrame:
    if len(frame) != len(selected_actions):
        raise ValueError("selected actions must align with frame rows")
    rows = []
    for episode, action in zip(frame.itertuples(index=False), selected_actions):
        action = str(action)
        label = int(episode.label_supported)
        if action == "ABSTAIN":
            decision = None
            cost = float(router_overhead_ms)
            call_count = 0
        else:
            if action not in thresholds or action not in action_cost_ms:
                raise ValueError(f"unknown selected action: {action}")
            decision = _decision(
                getattr(episode, f"{action}__score"), thresholds[action]
            )
            cost = float(router_overhead_ms) + float(action_cost_ms[action])
            call_count = 1
        stopped = decision is not None
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "doc_group_key": str(episode.doc_group_key),
                "dataset": str(episode.dataset),
                "label_supported": label,
                "selected_action": action,
                "stop_decision": decision,
                "stopped": stopped,
                "stop_correct": bool(stopped and decision == label),
                "wrong_stop": bool(stopped and decision != label),
                "abstained": not stopped,
                "call_count": call_count,
                "path_cost_ms": cost,
            }
        )
    return pd.DataFrame(rows)


def evaluate_sequential_policy(
    frame: pd.DataFrame,
    *,
    candidate_actions: Sequence[str],
    activate: Sequence[bool],
    fixed_policy: Sequence[str],
    thresholds: Mapping[str, Mapping[str, Any]],
    action_cost_ms: Mapping[str, float],
    router_overhead_ms: float,
) -> pd.DataFrame:
    if not (len(frame) == len(candidate_actions) == len(activate)):
        raise ValueError("sequential policy inputs must align")
    fixed_policy = tuple(fixed_policy)
    if not fixed_policy:
        raise ValueError("sequential fallback policy cannot be empty")
    rows = []
    for episode, candidate, enabled in zip(
        frame.itertuples(index=False), candidate_actions, activate
    ):
        candidate = str(candidate)
        label = int(episode.label_supported)
        if not bool(enabled):
            policy: tuple[str, ...] = ()
        else:
            if candidate not in thresholds or candidate not in action_cost_ms:
                raise ValueError(f"unknown sequential candidate action: {candidate}")
            policy = (candidate,) + tuple(
                action for action in fixed_policy if action != candidate
            )
        calls = []
        decision = None
        stop_action = None
        cost = float(router_overhead_ms)
        for action in policy:
            calls.append(action)
            cost += float(action_cost_ms[action])
            decision = _decision(
                getattr(episode, f"{action}__score"), thresholds[action]
            )
            if decision is not None:
                stop_action = action
                break
        stopped = decision is not None
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "doc_group_key": str(episode.doc_group_key),
                "dataset": str(episode.dataset),
                "label_supported": label,
                "selected_action": candidate if enabled else "ABSTAIN",
                "policy": "->".join(policy) if policy else "ABSTAIN",
                "calls_json": json.dumps(calls),
                "stop_action": stop_action,
                "stop_decision": decision,
                "stopped": stopped,
                "stop_correct": bool(stopped and decision == label),
                "wrong_stop": bool(stopped and decision != label),
                "abstained": not stopped,
                "call_count": len(calls),
                "path_cost_ms": cost,
            }
        )
    return pd.DataFrame(rows)


def policy_metrics(outcomes: pd.DataFrame) -> dict[str, Any]:
    rows = len(outcomes)
    stops = int(outcomes["stopped"].sum())
    correct = int(outcomes["stop_correct"].sum())
    wrong = int(outcomes["wrong_stop"].sum())
    costs = pd.to_numeric(outcomes["path_cost_ms"], errors="raise").astype(float)
    return {
        "rows": int(rows),
        "documents": int(outcomes["doc_group_key"].nunique()),
        "stops": stops,
        "correct_stops": correct,
        "wrong_stops": wrong,
        "abstains": int(rows - stops),
        "coverage": float(stops / rows),
        "correct_stop_rate": float(correct / rows),
        "wrong_stop_rate": float(wrong / rows),
        "conditional_error": float(wrong / stops) if stops else None,
        "mean_cost_ms": float(costs.mean()),
        "p50_cost_ms": float(costs.quantile(0.5)),
        "p95_cost_ms": float(costs.quantile(0.95)),
        "mean_calls": float(outcomes["call_count"].mean()),
        "action_counts": {
            str(action): int(count)
            for action, count in outcomes["selected_action"].value_counts().items()
        },
    }


def select_operating_point(
    frame: pd.DataFrame,
    *,
    candidate_actions: Sequence[str],
    confidence: Sequence[float],
    thresholds: Mapping[str, Mapping[str, Any]],
    action_cost_ms: Mapping[str, float],
    risk_budget: float,
    router_overhead_ms: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if len(frame) != len(candidate_actions) or len(frame) != len(confidence_array):
        raise ValueError("operating-point inputs must align")
    if not np.isfinite(confidence_array).all():
        raise ValueError("router confidence must be finite")
    grid = sorted(set(float(value) for value in confidence_array))
    grid.append(float(np.nextafter(max(grid), math.inf)))
    rows = []
    for threshold in grid:
        selected_actions = [
            str(action) if score >= threshold else "ABSTAIN"
            for action, score in zip(candidate_actions, confidence_array)
        ]
        outcomes = evaluate_direct_policy(
            frame,
            selected_actions=selected_actions,
            thresholds=thresholds,
            action_cost_ms=action_cost_ms,
            router_overhead_ms=router_overhead_ms,
        )
        metrics = policy_metrics(outcomes)
        rows.append(
            {
                "threshold": float(threshold),
                "selection_feasible": metrics["wrong_stop_rate"] <= risk_budget,
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    feasible = table.loc[table["selection_feasible"]].copy()
    if feasible.empty:
        selected = table.sort_values(
            ["wrong_stop_rate", "correct_stop_rate", "mean_cost_ms", "threshold"],
            ascending=[True, False, True, False],
        ).iloc[0]
    else:
        selected = feasible.sort_values(
            ["correct_stop_rate", "wrong_stop_rate", "mean_cost_ms", "threshold"],
            ascending=[False, True, True, False],
        ).iloc[0]
    table["selected"] = table["threshold"].eq(float(selected["threshold"]))
    return selected.to_dict(), table


def select_sequential_operating_point(
    frame: pd.DataFrame,
    *,
    candidate_actions: Sequence[str],
    confidence: Sequence[float],
    fixed_policy: Sequence[str],
    thresholds: Mapping[str, Mapping[str, Any]],
    action_cost_ms: Mapping[str, float],
    risk_budget: float,
    correct_stop_tolerance: float,
    router_overhead_ms: float,
    fixed_outcomes: pd.DataFrame,
    grid_size: int = 201,
) -> tuple[dict[str, Any], pd.DataFrame]:
    confidence_array = np.asarray(confidence, dtype=np.float64)
    if len(frame) != len(candidate_actions) or len(frame) != len(confidence_array):
        raise ValueError("sequential operating-point inputs must align")
    if not np.isfinite(confidence_array).all():
        raise ValueError("sequential Router confidence must be finite")
    if not 0 <= correct_stop_tolerance < 1:
        raise ValueError("correct-stop tolerance must be in [0, 1)")
    fixed = policy_metrics(fixed_outcomes)
    required_correct = max(
        0.0, float(fixed["correct_stop_rate"]) - float(correct_stop_tolerance)
    )
    quantiles = np.linspace(0.0, 1.0, min(int(grid_size), len(frame)))
    grid = np.unique(np.quantile(confidence_array, quantiles))
    grid = np.append(grid, np.nextafter(float(grid.max()), math.inf))
    rows = []
    for threshold in grid:
        enabled = confidence_array >= float(threshold)
        outcomes = evaluate_sequential_policy(
            frame,
            candidate_actions=candidate_actions,
            activate=enabled,
            fixed_policy=fixed_policy,
            thresholds=thresholds,
            action_cost_ms=action_cost_ms,
            router_overhead_ms=router_overhead_ms,
        )
        metrics = policy_metrics(outcomes)
        rows.append(
            {
                "policy_mode": "sequential_fallback_or_abstain",
                "threshold": float(threshold),
                "active_rate": float(enabled.mean()),
                "required_correct_stop_rate": required_correct,
                "selection_feasible": (
                    metrics["wrong_stop_rate"] <= risk_budget
                    and metrics["correct_stop_rate"] >= required_correct
                ),
                "cost_saving_vs_fixed": 1.0
                - metrics["mean_cost_ms"] / float(fixed["mean_cost_ms"]),
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    feasible = table.loc[table["selection_feasible"]].copy()
    if feasible.empty:
        selected = table.sort_values(
            [
                "wrong_stop_rate",
                "correct_stop_rate",
                "mean_cost_ms",
                "threshold",
            ],
            ascending=[True, False, True, False],
        ).iloc[0]
    else:
        selected = feasible.sort_values(
            ["mean_cost_ms", "correct_stop_rate", "wrong_stop_rate", "threshold"],
            ascending=[True, False, True, False],
        ).iloc[0]
    table["selected"] = table["threshold"].eq(float(selected["threshold"]))
    return selected.to_dict(), table


def _new_lr_hard_ce(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def predict_lr_hard_ce(
    model: Any,
    features: pd.DataFrame,
    *,
    actions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    actions = tuple(actions)
    probabilities = model.predict_proba(features.to_numpy(dtype=np.float64))
    classes = [str(value) for value in model.named_steps["model"].classes_]
    by_class = {
        name: probabilities[:, index] for index, name in enumerate(classes)
    }
    action_probabilities = np.column_stack(
        [by_class.get(action, np.zeros(len(features))) for action in actions]
    )
    best_indices = np.argmax(action_probabilities, axis=1)
    candidates = np.asarray([actions[index] for index in best_indices], dtype=object)
    best_probability = action_probabilities[np.arange(len(features)), best_indices]
    abstain_probability = by_class.get("ABSTAIN", np.zeros(len(features)))
    confidence = best_probability - abstain_probability
    raw = pd.DataFrame(
        {
            **{
                f"probability__{action}": action_probabilities[:, index]
                for index, action in enumerate(actions)
            },
            "probability__ABSTAIN": abstain_probability,
        },
        index=features.index,
    )
    return candidates, confidence.astype(np.float64), raw


def cross_fit_lr_hard_ce(
    features: pd.DataFrame,
    oracle_action: pd.Series,
    groups: pd.Series,
    *,
    actions: Sequence[str],
    n_splits: int = 5,
    seed: int = 73,
) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    if not (len(features) == len(oracle_action) == len(groups)):
        raise ValueError("LR cross-fit inputs must align")
    unique_groups = groups.astype(str).nunique()
    splits = min(int(n_splits), int(unique_groups))
    if splits < 2:
        raise ValueError("LR cross-fit requires at least two document groups")
    x = features.reset_index(drop=True)
    y = oracle_action.astype(str).reset_index(drop=True)
    group_values = groups.astype(str).reset_index(drop=True)
    oof_candidates = np.empty(len(x), dtype=object)
    oof_confidence = np.full(len(x), np.nan, dtype=np.float64)
    oof_fold = np.full(len(x), -1, dtype=np.int64)
    leakage = False
    splitter = GroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(x, y, group_values)
    ):
        train_groups = set(group_values.iloc[train_index])
        validation_groups = set(group_values.iloc[validation_index])
        leakage = leakage or bool(train_groups & validation_groups)
        model = _new_lr_hard_ce(seed + fold)
        model.fit(x.iloc[train_index].to_numpy(dtype=np.float64), y.iloc[train_index])
        candidates, confidence, _ = predict_lr_hard_ce(
            model,
            x.iloc[validation_index],
            actions=actions,
        )
        oof_candidates[validation_index] = candidates
        oof_confidence[validation_index] = confidence
        oof_fold[validation_index] = fold
    if not np.isfinite(oof_confidence).all() or (oof_fold < 0).any():
        raise AssertionError("LR cross-fit left rows without held-out predictions")
    final_model = _new_lr_hard_ce(seed)
    final_model.fit(x.to_numpy(dtype=np.float64), y)
    return {
        "model": final_model,
        "oof_candidate_actions": oof_candidates,
        "oof_confidence": oof_confidence,
        "oof_fold": oof_fold,
        "oof_group_leakage": leakage,
        "n_splits": splits,
        "classes": sorted(set(y)),
        "protocol": "multinomial_logistic_regression_balanced_hard_oracle_action_v1",
    }


def _new_hgb_cost_to_go(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=seed,
    )


def predict_hgb_cost_to_go(
    models: Mapping[str, Any],
    features: pd.DataFrame,
    *,
    actions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    actions = tuple(actions)
    predicted = np.column_stack(
        [
            np.asarray(
                models[action].predict(features.to_numpy(dtype=np.float64)),
                dtype=np.float64,
            )
            for action in actions
        ]
    )
    if not np.isfinite(predicted).all():
        raise ValueError("HGB predicted non-finite cost-to-go")
    best_indices = np.argmin(predicted, axis=1)
    candidates = np.asarray([actions[index] for index in best_indices], dtype=object)
    best_cost = predicted[np.arange(len(features)), best_indices]
    confidence = 1.0 - best_cost
    raw = pd.DataFrame(
        {
            f"predicted_cost_to_go__{action}": predicted[:, index]
            for index, action in enumerate(actions)
        },
        index=features.index,
    )
    return candidates, confidence.astype(np.float64), raw


def cross_fit_hgb_cost_to_go(
    features: pd.DataFrame,
    cost_targets: pd.DataFrame,
    groups: pd.Series,
    *,
    actions: Sequence[str],
    n_splits: int = 5,
    seed: int = 73,
) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    if not (len(features) == len(cost_targets) == len(groups)):
        raise ValueError("HGB cross-fit inputs must align")
    required = {f"{action}__cost_to_go" for action in actions}
    missing = sorted(required - set(cost_targets.columns))
    if missing:
        raise ValueError(f"HGB cost targets missing columns: {missing}")
    unique_groups = groups.astype(str).nunique()
    splits = min(int(n_splits), int(unique_groups))
    if splits < 2:
        raise ValueError("HGB cross-fit requires at least two document groups")
    x = features.reset_index(drop=True)
    targets = cost_targets.reset_index(drop=True)
    group_values = groups.astype(str).reset_index(drop=True)
    oof_candidates = np.empty(len(x), dtype=object)
    oof_confidence = np.full(len(x), np.nan, dtype=np.float64)
    oof_fold = np.full(len(x), -1, dtype=np.int64)
    leakage = False
    splitter = GroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(x, groups=group_values)
    ):
        train_groups = set(group_values.iloc[train_index])
        validation_groups = set(group_values.iloc[validation_index])
        leakage = leakage or bool(train_groups & validation_groups)
        fold_models = {}
        for action in actions:
            model = _new_hgb_cost_to_go(seed + fold)
            model.fit(
                x.iloc[train_index].to_numpy(dtype=np.float64),
                targets.iloc[train_index][f"{action}__cost_to_go"].to_numpy(
                    dtype=np.float64
                ),
            )
            fold_models[action] = model
        candidates, confidence, _ = predict_hgb_cost_to_go(
            fold_models,
            x.iloc[validation_index],
            actions=actions,
        )
        oof_candidates[validation_index] = candidates
        oof_confidence[validation_index] = confidence
        oof_fold[validation_index] = fold
    if not np.isfinite(oof_confidence).all() or (oof_fold < 0).any():
        raise AssertionError("HGB cross-fit left rows without held-out predictions")
    final_models = {}
    for action in actions:
        model = _new_hgb_cost_to_go(seed)
        model.fit(
            x.to_numpy(dtype=np.float64),
            targets[f"{action}__cost_to_go"].to_numpy(dtype=np.float64),
        )
        final_models[action] = model
    return {
        "model": final_models,
        "oof_candidate_actions": oof_candidates,
        "oof_confidence": oof_confidence,
        "oof_fold": oof_fold,
        "oof_group_leakage": leakage,
        "n_splits": splits,
        "protocol": "per_action_hist_gradient_boosting_cost_to_go_regression_v1",
    }


def _new_binary_gate(family: str, seed: int):
    if family == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        l1_ratio=0.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=2_000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if family == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
    raise ValueError(f"unknown two-stage gate family: {family}")


def _positive_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features.to_numpy(dtype=np.float64))
    classes = (
        model.named_steps["model"].classes_
        if hasattr(model, "named_steps")
        else model.classes_
    )
    indices = np.flatnonzero(np.asarray(classes) == 1)
    if len(indices) != 1:
        raise ValueError("binary gate has no unique positive class")
    return np.asarray(probabilities[:, int(indices[0])], dtype=np.float64)


def predict_two_stage_router(
    model: Mapping[str, Any],
    features: pd.DataFrame,
    *,
    actions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    confidence = _positive_probability(model["gate"], features)
    candidates, _, selector_raw = predict_lr_hard_ce(
        model["selector"], features, actions=actions
    )
    raw = selector_raw.copy()
    raw.insert(0, "probability__oracle_available", confidence)
    return candidates, confidence, raw


def cross_fit_two_stage_router(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    groups: pd.Series,
    *,
    actions: Sequence[str],
    gate_family: str,
    n_splits: int = 5,
    seed: int = 73,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    required = {"oracle_action", "oracle_available"}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"two-stage targets missing columns: {missing}")
    if not (len(features) == len(targets) == len(groups)):
        raise ValueError("two-stage cross-fit inputs must align")
    unique_groups = groups.astype(str).nunique()
    splits = min(int(n_splits), int(unique_groups))
    if splits < 2:
        raise ValueError("two-stage cross-fit requires at least two document groups")
    x = features.reset_index(drop=True)
    target = targets.reset_index(drop=True)
    available = target["oracle_available"].astype(int).to_numpy()
    group_values = groups.astype(str).reset_index(drop=True)
    oof_candidates = np.empty(len(x), dtype=object)
    oof_confidence = np.full(len(x), np.nan, dtype=np.float64)
    oof_fold = np.full(len(x), -1, dtype=np.int64)
    leakage = False
    splitter = GroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(x, available, group_values)
    ):
        train_groups = set(group_values.iloc[train_index])
        validation_groups = set(group_values.iloc[validation_index])
        leakage = leakage or bool(train_groups & validation_groups)
        gate = _new_binary_gate(gate_family, seed + fold)
        gate.fit(x.iloc[train_index].to_numpy(dtype=np.float64), available[train_index])
        selector_index = train_index[available[train_index] == 1]
        selector_targets = target.iloc[selector_index]["oracle_action"].astype(str)
        if selector_targets.nunique() < 2:
            raise ValueError("two-stage selector fold has fewer than two action classes")
        selector = _new_lr_hard_ce(seed + 100 + fold)
        selector.fit(
            x.iloc[selector_index].to_numpy(dtype=np.float64), selector_targets
        )
        fold_model = {"gate": gate, "selector": selector}
        candidates, confidence, _ = predict_two_stage_router(
            fold_model,
            x.iloc[validation_index],
            actions=actions,
        )
        oof_candidates[validation_index] = candidates
        oof_confidence[validation_index] = confidence
        oof_fold[validation_index] = fold
    if not np.isfinite(oof_confidence).all() or (oof_fold < 0).any():
        raise AssertionError("two-stage cross-fit left rows without held-out predictions")
    final_gate = _new_binary_gate(gate_family, seed)
    final_gate.fit(x.to_numpy(dtype=np.float64), available)
    final_selector = _new_lr_hard_ce(seed + 100)
    final_selector.fit(
        x.loc[available == 1].to_numpy(dtype=np.float64),
        target.loc[available == 1, "oracle_action"].astype(str),
    )
    return {
        "model": {
            "gate": final_gate,
            "selector": final_selector,
            "gate_family": gate_family,
        },
        "oof_candidate_actions": oof_candidates,
        "oof_confidence": oof_confidence,
        "oof_fold": oof_fold,
        "oof_group_leakage": leakage,
        "oof_gate_auc": float(roc_auc_score(available, oof_confidence)),
        "n_splits": splits,
        "protocol": f"two_stage_{gate_family}_defer_gate_plus_lr_action_selector_v1",
    }
