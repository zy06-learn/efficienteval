from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pysbd
import torch
from scipy.stats import kendalltau, pointbiserialr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F


SCHEMA_VERSION = "summary_router_compact16_direct_v1"
MATRIX_RELATIVE = Path("results/unified_summary_verifiers_v1/ROUTER_TRAINING_MATRIX.parquet")
INPUT_RELATIVE = Path(
    "results/unified_summary_verifiers_v1/inputs/mixed_complete_summaries_v1.parquet"
)
OUTPUT_RELATIVE = Path("results/summary_router_compact16_direct_v1")
EXPECTED_MATRIX_SHA256 = "ce851c5536b653c1504f90eed9e083c76fc719afda1231cc7dcb13654a4d2c92"
EXPECTED_INPUT_SHA256 = "5cef5d4614339588aaf88a12aa9032fb7a5802d3f1e6bf5c5161e2d1479afe7f"

ACTIONS = ("factkb", "granite_guardian_3_1_2b", "qwen30_fast")
ACTION_TIERS = {"factkb": "Low", "granite_guardian_3_1_2b": "Mid", "qwen30_fast": "High"}
ACTION_COSTS_MS = {
    "factkb": 42.59736219641963,
    "granite_guardian_3_1_2b": 191.55966983423195,
    "qwen30_fast": 292.76183418827605,
}
METHODS = ("LR-S1", "HGB-S1", "MLP-S1", "MLP-S2", "MLP-S3")
BETAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
SEEDS = (17, 29, 43)
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOTSTRAP_DRAWS = 2_000

FEATURE_COLUMNS = (
    "log_source_token_count",
    "summary_sentence_count",
    "max_summary_sentence_tokens",
    "fact_mention_density",
    "logic_marker_density",
    "attribution_marker_density",
    "pronoun_reference_density",
    "structured_source_line_ratio",
    "source_lexical_entropy",
    "idf_weighted_coverage_mean",
    "weakest_sentence_bm25",
    "evidence_ambiguity",
    "evidence_span_normalized",
    "distinct_evidence_ratio",
    "entity_value_colocation",
    "conflicting_value_rate",
)

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
DISPLAY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|\d[\d,]*(?:\.\d+)?")
VALUE_RE = re.compile(
    r"(?<!\w)(?:[$£€]\s*)?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent|million|billion|"
    r"thousand|km|kg|miles?|years?|months?|days?|hours?|minutes?))?(?!\w)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
MULTI_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'-]+)(?:\s+(?:of|the|and|&)?\s*[A-Z][A-Za-z0-9'-]+)+\b"
)
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
STRUCTURED_LINE_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+|\t|\|.*\||^\s*[^:]{1,40}:\s*\S"
)

STOPWORDS = frozenset(
    "a an and are as at be been being by for from had has have he her hers him his i in into is "
    "it its me my of on or our ours she that the their theirs them they this those to was we were "
    "what when where which who will with you your".split()
)
LOGIC_MARKERS = frozenset(
    "no not never none neither nor without less least more most than every all any some only if unless "
    "because therefore thus hence so although though however but yet whereas while despite unlike before "
    "after may might could would should must cannot can't won't isn't wasn't aren't weren't".split()
)
LOGIC_PHRASES = (
    "even though",
    "rather than",
    "as a result",
    "due to",
    "in contrast",
    "on the other hand",
)
ATTRIBUTION_WORDS = frozenset(
    "said says say reported reports report claimed claims claim stated states state told tells according "
    "announced announces wrote writes noted notes alleged alleges".split()
)
ATTRIBUTION_PHRASES = ("according to", "was quoted", "were quoted")
PRONOUNS = frozenset(
    "he she it they him her them his hers its their theirs this that these those himself herself itself "
    "themselves such former latter".split()
)
ENTITY_EXCLUSIONS = frozenset(
    "a an and as at but by for from he her his i in it its no not of on or she that the their they this "
    "those to we what when where which who you".split()
)

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, *parts: Any) -> int:
    value = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16)


def _tokens(text: Any) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(str(text or ""))]


def _sentences(text: Any) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return [""]
    sentences = [part.strip() for part in _SEGMENTER.segment(value) if part.strip()]
    return sentences or [value]


def _values(text: Any) -> set[str]:
    values = {
        re.sub(r"\s+", "", match.group(0).casefold().replace(",", ""))
        for match in VALUE_RE.finditer(str(text or ""))
    }
    values.update(re.sub(r"\s+", "", item.casefold()) for item in DATE_RE.findall(str(text or "")))
    return values


def _entity_spans(sentence: str) -> set[str]:
    entities = {re.sub(r"\s+", " ", item.group(0)).casefold() for item in MULTI_ENTITY_RE.finditer(sentence)}
    entities.update(item.group(0).casefold() for item in ACRONYM_RE.finditer(sentence))
    display = DISPLAY_TOKEN_RE.findall(sentence)
    for index, token in enumerate(display):
        normalized = token.casefold()
        if (
            index > 0
            and token[:1].isupper()
            and token[1:].islower()
            and normalized not in ENTITY_EXCLUSIONS
        ):
            entities.add(normalized)
    return entities


def _summary_entity_spans(sentence: str, source_token_set: frozenset[str]) -> set[str]:
    entities = _entity_spans(sentence)
    display = DISPLAY_TOKEN_RE.findall(sentence)
    if display:
        first = display[0]
        normalized = first.casefold()
        if (
            first[:1].isupper()
            and first[1:].islower()
            and normalized not in ENTITY_EXCLUSIONS
            and normalized in source_token_set
        ):
            entities.add(normalized)
    return entities


@dataclass(frozen=True)
class _LocalIndex:
    sentences: tuple[str, ...]
    tokens: tuple[tuple[str, ...], ...]
    counts: tuple[Counter[str], ...]
    document_frequency: Counter[str]
    average_length: float
    source_token_set: frozenset[str]


def _build_local_index(source: str) -> _LocalIndex:
    sentences = tuple(_sentences(source))
    sentence_tokens = tuple(tuple(_tokens(sentence)) for sentence in sentences)
    document_frequency: Counter[str] = Counter()
    for tokens in sentence_tokens:
        document_frequency.update(set(tokens))
    lengths = [len(tokens) for tokens in sentence_tokens]
    return _LocalIndex(
        sentences=sentences,
        tokens=sentence_tokens,
        counts=tuple(Counter(tokens) for tokens in sentence_tokens),
        document_frequency=document_frequency,
        average_length=max(float(np.mean(lengths)), 1.0),
        source_token_set=frozenset(token for sentence in sentence_tokens for token in sentence),
    )


def _idf(term: str, index: _LocalIndex) -> float:
    documents = len(index.sentences)
    frequency = index.document_frequency.get(term, 0)
    return math.log(1.0 + (documents - frequency + 0.5) / (frequency + 0.5))


def _bm25_scores(query_tokens: Sequence[str], index: _LocalIndex) -> np.ndarray:
    scores = np.zeros(len(index.sentences), dtype=np.float64)
    query_counts = Counter(query_tokens)
    for sentence_index, counts in enumerate(index.counts):
        length = len(index.tokens[sentence_index])
        for term, query_frequency in query_counts.items():
            term_frequency = counts.get(term, 0)
            if term_frequency <= 0:
                continue
            denominator = term_frequency + 1.5 * (
                1.0 - 0.75 + 0.75 * length / index.average_length
            )
            scores[sentence_index] += (
                query_frequency
                * _idf(term, index)
                * term_frequency
                * 2.5
                / max(denominator, 1e-12)
            )
    return scores


def _normalized_entropy(values: np.ndarray) -> float:
    scores = np.asarray(values, dtype=float)
    if len(scores) <= 1:
        return 0.0
    total = float(scores.sum())
    probability = scores / total if total > 0 else np.full(len(scores), 1.0 / len(scores))
    positive = probability[probability > 0]
    return float(-np.sum(positive * np.log(positive)) / math.log(len(scores)))


def _structured_line_ratio(source: str) -> float:
    lines = [line for line in str(source).splitlines() if line.strip()]
    if not lines:
        return 0.0
    structured = 0
    for line in lines:
        stripped = line.strip()
        header = len(_tokens(stripped)) <= 8 and (
            stripped.endswith(":") or (any(char.isalpha() for char in stripped) and stripped.upper() == stripped)
        )
        structured += int(bool(STRUCTURED_LINE_RE.search(line) or header))
    return float(structured / len(lines))


def _source_entropy(source_tokens: Sequence[str]) -> float:
    content = [token for token in source_tokens if token not in STOPWORDS]
    counts = Counter(content)
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts.values()))
    probability = np.asarray(list(counts.values()), dtype=float) / total
    return float(-np.sum(probability * np.log(probability)) / math.log(len(counts)))


def _row_features(source: str, summary: str) -> dict[str, float]:
    index = _build_local_index(source)
    source_tokens = _tokens(source)
    summary_tokens = _tokens(summary)
    summary_sentences = _sentences(summary)
    summary_sentence_tokens = [_tokens(sentence) for sentence in summary_sentences]
    denominator = max(len(summary_tokens), 1)
    entities_by_sentence = [
        _summary_entity_spans(sentence, index.source_token_set)
        for sentence in summary_sentences
    ]
    values_by_sentence = [_values(sentence) for sentence in summary_sentences]
    entity_mentions = sum(len(items) for items in entities_by_sentence)
    value_mentions = sum(len(items) for items in values_by_sentence)

    logic_count = sum(token in LOGIC_MARKERS for token in summary_tokens)
    folded_summary = summary.casefold()
    logic_count += sum(folded_summary.count(phrase) for phrase in LOGIC_PHRASES)
    attribution_count = sum(token in ATTRIBUTION_WORDS for token in summary_tokens)
    attribution_count += sum(folded_summary.count(phrase) for phrase in ATTRIBUTION_PHRASES)
    attribution_count += len(re.findall(r"[\"“”‘’']", summary)) // 2
    pronoun_count = sum(token in PRONOUNS for token in summary_tokens)

    coverage_values: list[float] = []
    normalized_top_scores: list[float] = []
    ambiguities: list[float] = []
    best_indices: list[int] = []
    top_indices: list[np.ndarray] = []
    for tokens in summary_sentence_tokens:
        content = sorted(set(token for token in tokens if token not in STOPWORDS))
        weights = np.asarray([_idf(token, index) for token in content], dtype=float)
        total_weight = float(weights.sum())
        coverage_values.append(
            float(
                sum(weight for token, weight in zip(content, weights) if token in index.source_token_set)
                / total_weight
            )
            if total_weight > 0
            else 1.0
        )
        scores = _bm25_scores(content, index)
        order = np.argsort(-scores, kind="stable")
        top_indices.append(order[: min(3, len(order))])
        best_indices.append(int(order[0]) if len(order) else 0)
        normalized_top_scores.append(float(scores[order[0]] / total_weight) if len(order) and total_weight > 0 else 0.0)
        ambiguities.append(_normalized_entropy(scores))

    pairs: list[tuple[str, str, int]] = []
    for sentence_index, (entities, values) in enumerate(zip(entities_by_sentence, values_by_sentence)):
        pairs.extend((entity, value, sentence_index) for entity in sorted(entities) for value in sorted(values))
    colocated = 0
    conflicting = 0
    for entity, value, sentence_index in pairs:
        evidence = [index.sentences[position] for position in top_indices[sentence_index]]
        matching = False
        different = False
        for sentence in evidence:
            sentence_entities = _entity_spans(sentence)
            sentence_values = _values(sentence)
            entity_present = entity in sentence.casefold() or entity in sentence_entities
            if not entity_present:
                continue
            matching |= value in sentence_values
            different |= bool(sentence_values - {value})
        colocated += int(matching)
        conflicting += int((not matching) and different)

    source_sentence_denominator = max(len(index.sentences) - 1, 1)
    evidence_span = (max(best_indices) - min(best_indices)) / source_sentence_denominator if best_indices else 0.0
    return {
        "log_source_token_count": math.log1p(len(source_tokens)),
        "summary_sentence_count": float(len(summary_sentences)),
        "max_summary_sentence_tokens": float(max((len(tokens) for tokens in summary_sentence_tokens), default=0)),
        "fact_mention_density": float((entity_mentions + value_mentions) / denominator),
        "logic_marker_density": float(logic_count / denominator),
        "attribution_marker_density": float(attribution_count / denominator),
        "pronoun_reference_density": float(pronoun_count / denominator),
        "structured_source_line_ratio": _structured_line_ratio(source),
        "source_lexical_entropy": _source_entropy(source_tokens),
        "idf_weighted_coverage_mean": float(np.mean(coverage_values)),
        "weakest_sentence_bm25": float(min(normalized_top_scores, default=0.0)),
        "evidence_ambiguity": float(np.mean(ambiguities)),
        "evidence_span_normalized": float(evidence_span),
        "distinct_evidence_ratio": float(len(set(best_indices)) / max(len(summary_sentences), 1)),
        "entity_value_colocation": float(colocated / len(pairs)) if pairs else 1.0,
        "conflicting_value_rate": float(conflicting / len(pairs)) if pairs else 0.0,
    }


def build_compact16_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"source_document", "candidate_summary"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature input missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        started = time.perf_counter()
        values = _row_features(str(row["source_document"]), str(row["candidate_summary"]))
        values["feature_latency_ms"] = (time.perf_counter() - started) * 1_000.0
        values["__index"] = index
        rows.append(values)
    result = pd.DataFrame(rows).set_index("__index")
    result.index.name = frame.index.name
    if tuple(result.columns[: len(FEATURE_COLUMNS)]) != FEATURE_COLUMNS:
        raise AssertionError("Compact-16 feature schema drift")
    if not np.isfinite(result.loc[:, FEATURE_COLUMNS].to_numpy(float)).all():
        raise AssertionError("Compact-16 contains non-finite values")
    return result


def build_group_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    group_column = "group_id" if "group_id" in frame.columns else "doc_group_key"
    groups = frame[group_column].astype(str).to_numpy()
    dataset = frame["dataset_key"].astype(str) if "dataset_key" in frame else pd.Series("pool", index=frame.index)
    labels = frame["label_supported"].astype(int) if "label_supported" in frame else pd.Series(0, index=frame.index)
    strata = (dataset + "::" + labels.astype(str)).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    folds = np.full(len(frame), -1, dtype=np.int16)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(frame)), strata, groups)):
        folds[validation] = fold
    validate_group_folds(frame, folds, n_splits=n_splits)
    return folds


def validate_group_folds(frame: pd.DataFrame, folds: Sequence[int], n_splits: int) -> None:
    values = np.asarray(folds, dtype=int)
    if len(values) != len(frame) or set(values) != set(range(int(n_splits))):
        raise ValueError("source group fold assignment is incomplete")
    group_column = "group_id" if "group_id" in frame.columns else "doc_group_key"
    audit = pd.DataFrame({"group": frame[group_column].astype(str).to_numpy(), "fold": values})
    if audit.groupby("group")["fold"].nunique().max() != 1:
        raise ValueError("source group crosses folds")


def cost_discounts(beta: float, costs: Mapping[str, float]) -> np.ndarray:
    denominator = float(costs[ACTIONS[-1]])
    return np.exp(-float(beta) * np.asarray([float(costs[action]) for action in ACTIONS]) / denominator)


def masked_argmax(utility: np.ndarray, availability: np.ndarray) -> np.ndarray:
    values = np.asarray(utility, dtype=float)
    available = np.asarray(availability, dtype=bool)
    if values.shape != available.shape or values.ndim != 2:
        raise ValueError("utility/availability shape mismatch")
    if np.any(~available.any(axis=1)):
        raise ValueError("row has no available Router action")
    return np.argmax(np.where(available, values, -np.inf), axis=1).astype(int)


def pairwise_preferences(correct: np.ndarray, availability: np.ndarray, discount: np.ndarray) -> dict[str, np.ndarray]:
    z = np.nan_to_num(np.asarray(correct, dtype=float), nan=0.0)
    available = np.asarray(availability, dtype=bool)
    rewards = z * np.asarray(discount, dtype=float)[None, :]
    valid_row = ((z > 0) & available).any(axis=1)
    gap = rewards[:, :, None] - rewards[:, None, :]
    pair_valid = (
        valid_row[:, None, None]
        & available[:, :, None]
        & available[:, None, :]
        & (np.abs(gap) > 1e-12)
    )
    diagonal = np.arange(rewards.shape[1])
    pair_valid[:, diagonal, diagonal] = False
    return {
        "valid": pair_valid.any(axis=2),
        "pair_valid": pair_valid,
        "target": (gap > 0).astype(float),
        "gap": np.abs(gap),
        "reward": rewards,
    }


def all_configurations() -> list[dict[str, Any]]:
    return [
        {"method": method, "beta": float(beta), "seed": int(seed)}
        for method, beta, seed in product(METHODS, BETAS, SEEDS)
    ]


class _ConstantHead:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = np.full(len(features), self.probability)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class _SklearnHeads:
    heads: list[Any]

    def predict_raw(self, features: np.ndarray, availability: np.ndarray) -> np.ndarray:
        probabilities = np.column_stack([head.predict_proba(features)[:, 1] for head in self.heads])
        return np.where(availability, probabilities, 0.0)


class _CompactMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, len(ACTIONS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass
class _TorchBundle:
    model: _CompactMLP
    mean: np.ndarray
    scale: np.ndarray
    supervision: str
    best_epoch: int

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(
            (np.asarray(features, dtype=np.float32) - self.mean) / self.scale,
            dtype=torch.float32,
        )
        self.model.eval()
        with torch.no_grad():
            return self.model(values).cpu().numpy()

    def predict_raw(self, features: np.ndarray, availability: np.ndarray) -> np.ndarray:
        logits = self.logits(features)
        if self.supervision == "S3":
            masked = np.where(availability, logits, -1e9)
            shifted = masked - masked.max(axis=1, keepdims=True)
            values = np.exp(shifted) * availability
            return values / values.sum(axis=1, keepdims=True)
        return np.where(availability, 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40))), 0.0)


class _Platt:
    def __init__(self, constant: float | None, model: LogisticRegression | None) -> None:
        self.constant = constant
        self.model = model

    @classmethod
    def fit(cls, raw: np.ndarray, target: np.ndarray) -> "_Platt":
        y = np.asarray(target, dtype=int)
        if len(np.unique(y)) == 1:
            return cls(float(y[0]), None)
        probability = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(probability / (1 - probability)).reshape(-1, 1)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2_000)
        model.fit(logits, y)
        return cls(None, model)

    def predict(self, raw: np.ndarray) -> np.ndarray:
        probability = np.clip(np.asarray(raw, dtype=float), 1e-6, 1 - 1e-6)
        if self.model is None:
            return np.full(len(probability), float(self.constant))
        logits = np.log(probability / (1 - probability)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _masked_bce(probability: np.ndarray, correct: np.ndarray, availability: np.ndarray) -> float:
    mask = np.asarray(availability, dtype=bool)
    target = np.nan_to_num(np.asarray(correct, dtype=float), nan=0.0)
    values = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return float(-(target[mask] * np.log(values[mask]) + (1 - target[mask]) * np.log(1 - values[mask])).mean())


def _realized_selection_metrics(
    raw: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    decisions: np.ndarray,
    labels: np.ndarray,
    latency: np.ndarray,
    beta: float,
    supervision: str,
) -> tuple[float, float, float]:
    discount = cost_discounts(beta, ACTION_COSTS_MS)
    selected = masked_argmax(raw if supervision == "S3" else raw * discount[None, :], availability)
    rows = np.arange(len(selected))
    reward = np.nan_to_num(correct, nan=0.0) * discount[None, :]
    utility = float(reward[rows, selected].mean())
    hard = decisions[rows, selected]
    bacc = float(balanced_accuracy_score(labels, hard))
    mean_latency = float(latency[rows, selected].mean())
    return utility, bacc, mean_latency


def _sklearn_grid(method: str, quick: bool) -> list[dict[str, Any]]:
    if method == "LR-S1":
        values = (1.0,) if quick else (0.1, 1.0, 10.0)
        return [{"C": value} for value in values]
    if method == "HGB-S1":
        if quick:
            return [{"max_leaf_nodes": 7, "learning_rate": 0.1, "min_samples_leaf": 20}]
        return [
            {"max_leaf_nodes": leaf, "learning_rate": rate, "min_samples_leaf": minimum}
            for leaf, rate, minimum in product((7, 15), (0.03, 0.1), (20, 50))
        ]
    raise ValueError(f"no sklearn grid for {method}")


def _mlp_grid(quick: bool) -> list[dict[str, Any]]:
    if quick:
        return [{"lr": 1e-3, "dropout": 0.1}]
    return [
        {"lr": learning_rate, "dropout": dropout}
        for learning_rate, dropout in product((3e-4, 1e-3), (0.1, 0.2))
    ]


def _fit_sklearn_heads(
    method: str,
    features: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    parameters: Mapping[str, Any],
    seed: int,
) -> _SklearnHeads:
    heads: list[Any] = []
    for action_index in range(len(ACTIONS)):
        mask = availability[:, action_index]
        target = np.asarray(correct[mask, action_index], dtype=int)
        if len(np.unique(target)) == 1:
            heads.append(_ConstantHead(float(target[0])))
            continue
        if method == "LR-S1":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=float(parameters["C"]),
                    max_iter=2_000,
                    solver="lbfgs",
                    random_state=int(seed),
                ),
            )
        elif method == "HGB-S1":
            model = HistGradientBoostingClassifier(
                max_leaf_nodes=int(parameters["max_leaf_nodes"]),
                learning_rate=float(parameters["learning_rate"]),
                min_samples_leaf=int(parameters["min_samples_leaf"]),
                l2_regularization=1.0,
                max_iter=100,
                early_stopping=False,
                random_state=int(seed),
            )
        else:
            raise ValueError(method)
        model.fit(features[mask], target)
        heads.append(model)
    return _SklearnHeads(heads)


def _torch_loss(
    logits: torch.Tensor,
    correct: torch.Tensor,
    availability: torch.Tensor,
    beta: float,
    supervision: str,
) -> torch.Tensor:
    mask = availability.bool()
    target = torch.nan_to_num(correct, nan=0.0)
    if supervision in {"S1", "S2"}:
        bce = F.binary_cross_entropy_with_logits(logits[mask], target[mask])
        if supervision == "S1":
            return bce
        discounts = torch.as_tensor(cost_discounts(beta, ACTION_COSTS_MS), dtype=logits.dtype, device=logits.device)
        rewards = target * discounts[None, :]
        any_correct = ((target > 0) & mask).any(dim=1)
        pair_losses: list[torch.Tensor] = []
        pair_weights: list[torch.Tensor] = []
        for left in range(len(ACTIONS)):
            for right in range(left + 1, len(ACTIONS)):
                gap = rewards[:, left] - rewards[:, right]
                valid = any_correct & mask[:, left] & mask[:, right] & (gap.abs() > 1e-12)
                if valid.any():
                    target_pair = (gap[valid] > 0).to(logits.dtype)
                    pair_losses.append(
                        F.binary_cross_entropy_with_logits(
                            logits[valid, left] - logits[valid, right], target_pair, reduction="none"
                        )
                    )
                    pair_weights.append(gap[valid].abs())
        if not pair_losses:
            return bce
        losses = torch.cat(pair_losses)
        weights = torch.cat(pair_weights)
        return bce + (losses * weights).sum() / weights.sum().clamp_min(1e-12)
    if supervision != "S3":
        raise ValueError(supervision)
    valid = ((target > 0) & mask).any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    masked_logits = logits.masked_fill(~mask, -1e9)
    policy = torch.softmax(masked_logits, dim=1)
    discounts = torch.as_tensor(cost_discounts(beta, ACTION_COSTS_MS), dtype=logits.dtype, device=logits.device)
    rewards = target * discounts[None, :]
    expected_reward = (policy[valid] * rewards[valid]).sum(dim=1).mean()
    entropy = -(policy[valid] * torch.log(policy[valid].clamp_min(1e-12))).sum(dim=1).mean()
    return -expected_reward - 0.01 * entropy


def _fit_mlp(
    features: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    parameters: Mapping[str, Any],
    beta: float,
    supervision: str,
    seed: int,
    *,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    fixed_epochs: int | None = None,
    quick: bool = False,
) -> _TorchBundle:
    _seed_everything(seed)
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    device = torch.device("cpu" if quick or not torch.cuda.is_available() else "cuda")
    model = _CompactMLP(x.shape[1], float(parameters["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parameters["lr"]), weight_decay=1e-4)
    train_x = torch.as_tensor((x - mean) / scale, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(correct, dtype=torch.float32, device=device)
    train_available = torch.as_tensor(availability, dtype=torch.bool, device=device)
    if validation is not None:
        val_x_np, val_y_np, val_available_np = validation
        val_x = torch.as_tensor((val_x_np.astype(np.float32) - mean) / scale, dtype=torch.float32, device=device)
        val_y = torch.as_tensor(val_y_np, dtype=torch.float32, device=device)
        val_available = torch.as_tensor(val_available_np, dtype=torch.bool, device=device)
    else:
        val_x = val_y = val_available = None
    max_epochs = int(fixed_epochs or (4 if quick else 300))
    patience = 2 if quick else 30
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_loss = math.inf
    best_epoch = max_epochs
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 256):
            batch = order[start : start + 256].to(device)
            loss = _torch_loss(
                model(train_x[batch]), train_y[batch], train_available[batch], beta, supervision
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if validation is None:
            continue
        model.eval()
        with torch.no_grad():
            validation_loss = float(_torch_loss(model(val_x), val_y, val_available, beta, supervision).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    return _TorchBundle(model=model, mean=mean, scale=scale, supervision=supervision, best_epoch=best_epoch)


def _model_predict(model: Any, features: np.ndarray, availability: np.ndarray) -> np.ndarray:
    return model.predict_raw(np.asarray(features, dtype=float), np.asarray(availability, dtype=bool))


def _inner_folds(frame: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    return build_group_folds(frame, n_splits=count, seed=seed)


@dataclass
class _NestedResult:
    model: Any
    calibrated_test: np.ndarray
    raw_test: np.ndarray
    calibrators: list[_Platt] | None
    selection: dict[str, Any]
    router_latency_ms: float


def _measure_batch1_ms(callback: Callable[[np.ndarray, np.ndarray], Any], row: np.ndarray, availability: np.ndarray) -> float:
    sample = np.asarray(row)[None, :]
    mask = np.asarray(availability)[None, :]
    for _ in range(5):
        callback(sample, mask)
    started = time.perf_counter()
    for _ in range(100):
        callback(sample, mask)
    return float((time.perf_counter() - started) * 10.0)


def _nested_fit(
    train_frame: pd.DataFrame,
    train_features: np.ndarray,
    test_features: np.ndarray,
    test_availability: np.ndarray,
    method: str,
    beta: float,
    seed: int,
    inner_folds: int,
    quick: bool,
) -> _NestedResult:
    correct = np.column_stack(
        [pd.to_numeric(train_frame[f"correct__{action}"], errors="coerce").to_numpy(float) for action in ACTIONS]
    )
    availability = np.column_stack(
        [train_frame[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS]
    )
    decisions = np.column_stack(
        [pd.to_numeric(train_frame[f"decision__{action}"], errors="coerce").fillna(0).to_numpy(int) for action in ACTIONS]
    )
    labels = train_frame["label_supported"].to_numpy(int)
    latency = np.column_stack(
        [train_frame[f"latency_ms__{action}"].to_numpy(float) for action in ACTIONS]
    )
    folds = _inner_folds(train_frame, inner_folds, stable_seed(seed, method, beta, "inner"))
    candidates = _mlp_grid(quick) if method.startswith("MLP") else _sklearn_grid(method, quick)
    supervision = method.split("-")[-1]
    candidate_rows: list[dict[str, Any]] = []
    candidate_oof: list[np.ndarray] = []
    candidate_epochs: list[list[int]] = []
    for candidate_index, parameters in enumerate(candidates):
        raw_oof = np.zeros_like(correct, dtype=float)
        epochs: list[int] = []
        for fold in range(inner_folds):
            fit = folds != fold
            validation = folds == fold
            model_seed = stable_seed(seed, method, beta, candidate_index, fold)
            if method.startswith("MLP"):
                model = _fit_mlp(
                    train_features[fit],
                    correct[fit],
                    availability[fit],
                    parameters,
                    beta,
                    supervision,
                    model_seed,
                    validation=(train_features[validation], correct[validation], availability[validation]),
                    quick=quick,
                )
                epochs.append(model.best_epoch)
            else:
                model = _fit_sklearn_heads(
                    method,
                    train_features[fit],
                    correct[fit],
                    availability[fit],
                    parameters,
                    model_seed,
                )
            raw_oof[validation] = _model_predict(model, train_features[validation], availability[validation])
        if supervision == "S1":
            primary = -_masked_bce(raw_oof, correct, availability)
            realized_utility, inner_bacc, inner_latency = _realized_selection_metrics(
                raw_oof, correct, availability, decisions, labels, latency, beta, supervision
            )
        else:
            realized_utility, inner_bacc, inner_latency = _realized_selection_metrics(
                raw_oof, correct, availability, decisions, labels, latency, beta, supervision
            )
            primary = realized_utility
        candidate_rows.append(
            {
                "candidate": candidate_index,
                "parameters": json.dumps(parameters, sort_keys=True),
                "primary": float(primary),
                "inner_utility": float(realized_utility),
                "inner_bacc": float(inner_bacc),
                "inner_latency_ms": float(inner_latency),
                "epoch": int(round(np.median(epochs))) if epochs else None,
            }
        )
        candidate_oof.append(raw_oof)
        candidate_epochs.append(epochs)
    best = max(
        candidate_rows,
        key=lambda row: (
            row["primary"],
            row["inner_bacc"],
            -row["inner_latency_ms"],
            -len(row["parameters"]),
            row["parameters"],
        ),
    )
    best_index = int(best["candidate"])
    parameters = candidates[best_index]
    raw_oof = candidate_oof[best_index]
    calibrators: list[_Platt] | None = None
    if supervision in {"S1", "S2"}:
        calibrators = []
        for action_index in range(len(ACTIONS)):
            mask = availability[:, action_index]
            calibrators.append(_Platt.fit(raw_oof[mask, action_index], correct[mask, action_index]))
    final_seed = stable_seed(seed, method, beta, "outer_final")
    if method.startswith("MLP"):
        fixed_epochs = int(best["epoch"] or (4 if quick else 300))
        model = _fit_mlp(
            train_features,
            correct,
            availability,
            parameters,
            beta,
            supervision,
            final_seed,
            validation=None,
            fixed_epochs=fixed_epochs,
            quick=quick,
        )
    else:
        model = _fit_sklearn_heads(
            method, train_features, correct, availability, parameters, final_seed
        )
    raw_test = _model_predict(model, test_features, test_availability)
    if calibrators is None:
        calibrated_test = raw_test
    else:
        calibrated_test = np.column_stack(
            [
                calibrator.predict(raw_test[:, action_index])
                for action_index, calibrator in enumerate(calibrators)
            ]
        )
        calibrated_test = np.where(test_availability, calibrated_test, 0.0)

    def callback(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        raw = _model_predict(model, values, mask)
        if calibrators is None:
            return masked_argmax(raw, mask)
        calibrated = np.column_stack(
            [calibrators[index].predict(raw[:, index]) for index in range(len(ACTIONS))]
        )
        return masked_argmax(calibrated * cost_discounts(beta, ACTION_COSTS_MS)[None, :], mask)

    router_latency = _measure_batch1_ms(callback, test_features[0], test_availability[0])
    selection = {
        **best,
        "method": method,
        "beta": float(beta),
        "seed": int(seed),
        "train_rows": int(len(train_frame)),
        "inner_folds": int(inner_folds),
        "candidate_count": int(len(candidates)),
        "all_candidates": candidate_rows,
    }
    return _NestedResult(
        model=model,
        calibrated_test=calibrated_test,
        raw_test=raw_test,
        calibrators=calibrators,
        selection=selection,
        router_latency_ms=router_latency,
    )


def _beta_slug(beta: float) -> str:
    return f"{float(beta):g}".replace(".", "p")


def _part_paths(output_dir: Path, method: str, beta: float, seed: int, fold: int) -> tuple[Path, Path]:
    stem = output_dir / "parts" / method.casefold().replace("-", "_") / f"seed_{seed}"
    name = f"beta_{_beta_slug(beta)}__fold_{fold}"
    return stem / f"{name}.parquet", stem / f"{name}.json"


def _prediction_part(
    test: pd.DataFrame,
    test_features: pd.DataFrame,
    probabilities: np.ndarray,
    result: _NestedResult,
    method: str,
    beta: float,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    availability = np.column_stack(
        [test[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS]
    )
    utility = probabilities if method == "MLP-S3" else probabilities * cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    selected = masked_argmax(utility, availability)
    rows = np.arange(len(test))
    decisions = np.column_stack(
        [pd.to_numeric(test[f"decision__{action}"], errors="coerce").fillna(0).to_numpy(int) for action in ACTIONS]
    )
    scores = np.column_stack(
        [pd.to_numeric(test[f"score__{action}"], errors="coerce").fillna(0.5).to_numpy(float) for action in ACTIONS]
    )
    verifier_latency = np.column_stack(
        [test[f"latency_ms__{action}"].to_numpy(float) for action in ACTIONS]
    )
    selected_names = np.asarray(ACTIONS, dtype=object)[selected]
    output = test[["episode_key", "dataset_key", "group_id", "label_supported"]].copy()
    output["outer_fold"] = int(fold)
    output["method"] = method
    output["beta"] = float(beta)
    output["seed"] = int(seed)
    output["selected_action"] = selected_names
    output["router_decision"] = decisions[rows, selected]
    output["probability_supported"] = scores[rows, selected]
    output["correct"] = (output["router_decision"].to_numpy(int) == output["label_supported"].to_numpy(int)).astype(np.int8)
    output["verifier_latency_ms"] = verifier_latency[rows, selected]
    output["feature_latency_ms"] = test_features["feature_latency_ms"].to_numpy(float)
    output["router_latency_ms"] = float(result.router_latency_ms)
    output["end_to_end_latency_ms"] = (
        output["verifier_latency_ms"] + output["feature_latency_ms"] + output["router_latency_ms"]
    )
    output["forced_upgrade"] = ~availability[:, 0]
    correct = np.column_stack(
        [pd.to_numeric(test[f"correct__{action}"], errors="coerce").to_numpy(float) for action in ACTIONS]
    )
    for action_index, action in enumerate(ACTIONS):
        output[f"available__{action}"] = availability[:, action_index]
        output[f"correct__{action}"] = correct[:, action_index]
        output[f"router_probability__{action}"] = probabilities[:, action_index]
        output[f"raw_router_probability__{action}"] = result.raw_test[:, action_index]
    return output.reset_index(drop=True)


def _safe_statistic(callback: Callable[..., Any], *values: Any) -> float:
    try:
        result = callback(*values)
        statistic = result.statistic if hasattr(result, "statistic") else result[0] if isinstance(result, tuple) else result
        return float(statistic) if math.isfinite(float(statistic)) else float("nan")
    except (ValueError, TypeError, FloatingPointError):
        return float("nan")


def _quality_metrics(labels: np.ndarray, decisions: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    hard = np.asarray(decisions, dtype=int)
    score = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    both = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, hard)),
        "balanced_accuracy": float(balanced_accuracy_score(y, hard)) if both else float("nan"),
        "macro_f1": float(f1_score(y, hard, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, hard)) if len(np.unique(hard)) > 1 else 0.0,
        "auroc": float(roc_auc_score(y, score)) if both else float("nan"),
        "supported_recall": float(recall_score(y, hard, pos_label=1, zero_division=0)),
        "unsupported_recall": float(recall_score(y, hard, pos_label=0, zero_division=0)),
        "point_biserial": _safe_statistic(pointbiserialr, y, score),
        "spearman": _safe_statistic(spearmanr, y, score),
        "kendall_tau_b": _safe_statistic(lambda left, right: kendalltau(left, right, variant="b"), y, score),
    }


def _prediction_metrics(part: pd.DataFrame, keys: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = dict(keys or {})
    result.update(
        {
            "rows": int(len(part)),
            "source_groups": int(part["group_id"].nunique()),
            **_quality_metrics(
                part["label_supported"].to_numpy(int),
                part["router_decision"].to_numpy(int),
                part["probability_supported"].to_numpy(float),
            ),
            "mean_verifier_latency_ms": float(part["verifier_latency_ms"].mean()),
            "mean_feature_latency_ms": float(part["feature_latency_ms"].mean()),
            "mean_router_latency_ms": float(part["router_latency_ms"].mean()),
            "mean_end_to_end_latency_ms": float(part["end_to_end_latency_ms"].mean()),
            "forced_upgrade_rate": float(part["forced_upgrade"].mean()),
        }
    )
    counts = part["selected_action"].value_counts().to_dict()
    for action in ACTIONS:
        result[f"rate__{action}"] = float(counts.get(action, 0) / len(part))
    return result


def _metrics_table(predictions: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, part in predictions.groupby(list(group_columns), sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        rows.append(_prediction_metrics(part, dict(zip(group_columns, key_values))))
    return pd.DataFrame(rows)


def _seed_summary(metrics: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    numeric = [column for column in metrics.columns if column not in {*keys, "seed"} and pd.api.types.is_numeric_dtype(metrics[column])]
    rows: list[dict[str, Any]] = []
    for key, part in metrics.groupby(list(keys), sort=True, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(keys, values))
        row["seed_count"] = int(part["seed"].nunique())
        for column in numeric:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _pareto(summary: pd.DataFrame) -> pd.DataFrame:
    work = summary.copy()
    keep: list[bool] = []
    for index, row in work.iterrows():
        dominated = (
            (work["balanced_accuracy_mean"] >= float(row["balanced_accuracy_mean"]))
            & (work["mean_end_to_end_latency_ms_mean"] <= float(row["mean_end_to_end_latency_ms_mean"]))
            & (
                (work["balanced_accuracy_mean"] > float(row["balanced_accuracy_mean"]))
                | (work["mean_end_to_end_latency_ms_mean"] < float(row["mean_end_to_end_latency_ms_mean"]))
            )
        )
        dominated.loc[index] = False
        keep.append(not bool(dominated.any()))
    work["pareto"] = keep
    return work[work["pareto"]].sort_values(
        ["mean_end_to_end_latency_ms_mean", "balanced_accuracy_mean"], ascending=[True, False]
    )


def _ece(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(labels, dtype=int)
    score = np.clip(np.asarray(probability, dtype=float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(score, edges[1:-1], right=True), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            value += float(mask.mean()) * abs(float(score[mask].mean()) - float(y[mask].mean()))
    return value


def _correctness_calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = predictions[~predictions["method"].eq("MLP-S3")]
    for key, part in work.groupby(["method", "beta", "seed"], sort=True):
        method, beta, seed = key
        for action in ACTIONS:
            mask = part[f"available__{action}"].astype(bool).to_numpy()
            target = part.loc[mask, f"correct__{action}"].to_numpy(int)
            probability = np.clip(part.loc[mask, f"router_probability__{action}"].to_numpy(float), 1e-7, 1 - 1e-7)
            rows.append(
                {
                    "method": method,
                    "beta": float(beta),
                    "seed": int(seed),
                    "action": action,
                    "rows": int(mask.sum()),
                    "nll": float(log_loss(target, probability, labels=[0, 1])),
                    "brier": float(brier_score_loss(target, probability)),
                    "ece_15": _ece(target, probability),
                    "auroc": float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _baseline_prediction(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    selected: np.ndarray,
    *,
    feature_latency: np.ndarray | None = None,
    router_latency: np.ndarray | None = None,
) -> pd.DataFrame:
    rows = np.arange(len(frame))
    availability = np.column_stack([frame[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS])
    decisions = np.column_stack(
        [pd.to_numeric(frame[f"decision__{action}"], errors="coerce").fillna(0).to_numpy(int) for action in ACTIONS]
    )
    scores = np.column_stack(
        [pd.to_numeric(frame[f"score__{action}"], errors="coerce").fillna(0.5).to_numpy(float) for action in ACTIONS]
    )
    latency = np.column_stack([frame[f"latency_ms__{action}"].to_numpy(float) for action in ACTIONS])
    output = frame[["episode_key", "dataset_key", "group_id", "label_supported"]].copy()
    output["selected_action"] = np.asarray(ACTIONS, dtype=object)[selected]
    output["router_decision"] = decisions[rows, selected]
    output["probability_supported"] = scores[rows, selected]
    output["verifier_latency_ms"] = latency[rows, selected]
    output["feature_latency_ms"] = 0.0 if feature_latency is None else feature_latency
    output["router_latency_ms"] = 0.0 if router_latency is None else router_latency
    output["end_to_end_latency_ms"] = output["verifier_latency_ms"] + output["feature_latency_ms"] + output["router_latency_ms"]
    output["forced_upgrade"] = ~availability[:, 0]
    return output


def _baseline_metrics(frame: pd.DataFrame, features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    availability = np.column_stack([frame[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS])
    correct = np.column_stack(
        [pd.to_numeric(frame[f"correct__{action}"], errors="coerce").fillna(0).to_numpy(float) for action in ACTIONS]
    )
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(ACTIONS):
        preferred = np.zeros_like(availability, dtype=float)
        preferred[:, index] = 1.0
        selected = masked_argmax(preferred, availability)
        part = _baseline_prediction(frame, features, selected)
        rows.append(_prediction_metrics(part, {"baseline": f"always_{action}"}))
    oracle_utility = correct * cost_discounts(0.0, ACTION_COSTS_MS)[None, :]
    costs = np.asarray([ACTION_COSTS_MS[action] for action in ACTIONS])
    oracle_utility += (costs.max() - costs)[None, :] * 1e-9
    oracle_selected = masked_argmax(oracle_utility, availability)
    rows.append(_prediction_metrics(_baseline_prediction(frame, features, oracle_selected), {"baseline": "cheapest_correct_oracle"}))
    config_keys = ["method", "beta", "seed"]
    for key, router in predictions.groupby(config_keys, sort=True):
        method, beta, seed = key
        rates = router["selected_action"].value_counts(normalize=True).reindex(ACTIONS, fill_value=0).to_numpy(float)
        generator = np.random.default_rng(stable_seed(int(seed), method, beta, "matched_random"))
        weights = np.tile(rates, (len(frame), 1)) * availability
        weights /= weights.sum(axis=1, keepdims=True)
        draws = generator.random(len(frame))
        selected = (draws[:, None] > np.cumsum(weights, axis=1)).sum(axis=1)
        router_order = router.set_index("episode_key").loc[frame["episode_key"]]
        part = _baseline_prediction(
            frame,
            features,
            selected,
            feature_latency=router_order["feature_latency_ms"].to_numpy(float),
            router_latency=router_order["router_latency_ms"].to_numpy(float),
        )
        rows.append(
            _prediction_metrics(
                part,
                {
                    "baseline": "matched_random",
                    "reference_method": method,
                    "reference_beta": float(beta),
                    "reference_seed": int(seed),
                },
            )
        )
    return pd.DataFrame(rows)


def _bootstrap_ci(predictions: pd.DataFrame, frame: pd.DataFrame, draws: int) -> pd.DataFrame:
    groups = sorted(frame["group_id"].astype(str).unique())
    group_to_index = {group: index for index, group in enumerate(groups)}
    base_mid = _baseline_prediction(
        frame,
        pd.DataFrame(index=frame.index),
        np.full(len(frame), 1, dtype=int),
    )
    base_mid["group_id"] = base_mid["group_id"].astype(str)

    def group_arrays(part: pd.DataFrame) -> dict[str, np.ndarray]:
        arrays = {name: np.zeros(len(groups), dtype=float) for name in ("g0", "g1", "c0", "c1", "rows", "lat")}
        for group, values in part.groupby("group_id", sort=False):
            index = group_to_index[str(group)]
            labels = values["label_supported"].to_numpy(int)
            decisions = values["router_decision"].to_numpy(int)
            arrays["g0"][index] = np.sum(labels == 0)
            arrays["g1"][index] = np.sum(labels == 1)
            arrays["c0"][index] = np.sum((labels == 0) & (decisions == 0))
            arrays["c1"][index] = np.sum((labels == 1) & (decisions == 1))
            arrays["rows"][index] = len(values)
            arrays["lat"][index] = values["end_to_end_latency_ms"].sum()
        return arrays

    baseline_arrays = group_arrays(base_mid)
    output: list[dict[str, Any]] = []
    for (method, beta), method_part in predictions.groupby(["method", "beta"], sort=True):
        seed_arrays = {int(seed): group_arrays(part) for seed, part in method_part.groupby("seed", sort=True)}
        generator = np.random.default_rng(stable_seed(2026, method, beta, "bootstrap"))
        bacc_values: list[float] = []
        latency_values: list[float] = []
        delta_bacc: list[float] = []
        delta_latency: list[float] = []
        for _ in range(int(draws)):
            sample = generator.integers(0, len(groups), size=len(groups))
            base_g0 = baseline_arrays["g0"][sample].sum()
            base_g1 = baseline_arrays["g1"][sample].sum()
            base_rows = baseline_arrays["rows"][sample].sum()
            base_bacc = 0.5 * (
                baseline_arrays["c0"][sample].sum() / base_g0
                + baseline_arrays["c1"][sample].sum() / base_g1
            )
            base_latency = baseline_arrays["lat"][sample].sum() / base_rows
            seed_bacc = []
            seed_latency = []
            for arrays in seed_arrays.values():
                g0 = arrays["g0"][sample].sum()
                g1 = arrays["g1"][sample].sum()
                seed_bacc.append(0.5 * (arrays["c0"][sample].sum() / g0 + arrays["c1"][sample].sum() / g1))
                seed_latency.append(arrays["lat"][sample].sum() / arrays["rows"][sample].sum())
            bacc = float(np.mean(seed_bacc))
            latency = float(np.mean(seed_latency))
            bacc_values.append(bacc)
            latency_values.append(latency)
            delta_bacc.append(bacc - base_bacc)
            delta_latency.append(latency - base_latency)
        output.append(
            {
                "method": method,
                "beta": float(beta),
                "draws": int(draws),
                "balanced_accuracy_ci_low": float(np.quantile(bacc_values, 0.025)),
                "balanced_accuracy_ci_high": float(np.quantile(bacc_values, 0.975)),
                "end_to_end_latency_ms_ci_low": float(np.quantile(latency_values, 0.025)),
                "end_to_end_latency_ms_ci_high": float(np.quantile(latency_values, 0.975)),
                "delta_bacc_vs_always_mid_ci_low": float(np.quantile(delta_bacc, 0.025)),
                "delta_bacc_vs_always_mid_ci_high": float(np.quantile(delta_bacc, 0.975)),
                "delta_latency_vs_always_mid_ci_low": float(np.quantile(delta_latency, 0.025)),
                "delta_latency_vs_always_mid_ci_high": float(np.quantile(delta_latency, 0.975)),
            }
        )
    return pd.DataFrame(output)


def build_oof_reports(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_draws: int,
) -> dict[str, pd.DataFrame]:
    pooled = _metrics_table(predictions, ["method", "beta", "seed"])
    datasets = _metrics_table(predictions, ["method", "beta", "seed", "dataset_key"])
    numeric_dataset = [
        column
        for column in datasets.columns
        if column not in {"method", "beta", "seed", "dataset_key"} and pd.api.types.is_numeric_dtype(datasets[column])
    ]
    family_macro = datasets.groupby(["method", "beta", "seed"], as_index=False)[numeric_dataset].mean()
    seed_summary = _seed_summary(pooled, ["method", "beta"])
    pareto = _pareto(seed_summary)
    calibration = _correctness_calibration(predictions)
    baselines = _baseline_metrics(frame, features, predictions)
    bootstrap = _bootstrap_ci(predictions, frame, bootstrap_draws) if bootstrap_draws > 0 else pd.DataFrame()
    outputs = {
        "OOF_CONFIG_METRICS_BY_SEED.csv": pooled,
        "OOF_DATASET_METRICS_BY_SEED.csv": datasets,
        "OOF_FAMILY_MACRO_BY_SEED.csv": family_macro,
        "OOF_SEED_SUMMARY.csv": seed_summary,
        "PARETO_FRONTIER.csv": pareto,
        "CORRECTNESS_CALIBRATION.csv": calibration,
        "BASELINE_METRICS.csv": baselines,
        "BOOTSTRAP_CI.csv": bootstrap,
    }
    for name, table in outputs.items():
        write_csv(output_dir / name, table)
    return outputs


def run_oof(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    *,
    methods: Sequence[str] = METHODS,
    betas: Sequence[float] = BETAS,
    seeds: Sequence[int] = SEEDS,
    inner_folds: int = INNER_FOLDS,
    quick: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_methods = tuple(methods)
    requested_betas = tuple(float(value) for value in betas)
    requested_seeds = tuple(int(value) for value in seeds)
    unknown = sorted(set(requested_methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    if len(frame) != len(features):
        raise ValueError("training frame/features row mismatch")
    values = features.loc[:, FEATURE_COLUMNS].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("training features contain non-finite values")
    folds = frame["fold"].to_numpy(int)
    fold_values = sorted(np.unique(folds).tolist())
    validate_group_folds(frame, folds, n_splits=len(fold_values))
    selection_rows: list[dict[str, Any]] = []
    for method in requested_methods:
        for seed in requested_seeds:
            for fold in fold_values:
                train_mask = folds != fold
                test_mask = folds == fold
                train = frame.loc[train_mask].reset_index(drop=True)
                test = frame.loc[test_mask].reset_index(drop=True)
                x_train = values[train_mask]
                x_test = values[test_mask]
                test_features = features.loc[test_mask].reset_index(drop=True)
                test_availability = np.column_stack(
                    [test[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS]
                )
                if method.endswith("S1"):
                    missing = [
                        beta
                        for beta in requested_betas
                        if not all(path.is_file() for path in _part_paths(output_dir, method, beta, seed, fold))
                    ]
                    if not missing:
                        continue
                    print(f"[oof] start method={method} seed={seed} fold={fold} trained_beta=0", flush=True)
                    nested = _nested_fit(
                        train,
                        x_train,
                        x_test,
                        test_availability,
                        method,
                        0.0,
                        seed,
                        inner_folds,
                        quick,
                    )
                    for beta in missing:
                        part_path, selection_path = _part_paths(output_dir, method, beta, seed, fold)
                        part = _prediction_part(
                            test,
                            test_features,
                            nested.calibrated_test,
                            nested,
                            method,
                            beta,
                            seed,
                            fold,
                        )
                        selection = {
                            **nested.selection,
                            "trained_beta": 0.0,
                            "decision_beta": float(beta),
                            "outer_fold": int(fold),
                            "test_rows": int(len(test)),
                            "router_latency_ms": float(nested.router_latency_ms),
                        }
                        write_json(selection_path, selection)
                        write_parquet(part_path, part)
                        selection_rows.append(selection)
                    print(f"[oof] complete method={method} seed={seed} fold={fold} betas={len(missing)}", flush=True)
                else:
                    for beta in requested_betas:
                        part_path, selection_path = _part_paths(output_dir, method, beta, seed, fold)
                        if part_path.is_file() and selection_path.is_file():
                            continue
                        print(f"[oof] start method={method} beta={beta:g} seed={seed} fold={fold}", flush=True)
                        nested = _nested_fit(
                            train,
                            x_train,
                            x_test,
                            test_availability,
                            method,
                            beta,
                            seed,
                            inner_folds,
                            quick,
                        )
                        part = _prediction_part(
                            test,
                            test_features,
                            nested.calibrated_test,
                            nested,
                            method,
                            beta,
                            seed,
                            fold,
                        )
                        selection = {
                            **nested.selection,
                            "trained_beta": float(beta),
                            "decision_beta": float(beta),
                            "outer_fold": int(fold),
                            "test_rows": int(len(test)),
                            "router_latency_ms": float(nested.router_latency_ms),
                        }
                        write_json(selection_path, selection)
                        write_parquet(part_path, part)
                        selection_rows.append(selection)
                        print(f"[oof] complete method={method} beta={beta:g} seed={seed} fold={fold}", flush=True)
    parts: list[pd.DataFrame] = []
    all_selection: list[dict[str, Any]] = []
    for method, beta, seed, fold in product(requested_methods, requested_betas, requested_seeds, fold_values):
        part_path, selection_path = _part_paths(output_dir, method, beta, seed, fold)
        if not part_path.is_file() or not selection_path.is_file():
            raise AssertionError(f"missing OOF part: {part_path}")
        parts.append(pd.read_parquet(part_path))
        all_selection.append(json.loads(selection_path.read_text()))
    predictions = pd.concat(parts, ignore_index=True)
    expected = len(frame) * len(requested_methods) * len(requested_betas) * len(requested_seeds)
    if len(predictions) != expected:
        raise AssertionError(f"expected {expected} OOF predictions, found {len(predictions)}")
    keys = ["episode_key", "method", "beta", "seed"]
    if predictions.duplicated(keys).any():
        raise AssertionError("duplicate OOF predictions")
    sizes = predictions.groupby(["method", "beta", "seed"]).size()
    if not sizes.eq(len(frame)).all():
        raise AssertionError("OOF configuration coverage is incomplete")
    write_parquet(output_dir / "OOF_PREDICTIONS.parquet", predictions)
    selection_frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "all_candidates"}
            for row in all_selection
        ]
    )
    write_csv(output_dir / "NESTED_SELECTION.csv", selection_frame)
    reports = build_oof_reports(
        predictions,
        frame.reset_index(drop=True),
        features.reset_index(drop=True),
        output_dir,
        bootstrap_draws=0 if quick else BOOTSTRAP_DRAWS,
    )
    return {
        "predictions": predictions,
        "selection": selection_frame,
        "reports": reports,
        "expected_rows": expected,
    }


def feature_rule_manifest() -> dict[str, Any]:
    return {
        "sentence_splitter": "pysbd:en:clean_false",
        "lexical_token_regex": TOKEN_RE.pattern,
        "value_regex": VALUE_RE.pattern,
        "date_regex": DATE_RE.pattern,
        "entity_rules": [
            "multi_token_proper_names",
            "uppercase_acronyms",
            "non_sentence_initial_capitalized_tokens",
        ],
        "bm25": {"k1": 1.5, "b": 0.75, "idf": "log(1+(N-df+0.5)/(df+0.5))"},
        "idf_scope": "current_source_sentences_only",
        "no_entity_value_pair_defaults": {
            "entity_value_colocation": 1.0,
            "conflicting_value_rate": 0.0,
        },
        "logic_markers": sorted(LOGIC_MARKERS),
        "logic_phrases": list(LOGIC_PHRASES),
        "attribution_words": sorted(ATTRIBUTION_WORDS),
        "attribution_phrases": list(ATTRIBUTION_PHRASES),
        "pronouns": sorted(PRONOUNS),
        "stopwords": sorted(STOPWORDS),
    }


def preregistration_payload(
    *,
    matrix_sha256: str,
    input_sha256: str,
    code_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "PREREGISTERED_BEFORE_OOF",
        "scope": "four_dataset_mixed_pool_source_group_oof_only",
        "stop_after": "completed_oof_audit_before_external_evaluation",
        "rows": 6_850,
        "source_groups": 1_241,
        "gold": {"supported": 3_584, "unsupported": 3_266},
        "datasets": {
            "cogensumm_val": 535,
            "frank_train": 2_238,
            "ragtruth_train": 2_988,
            "unisumeval_train": 1_089,
        },
        "assets": {
            "matrix_path": str(MATRIX_RELATIVE),
            "matrix_sha256": matrix_sha256,
            "complete_summary_input_path": str(INPUT_RELATIVE),
            "complete_summary_input_sha256": input_sha256,
        },
        "actions": [
            {"tier": ACTION_TIERS[action], "verifier": action, "mean_latency_ms": ACTION_COSTS_MS[action]}
            for action in ACTIONS
        ],
        "unavailable_policy": "retain_row_mask_action_force_available_upgrade",
        "expected_factkb_unavailable": 8,
        "features": {
            "columns": list(FEATURE_COLUMNS),
            "count": len(FEATURE_COLUMNS),
            "row_local": True,
            "explicit_dataset_or_domain_input": False,
            "verifier_output_input": False,
            "teacher_input": False,
            "neural_encoder_input": False,
            "rules": feature_rule_manifest(),
        },
        "methods": list(METHODS),
        "betas": list(BETAS),
        "seeds": list(SEEDS),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "expected_configurations": len(METHODS) * len(BETAS) * len(SEEDS),
        "expected_oof_prediction_rows": 6_850 * len(METHODS) * len(BETAS) * len(SEEDS),
        "utility": "correctness*exp(-beta*mean_action_latency/high_mean_latency)",
        "supervision": {
            "S1": "unweighted_masked_correctness_bce_then_beta_replay",
            "S2": "unweighted_masked_correctness_bce_plus_utility_gap_weighted_pairwise",
            "S3": "full_information_one_step_expected_reward_plus_entropy_0.01",
            "all_wrong": "correctness_only_for_S1_S2_excluded_from_pairwise_and_policy",
        },
        "hyperparameters": {
            "LR-S1": {"C": [0.1, 1.0, 10.0]},
            "HGB-S1": {
                "max_leaf_nodes": [7, 15],
                "learning_rate": [0.03, 0.1],
                "min_samples_leaf": [20, 50],
                "l2_regularization": 1.0,
                "max_iter": 100,
                "early_stopping": False,
            },
            "MLP": {
                "architecture": [16, 64, 64, 3],
                "normalization": "LayerNorm",
                "activation": "GELU",
                "learning_rate": [3e-4, 1e-3],
                "dropout": [0.1, 0.2],
                "weight_decay": 1e-4,
                "batch_size": 256,
                "max_epochs": 300,
                "patience": 30,
                "gradient_clip": 1.0,
            },
        },
        "inner_selection": {
            "S1": "mean_correctness_nll",
            "S2": "realized_multiplicative_utility_then_bacc_latency_smaller_model",
            "S3": "realized_multiplicative_utility_then_bacc_latency_smaller_model",
        },
        "calibration": "new_inner_oof_platt_per_correctness_head_for_S1_S2_only",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "code_hashes": dict(code_hashes),
        "official_or_sealed_test_rows_read": 0,
    }


def build_preregistration(
    project_root: Path,
    output_dir: Path,
    *,
    code_paths: Sequence[Path],
) -> dict[str, Any]:
    matrix = project_root / MATRIX_RELATIVE
    source_input = project_root / INPUT_RELATIVE
    matrix_hash = sha256_file(matrix)
    input_hash = sha256_file(source_input)
    if matrix_hash != EXPECTED_MATRIX_SHA256:
        raise ValueError(f"frozen matrix hash mismatch: {matrix_hash}")
    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(f"frozen complete-summary input hash mismatch: {input_hash}")
    code_hashes = {str(path.relative_to(project_root)): sha256_file(path) for path in code_paths}
    payload = preregistration_payload(
        matrix_sha256=matrix_hash,
        input_sha256=input_hash,
        code_hashes=code_hashes,
    )
    write_json(output_dir / "PREREG.json", payload)
    return payload


def validate_preregistration(
    project_root: Path,
    output_dir: Path,
    *,
    code_paths: Sequence[Path],
) -> dict[str, Any]:
    path = output_dir / "PREREG.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    expected = build_preregistration_payload_without_write(project_root, code_paths)
    for key in ("assets", "actions", "features", "methods", "betas", "seeds", "hyperparameters", "code_hashes"):
        if payload.get(key) != expected.get(key):
            raise ValueError(f"preregistration drift in {key}")
    return {"status": "PASS", "validated_at_utc": utc_now(), "prereg_path": str(path)}


def build_preregistration_payload_without_write(project_root: Path, code_paths: Sequence[Path]) -> dict[str, Any]:
    return preregistration_payload(
        matrix_sha256=sha256_file(project_root / MATRIX_RELATIVE),
        input_sha256=sha256_file(project_root / INPUT_RELATIVE),
        code_hashes={str(path.relative_to(project_root)): sha256_file(path) for path in code_paths},
    )


def load_frozen_training_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrix_path = project_root / MATRIX_RELATIVE
    input_path = project_root / INPUT_RELATIVE
    if sha256_file(matrix_path) != EXPECTED_MATRIX_SHA256:
        raise ValueError("frozen Router matrix hash mismatch")
    if sha256_file(input_path) != EXPECTED_INPUT_SHA256:
        raise ValueError("frozen complete-summary input hash mismatch")
    matrix = pd.read_parquet(matrix_path)
    source_input = pd.read_parquet(input_path)
    required_matrix = {
        "episode_key",
        "dataset_key",
        "group_id",
        "doc_group_key",
        "fold",
        "label_supported",
        *[f"score__{action}" for action in ACTIONS],
        *[f"available__{action}" for action in ACTIONS],
        *[f"decision__{action}" for action in ACTIONS],
        *[f"correct__{action}" for action in ACTIONS],
        *[f"latency_ms__{action}" for action in ACTIONS],
    }
    missing = sorted(required_matrix - set(matrix.columns))
    if missing:
        raise ValueError(f"frozen matrix missing columns: {missing}")
    required_input = {"episode_key", "doc_group_key", "source_document", "candidate_summary"}
    missing_input = sorted(required_input - set(source_input.columns))
    if missing_input:
        raise ValueError(f"complete-summary input missing columns: {missing_input}")
    if len(matrix) != 6_850 or len(source_input) != 6_850:
        raise ValueError("frozen input row count drift")
    if matrix["episode_key"].duplicated().any() or source_input["episode_key"].duplicated().any():
        raise ValueError("frozen input episode keys are not unique")
    matrix = matrix.sort_values("episode_key").reset_index(drop=True)
    source_input = source_input.sort_values("episode_key").reset_index(drop=True)
    if not matrix["episode_key"].eq(source_input["episode_key"]).all():
        raise ValueError("matrix/complete-summary episode alignment mismatch")
    if not matrix["doc_group_key"].astype(str).eq(source_input["doc_group_key"].astype(str)).all():
        raise ValueError("matrix/complete-summary source group mismatch")
    matrix["label_supported"] = matrix["label_supported"].astype(int)
    if matrix["label_supported"].value_counts().to_dict() != {1: 3_584, 0: 3_266}:
        raise ValueError("frozen gold distribution drift")
    if matrix["group_id"].nunique() != 1_241:
        raise ValueError("frozen source group count drift")
    validate_group_folds(matrix, matrix["fold"].to_numpy(int), n_splits=OUTER_FOLDS)
    unavailable = {action: int((~matrix[f"available__{action}"].astype(bool)).sum()) for action in ACTIONS}
    if unavailable != {"factkb": 8, "granite_guardian_3_1_2b": 0, "qwen30_fast": 0}:
        raise ValueError(f"action availability drift: {unavailable}")
    measured_costs = {
        action: float(matrix[f"latency_ms__{action}"].mean())
        for action in ACTIONS
    }
    for action in ACTIONS:
        if not math.isclose(measured_costs[action], ACTION_COSTS_MS[action], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"frozen action latency drift for {action}: {measured_costs[action]}")
    audit = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "rows": int(len(matrix)),
        "source_groups": int(matrix["group_id"].nunique()),
        "supported": int(matrix["label_supported"].sum()),
        "unsupported": int((1 - matrix["label_supported"]).sum()),
        "datasets": matrix["dataset_key"].value_counts().sort_index().to_dict(),
        "outer_fold_rows": matrix["fold"].value_counts().sort_index().to_dict(),
        "unavailable": unavailable,
        "mean_action_latency_ms": measured_costs,
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "complete_summary_input_sha256": EXPECTED_INPUT_SHA256,
        "official_or_sealed_test_rows_read": 0,
    }
    return matrix, source_input, audit


def build_feature_asset(
    matrix: pd.DataFrame,
    source_input: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_frame = matrix[["episode_key"]].merge(
        source_input[["episode_key", "source_document", "candidate_summary"]],
        on="episode_key",
        how="left",
        validate="one_to_one",
    )
    features = build_compact16_feature_frame(input_frame)
    output = pd.concat(
        [input_frame[["episode_key"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1
    )
    write_parquet(output_dir / "COMPACT16_FEATURES.parquet", output)
    audit = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "rows": int(len(output)),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "all_finite": bool(np.isfinite(output.loc[:, FEATURE_COLUMNS].to_numpy(float)).all()),
        "mean_feature_latency_ms": float(output["feature_latency_ms"].mean()),
        "p95_feature_latency_ms": float(output["feature_latency_ms"].quantile(0.95)),
        "input_sha256": EXPECTED_INPUT_SHA256,
        "feature_file_sha256": sha256_file(output_dir / "COMPACT16_FEATURES.parquet"),
        "rules": feature_rule_manifest(),
        "official_or_sealed_test_rows_read": 0,
    }
    write_json(output_dir / "FEATURE_AUDIT.json", audit)
    return output, audit


def load_feature_asset(matrix: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    path = output_dir / "COMPACT16_FEATURES.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    features = pd.read_parquet(path)
    if len(features) != len(matrix) or not features["episode_key"].astype(str).eq(matrix["episode_key"].astype(str)).all():
        raise ValueError("Compact-16 feature asset alignment mismatch")
    if not np.isfinite(features.loc[:, FEATURE_COLUMNS].to_numpy(float)).all():
        raise ValueError("Compact-16 feature asset contains non-finite values")
    return features


def canary_indices(matrix: pd.DataFrame, groups_per_fold: int = 8) -> np.ndarray:
    selected: set[str] = set()
    for fold in sorted(matrix["fold"].unique()):
        groups = sorted(
            matrix.loc[matrix["fold"].eq(fold), "group_id"].astype(str).unique(),
            key=lambda group: (hashlib.sha256(f"canary:{fold}:{group}".encode()).hexdigest(), group),
        )
        selected.update(groups[:groups_per_fold])
    return matrix["group_id"].astype(str).isin(selected).to_numpy()


def audit_completed_run(output_dir: Path) -> dict[str, Any]:
    predictions_path = output_dir / "OOF_PREDICTIONS.parquet"
    predictions = pd.read_parquet(predictions_path)
    expected = 6_850 * len(METHODS) * len(BETAS) * len(SEEDS)
    configurations = predictions[["method", "beta", "seed"]].drop_duplicates()
    files = [
        "OOF_PREDICTIONS.parquet",
        "NESTED_SELECTION.csv",
        "OOF_CONFIG_METRICS_BY_SEED.csv",
        "OOF_DATASET_METRICS_BY_SEED.csv",
        "OOF_FAMILY_MACRO_BY_SEED.csv",
        "OOF_SEED_SUMMARY.csv",
        "PARETO_FRONTIER.csv",
        "CORRECTNESS_CALIBRATION.csv",
        "BASELINE_METRICS.csv",
        "BOOTSTRAP_CI.csv",
        "COMPACT16_FEATURES.parquet",
        "FEATURE_AUDIT.json",
        "PREREG.json",
    ]
    missing = [name for name in files if not (output_dir / name).is_file()]
    status = (
        len(predictions) == expected
        and len(configurations) == len(METHODS) * len(BETAS) * len(SEEDS)
        and not predictions.duplicated(["episode_key", "method", "beta", "seed"]).any()
        and not missing
    )
    audit = {
        "status": "PASS" if status else "FAIL",
        "schema_version": SCHEMA_VERSION,
        "completed_at_utc": utc_now(),
        "rows": int(len(predictions)),
        "expected_rows": int(expected),
        "configurations": int(len(configurations)),
        "expected_configurations": int(len(METHODS) * len(BETAS) * len(SEEDS)),
        "missing_files": missing,
        "duplicate_predictions": int(predictions.duplicated(["episode_key", "method", "beta", "seed"]).sum()),
        "files": {name: sha256_file(output_dir / name) for name in files if (output_dir / name).is_file()},
        "matrix_sha256": EXPECTED_MATRIX_SHA256,
        "complete_summary_input_sha256": EXPECTED_INPUT_SHA256,
        "official_or_sealed_test_rows_read": 0,
        "external_evaluation_rows_read": 0,
    }
    write_json(output_dir / "AUDIT.json", audit)
    if not status:
        raise AssertionError(f"completed run audit failed: {audit}")
    return audit
