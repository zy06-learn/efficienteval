from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


SCHEMA_VERSION = "unified_summary_verifiers_v1"
DATASET_NAME = "mixed_complete_summaries_v1"
CANARY_NAME = "mixed_complete_summaries_v1_canary"

VERIFIERS = (
    "factkb",
    "factcc",
    "lettuce_v2",
    "hhem",
    "factcg",
    "minicheck_dbta",
    "minicheck_ft5",
    "alignscore",
    "wecheck",
    "granite_guardian_3_2_8b_factuality",
    "qwen30_judge",
    "granite_guardian_4_1_3b_factuality_lora",
    "granite_guardian_3_1_2b",
    "granite_guardian_3_2_3b_a800m",
    "qwen30_fast",
)

LOCAL_VERIFIERS = VERIFIERS[:9]
API_VERIFIERS = VERIFIERS[9:]

MODEL_REVISIONS = {
    "granite_guardian_3_2_8b_factuality": (
        "ibm-granite/granite-guardian-3.2-8b-factuality-detection@"
        "de0c27b0ed657529269b573e106a3c72d18f85f9"
    ),
    "qwen30_judge": (
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8@"
        "5a5a776300a41aaa681dd7ff0106608ef2bc90db"
    ),
    "granite_guardian_4_1_3b_factuality_lora": (
        "ibm-granite/granite-4.1-3b@c0650403e44e78ec0262dab1c90914c65b196c4e+"
        "ibm-granite/granitelib-guardian-r1.0@c93c0545e8e74ec75ce60ba642065fc800589c22:"
        "factuality-detection/granite-4.1-3b/lora"
    ),
    "granite_guardian_3_1_2b": (
        "ibm-granite/granite-guardian-3.1-2b@81145486e85c6c82c01e759c0356d9d6da4d21a5"
    ),
    "granite_guardian_3_2_3b_a800m": (
        "ibm-granite/granite-guardian-3.2-3b-a800m@"
        "3de033d89b499a18d9a573b5192bf3b967ef48c5"
    ),
    "qwen30_fast": (
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8@"
        "5a5a776300a41aaa681dd7ff0106608ef2bc90db"
    ),
}

PROTOCOLS = {
    "factkb": "summary_once_fullclaim_sourcewin512_overlap64_max_oof_v1",
    "factcc": "summary_once_fullclaim_sourcewin512_overlap64_max_oof_v1",
    "lettuce_v2": "summary_once_official_sourcewin512_overlap64_span_oof_v1",
    "hhem": "summary_once_fullclaim_sourcewin512_overlap64_max_oof_v1",
    "factcg": "summary_once_fullclaim_sourcewin2048_overlap64_max_oof_v1",
    "minicheck_dbta": "summary_once_fullclaim_sourcewin2048_overlap64_max_oof_v1",
    "minicheck_ft5": "summary_once_fullclaim_sourcewin2048_overlap64_max_oof_v1",
    "alignscore": "summary_once_native_nli_sp_fullsource_oof_v1",
    "wecheck": "summary_once_fullclaim_sourcewin512_overlap64_max_oof_v1",
    "granite_guardian_3_2_8b_factuality": "summary_once_official_factuality_evidence_select_oof_v1",
    "qwen30_judge": "summary_once_json_prefix_label_probability_no_cot_oof_v2",
    "granite_guardian_4_1_3b_factuality_lora": "summary_once_official_lora_json_evidence_select_oof_v1",
    "granite_guardian_3_1_2b": "summary_once_official_groundedness_evidence_select_oof_v1",
    "granite_guardian_3_2_3b_a800m": "summary_once_official_groundedness_evidence_select_oof_v1",
    "qwen30_fast": "summary_once_binary_token_logprob_no_cot_oof_v1",
}

SCORING_COLUMNS = [
    "episode_key",
    "episode_id",
    "dataset_key",
    "role",
    "doc_group_key",
    "source_document",
    "candidate_summary",
    "source_token_count",
    "summary_token_count",
    "semantic_input_tokens",
]

ROUTER_FEATURE_COLUMNS = [
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


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def prepare_inputs(cohort_path: Path, output_dir: Path) -> dict[str, Any]:
    cohort_path = Path(cohort_path)
    output_dir = Path(output_dir)
    cohort = pd.read_parquet(cohort_path)
    required = {
        "episode_key",
        "episode_id",
        "dataset_key",
        "role",
        "doc_group_key",
        "group_id",
        "fold",
        "label_supported",
        "source_document",
        "candidate_sentence",
        "source_token_count",
        "claim_token_count",
        "content_doc_key",
        "feature_latency_ms",
        *ROUTER_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"mixed cohort missing columns: {missing}")
    if len(cohort) != 6850 or cohort["episode_key"].duplicated().any():
        raise ValueError("mixed cohort identity/row-count contract failed")
    if set(cohort["fold"].astype(int)) != set(range(5)):
        raise ValueError("mixed cohort must contain five folds")
    if cohort.groupby("group_id")["fold"].nunique().max() != 1:
        raise ValueError("source-group fold leakage detected")

    scoring = pd.DataFrame(
        {
            "episode_key": cohort["episode_key"].astype(str),
            "episode_id": cohort["episode_id"].astype(str),
            "dataset_key": cohort["dataset_key"].astype(str),
            "role": cohort["role"].astype(str),
            "doc_group_key": cohort["doc_group_key"].astype(str),
            "source_document": cohort["source_document"].astype(str),
            "candidate_summary": cohort["candidate_sentence"].astype(str),
            "source_token_count": cohort["source_token_count"].astype(int),
            "summary_token_count": cohort["claim_token_count"].astype(int),
        }
    )
    scoring["semantic_input_tokens"] = (
        scoring["source_token_count"] + scoring["summary_token_count"]
    )
    scoring = scoring[SCORING_COLUMNS]
    train_index = cohort[
        [
            "episode_key",
            "episode_id",
            "dataset_key",
            "role",
            "group_id",
            "doc_group_key",
            "content_doc_key",
            "fold",
            "label_supported",
            "feature_latency_ms",
            *ROUTER_FEATURE_COLUMNS,
        ]
    ].copy()
    train_index["label_supported"] = train_index["label_supported"].astype(np.int8)
    numeric = train_index[["feature_latency_ms", *ROUTER_FEATURE_COLUMNS]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("router feature matrix contains non-finite values")
    scoring_path = output_dir / f"{DATASET_NAME}.parquet"
    index_path = output_dir / "TRAIN_INDEX.parquet"
    atomic_parquet(scoring_path, scoring)
    atomic_parquet(index_path, train_index)

    canary_keys = (
        scoring.sort_values(
            ["summary_token_count", "source_token_count", "episode_key"],
            ascending=[False, False, True],
        )
        .drop_duplicates("doc_group_key")
        .head(20)["episode_key"]
    )
    canary = scoring[scoring["episode_key"].isin(set(canary_keys))].copy()
    canary = canary.set_index("episode_key").loc[list(canary_keys)].reset_index()
    canary_path = output_dir / f"{CANARY_NAME}.parquet"
    atomic_parquet(canary_path, canary)

    cohort_sha = sha256_file(cohort_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "READY_FOR_GOLD_FREE_SCORING",
        "created_at_utc": utc_now(),
        "rows": len(scoring),
        "canary_rows": len(canary),
        "dataset_rows": scoring["dataset_key"].value_counts().sort_index().to_dict(),
        "scoring_unit": "one_complete_source_plus_one_complete_summary",
        "summary_splitting": False,
        "source_handling": "full context or fixed label-free source windows/evidence selection",
        "scoring_input_has_gold": False,
        "official_or_sealed_test_rows_read": 0,
        "cohort_path": str(cohort_path),
        "cohort_sha256": cohort_sha,
        "scoring_path": str(scoring_path),
        "scoring_sha256": sha256_file(scoring_path),
        "canary_path": str(canary_path),
        "canary_sha256": sha256_file(canary_path),
        "train_index_path": str(index_path),
        "train_index_sha256": sha256_file(index_path),
    }
    atomic_json(output_dir / "INPUT_MANIFEST.json", manifest)
    prereg = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_SCORING",
        "frozen_at_utc": utc_now(),
        "verifiers": list(VERIFIERS),
        "rows": 6850,
        "folds": 5,
        "threshold": "held-out fold excluded; pooled train-fold BAcc maximum",
        "primary_quality": "pooled OOF balanced_accuracy",
        "primary_speed": "strict batch-1 steady-state mean latency_ms per summary",
        "efficiency_metrics": [
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "summaries_per_second",
            "semantic_tokens_per_second",
            "model_input_tokens_mean",
            "model_forward_calls_mean",
        ],
        "training_asset": "wide OOF score/decision/correctness/cost/availability matrix",
        "router_feature_columns": list(ROUTER_FEATURE_COLUMNS),
        "factkb_overflow": "no summary truncation; unavailable and forced escalation",
        "model_load_excluded": True,
        "batch_size": 1,
        "warmup_forwards": 5,
        "protocols": PROTOCOLS,
        "input_manifest_sha256": sha256_file(output_dir / "INPUT_MANIFEST.json"),
        "forbidden": [
            "summary splitting",
            "summary truncation",
            "test-fold threshold tuning",
            "sealed or official test access",
            "overwriting legacy results",
        ],
    }
    atomic_json(output_dir.parent / "PREREG.json", prereg)
    return manifest


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    original = getattr(tokenizer, "model_max_length", None)
    if original is not None:
        tokenizer.model_max_length = max(int(original), 1_000_000)
    try:
        values = tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]
    finally:
        if original is not None:
            tokenizer.model_max_length = original
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _encoded_length(encode_pair: Callable[[str], Mapping[str, Any]], chunk: str) -> int:
    values = encode_pair(chunk)["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    return len(values)


def source_windows_preserving_summary(
    tokenizer: Any,
    source: str,
    encode_pair: Callable[[str], Mapping[str, Any]],
    *,
    max_length: int,
    overlap_tokens: int = 64,
) -> dict[str, Any]:
    empty_length = _encoded_length(encode_pair, "")
    if empty_length > max_length:
        return {
            "available": False,
            "reason": f"context_overflow:summary_prompt_tokens={empty_length}>max={max_length}",
        }
    source_ids = _token_ids(tokenizer, source)
    budget = max(1, max_length - empty_length)
    overlap = min(int(overlap_tokens), max(0, budget // 4))
    chunks: list[str] = []
    prompt_counts: list[int] = []
    start = 0
    while start < max(1, len(source_ids)):
        ids = source_ids[start : start + budget] if source_ids else []
        chunk = tokenizer.decode(ids, skip_special_tokens=True) if ids else ""
        length = _encoded_length(encode_pair, chunk)
        while length > max_length and len(ids) > 1:
            ids = ids[: max(1, len(ids) - (length - max_length) - 1)]
            chunk = tokenizer.decode(ids, skip_special_tokens=True)
            length = _encoded_length(encode_pair, chunk)
        if length > max_length:
            return {
                "available": False,
                "reason": f"context_overflow:minimum_prompt_tokens={length}>max={max_length}",
            }
        chunks.append(chunk)
        prompt_counts.append(length)
        if not source_ids or start + len(ids) >= len(source_ids):
            break
        advance = len(ids) - overlap
        if advance <= 0:
            raise ValueError("source window cannot advance")
        start += advance
    return {
        "available": True,
        "chunks": chunks,
        "prompt_token_counts": prompt_counts,
        "source_model_tokens": len(source_ids),
        "overlap_tokens": overlap,
    }


def unavailable_output(reason: str) -> dict[str, Any]:
    return {
        "score": 0.5,
        "parse_ok": True,
        "aux": {
            "available": False,
            "context_overflow": True,
            "unavailable_reason": reason,
            "native_label": "UNAVAILABLE",
            "num_model_calls": 0,
            "model_forward_calls": 0,
            "forward_items": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "summary_split": False,
        },
    }


class PairWindowScorer:
    def __init__(self, base: Any, *, verifier: str, order: str, max_length: int = 512):
        self.base = base
        self.verifier = verifier
        self.order = order
        self.max_length = int(max_length)
        self.tokenizer = base.tokenizer
        self.model_id = str(base.model_id)

    def _encode(self, summary: str, chunk: str) -> Mapping[str, Any]:
        left, right = (summary, chunk) if self.order == "claim_doc" else (chunk, summary)
        return self.tokenizer(
            left,
            right,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != 1 or len(claims) != 1:
            raise ValueError(f"{self.verifier} requires strict batch=1")
        summary = str(claims[0])
        packed = source_windows_preserving_summary(
            self.tokenizer,
            str(docs[0]),
            lambda chunk: self._encode(summary, chunk),
            max_length=self.max_length,
        )
        if not packed["available"]:
            return [unavailable_output(str(packed["reason"]))]
        values = [self.base.score_batch([chunk], [summary])[0] for chunk in packed["chunks"]]
        best = max(range(len(values)), key=lambda index: float(values[index]["score"]))
        output = dict(values[best])
        aux = dict(output.get("aux") or {})
        aux.update(
            {
                "available": True,
                "summary_split": False,
                "source_chunked": len(values) > 1,
                "source_window_count": len(values),
                "prompt_token_counts": packed["prompt_token_counts"],
                "input_tokens": int(sum(packed["prompt_token_counts"])),
                "num_model_calls": len(values),
                "model_forward_calls": len(values),
                "forward_items": len(values),
                "aggregation_rule": "max_support_over_source_windows",
            }
        )
        output["aux"] = aux
        return [output]

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


class SummaryPreservingMiniCheck:
    def __init__(self, base: Any, *, verifier: str, max_length: int = 2048):
        self.base = base
        self.verifier = verifier
        self.max_length = int(max_length)
        self.tokenizer = base.tokenizer
        self.model_id = str(base.model_id)

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != 1 or len(claims) != 1:
            raise ValueError(f"{self.verifier} requires strict batch=1")
        summary = str(claims[0])
        render = lambda chunk: self.base.input_prefix + self.tokenizer.eos_token.join([chunk, summary])
        encode = lambda chunk: self.tokenizer(
            render(chunk), add_special_tokens=True, truncation=False, padding=False
        )
        packed = source_windows_preserving_summary(
            self.tokenizer, str(docs[0]), encode, max_length=self.max_length
        )
        if not packed["available"]:
            return [unavailable_output(str(packed["reason"]))]
        texts = [render(chunk) for chunk in packed["chunks"]]
        probs, logits = self.base._support_probs_with_logits(texts)
        best = max(range(len(probs)), key=probs.__getitem__)
        return [
            {
                "score": float(probs[best]),
                "aux": {
                    "available": True,
                    "summary_split": False,
                    "source_chunked": len(texts) > 1,
                    "source_window_count": len(texts),
                    "prompt_token_counts": packed["prompt_token_counts"],
                    "input_tokens": int(sum(packed["prompt_token_counts"])),
                    "num_model_calls": 1,
                    "model_forward_calls": 1,
                    "forward_items": len(texts),
                    "native_label": self.base.label_space[int(probs[best] >= 0.5)],
                    "native_label_space": list(self.base.label_space),
                    "native_logits": [float(value) for value in logits[best]],
                    "aggregation_rule": "max_support_over_source_windows",
                },
            }
        ]


class SummaryPreservingFactCG:
    max_length = 2048

    def __init__(self, base: Any):
        self.base = base
        self.tokenizer = base.tokenizer
        self.model_id = str(base.model_id)

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != 1 or len(claims) != 1:
            raise ValueError("factcg requires strict batch=1")
        summary = str(claims[0])
        render = lambda chunk: self.base._prompt.format(document=chunk, claim=summary)
        encode = lambda chunk: self.tokenizer(
            render(chunk), add_special_tokens=True, truncation=False, padding=False
        )
        packed = source_windows_preserving_summary(
            self.tokenizer, str(docs[0]), encode, max_length=self.max_length
        )
        if not packed["available"]:
            return [unavailable_output(str(packed["reason"]))]
        support: list[float] = []
        logits_rows: list[list[float]] = []
        for chunk in packed["chunks"]:
            encoded = self.tokenizer(
                [render(chunk)],
                max_length=self.max_length,
                truncation=False,
                padding=False,
                return_tensors="pt",
            ).to(self.base.device)
            with self.base._torch.inference_mode():
                logits = self.base.model(**encoded).logits
            probs = self.base._torch.softmax(logits, dim=-1)
            support.append(float(probs[0, 1].cpu()))
            logits_rows.append([float(value) for value in logits[0].cpu().tolist()])
        best = max(range(len(support)), key=support.__getitem__)
        return [
            {
                "score": support[best],
                "aux": {
                    "available": True,
                    "summary_split": False,
                    "source_chunked": len(support) > 1,
                    "source_window_count": len(support),
                    "prompt_token_counts": packed["prompt_token_counts"],
                    "input_tokens": int(sum(packed["prompt_token_counts"])),
                    "num_model_calls": len(support),
                    "model_forward_calls": len(support),
                    "forward_items": len(support),
                    "native_label": self.base.label_space[int(support[best] >= 0.5)],
                    "native_label_space": list(self.base.label_space),
                    "native_logits": logits_rows[best],
                    "aggregation_rule": "max_support_over_source_windows",
                },
            }
        ]


class MetadataScorer:
    def __init__(self, base: Any, *, verifier: str):
        self.base = base
        self.verifier = verifier
        self.model_id = str(base.model_id)

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        try:
            output = dict(self.base.score_batch(docs, claims)[0])
        except ValueError as exc:
            if "cannot fit" in str(exc) or "no usable source budget" in str(exc):
                return [unavailable_output(f"context_overflow:{exc}")]
            raise
        aux = dict(output.get("aux") or {})
        aux.setdefault("available", True)
        aux.setdefault("summary_split", False)
        aux.setdefault("num_model_calls", int(aux.get("n_chunks", 1)))
        aux.setdefault("model_forward_calls", int(aux.get("num_model_calls", aux.get("n_chunks", 1))))
        aux.setdefault("forward_items", int(aux.get("n_chunks", aux.get("source_window_count", 1))))
        prompt_counts = aux.get("prompt_token_counts")
        if aux.get("input_tokens") is None and isinstance(prompt_counts, list):
            aux["input_tokens"] = int(sum(int(value) for value in prompt_counts))
        aux.setdefault("output_tokens", 0)
        output["aux"] = aux
        return [output]

    def close(self) -> None:
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


def _normalized_token(value: str) -> str:
    return re.sub(r"[^a-z]", "", str(value).casefold())


def support_probability_from_logprobs(top_logprobs: Any) -> float | None:
    for position in top_logprobs or []:
        if not isinstance(position, Mapping):
            continue
        mass = {"yes": 0.0, "no": 0.0, "supported": 0.0, "unsupported": 0.0}
        for token, logprob in position.items():
            normalized = _normalized_token(token)
            if normalized in mass:
                mass[normalized] += math.exp(float(logprob))
        supported = mass["no"] + mass["supported"]
        unsupported = mass["yes"] + mass["unsupported"]
        if supported + unsupported > 0:
            return float(supported / (supported + unsupported))
    return None


def _lexical_terms(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", str(text).casefold()))


def _sentences(text: str) -> list[str]:
    try:
        from verifier_wrappers.global_gamma_calibration import _sentence_offsets

        rows = [str(text)[start:end].strip() for start, end in _sentence_offsets(str(text))]
        return [row for row in rows if row]
    except Exception:
        rows = re.split(r"(?<=[.!?])\s+|\n+", str(text))
        return [row.strip() for row in rows if row.strip()]


class CompletionAPIScorer:
    def __init__(
        self,
        *,
        verifier: str,
        tokenizer_path: Path,
        api_base: str,
        served_model: str,
        max_context: int,
    ) -> None:
        from openai import OpenAI
        from transformers import AutoTokenizer

        self.verifier = verifier
        self.model_id = MODEL_REVISIONS[verifier]
        self.client = OpenAI(base_url=api_base, api_key="EMPTY", timeout=600.0)
        served = {value.id for value in self.client.models.list().data}
        if served_model not in served:
            raise RuntimeError(f"{served_model} not served: {sorted(served)}")
        self.served_model = served_model
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        self.max_context = int(max_context)
        self.max_tokens = 64 if verifier == "qwen30_judge" else 20
        if verifier == "qwen30_fast":
            self.max_tokens = 4

    def _guardian_instruction(self) -> str:
        return (
            "<guardian>As a judge agent, your role is to help assess whether the provided "
            "text meets the given judging criteria, utilizing all available information, "
            "including conversations, documents, and tools.\n\n### Criteria: A factually "
            "incorrect response occurs when the assistant's message contains one or more "
            "factual claims that are unsupported by, inconsistent with, or directly "
            "contradicted by the information provided in the documents or context. This "
            "includes situations where the assistant: introduces details not grounded in "
            "the context, misstates or distorts facts contained within the context, "
            "misinterprets the meaning or implications of the context, supplies erroneous "
            "or conflicting information relative to the context. Even if only a small "
            "portion of the response contains such inaccuracies, the overall message is "
            "considered factually incorrect.\n\n### Scoring Schema: If the last assistant's "
            "text meets the criteria, return 'yes'; otherwise, return 'no'."
        )

    def _prompt(self, source: str, summary: str) -> str:
        if self.verifier in {"granite_guardian_3_1_2b", "granite_guardian_3_2_3b_a800m"}:
            messages = [
                {"role": "context", "content": source},
                {"role": "assistant", "content": summary},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                guardian_config={"risk_name": "groundedness"},
                tokenize=False,
                add_generation_prompt=True,
            )
        if self.verifier == "granite_guardian_3_2_8b_factuality":
            messages = [
                {"role": "context", "content": source},
                {"role": "assistant", "content": summary},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if self.verifier == "granite_guardian_4_1_3b_factuality_lora":
            messages = [
                {"role": "assistant", "content": summary},
                {"role": "user", "content": self._guardian_instruction()},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                documents=[{"doc_id": "0", "text": source}],
                tokenize=False,
                add_generation_prompt=True,
            )
        if self.verifier == "qwen30_fast":
            return (
                "Judge whether the COMPLETE SUMMARY is fully supported by the SOURCE. "
                "Output exactly SUPPORTED or UNSUPPORTED.\nSOURCE:\n"
                f"{source}\n\nCOMPLETE SUMMARY:\n{summary}\n\nLABEL:"
            )
        return (
            "Judge the COMPLETE SUMMARY only against the SOURCE. Do not use outside "
            "knowledge. Return exactly one JSON object with label SUPPORTED or UNSUPPORTED "
            "and support_probability in [0,1]. Use UNSUPPORTED if any part is unsupported "
            "or contradicted.\nSOURCE:\n"
            f"{source}\n\nCOMPLETE SUMMARY:\n{summary}\n\nJSON:"
        )

    def _prompt_tokens(self, source: str, summary: str) -> int:
        return len(_token_ids(self.tokenizer, self._prompt(source, summary)))

    def _select_source(self, source: str, summary: str) -> tuple[str, dict[str, Any]]:
        limit = self.max_context - self.max_tokens
        full_tokens = self._prompt_tokens(source, summary)
        if full_tokens <= limit:
            return source, {
                "source_selected": False,
                "full_prompt_tokens": full_tokens,
                "selected_prompt_tokens": full_tokens,
                "source_sentence_coverage": 1.0,
            }
        empty_tokens = self._prompt_tokens("", summary)
        if empty_tokens > limit:
            raise ValueError(
                f"context_overflow:summary_prompt_tokens={empty_tokens}>limit={limit}"
            )
        rows = _sentences(source)
        terms = _lexical_terms(summary)
        ranked = sorted(
            range(len(rows)),
            key=lambda index: (
                -len(_lexical_terms(rows[index]) & terms),
                index,
            ),
        )
        selected: list[int] = []
        for index in ranked:
            candidate_indices = sorted([*selected, index])
            candidate = "\n".join(rows[value] for value in candidate_indices)
            if self._prompt_tokens(candidate, summary) <= limit:
                selected.append(index)
        if not selected and rows:
            raise ValueError("context_overflow:no_source_sentence_fits_with_complete_summary")
        selected_source = "\n".join(rows[value] for value in sorted(selected))
        selected_tokens = self._prompt_tokens(selected_source, summary)
        return selected_source, {
            "source_selected": True,
            "full_prompt_tokens": full_tokens,
            "selected_prompt_tokens": selected_tokens,
            "source_sentence_coverage": len(selected) / max(len(rows), 1),
            "source_sentence_count": len(rows),
            "selected_sentence_count": len(selected),
            "selection_rule": "label_free_summary_lexical_overlap_then_original_order",
        }

    def _parse(self, text: str, top_logprobs: Any) -> tuple[str | None, float | None]:
        if self.verifier == "qwen30_judge":
            try:
                payload, _ = json.JSONDecoder().raw_decode(str(text).lstrip())
                label = str(payload["label"]).upper()
                probability = float(payload["support_probability"])
                if label in {"SUPPORTED", "UNSUPPORTED"} and 0 <= probability <= 1:
                    return label, probability
            except Exception:
                return None, None
            return None, None
        match = re.search(r"\b(yes|no|supported|unsupported)\b", text, flags=re.I)
        if not match:
            return None, None
        token = match.group(1).casefold()
        label = "SUPPORTED" if token in {"no", "supported"} else "UNSUPPORTED"
        probability = support_probability_from_logprobs(top_logprobs)
        if probability is None:
            probability = 1.0 if label == "SUPPORTED" else 0.0
        return label, probability

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != 1 or len(claims) != 1:
            raise ValueError(f"{self.verifier} requires strict batch=1")
        preprocessing_started = time.perf_counter()
        try:
            source, selection = self._select_source(str(docs[0]), str(claims[0]))
        except ValueError as exc:
            return [unavailable_output(str(exc))]
        prompt = self._prompt(source, str(claims[0]))
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        attempts = 0
        inference_ms = 0.0
        raw = ""
        label = None
        probability = None
        input_tokens = None
        output_tokens = None
        parse_error = None
        while attempts < 2 and label is None:
            attempts += 1
            started = time.perf_counter()
            try:
                response = self.client.completions.create(
                    model=self.served_model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    logprobs=20,
                )
                inference_ms += (time.perf_counter() - started) * 1000.0
                choice = response.choices[0]
                raw = choice.text or ""
                top = getattr(getattr(choice, "logprobs", None), "top_logprobs", None)
                label, probability = self._parse(raw, top)
                usage = getattr(response, "usage", None)
                input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                output_tokens = getattr(usage, "completion_tokens", None) if usage else None
                if label is None:
                    parse_error = f"unparseable:{raw[:200]!r}"
            except Exception as exc:
                inference_ms += (time.perf_counter() - started) * 1000.0
                parse_error = f"{type(exc).__name__}: {exc}"[:500]
        parse_ok = label in {"SUPPORTED", "UNSUPPORTED"} and probability is not None
        return [
            {
                "score": probability if parse_ok else None,
                "parse_ok": parse_ok,
                "parse_error": None if parse_ok else parse_error,
                "aux": {
                    "available": True,
                    "summary_split": False,
                    "native_label": label,
                    "raw_response": raw,
                    "num_model_calls": attempts,
                    "model_forward_calls": attempts,
                    "forward_items": attempts,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_preprocessing_ms": preprocessing_ms,
                    "latency_inference_ms": inference_ms,
                    **selection,
                },
            }
        ]


def build_scorer(
    verifier: str,
    *,
    device: str,
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    api_base: str = "http://127.0.0.1:8001/v1",
    served_model: str = "unified-summary-verifier",
    max_context: int = 16384,
) -> Any:
    if verifier in API_VERIFIERS:
        if tokenizer_path is None:
            raise ValueError(f"{verifier} requires tokenizer_path")
        return CompletionAPIScorer(
            verifier=verifier,
            tokenizer_path=tokenizer_path,
            api_base=api_base,
            served_model=served_model,
            max_context=max_context,
        )
    if verifier == "wecheck":
        if model_path is None:
            raise ValueError("wecheck requires model_path")
        from verifier_wrappers.additional_verifier_scorers_v1 import WeCheckScorer

        return PairWindowScorer(
            WeCheckScorer(model_path=model_path, device=device),
            verifier=verifier,
            order="doc_claim",
        )
    if verifier in {"factkb", "factcc"}:
        from verifier_wrappers.unified_scoring import FactCCPinnedScorer, FactKBPinnedScorer

        if verifier == "factkb":
            base = FactKBPinnedScorer(device=device)
            return PairWindowScorer(base, verifier=verifier, order="claim_doc")
        base = FactCCPinnedScorer(device=device)
        return PairWindowScorer(base, verifier=verifier, order="doc_claim")
    if verifier == "lettuce_v2":
        from verifier_wrappers.extended_scorers import LettuceV2Scorer

        return MetadataScorer(LettuceV2Scorer(device=device), verifier=verifier)
    if verifier in {"hhem", "alignscore"}:
        from verifier_wrappers.primary_scoring import build_primary_scorer

        return MetadataScorer(build_primary_scorer(verifier, device=device), verifier=verifier)
    if verifier == "factcg":
        from verifier_wrappers.primary_scoring import build_primary_scorer

        return SummaryPreservingFactCG(build_primary_scorer(verifier, device=device))
    if verifier in {"minicheck_dbta", "minicheck_ft5"}:
        from verifier_wrappers.primary_scoring import build_primary_scorer

        return SummaryPreservingMiniCheck(
            build_primary_scorer(verifier, device=device), verifier=verifier
        )
    raise ValueError(f"unsupported verifier: {verifier}")


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def score_frame(
    *,
    scorer: Any,
    verifier: str,
    frame: pd.DataFrame,
    input_sha256: str,
    output_dir: Path,
    warmup: int = 5,
    warmup_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_sha256 = sha256_file(Path(__file__))
    cache_path = output_dir / f"{verifier}.jsonl"
    parquet_path = output_dir / f"{verifier}.parquet"
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line_number, line in enumerate(cache_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("verifier") != verifier
                or row.get("input_sha256") != input_sha256
                or row.get("harness_sha256") != harness_sha256
            ):
                raise ValueError(f"cache identity mismatch: {cache_path}:{line_number}")
            cached[str(row["episode_key"])] = row

    warmed = 0
    warmup_source = warmup_frame if warmup_frame is not None else frame
    for row in warmup_source.itertuples(index=False):
        _cuda_sync()
        warm_output = scorer.score_batch(
            [str(row.source_document)], [str(row.candidate_summary)]
        )[0]
        _cuda_sync()
        warm_aux = dict(warm_output.get("aux") or {})
        if bool(warm_aux.get("available", True)) and int(
            warm_aux.get("num_model_calls", 1)
        ) > 0:
            warmed += 1
        if warmed >= warmup:
            break
    if warmed < warmup:
        raise ValueError(f"{verifier} could not complete {warmup} warm-up forwards")

    pending = frame[~frame["episode_key"].astype(str).isin(cached)]
    started_all = time.perf_counter()
    with cache_path.open("a", encoding="utf-8") as sink:
        for offset, episode in enumerate(pending.itertuples(index=False), 1):
            _cuda_sync()
            started = time.perf_counter()
            output = scorer.score_batch(
                [str(episode.source_document)], [str(episode.candidate_summary)]
            )[0]
            _cuda_sync()
            total_ms = (time.perf_counter() - started) * 1000.0
            aux = dict(output.get("aux") or {})
            available = bool(aux.get("available", True))
            parse_ok = bool(output.get("parse_ok", True))
            score = output.get("score")
            if available and (not parse_ok or score is None or not math.isfinite(float(score))):
                raise ValueError(f"{verifier} invalid available output for {episode.episode_key}")
            record = {
                "schema_version": SCHEMA_VERSION,
                "episode_key": str(episode.episode_key),
                "episode_id": str(episode.episode_id),
                "dataset_key": str(episode.dataset_key),
                "doc_group_key": str(episode.doc_group_key),
                "verifier": verifier,
                "model_revision": str(getattr(scorer, "model_id", MODEL_REVISIONS.get(verifier, verifier))),
                "protocol_version": PROTOCOLS[verifier],
                "input_sha256": input_sha256,
                "harness_sha256": harness_sha256,
                "available": available,
                "context_overflow": bool(aux.get("context_overflow", False)),
                "unavailable_reason": aux.get("unavailable_reason"),
                "parse_ok": parse_ok,
                "parse_error": output.get("parse_error"),
                "score": float(score) if score is not None else None,
                "native_label": aux.get("native_label"),
                "latency_total_ms": float(total_ms),
                "latency_preprocessing_ms": aux.get("latency_preprocessing_ms"),
                "latency_inference_ms": aux.get("latency_inference_ms"),
                "semantic_input_tokens": int(episode.semantic_input_tokens),
                "model_input_tokens": int(aux["input_tokens"]) if aux.get("input_tokens") is not None else None,
                "output_tokens": int(aux["output_tokens"]) if aux.get("output_tokens") is not None else None,
                "model_calls": int(aux.get("num_model_calls", 1)),
                "model_forward_calls": int(aux.get("model_forward_calls", aux.get("num_model_calls", 1))),
                "forward_items": int(aux.get("forward_items", 1)),
                "source_window_count": int(
                    aux.get(
                        "source_window_count",
                        aux.get("n_chunks", 1 if available else 0),
                    )
                ),
                "source_selected": bool(aux.get("source_selected", False)),
                "source_sentence_coverage": aux.get("source_sentence_coverage"),
                "source_sentence_count": aux.get("source_sentence_count"),
                "selected_sentence_count": aux.get("selected_sentence_count"),
                "summary_split": bool(aux.get("summary_split", False)),
                "aux_json": json.dumps(aux, ensure_ascii=False, sort_keys=True, default=str),
            }
            sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            cached[record["episode_key"]] = record
            done = len(cached)
            if done % 100 == 0 or done == len(frame):
                elapsed = time.perf_counter() - started_all
                print(
                    f"[{verifier}] {done}/{len(frame)} elapsed_s={elapsed:.1f} "
                    f"last_ms={total_ms:.2f}",
                    flush=True,
                )
    ordered = pd.DataFrame([cached[str(key)] for key in frame["episode_key"].astype(str)])
    if len(ordered) != len(frame) or ordered["episode_key"].duplicated().any():
        raise ValueError(f"{verifier} output coverage mismatch")
    if ordered["summary_split"].astype(bool).any():
        raise ValueError(f"{verifier} violated no-summary-split contract")
    available = ordered["available"].astype(bool)
    if not ordered.loc[available, "parse_ok"].astype(bool).all():
        raise ValueError(f"{verifier} parse failure on available rows")
    atomic_parquet(parquet_path, ordered)
    total_seconds = float(ordered["latency_total_ms"].sum()) / 1000.0
    available_seconds = float(ordered.loc[available, "latency_total_ms"].sum()) / 1000.0
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "SCORED",
        "created_at_utc": utc_now(),
        "verifier": verifier,
        "rows": len(ordered),
        "coverage": 1.0,
        "availability": float(available.mean()),
        "context_overflow_rows": int(ordered["context_overflow"].astype(bool).sum()),
        "parse_ok_rate_available": float(ordered.loc[available, "parse_ok"].astype(bool).mean()),
        "summary_split_rows": int(ordered["summary_split"].astype(bool).sum()),
        "strict_batch1": True,
        "warmup_forwards": warmup,
        "model_load_excluded": True,
        "input_sha256": input_sha256,
        "harness_sha256": harness_sha256,
        "model_revision": str(ordered["model_revision"].iloc[0]),
        "protocol_version": PROTOCOLS[verifier],
        "latency_mean_ms": float(ordered["latency_total_ms"].mean()),
        "latency_p50_ms": float(ordered["latency_total_ms"].quantile(0.5)),
        "latency_p95_ms": float(ordered["latency_total_ms"].quantile(0.95)),
        "attempted_summaries_per_second": float(len(ordered) / total_seconds),
        "available_summaries_per_second": (
            float(available.sum() / available_seconds) if available_seconds > 0 else None
        ),
        "common_semantic_tokens_per_second": float(
            ordered["semantic_input_tokens"].sum() / total_seconds
        ),
        "parquet_path": str(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "official_or_sealed_test_rows_read": 0,
    }
    atomic_json(output_dir / f"{verifier}.manifest.json", manifest)
    return manifest


def choose_bacc_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    bacc = 0.5 * (tpr + 1.0 - fpr)
    candidates = np.flatnonzero(finite & np.isclose(bacc, np.max(bacc[finite])))
    specificity = 1.0 - fpr[candidates]
    candidates = candidates[np.isclose(specificity, np.max(specificity))]
    return float(np.max(thresholds[candidates]))


def apply_oof_thresholds(
    joined: pd.DataFrame, *, verifier: str
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = joined.copy()
    folds = sorted(pd.to_numeric(result["fold"], errors="raise").astype(int).unique())
    if folds != list(range(5)):
        raise ValueError(f"{verifier} must contain folds 0 through 4")
    result["threshold"] = np.nan
    result["predicted_supported"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    threshold_rows: list[dict[str, Any]] = []
    available = result["available"].astype(bool)
    for held_fold in folds:
        train = result[result["fold"].ne(held_fold) & available]
        held = result[result["fold"].eq(held_fold) & available]
        train_folds = sorted(train["fold"].astype(int).unique())
        if held_fold in train_folds or train_folds != [fold for fold in folds if fold != held_fold]:
            raise ValueError(f"{verifier} held fold leaked into threshold training")
        if set(train["label_supported"].astype(int)) != {0, 1}:
            raise ValueError(f"{verifier} threshold training requires both classes")
        threshold = choose_bacc_threshold(
            train["score"].to_numpy(float), train["label_supported"].to_numpy(int)
        )
        result.loc[held.index, "threshold"] = threshold
        result.loc[held.index, "predicted_supported"] = (
            held["score"].to_numpy(float) >= threshold
        ).astype(np.int8)
        threshold_rows.append(
            {
                "verifier": verifier,
                "held_out_fold": held_fold,
                "train_folds": ",".join(str(value) for value in train_folds),
                "threshold": threshold,
                "train_rows": len(train),
                "train_groups": train["group_id"].nunique(),
                "evaluation_rows": len(held),
                "evaluation_groups": held["group_id"].nunique(),
            }
        )
    result["correct"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[available, "correct"] = (
        result.loc[available, "predicted_supported"].astype(int)
        == result.loc[available, "label_supported"].astype(int)
    ).astype(np.int8)
    return result, threshold_rows


def build_training_matrix(
    index: pd.DataFrame, long: pd.DataFrame, *, verifiers: tuple[str, ...] = VERIFIERS
) -> pd.DataFrame:
    if index["episode_key"].astype(str).duplicated().any():
        raise ValueError("training index contains duplicate episode keys")
    expected = len(index) * len(verifiers)
    if len(long) != expected:
        raise ValueError(f"long prediction coverage mismatch: {len(long)} != {expected}")
    wide = index.copy()
    value_columns = (
        "score",
        "available",
        "threshold",
        "predicted_supported",
        "correct",
        "latency_total_ms",
        "semantic_input_tokens",
        "model_input_tokens",
        "output_tokens",
        "model_calls",
        "model_forward_calls",
        "forward_items",
        "source_window_count",
        "source_selected",
        "source_sentence_coverage",
        "context_overflow",
    )
    prefixes = {
        "predicted_supported": "decision",
        "latency_total_ms": "latency_ms",
        "semantic_input_tokens": "semantic_tokens",
    }
    nullable_ints = {"predicted_supported", "correct"}
    booleans = {"available", "source_selected", "context_overflow"}
    for verifier in verifiers:
        part = long[long["verifier"].eq(verifier)].copy()
        if len(part) != len(index) or part["episode_key"].astype(str).duplicated().any():
            raise ValueError(f"training coverage mismatch for {verifier}")
        part = part.set_index("episode_key")
        for column in value_columns:
            name = f"{prefixes.get(column, column)}__{verifier}"
            values = wide["episode_key"].map(part[column])
            if column in nullable_ints:
                values = values.astype("Int8")
            elif column in booleans:
                values = values.astype(bool)
            wide[name] = values
    return wide


def audit_canary(input_dir: Path, score_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_path = input_dir / f"{CANARY_NAME}.parquet"
    expected_hash = sha256_file(input_path)
    rows: list[dict[str, Any]] = []
    for verifier in VERIFIERS:
        path = score_dir / f"{verifier}.parquet"
        score = pd.read_parquet(path)
        if len(score) != 20 or score["episode_key"].astype(str).duplicated().any():
            raise ValueError(f"invalid canary coverage: {verifier}")
        if set(score["input_sha256"].astype(str)) != {expected_hash}:
            raise ValueError(f"canary input hash mismatch: {verifier}")
        available = score["available"].astype(bool)
        split_rows = int(score["summary_split"].astype(bool).sum())
        parse_failures = int((available & ~score["parse_ok"].astype(bool)).sum())
        if split_rows or parse_failures:
            raise ValueError(
                f"canary protocol failure: {verifier} split={split_rows} parse={parse_failures}"
            )
        rows.append(
            {
                "verifier": verifier,
                "rows": len(score),
                "availability": float(available.mean()),
                "parse_ok_rate_available": float(score.loc[available, "parse_ok"].mean()),
                "summary_split_rows": split_rows,
                "latency_mean_ms": float(score["latency_total_ms"].mean()),
                "model_input_token_coverage": float(score["model_input_tokens"].notna().mean()),
                "source_selection_rate": float(score["source_selected"].astype(bool).mean()),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "CANARY_METRICS.csv", index=False)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "verifiers": len(rows),
        "rows_per_verifier": 20,
        "summary_split_rows": int(frame["summary_split_rows"].sum()),
        "official_or_sealed_test_rows_read": 0,
        "canary_metrics_sha256": sha256_file(output_dir / "CANARY_METRICS.csv"),
    }
    atomic_json(output_dir / "CANARY_AUDIT.json", audit)
    return audit


def finalize_results(input_dir: Path, score_dir: Path, output_dir: Path) -> dict[str, Any]:
    from scipy.stats import kendalltau, pointbiserialr, spearmanr
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    manifest = json.loads((input_dir / "INPUT_MANIFEST.json").read_text(encoding="utf-8"))
    index = pd.read_parquet(input_dir / "TRAIN_INDEX.parquet")
    parts: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for verifier in VERIFIERS:
        path = score_dir / f"{verifier}.parquet"
        score = pd.read_parquet(path)
        if len(score) != 6850 or score["episode_key"].duplicated().any():
            raise ValueError(f"invalid score asset: {verifier}")
        if set(score["input_sha256"].astype(str)) != {manifest["scoring_sha256"]}:
            raise ValueError(f"input hash mismatch: {verifier}")
        joined = index.merge(
            score,
            on=["episode_key", "episode_id", "dataset_key"],
            validate="one_to_one",
        )
        joined, verifier_threshold_rows = apply_oof_thresholds(joined, verifier=verifier)
        threshold_rows.extend(verifier_threshold_rows)
        available = joined["available"].astype(bool)
        evaluated = joined[available].copy()
        y = evaluated["label_supported"].to_numpy(int)
        p = evaluated["predicted_supported"].to_numpy(int)
        s = evaluated["score"].to_numpy(float)
        latency_seconds = joined["latency_total_ms"].sum() / 1000.0
        model_token_values = pd.to_numeric(joined["model_input_tokens"], errors="coerce")
        metric_rows.append(
            {
                "verifier": verifier,
                "rows": len(joined),
                "available_rows": len(evaluated),
                "availability": float(available.mean()),
                "balanced_accuracy": float(balanced_accuracy_score(y, p)),
                "auroc": float(roc_auc_score(y, s)),
                "mcc": float(matthews_corrcoef(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro")),
                "accuracy": float(accuracy_score(y, p)),
                "supported_recall": float(p[y == 1].mean()),
                "unsupported_recall": float((p[y == 0] == 0).mean()),
                "point_biserial_pearson": float(pointbiserialr(y, s).statistic),
                "spearman_rho": float(spearmanr(y, s).statistic),
                "kendall_tau_b": float(kendalltau(y, s, variant="b").statistic),
                "latency_mean_ms": float(joined["latency_total_ms"].mean()),
                "latency_p50_ms": float(joined["latency_total_ms"].quantile(0.5)),
                "latency_p95_ms": float(joined["latency_total_ms"].quantile(0.95)),
                "attempted_summaries_per_second": float(len(joined) / latency_seconds),
                "available_summaries_per_second": float(
                    len(evaluated) / (evaluated["latency_total_ms"].sum() / 1000.0)
                ),
                "common_semantic_tokens_per_second": float(
                    joined["semantic_input_tokens"].sum() / latency_seconds
                ),
                "model_input_tokens_mean": float(model_token_values.mean()) if model_token_values.notna().any() else np.nan,
                "model_input_token_coverage": float(model_token_values.notna().mean()),
                "model_calls_mean": float(joined["model_calls"].mean()),
                "model_forward_calls_mean": float(joined["model_forward_calls"].mean()),
                "forward_items_mean": float(joined["forward_items"].mean()),
                "source_selection_rate": float(joined["source_selected"].astype(bool).mean()),
                "context_overflow_rows": int(joined["context_overflow"].astype(bool).sum()),
            }
        )
        parts.append(joined)

    long = pd.concat(parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("latency_mean_ms").reset_index(drop=True)
    thresholds = pd.DataFrame(threshold_rows)
    wide_base = build_training_matrix(index, long)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(output_dir / "OOF_LONG_PREDICTIONS.parquet", long)
    atomic_parquet(output_dir / "ROUTER_TRAINING_MATRIX.parquet", wide_base)
    metrics.to_csv(output_dir / "POOLED_METRICS.csv", index=False)
    thresholds.to_csv(output_dir / "THRESHOLD_AUDIT.csv", index=False)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "verifiers": len(VERIFIERS),
        "rows": len(index),
        "prediction_rows": len(long),
        "expected_prediction_rows": len(index) * len(VERIFIERS),
        "threshold_rows": len(thresholds),
        "router_feature_columns": list(ROUTER_FEATURE_COLUMNS),
        "summary_split_rows": int(long["summary_split"].astype(bool).sum()),
        "parse_failures_available": int((long["available"].astype(bool) & ~long["parse_ok"].astype(bool)).sum()),
        "official_or_sealed_test_rows_read": 0,
        "training_matrix_sha256": sha256_file(output_dir / "ROUTER_TRAINING_MATRIX.parquet"),
        "long_predictions_sha256": sha256_file(output_dir / "OOF_LONG_PREDICTIONS.parquet"),
        "metrics_sha256": sha256_file(output_dir / "POOLED_METRICS.csv"),
        "threshold_audit_sha256": sha256_file(output_dir / "THRESHOLD_AUDIT.csv"),
    }
    atomic_json(output_dir / "AUDIT.json", audit)
    return audit
