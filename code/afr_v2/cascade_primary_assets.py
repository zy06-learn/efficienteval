from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from afr_v2.router_feature_learnability import FEATURE_COLUMNS, build_cheap_feature_frame


PRIMARY_SPLIT = "primary_train"
FEATURE_VERSION = "cheap27_v1_full_source_cascade"

EXPECTED = {
    "rows": 21_619,
    "groups": 1_265,
    "supported": 17_818,
    "unsupported": 3_801,
    "ragtruth_rows": 16_991,
    "ragtruth_groups": 500,
    "usb_rows": 4_628,
    "usb_groups": 765,
    "usb_supported": 1_846,
    "usb_unsupported": 2_782,
}

SCORING_COLUMNS = [
    "episode_id",
    "doc_group_key",
    "generator_id",
    "dataset",
    "split",
    "is_official_test",
    "source_document",
    "candidate_sentence",
]

EPISODE_COLUMNS = SCORING_COLUMNS + [
    "content_doc_key",
    "original_dataset",
    "original_split",
    "role",
    "source_document_id",
    "target_position",
    "raw_gold_label",
    "label_supported",
    "human_span_json",
    "human_evidence_json",
    "human_correction",
    "annotation_provenance",
    "license",
    "dataset_version",
]


def normalize_text(text: Any) -> str:
    """Project-wide identity normalization used by the existing RAGTruth split."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join(normalized.split())


def normalized_sha256(text: Any) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"invalid JSON object at {path}:{line_number}")
            yield row


def materialize_usb_all_annotations(path: Path, *, source_split: str) -> pd.DataFrame:
    """Expand USB all_annotations into full-source, pre-edit sentence episodes."""

    rows: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    for document_index, document in enumerate(_iter_jsonl(path)):
        required = {"id", "source", "summary"}
        missing = sorted(required - set(document))
        if missing:
            raise ValueError(f"USB document {document_index} missing {missing}")
        document_id = str(document["id"])
        if document_id in document_ids:
            raise ValueError(f"duplicate USB document id: {document_id}")
        document_ids.add(document_id)
        if not isinstance(document["source"], list) or not isinstance(document["summary"], list):
            raise ValueError(f"USB document {document_id} source/summary must be lists")

        source_sentences = [str(item.get("txt", "")).strip() for item in document["source"]]
        if not any(source_sentences):
            raise ValueError(f"USB document {document_id} has no source text")
        source_document = "\n".join(source_sentences)
        doc_group_key = normalized_sha256(source_document)
        domain = document_id.split("/", maxsplit=1)[0]

        for position, sentence in enumerate(document["summary"]):
            pre_edit = str(sentence.get("pre_edit", "")).strip()
            post_edit = str(sentence.get("post_edit", "")).strip()
            if not pre_edit:
                raise ValueError(f"USB {document_id} sentence {position} has empty pre_edit")
            supported = int(normalize_text(pre_edit) == normalize_text(post_edit))
            if supported:
                raw_label = "SUPPORTED_UNCHANGED"
            elif not post_edit:
                raw_label = "UNSUPPORTED_HUMAN_DELETE"
            else:
                raw_label = "UNSUPPORTED_HUMAN_EDIT"
            evidence = [int(value) for value in sentence.get("evidence", [])]
            if any(index < 0 or index >= len(source_sentences) for index in evidence):
                raise ValueError(f"USB {document_id} sentence {position} has bad evidence index")
            rows.append(
                {
                    "episode_id": f"usb:{source_split}:{doc_group_key[:16]}:s{position:03d}",
                    "doc_group_key": doc_group_key,
                    "generator_id": "wikipedia_lead_human",
                    "dataset": "USB-full-source",
                    "split": PRIMARY_SPLIT,
                    "is_official_test": False,
                    "source_document": source_document,
                    "candidate_sentence": pre_edit,
                    "content_doc_key": doc_group_key,
                    "original_dataset": "kundank/usb",
                    "original_split": source_split,
                    "role": PRIMARY_SPLIT,
                    "source_document_id": document_id,
                    "target_position": int(position),
                    "raw_gold_label": raw_label,
                    "label_supported": supported,
                    "human_span_json": None,
                    "human_evidence_json": _json(evidence),
                    "human_correction": post_edit,
                    "annotation_provenance": "USB all_annotations human pre/post edit",
                    "license": "Apache-2.0",
                    "dataset_version": "processed_data_sha256_2d043d940120558b",
                    "domain": domain,
                }
            )
    frame = pd.DataFrame(rows)
    if frame["episode_id"].duplicated().any():
        raise ValueError("duplicate USB episode_id")
    return frame


def materialize_ragtruth(path: Path) -> pd.DataFrame:
    source = pd.read_parquet(path)
    required = {
        "episode_id",
        "doc_group_key",
        "generator_id",
        "source_document",
        "candidate_sentence",
        "raw_gold_label",
        "label_supported",
        "unsupported_spans_json",
        "is_official_test",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"RAGTruth primary file missing {missing}")
    if source["is_official_test"].fillna(False).astype(bool).any():
        raise ValueError("RAGTruth primary contains official-test rows")
    target_position = pd.to_numeric(
        source["sentence_id"].astype(str).str.extract(r":s(\d+)$", expand=False),
        errors="raise",
    ).astype(int)
    frame = pd.DataFrame(
        {
            "episode_id": source["episode_id"].astype(str),
            "doc_group_key": source["doc_group_key"].astype(str),
            "generator_id": source["generator_id"].astype(str),
            "dataset": "RAGTruth-Summary",
            "split": PRIMARY_SPLIT,
            "is_official_test": False,
            "source_document": source["source_document"].astype(str),
            "candidate_sentence": source["candidate_sentence"].astype(str),
            "content_doc_key": source["doc_group_key"].astype(str),
            "original_dataset": "RAGTruth",
            "original_split": "official_train_reallocated_router_train",
            "role": PRIMARY_SPLIT,
            "source_document_id": source["doc_group_key"].astype(str),
            "target_position": target_position,
            "raw_gold_label": source["raw_gold_label"].astype(str),
            "label_supported": source["label_supported"].astype(int),
            "human_span_json": source["unsupported_spans_json"],
            "human_evidence_json": None,
            "human_correction": None,
            "annotation_provenance": "RAGTruth human sentence/span annotation",
            "license": "RAGTruth upstream terms",
            "dataset_version": "ragtruth_unseal_v2_seed_20260719",
        }
    )
    return frame


def scoring_inputs(episodes: pd.DataFrame) -> pd.DataFrame:
    result = episodes[SCORING_COLUMNS].copy()
    forbidden = [
        column
        for column in result.columns
        if column.lower().startswith("label")
        or "gold" in column.lower()
        or "human" in column.lower()
        or "correction" in column.lower()
    ]
    if forbidden:
        raise AssertionError(f"scoring inputs leak supervision: {forbidden}")
    return result


def validate_primary(episodes: pd.DataFrame) -> dict[str, Any]:
    if episodes["episode_id"].duplicated().any():
        raise ValueError("primary episode_id must be unique")
    if episodes["is_official_test"].fillna(False).astype(bool).any():
        raise ValueError("primary includes official-test rows")
    if set(episodes["split"].astype(str)) != {PRIMARY_SPLIT}:
        raise ValueError("primary split mismatch")
    if episodes[["source_document", "candidate_sentence"]].isna().any().any():
        raise ValueError("primary text fields must be non-null")

    rag = episodes.loc[episodes["dataset"].eq("RAGTruth-Summary")]
    usb = episodes.loc[episodes["dataset"].eq("USB-full-source")]
    stats = {
        "rows": int(len(episodes)),
        "groups": int(episodes["doc_group_key"].nunique()),
        "supported": int((episodes["label_supported"] == 1).sum()),
        "unsupported": int((episodes["label_supported"] == 0).sum()),
        "ragtruth_rows": int(len(rag)),
        "ragtruth_groups": int(rag["doc_group_key"].nunique()),
        "usb_rows": int(len(usb)),
        "usb_groups": int(usb["doc_group_key"].nunique()),
        "usb_supported": int((usb["label_supported"] == 1).sum()),
        "usb_unsupported": int((usb["label_supported"] == 0).sum()),
    }
    if stats != EXPECTED:
        raise ValueError(f"primary counts differ from frozen protocol: {stats}")

    by_hash = episodes.groupby("doc_group_key")["dataset"].nunique()
    if (by_hash > 1).any():
        raise ValueError("cross-dataset primary source-hash overlap")
    return stats


def build_features(episodes: pd.DataFrame) -> pd.DataFrame:
    feature_input = episodes[
        ["episode_id", "doc_group_key", "source_document", "candidate_sentence"]
    ].copy()
    computed = build_cheap_feature_frame(feature_input)
    computed.insert(0, "episode_id", feature_input["episode_id"].to_numpy())
    computed["feature_version"] = FEATURE_VERSION
    if len(computed) != len(episodes):
        raise ValueError("feature row count mismatch")
    if computed["episode_id"].duplicated().any():
        raise ValueError("feature episode_id must be unique")
    if computed[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("cheap features contain missing values")
    return computed


def source_hashes_from_usb(path: Path) -> set[str]:
    hashes: set[str] = set()
    for document in _iter_jsonl(path):
        source = "\n".join(str(item.get("txt", "")).strip() for item in document["source"])
        hashes.add(normalized_sha256(source))
    return hashes


def audit_reference_overlap(
    primary: pd.DataFrame,
    *,
    ragtruth_screening_path: Path,
    ragtruth_fresh_inputs_path: Path,
    usb_validation_path: Path,
) -> dict[str, int]:
    primary_hashes = set(primary["doc_group_key"].astype(str))
    screening = pd.read_parquet(ragtruth_screening_path)
    fresh = pd.read_parquet(ragtruth_fresh_inputs_path)
    references = {
        "ragtruth_screening": set(screening["doc_group_key"].astype(str)),
        "ragtruth_fresh_calibration": set(fresh["doc_group_key"].astype(str)),
        "usb_validation": source_hashes_from_usb(usb_validation_path),
    }
    overlap = {name: len(primary_hashes & values) for name, values in references.items()}
    if any(overlap.values()):
        raise ValueError(f"primary/reference source overlap: {overlap}")
    return overlap


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
