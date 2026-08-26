#!/usr/bin/env python3
"""Build the official-split EfficientEval v2 train and sealed test assets.

This is the only P0 program allowed to read official-test gold.  It emits a
separate fail-closed TEST_SCORING parquet with every gold-derived field removed;
P1 must read that projection and never TEST.parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PAPER_ROOT.parent
sys.path.insert(0, str(PAPER_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import config_v2 as C  # noqa: E402
from verifier_wrappers.global_gamma_calibration import (  # noqa: E402
    parse_ragtruth_spans,
    segment_ragtruth_response,
)
from router_feature_learnability_frozen import (  # noqa: E402
    FEATURE_COLUMNS as CHEAP27,
    build_cheap_feature_frame,
)
from summary_router_compact16_direct_v1_frozen import (  # noqa: E402
    FEATURE_COLUMNS as COMPACT16,
    build_compact16_feature_frame,
)


SCHEMA_VERSION = "efficienteval_official_split_v2"
RAGTRUTH_SUFFIX = "\n\noutput:"
COGEN_TEST_MEMBERS = (
    "test_chen18_org.json",
    "test_chen18_reranked.json",
    "test_gehrmann18_org.json",
    "test_see17_org.json",
)
GOLD_FIELD_TOKENS = (
    "label",
    "gold",
    "correct",
    "quality",
    "unsupported",
    "error_type",
    "annotation",
    "target",
)
RAW_SCORE_FIELDS = (
    "score",
    "available",
    "latency_ms",
    "semantic_tokens",
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(normalized.split())


def normalized_sha256(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def verify_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in C.SOURCE_SHA256.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source hash drift: {relative}: {actual} != {expected}")
        observed[relative] = actual
    return observed


def raw_score_columns() -> list[str]:
    return [f"{field}__{verifier}" for verifier in C.VERIFIERS for field in RAW_SCORE_FIELDS]


def _standardize_old_rows() -> pd.DataFrame:
    matrix = pd.read_parquet(
        PROJECT_ROOT / "results/unified_summary_verifiers_v1/ROUTER_TRAINING_MATRIX.parquet"
    )
    cohort_columns = [
        "episode_key",
        "source_document",
        "candidate_sentence",
        "feature_query_latency_ms",
        "feature_document_setup_ms",
    ]
    cohort = pd.read_parquet(
        PROJECT_ROOT / "results/mixed_dataset_v1/MIXED_COHORT.parquet",
        columns=cohort_columns,
    )
    compact = pd.read_parquet(
        PROJECT_ROOT / "results/summary_router_compact16_direct_v1/COMPACT16_FEATURES.parquet"
    ).rename(columns={"feature_latency_ms": "compact16_feature_latency_ms"})
    if matrix["episode_key"].duplicated().any() or cohort["episode_key"].duplicated().any():
        raise ValueError("legacy episode_key is not unique")
    frame = matrix.merge(cohort, on="episode_key", how="left", validate="one_to_one")
    frame = frame.merge(compact, on="episode_key", how="left", validate="one_to_one")
    if frame[["source_document", "candidate_sentence"]].isna().any().any():
        raise ValueError("legacy text join is incomplete")

    frank = pd.read_json(
        PROJECT_ROOT / "data/unified_v1/frank/processed.jsonl", lines=True
    )
    split_map = frank.groupby("content_doc_key")["original_split"].agg(
        lambda values: values.mode().iat[0]
    )
    frank_mask = frame["dataset_key"].eq("frank_train")
    frame.loc[frank_mask, "official_split"] = frame.loc[
        frank_mask, "content_doc_key"
    ].astype(str).map(split_map)
    if frame.loc[frank_mask, "official_split"].isna().any():
        raise ValueError("FRANK content_doc_key split join contains NaN")
    frank_counts = frame.loc[frank_mask, "official_split"].value_counts().to_dict()
    if frank_counts != {"test": 1_569, "valid": 669}:
        raise ValueError(f"FRANK official split mismatch: {frank_counts}")

    fixed_dataset = frame["dataset_key"].astype(str).copy()
    fixed_dataset.loc[frank_mask & frame["official_split"].eq("valid")] = "frank_valid"
    fixed_dataset.loc[frank_mask & frame["official_split"].eq("test")] = "frank_test"
    frame["dataset_key"] = fixed_dataset
    frame["official_split"] = frame["official_split"].fillna(
        frame["dataset_key"].map(
            {
                "ragtruth_train": "train",
                "unisumeval_train": "train",
                "cogensumm_val": "val_reranking",
            }
        )
    )
    frame["candidate_summary"] = frame.pop("candidate_sentence").astype(str)
    frame["role"] = np.where(frame["dataset_key"].eq("frank_test"), "test", "train")
    frame["group_id"] = frame["dataset_key"].astype(str) + "::" + frame["doc_group_key"].astype(str)
    frame["episode_key"] = frame["dataset_key"].astype(str) + "::" + frame["episode_id"].astype(str)
    frame["source_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["summary_model"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["num_sentences"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    responses = {
        str(row["id"]): row
        for row in read_jsonl(PROJECT_ROOT / "data/external/RAGTruth/dataset/response.jsonl")
    }
    rag_mask = frame["dataset_key"].eq("ragtruth_train")
    response_ids = frame.loc[rag_mask, "episode_id"].str.extract(
        r"^ragtruth:([^:]+):summary$", expand=False
    )
    if response_ids.isna().any():
        raise ValueError("legacy RAGTruth episode ID drift")
    frame.loc[rag_mask, "source_id"] = response_ids.map(
        lambda value: str(responses[str(value)]["source_id"])
    ).to_numpy()
    frame.loc[rag_mask, "summary_model"] = response_ids.map(
        lambda value: str(responses[str(value)]["model"])
    ).to_numpy()

    for dataset in ("frank_valid", "frank_test", "unisumeval_train", "cogensumm_val"):
        mask = frame["dataset_key"].eq(dataset)
        frame.loc[mask, "source_id"] = frame.loc[mask, "doc_group_key"].astype(str).to_numpy()
    return _select_common_columns(frame)


def _select_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "episode_key",
        "episode_id",
        "dataset_key",
        "role",
        "official_split",
        "group_id",
        "doc_group_key",
        "content_doc_key",
        "source_id",
        "source_document",
        "candidate_summary",
        "summary_model",
        "label_supported",
        "num_sentences",
    ]
    features = [
        "feature_latency_ms",
        "feature_query_latency_ms",
        "feature_document_setup_ms",
        *CHEAP27,
        "compact16_feature_latency_ms",
        *COMPACT16,
    ]
    wanted = identity + features + raw_score_columns()
    result = frame.copy()
    for column in wanted:
        if column not in result:
            result[column] = np.nan
    return result[wanted]


def _feature_new_rows(frame: pd.DataFrame) -> pd.DataFrame:
    feature_input = frame.rename(columns={"candidate_summary": "candidate_sentence"})
    cheap = build_cheap_feature_frame(feature_input)
    compact_input = frame[["source_document", "candidate_summary"]]
    compact = build_compact16_feature_frame(compact_input).rename(
        columns={"feature_latency_ms": "compact16_feature_latency_ms"}
    )
    result = frame.copy()
    for column in CHEAP27:
        result[column] = cheap[column].to_numpy(float)
    result["feature_query_latency_ms"] = cheap["feature_query_latency_ms"].to_numpy(float)
    result["feature_document_setup_ms"] = cheap["feature_document_setup_ms"].to_numpy(float)
    result["feature_latency_ms"] = (
        result["feature_query_latency_ms"] + result["feature_document_setup_ms"]
    )
    for column in COMPACT16:
        result[column] = compact[column].to_numpy(float)
    result["compact16_feature_latency_ms"] = compact[
        "compact16_feature_latency_ms"
    ].to_numpy(float)
    return result


def _ragtruth_rows(*, limit_rows: int | None = None) -> pd.DataFrame:
    source_rows = read_jsonl(
        PROJECT_ROOT / "data/external/RAGTruth/dataset/source_info.jsonl"
    )
    sources = {
        str(row["source_id"]): row
        for row in source_rows
        if str(row.get("task_type")) == "Summary"
    }
    responses = read_jsonl(PROJECT_ROOT / "data/external/RAGTruth/dataset/response.jsonl")
    rows: list[dict[str, Any]] = []
    for response in responses:
        source_id = str(response["source_id"])
        if str(response.get("split")) != "test" or source_id not in sources:
            continue
        source_info = str(sources[source_id]["source_info"]).strip()
        source_document = source_info + RAGTRUTH_SUFFIX
        summary = str(response["response"]).strip()
        spans = parse_ragtruth_spans(response.get("labels"), response=summary)
        segmented = segment_ragtruth_response(
            response_id=str(response["id"]),
            doc_group_key=normalized_sha256(source_document),
            source_document=source_document,
            response=summary,
            generator_id=str(response["model"]),
            spans=spans,
        )
        supported = int(all(int(item["label_supported"]) == 1 for item in segmented))
        episode_id = f"ragtruth:{response['id']}:summary"
        doc_group_key = normalized_sha256(source_document)
        rows.append(
            {
                "episode_key": f"ragtruth_test::{episode_id}",
                "episode_id": episode_id,
                "dataset_key": "ragtruth_test",
                "role": "test",
                "official_split": "test",
                "group_id": f"ragtruth_test::{doc_group_key}",
                "doc_group_key": doc_group_key,
                "content_doc_key": normalized_sha256(source_info),
                "source_id": source_id,
                "source_document": source_document,
                "candidate_summary": summary,
                "summary_model": str(response["model"]),
                "label_supported": supported,
                "num_sentences": len(segmented),
            }
        )
        if limit_rows is not None and len(rows) >= limit_rows:
            break
    frame = pd.DataFrame(rows)
    if limit_rows is None and (len(frame) != 900 or frame["source_id"].nunique() != 150):
        raise ValueError("RAGTruth official test count mismatch")
    return _feature_new_rows(frame)


def _cogensumm_rows(*, limit_rows: int | None = None) -> pd.DataFrame:
    archive_path = (
        PROJECT_ROOT
        / "data/summary_level_raw_v1/cogensumm/official_archive/summary-correctness-v1.0.zip"
    )
    source = pd.read_parquet(
        PROJECT_ROOT
        / "data/summary_level_raw_v1/source_corpora/cnndm_3.0.0/cnndm-test-0000.parquet",
        columns=["id", "article"],
    )
    if source["id"].astype(str).duplicated().any():
        raise ValueError("CNN/DM test mirror contains duplicate IDs")
    article_by_id = source.assign(id=source["id"].astype(str)).set_index("id")[
        "article"
    ].to_dict()
    rows: list[dict[str, Any]] = []
    member_doc_ids: dict[str, set[str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in COGEN_TEST_MEMBERS:
            value = json.loads(archive.read(member))
            if not isinstance(value, dict) or len(value) != 100:
                raise ValueError(f"CoGenSumm {member} structure/count drift")
            member_doc_ids[member] = set(map(str, value))
            system = member.removeprefix("test_").removesuffix(".json")
            for source_id in sorted(value):
                entry = value[source_id]
                raw_label = str(entry.get("label"))
                if raw_label not in {"Correct", "Incorrect"}:
                    raise ValueError(f"invalid CoGenSumm top-level label: {member}/{source_id}")
                sentences = entry.get("sents")
                if not isinstance(sentences, dict) or not sentences:
                    raise ValueError(f"missing CoGenSumm sentences: {member}/{source_id}")
                ordered: list[str] = []
                for sentence_id in sorted(sentences, key=lambda item: int(item)):
                    sentence = sentences[sentence_id]
                    text = str(sentence.get("text", "")).strip()
                    if not text:
                        raise ValueError(
                            f"empty CoGenSumm sentence: {member}/{source_id}/{sentence_id}"
                        )
                    ordered.append(text)
                if str(source_id) not in article_by_id:
                    raise ValueError(f"missing CNN/DM article for {member}/{source_id}")
                article = str(article_by_id[str(source_id)]).strip()
                doc_key = normalized_sha256(article)
                episode_id = f"cogensumm:{source_id}:{system}:summary"
                if limit_rows is None or len(rows) < limit_rows:
                    rows.append(
                        {
                            "episode_key": f"cogensumm_test::{episode_id}",
                            "episode_id": episode_id,
                            "dataset_key": "cogensumm_test",
                            "role": "test",
                            "official_split": member,
                            "group_id": f"cogensumm_test::{doc_key}",
                            "doc_group_key": doc_key,
                            "content_doc_key": doc_key,
                            "source_id": str(source_id),
                            "source_document": article,
                            "candidate_summary": " ".join(ordered),
                            "summary_model": system,
                            # The official top-level summary label is authoritative.
                            "label_supported": int(raw_label == "Correct"),
                            "num_sentences": len(ordered),
                        }
                    )
    first = member_doc_ids[COGEN_TEST_MEMBERS[0]]
    if any(member_doc_ids[name] != first for name in COGEN_TEST_MEMBERS[1:]):
        raise ValueError("CoGenSumm four test systems do not share exactly 100 document IDs")
    frame = pd.DataFrame(rows)
    if limit_rows is None and (len(frame) != 400 or frame["content_doc_key"].nunique() != 100):
        raise ValueError("CoGenSumm official test count mismatch")
    return _feature_new_rows(frame)


def _unisumeval_rows() -> pd.DataFrame:
    scoring = pd.read_parquet(
        PROJECT_ROOT
        / "data/cross_granularity_eval_v1/scoring_inputs/unisumeval_dev_summary.parquet"
    )
    # The scoring asset stores a coarse whitespace count, while the frozen
    # Cheap-27 asset stores the canonical feature extractor's token count.
    # Keep only the latter so pandas cannot silently create _x/_y columns.
    scoring = scoring.drop(columns=["source_token_count"])
    gold = pd.read_parquet(
        PROJECT_ROOT / "data/cross_granularity_eval_v1/gold/unisumeval_dev_summary.parquet"
    )
    features = pd.read_parquet(
        PROJECT_ROOT
        / "data/cross_granularity_eval_v1/router_features/unisumeval_dev_summary.parquet"
    )
    frame = scoring.merge(
        gold[["episode_id", "label_supported", "num_sentences"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    ).merge(features, on="episode_id", how="left", validate="one_to_one")
    if frame[["label_supported", *CHEAP27]].isna().any().any():
        raise ValueError("UniSumEval dev gold/feature join incomplete")
    frame = frame.rename(columns={"candidate_sentence": "candidate_summary"})
    frame["episode_key"] = "unisumeval_dev::" + frame["episode_id"].astype(str)
    frame["dataset_key"] = "unisumeval_dev"
    frame["role"] = "test"
    frame["official_split"] = "dev"
    frame["group_id"] = "unisumeval_dev::" + frame["doc_group_key"].astype(str)
    frame["source_id"] = frame["doc_group_key"].astype(str)
    frame["feature_latency_ms"] = (
        frame["feature_query_latency_ms"] + frame["feature_document_setup_ms"]
    )
    compact = build_compact16_feature_frame(
        frame[["source_document", "candidate_summary"]]
    ).rename(columns={"feature_latency_ms": "compact16_feature_latency_ms"})
    for column in COMPACT16:
        frame[column] = compact[column].to_numpy(float)
    frame["compact16_feature_latency_ms"] = compact[
        "compact16_feature_latency_ms"
    ].to_numpy(float)

    for verifier in ("alignscore", "factkb", "lettuce_v2", "qwen30_judge"):
        score_path = (
            PROJECT_ROOT
            / f"data/cross_granularity_eval_v1/verifier_scores/unisumeval_dev_summary__{verifier}.parquet"
        )
        part = pd.read_parquet(score_path)
        if len(part) != len(frame) or part["episode_id"].duplicated().any():
            raise ValueError(f"UniSumEval existing {verifier} coverage mismatch")
        part = part.set_index("episode_id")
        mapped = frame["episode_id"].map(part["score"])
        frame[f"score__{verifier}"] = mapped.to_numpy(float)
        frame[f"available__{verifier}"] = frame["episode_id"].map(part["parse_ok"]).fillna(False)
        frame[f"latency_ms__{verifier}"] = frame["episode_id"].map(
            part["latency_total_ms"]
        ).to_numpy(float)
        frame[f"semantic_tokens__{verifier}"] = (
            frame["source_token_count"] + frame["claim_token_count"]
        ).astype(float)
        calls = frame["episode_id"].map(part["num_model_calls"]).astype(float)
        frame[f"model_calls__{verifier}"] = calls
        frame[f"model_forward_calls__{verifier}"] = calls
        frame[f"forward_items__{verifier}"] = calls
        frame[f"model_input_tokens__{verifier}"] = frame["episode_id"].map(part["input_tokens"])
        frame[f"output_tokens__{verifier}"] = frame["episode_id"].map(part["output_tokens"])
        frame[f"context_overflow__{verifier}"] = False
    if len(frame) != 367 or frame["doc_group_key"].nunique() != 45:
        raise ValueError("UniSumEval official dev count mismatch")
    return frame


def validate_ragtruth_training_replay() -> dict[str, int]:
    existing = pd.read_parquet(
        PROJECT_ROOT / "data/full_granularity_eval_v2/gold/ragtruth_summary.parquet"
    )
    response_by_id = {
        str(row["id"]): row
        for row in read_jsonl(PROJECT_ROOT / "data/external/RAGTruth/dataset/response.jsonl")
    }
    mismatch = Counter()
    for row in existing.itertuples(index=False):
        response_id = str(row.episode_id).split(":")[1]
        response = response_by_id[response_id]
        summary = str(response["response"])
        segmented = segment_ragtruth_response(
            response_id=response_id,
            doc_group_key=str(row.doc_group_key),
            source_document="training-replay",
            response=summary,
            generator_id=str(response["model"]),
            spans=parse_ragtruth_spans(response.get("labels"), response=summary),
        )
        supported = int(all(int(item["label_supported"]) == 1 for item in segmented))
        mismatch["label"] += int(supported != int(row.label_supported))
        mismatch["sentences"] += int(len(segmented) != int(row.num_sentences))
        mismatch["unsupported_sentences"] += int(
            sum(1 - int(item["label_supported"]) for item in segmented)
            != int(row.unsupported_sentences)
        )
    if any(mismatch.values()):
        raise ValueError(f"RAGTruth training mapping replay mismatch: {dict(mismatch)}")
    return {"rows": len(existing), **dict(mismatch)}


def _dataset_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        str(dataset): {
            "rows": int(len(part)),
            "groups": int(part["content_doc_key"].nunique()),
        }
        for dataset, part in frame.groupby("dataset_key", sort=True)
    }


def validate_contract(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    expected_train = {
        key: {"rows": rows, "groups": groups}
        for key, (rows, groups) in C.EXPECTED["train_by_dataset"].items()
    }
    expected_test = {
        key: {"rows": rows, "groups": groups}
        for key, (rows, groups) in C.EXPECTED["test_by_dataset"].items()
    }
    train_counts = _dataset_counts(train)
    test_counts = _dataset_counts(test)
    if train_counts != expected_train:
        raise ValueError(f"training count contract failed: {train_counts}")
    if test_counts != expected_test:
        raise ValueError(f"test count contract failed: {test_counts}")
    if len(train) != C.EXPECTED["train_rows"] or train["content_doc_key"].nunique() != C.EXPECTED["train_groups"]:
        raise ValueError("training total count contract failed")
    if len(test) != C.EXPECTED["test_rows"] or test["content_doc_key"].nunique() != C.EXPECTED["test_groups"]:
        raise ValueError("test total count contract failed")
    if pd.concat([train["episode_key"], test["episode_key"]]).duplicated().any():
        raise ValueError("episode_key is not globally unique")

    overlap = {
        column: len(set(train[column].astype(str)) & set(test[column].astype(str)))
        for column in ("content_doc_key", "doc_group_key", "group_id")
    }
    if any(overlap.values()):
        raise ValueError(f"train/test group isolation failed: {overlap}")

    train_source_ids = set(
        train.loc[train["dataset_key"].eq("ragtruth_train"), "source_id"].astype(str)
    )
    test_source_ids = set(
        test.loc[test["dataset_key"].eq("ragtruth_test"), "source_id"].astype(str)
    )
    ragtruth_source_overlap = len(train_source_ids & test_source_ids)
    if ragtruth_source_overlap:
        raise ValueError("RAGTruth train/test source_id overlap")

    cogen_val = set(
        train.loc[train["dataset_key"].eq("cogensumm_val"), "content_doc_key"].astype(str)
    )
    cogen_test = set(
        test.loc[test["dataset_key"].eq("cogensumm_test"), "content_doc_key"].astype(str)
    )
    cogen_overlap = len(cogen_val & cogen_test)
    if cogen_overlap:
        raise ValueError("CoGenSumm val/test document overlap")

    combined = pd.concat(
        [
            train[["dataset_key", "content_doc_key"]],
            test[["dataset_key", "content_doc_key"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    cross = combined.groupby("content_doc_key")["dataset_key"].agg(
        lambda values: tuple(sorted(set(map(str, values))))
    )
    cross = cross[cross.map(len).gt(1)]
    cross_pairs = Counter(pair for datasets in cross for pair in _pairs(datasets))
    if cross_pairs:
        raise ValueError(f"cross-corpus content overlap: {dict(cross_pairs)}")
    return {
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_test_overlap": overlap,
        "ragtruth_source_id_overlap": ragtruth_source_overlap,
        "cogensumm_val_test_overlap": cogen_overlap,
        "cross_corpus_content_overlap": {},
        "episode_key_duplicate_count": 0,
    }


def _pairs(values: Iterable[str]) -> Iterable[str]:
    ordered = list(values)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            yield f"{left}__{right}"


def scoring_projection(test: pd.DataFrame) -> pd.DataFrame:
    forbidden = [
        column
        for column in test.columns
        if any(token in column.casefold() for token in GOLD_FIELD_TOKENS)
    ]
    scoring = test.drop(columns=forbidden).copy()
    leaked = [
        column
        for column in scoring.columns
        if any(token in column.casefold() for token in GOLD_FIELD_TOKENS)
    ]
    if leaked:
        raise ValueError(f"TEST_SCORING contains forbidden fields: {leaked}")
    if len(scoring) != len(test) or scoring["episode_key"].duplicated().any():
        raise ValueError("TEST_SCORING identity/coverage mismatch")
    return scoring


def _smoke() -> dict[str, Any]:
    verify_sources()
    replay = validate_ragtruth_training_replay()
    rag = _ragtruth_rows(limit_rows=2)
    cogen = _cogensumm_rows(limit_rows=2)
    projection = scoring_projection(pd.concat([rag, cogen], ignore_index=True))
    return {
        "status": "SMOKE_PASS",
        "ragtruth_training_replay": replay,
        "feature_rows": len(projection),
        "gold_free_projection": True,
    }


def build() -> dict[str, Any]:
    source_hashes = verify_sources()
    replay = validate_ragtruth_training_replay()
    old = _standardize_old_rows()
    train = old[old["dataset_key"].isin(C.EXPECTED["train_by_dataset"])].copy()
    # Pre-declared deterministic leakage exclusion; see config_v2.TRAIN_EXCLUDED_CONTENT_DOC_KEYS.
    excluded_mask = train["content_doc_key"].astype(str).isin(C.TRAIN_EXCLUDED_CONTENT_DOC_KEYS)
    excluded_rows = int(excluded_mask.sum())
    if excluded_rows:
        train = train.loc[~excluded_mask].copy()
    globals()["_EXCLUSION_AUDIT"] = {
        "declared_keys": sorted(C.TRAIN_EXCLUDED_CONTENT_DOC_KEYS),
        "rows_removed": excluded_rows,
        "documents_removed": int(len(C.TRAIN_EXCLUDED_CONTENT_DOC_KEYS)),
    }
    frank_test = old[old["dataset_key"].eq("frank_test")].copy()
    test = pd.concat(
        [frank_test, _ragtruth_rows(), _cogensumm_rows(), _unisumeval_rows()],
        ignore_index=True,
        sort=False,
    )
    train = _select_common_columns(train)
    test = _select_common_columns(test)

    numeric_features = ["feature_latency_ms", *CHEAP27, "compact16_feature_latency_ms", *COMPACT16]
    if not np.isfinite(train[numeric_features].to_numpy(float)).all():
        raise ValueError("TRAIN contains non-finite feature values")
    if not np.isfinite(test[numeric_features].to_numpy(float)).all():
        raise ValueError("TEST contains non-finite feature values")
    audit = validate_contract(train, test)
    scoring = scoring_projection(test)

    train_path = C.DATA / "TRAIN.parquet"
    test_path = C.DATA / "TEST.parquet"
    scoring_path = C.DATA / "TEST_SCORING.parquet"
    atomic_parquet(train_path, train)
    atomic_parquet(test_path, test)
    atomic_parquet(scoring_path, scoring)
    os.chmod(test_path, 0o400)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "P0_COMPLETE_TEST_SEALED",
        "protocol_deviation_record": "ingest_and_scoring/PROTOCOL_DEVIATIONS.md",
        "train_leakage_exclusion": globals().get("_EXCLUSION_AUDIT"),
        "train": {
            "rows": len(train),
            "groups": int(train["content_doc_key"].nunique()),
            "by_dataset": audit["train_counts"],
            "path": str(train_path),
            "sha256": sha256_file(train_path),
        },
        # Test metadata is deliberately limited to counts/groups/file hashes.
        "test": {
            "rows": len(test),
            "groups": int(test["content_doc_key"].nunique()),
            "by_dataset": audit["test_counts"],
            "path": str(test_path),
            "sha256": sha256_file(test_path),
            "scoring_path": str(scoring_path),
            "scoring_sha256": sha256_file(scoring_path),
        },
        "isolation": {
            key: value
            for key, value in audit.items()
            if key not in {"train_counts", "test_counts"}
        },
        "ragtruth_training_mapping_replay": replay,
        "source_sha256": source_hashes,
        "feature_implementations": {
            "cheap27": {
                "copy": "ingest/router_feature_learnability_frozen.py",
                "sha256": sha256_file(PAPER_ROOT / "ingest/router_feature_learnability_frozen.py"),
                "columns": list(CHEAP27),
            },
            "compact16": {
                "copy": "ingest/summary_router_compact16_direct_v1_frozen.py",
                "sha256": sha256_file(
                    PAPER_ROOT / "ingest/summary_router_compact16_direct_v1_frozen.py"
                ),
                "columns": list(COMPACT16),
            },
        },
        "excluded": {
            "faithbench_rows": 0,
            "storysumm_rows": 0,
            "ragtruth_data2txt_rows": 0,
            "ragtruth_qa_rows": 0,
            "cogensumm_val_sentence_pairs_rows": 0,
        },
        "p1_boundary": {
            "only_allowed_input": str(scoring_path),
            "forbidden_gold_columns": [
                column for column in test.columns if column not in scoring.columns
            ],
            "scoring_has_gold": False,
        },
    }
    manifest_path = C.DATA / "SPLIT_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    atomic_json(
        C.STATUS / "P0.done",
        {
            "status": "complete",
            "created_at_utc": utc_now(),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    return {
        "status": "P0_COMPLETE_TEST_SEALED",
        "train_rows": len(train),
        "train_groups": int(train["content_doc_key"].nunique()),
        "test_rows": len(test),
        "test_groups": int(test["content_doc_key"].nunique()),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = _smoke() if args.smoke else build()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
