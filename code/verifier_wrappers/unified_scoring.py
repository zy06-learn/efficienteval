"""Gold-free, resumable scoring utilities for the unified-v1 evaluation line."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from verifier_wrappers.research_freeze import sha256_file


UNIFIED_DATASETS: dict[str, dict[str, Any]] = {
    "ragtruth": {
        "source_name": "ragtruth_summary",
        "rows": 16_991,
        "role": "TRAIN",
    },
    "frank": {
        "source_name": "frank",
        "rows": 4_942,
        "role": "TRAIN",
    },
    "tofueval": {
        "source_name": "tofueval",
        "rows": 4_947,
        "role": "BURNED_DIAGNOSTIC",
    },
}

CORE_VERIFIERS = (
    "factcg",
    "minicheck_dbta",
    "minicheck_ft5",
    "alignscore",
    "hhem",
    "qwen30_judge",
)
OPTIONAL_VERIFIERS = ("factcc", "factkb", "summac_zs")
EXTENDED_VERIFIERS = ("lettuce_v2", "fenice")
ALL_VERIFIERS = (
    "cheap_lexical",
    *CORE_VERIFIERS,
    *OPTIONAL_VERIFIERS,
    *EXTENDED_VERIFIERS,
)

CASCADE_ACTIONS = {
    "factcg": "factcg",
    "minicheck_dbta": "minicheck_dbta",
    "minicheck_ft5": "minicheck_ft5",
    "alignscore": "alignscore",
    "hhem": "hhem",
    "qwen30_judge": "qwen30",
}

PROTOCOLS = {
    "cheap_lexical": "bm25_top1_fullsource_sentence_retrieval_cpu_batch1_v1",
    "factcg": "factcg_deberta_v3_large_fullsource_chunked_promptbatch1_nativeaux_v2",
    "minicheck_dbta": "minicheck_deberta_v3_large_fullsource_chunked_nativeaux_batch1_v1",
    "minicheck_ft5": "minicheck_flan_t5_large_fullsource_chunked_predictprefix_nativeaux_batch1_v2",
    "alignscore": "alignscore_nli_sp_fullsource_persistent_batch1_rawscore_v1",
    "hhem": "hhem_2_1_open_fullsource_tokenwindow512_overlap64_max_batch1_v2",
    "qwen30_judge": "qwen3_30b_a3b_fp8_structured_judge_no_cot_cascade_v1",
    "factcc": "factcc_binary_longest_first512_native_batch1_pinned_v1",
    "factkb": "factkb_binary_claim_article512_native_batch1_pinned_v1",
    "summac_zs": "summac_zs_sentence_nli_max200_batch1_pinned_v1",
    "lettuce_v2": "lettucedetect_v2_mmbert_sourcewin512_overlap64_official4096_spans_batch1_v2",
    "fenice": "fenice_atomic_claim_nli_summary_batch1_persistent_v1",
}

INPUT_COLUMNS = [
    "episode_id",
    "dataset",
    "role",
    "original_dataset",
    "original_split",
    "doc_group_key",
    "content_doc_key",
    "source_document",
    "candidate_sentence",
    "summary_model",
    "split",
    "is_official_test",
    "source_char_count",
    "source_token_count",
]

OUTPUT_COLUMNS = [
    "episode_id",
    "dataset",
    "role",
    "original_split",
    "doc_group_key",
    "summary_model",
    "verifier",
    "model_revision",
    "protocol_version",
    "score",
    "raw_score",
    "native_label",
    "native_logits_json",
    "native_probs_json",
    "native_aux_json",
    "payload_json",
    "raw_response",
    "parse_ok",
    "parse_error",
    "latency_total_ms",
    "latency_preprocessing_ms",
    "latency_retrieval_ms",
    "latency_inference_ms",
    "latency_aggregation_ms",
    "latency_unattributed_ms",
    "num_model_calls",
    "input_tokens",
    "output_tokens",
    "source_token_count",
    "source_char_count",
    "device",
    "batch_size",
    "warmup_done",
    "reused_from_cascade_v1",
]

MODEL_REVISIONS = {
    "alignscore": "yzha/AlignScore-large.ckpt@sha256:ff4336312b377edcbcdad5694a2d09d73dc4225422c0422d810aa7e78485e32d",
    "factcc": "manueldeprada/FactCC@c7b3148015d4ddc263f6e2acb2689e90ac061669",
    "factkb": "bunsenfeng/FactKB@d56496a37c37331ed7c76df8af9a0a510ea7b28b;roberta-base@e2da8e2f811d1448a5b465c236feacd80ffbac7b",
    "summac_zs": "microsoft/deberta-large-mnli@7296194b9009373def4f7c5dad292651e4b5cf4e",
    "lettuce_v2": "KRLabsOrg/lettucedect-v2-mmbert-base@37fee7800fffa993dbfa4e79b638f6532a607d7a+KRLabsOrg/lettucedect-v2-taxonomy-head@b3c30dc2130253920768c3c8f1b6e35a123f39ae",
    "fenice": "Babelscape/FENICE@9741ec41996f1bf75825d7cdf29e931a066ce4f0;Babelscape/t5-base-summarization-claim-extractor@94775fb1c8dc2c3ef1bfec413f9f961e6ba5a1c8;MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli@b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    "qwen30_judge": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8@5a5a776300a41aaa681dd7ff0106608ef2bc90db",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if _is_missing(value):
        return None
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, dict, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def json_value(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            pass
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def forbidden_gold_columns(
    columns: Iterable[Any], *, allow_native_label: bool = False
) -> list[str]:
    forbidden = []
    for column in columns:
        name = str(column).lower()
        if allow_native_label and name.startswith("native_label"):
            continue
        if "label" in name or "gold" in name or "error_type" in name:
            forbidden.append(str(column))
    return sorted(forbidden)


def validate_gold_free_input(frame: pd.DataFrame, dataset: str) -> None:
    forbidden = forbidden_gold_columns(frame.columns)
    if forbidden:
        raise ValueError(f"gold columns are forbidden in scoring input: {forbidden}")
    if list(frame.columns) != INPUT_COLUMNS:
        raise ValueError(f"gold-free input schema mismatch: {list(frame.columns)}")
    if dataset not in UNIFIED_DATASETS:
        raise ValueError(f"unknown unified dataset: {dataset}")
    expected = UNIFIED_DATASETS[dataset]
    if len(frame) != int(expected["rows"]):
        raise ValueError(f"{dataset} row count {len(frame)} != {expected['rows']}")
    if frame["episode_id"].isna().any() or frame["episode_id"].astype(str).duplicated().any():
        raise ValueError(f"{dataset} episode_id must be non-null and unique")
    if set(frame["dataset"].astype(str)) != {dataset}:
        raise ValueError(f"{dataset} input contains a different dataset")
    if set(frame["role"].astype(str)) != {str(expected["role"])}:
        raise ValueError(f"{dataset} role mismatch")
    if set(frame["split"].astype(str)) != {"unified_v1_gold_free"}:
        raise ValueError(f"{dataset} split mismatch")
    if frame["is_official_test"].fillna(False).astype(bool).any():
        raise ValueError(f"{dataset} contains forbidden official/sealed rows")
    if frame[["source_document", "candidate_sentence"]].isna().any().any():
        raise ValueError(f"{dataset} contains null scoring text")


def build_gold_free_frame(source: pd.DataFrame, dataset: str) -> pd.DataFrame:
    spec = UNIFIED_DATASETS[dataset]
    part = source.loc[source["dataset"].eq(spec["source_name"])].copy()
    result = pd.DataFrame(
        {
            "episode_id": part["episode_id"].astype(str),
            "dataset": dataset,
            "role": str(spec["role"]),
            "original_dataset": part["original_dataset"].astype(str),
            "original_split": part["original_split"].astype(str),
            "doc_group_key": part["doc_group_key"].astype(str),
            "content_doc_key": part["content_doc_key"].astype(str),
            "source_document": part["document"].astype(str),
            "candidate_sentence": part["candidate_sentence"].astype(str),
            "summary_model": part["summary_model"].astype(str),
            "split": "unified_v1_gold_free",
            "is_official_test": False,
            "source_char_count": part["document"].astype(str).str.len().astype(int),
            "source_token_count": part["document"].astype(str).str.split().str.len().astype(int),
        }
    )[INPUT_COLUMNS]
    validate_gold_free_input(result, dataset)
    return result


def materialize_gold_free_inputs(sentences_path: Path, input_dir: Path) -> dict[str, Any]:
    sentences_path = Path(sentences_path)
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(sentences_path)
    source_columns = [
        "episode_id",
        "dataset",
        "original_dataset",
        "original_split",
        "doc_group_key",
        "content_doc_key",
        "document",
        "candidate_sentence",
        "summary_model",
    ]
    source = pd.read_parquet(sentences_path, columns=source_columns)
    outputs: dict[str, Any] = {}
    for dataset in UNIFIED_DATASETS:
        frame = build_gold_free_frame(source, dataset)
        parquet_path = input_dir / f"{dataset}.parquet"
        manifest_path = input_dir / f"{dataset}.manifest.json"
        if parquet_path.exists() or manifest_path.exists():
            if not parquet_path.exists() or not manifest_path.exists():
                raise ValueError(f"partial existing scoring input for {dataset}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = pd.read_parquet(parquet_path)
            validate_gold_free_input(existing, dataset)
            if manifest.get("sentences_sha256") != source_hash:
                raise ValueError(f"{dataset} scoring input source changed")
            if manifest.get("parquet_sha256") != sha256_file(parquet_path):
                raise ValueError(f"{dataset} scoring input hash mismatch")
            try:
                pd.testing.assert_frame_equal(
                    existing.reset_index(drop=True),
                    frame.reset_index(drop=True),
                    check_dtype=False,
                )
            except AssertionError as exc:
                raise ValueError(f"{dataset} scoring input content mismatch") from exc
        else:
            frame.to_parquet(parquet_path, index=False)
            manifest = {
                "schema_version": "unified_v1_gold_free_input_v1",
                "created_at_utc": utc_now(),
                "status": "READY_FOR_GOLD_FREE_SCORING",
                "dataset": dataset,
                "role": UNIFIED_DATASETS[dataset]["role"],
                "rows": len(frame),
                "columns": INPUT_COLUMNS,
                "scoring_input_has_gold": False,
                "gold_columns_read_from_parquet": False,
                "fresh_or_sealed_labels_read": False,
                "official_test_rows_read": 0,
                "sentences_path": str(sentences_path),
                "sentences_sha256": source_hash,
                "parquet_path": str(parquet_path),
                "parquet_sha256": sha256_file(parquet_path),
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        outputs[dataset] = manifest
    return outputs


def read_input(input_dir: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    parquet_path = Path(input_dir) / f"{dataset}.parquet"
    manifest_path = Path(input_dir) / f"{dataset}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(parquet_path)
    validate_gold_free_input(frame, dataset)
    if manifest.get("status") != "READY_FOR_GOLD_FREE_SCORING":
        raise ValueError(f"{dataset} input manifest is not ready")
    if manifest.get("scoring_input_has_gold") is not False:
        raise ValueError(f"{dataset} manifest does not prove gold-free input")
    if manifest.get("parquet_sha256") != sha256_file(parquet_path):
        raise ValueError(f"{dataset} input parquet hash mismatch")
    return frame, manifest, parquet_path


def write_file_hash_inventory(source_dir: Path, output_path: Path) -> dict[str, str]:
    source_dir = Path(source_dir)
    files = sorted(path for path in source_dir.iterdir() if path.is_file())
    hashes = {path.name: sha256_file(path) for path in files}
    output_path = Path(output_path)
    if output_path.exists():
        existing = {}
        for line in output_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            existing[name] = digest
        if existing != hashes:
            raise ValueError("cascade_v1 verifier-score baseline changed")
        return hashes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()),
        encoding="utf-8",
    )
    return hashes


def verify_file_hash_inventory(source_dir: Path, inventory_path: Path) -> dict[str, Any]:
    expected = {}
    for line in Path(inventory_path).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    actual = {
        path.name: sha256_file(path)
        for path in sorted(Path(source_dir).iterdir())
        if path.is_file()
    }
    changed = sorted(name for name in set(expected) | set(actual) if expected.get(name) != actual.get(name))
    return {
        "expected_files": len(expected),
        "actual_files": len(actual),
        "changed": changed,
        "unchanged": not changed,
    }


def _ensure_exact_ids(frame: pd.DataFrame, expected_ids: set[str], label: str) -> None:
    if frame["episode_id"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate episode_id")
    ids = set(frame["episode_id"].astype(str))
    if ids != expected_ids:
        raise ValueError(
            f"{label} episode mismatch: missing={len(expected_ids - ids)} extra={len(ids - expected_ids)}"
        )


def _validate_protocol_version(
    frame: pd.DataFrame,
    verifier: str,
    *,
    label: str,
) -> str:
    expected = PROTOCOLS[verifier]
    if "protocol_version" not in frame.columns:
        raise ValueError(f"{label} has no protocol_version column")
    series = frame["protocol_version"]
    actual = sorted(series.dropna().astype(str).unique().tolist())
    if series.isna().any():
        actual.append("<NULL>")
    if actual != [expected]:
        raise ValueError(
            f"{label} protocol mismatch: expected={expected!r} actual={actual!r}"
        )
    return expected


def validate_ragtruth_reuse(
    ragtruth_input: pd.DataFrame,
    cascade_episodes_path: Path,
    cascade_scores_dir: Path,
) -> dict[str, Any]:
    validate_gold_free_input(ragtruth_input, "ragtruth")
    expected_ids = set(ragtruth_input["episode_id"].astype(str))
    cascade = pd.read_parquet(
        cascade_episodes_path,
        columns=["episode_id", "dataset", "source_document", "candidate_sentence"],
    )
    cascade = cascade.loc[cascade["dataset"].eq("RAGTruth-Summary")].copy()
    _ensure_exact_ids(cascade, expected_ids, "cascade RAGTruth episodes")
    joined = ragtruth_input[
        ["episode_id", "source_document", "candidate_sentence"]
    ].merge(cascade, on="episode_id", suffixes=("_unified", "_cascade"), validate="one_to_one")
    doc_ok = joined["source_document_unified"].map(sha256_text).eq(
        joined["source_document_cascade"].map(sha256_text)
    )
    claim_ok = joined["candidate_sentence_unified"].map(sha256_text).eq(
        joined["candidate_sentence_cascade"].map(sha256_text)
    )
    if not doc_ok.all() or not claim_ok.all():
        raise ValueError(
            "RAGTruth reuse content hash mismatch: "
            f"documents={int((~doc_ok).sum())} claims={int((~claim_ok).sum())}"
        )

    score_audit = {}
    for verifier, action in CASCADE_ACTIONS.items():
        path = Path(cascade_scores_dir) / f"primary_train__{action}.parquet"
        try:
            score = pd.read_parquet(
                path, columns=["episode_id", "dataset", "protocol_version"]
            )
        except Exception as exc:
            raise ValueError(
                f"cascade score {action} is missing required reuse columns"
            ) from exc
        score = score.loc[score["dataset"].eq("RAGTruth-Summary")].copy()
        _ensure_exact_ids(score, expected_ids, f"cascade score {action}")
        protocol_version = _validate_protocol_version(
            score, verifier, label=f"cascade score {action}"
        )
        score_audit[verifier] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(score),
            "protocol_version": protocol_version,
        }
    return {
        "status": "PASS",
        "rows": len(ragtruth_input),
        "document_hash_matches": int(doc_ok.sum()),
        "candidate_hash_matches": int(claim_ok.sum()),
        "score_files": score_audit,
    }


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    value = row.get(name)
    if _is_missing(value):
        return None
    return value


def _native_aux_from_reused(row: Mapping[str, Any]) -> str | None:
    if _row_value(row, "aux_json") is not None:
        return json_value(row["aux_json"])
    prefixes = (
        "native_",
        "chunk_",
        "support_prob_",
        "prompt_token_",
        "aggregation_",
        "best_chunk_",
        "n_chunks",
        "payload.",
    )
    payload = {
        key: _json_default(value)
        for key, value in row.items()
        if key.startswith(prefixes) and _row_value(row, key) is not None
    }
    return json_value(payload) if payload else None


def _reused_model_calls(row: Mapping[str, Any], verifier: str) -> int:
    for key in ("n_chunks", "chunk_count"):
        value = _row_value(row, key)
        if value is not None:
            return max(1, int(value))
    return 1


def _reused_input_tokens(row: Mapping[str, Any]) -> int | None:
    value = _row_value(row, "usage.prompt_tokens")
    if value is not None:
        return int(value)
    prompt_counts = _row_value(row, "prompt_token_counts_json")
    if prompt_counts:
        return int(sum(json.loads(str(prompt_counts))))
    return None


def transform_reused_score(
    input_frame: pd.DataFrame,
    source_score: pd.DataFrame,
    verifier: str,
) -> pd.DataFrame:
    metadata = input_frame.set_index("episode_id")
    source = source_score.loc[source_score["dataset"].eq("RAGTruth-Summary")].copy()
    source = source.set_index("episode_id").reindex(metadata.index)
    parse_ok = source.get("parse_ok", pd.Series(True, index=source.index)).fillna(False).astype(bool)
    if source.index.has_duplicates or source.loc[parse_ok, "raw_score"].isna().any():
        raise ValueError(f"incomplete reused source for {verifier}")
    rows = []
    for episode_id, source_row in source.iterrows():
        item = source_row.to_dict()
        meta = metadata.loc[episode_id]
        total = float(item.get("total_latency_ms", item.get("latency_ms")))
        payload = _row_value(item, "payload_json") or _row_value(item, "payload")
        raw_score = _row_value(item, "raw_score")
        rows.append(
            {
                "episode_id": str(episode_id),
                "dataset": "ragtruth",
                "role": "TRAIN",
                "original_split": str(meta.original_split),
                "doc_group_key": str(meta.doc_group_key),
                "summary_model": str(meta.summary_model),
                "verifier": verifier,
                "model_revision": str(item.get("model_id")),
                "protocol_version": str(item.get("protocol_version")),
                "score": float(raw_score) if raw_score is not None else None,
                "raw_score": float(raw_score) if raw_score is not None else None,
                "native_label": _row_value(item, "native_label"),
                "native_logits_json": json_value(_row_value(item, "native_logits_json")),
                "native_probs_json": json_value(_row_value(item, "native_probs_json")),
                "native_aux_json": _native_aux_from_reused(item),
                "payload_json": json_value(payload),
                "raw_response": _row_value(item, "raw_response"),
                "parse_ok": bool(item.get("parse_ok", True)),
                "parse_error": _row_value(item, "parse_error"),
                "latency_total_ms": total,
                "latency_preprocessing_ms": None,
                "latency_retrieval_ms": None,
                "latency_inference_ms": None,
                "latency_aggregation_ms": None,
                "latency_unattributed_ms": total,
                "num_model_calls": _reused_model_calls(item, verifier),
                "input_tokens": _reused_input_tokens(item),
                "output_tokens": int(item["usage.completion_tokens"])
                if _row_value(item, "usage.completion_tokens") is not None
                else None,
                "source_token_count": int(meta.source_token_count),
                "source_char_count": int(meta.source_char_count),
                "device": None,
                "batch_size": 1,
                "warmup_done": None,
                "reused_from_cascade_v1": True,
            }
        )
    result = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    validate_score_frame(result, "ragtruth", verifier)
    return result


def materialize_reused_ragtruth_scores(
    input_dir: Path,
    cascade_episodes_path: Path,
    cascade_scores_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    ragtruth, input_manifest, input_path = read_input(input_dir, "ragtruth")
    audit = validate_ragtruth_reuse(ragtruth, cascade_episodes_path, cascade_scores_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for verifier, action in CASCADE_ACTIONS.items():
        source_path = Path(cascade_scores_dir) / f"primary_train__{action}.parquet"
        output_path = output_dir / f"ragtruth__{verifier}.parquet"
        manifest_path = output_dir / f"ragtruth__{verifier}.manifest.json"
        if output_path.exists() or manifest_path.exists():
            if not output_path.exists() or not manifest_path.exists():
                raise ValueError(f"partial reused artifact for {verifier}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_protocol = PROTOCOLS[verifier]
            if manifest.get("protocol_version") != expected_protocol:
                raise ValueError(
                    f"reused manifest protocol mismatch for {verifier}: "
                    f"expected={expected_protocol!r} "
                    f"actual={manifest.get('protocol_version')!r}"
                )
            frame = pd.read_parquet(output_path)
            validate_score_frame(frame, "ragtruth", verifier)
            if manifest.get("source_score_sha256") != sha256_file(source_path):
                raise ValueError(f"reused source changed for {verifier}")
            if manifest.get("parquet_sha256") != sha256_file(output_path):
                raise ValueError(f"reused output hash mismatch for {verifier}")
        else:
            source = pd.read_parquet(source_path)
            frame = transform_reused_score(ragtruth, source, verifier)
            frame.to_parquet(output_path, index=False)
            latency = frame["latency_total_ms"].astype(float)
            manifest = {
                "schema_version": "unified_v1_verifier_score_manifest_v1",
                "created_at_utc": utc_now(),
                "status": "SCORED",
                "dataset": "ragtruth",
                "role": "TRAIN",
                "verifier": verifier,
                "model_revision": str(frame["model_revision"].iloc[0]),
                "protocol_version": str(frame["protocol_version"].iloc[0]),
                "rows_requested": len(ragtruth),
                "rows": len(frame),
                "coverage": 1.0,
                "parse_ok_rate": float(frame["parse_ok"].mean()),
                "strict_batch1": True,
                "reused_from_cascade_v1": True,
                "reuse_note": "REUSED_FROM_CASCADE_V1; unavailable component latencies are null",
                "reused_rows": len(frame),
                "restart_count": 0,
                "warmup_done": None,
                "input_path": str(input_path),
                "input_sha256": input_manifest["parquet_sha256"],
                "source_score_path": str(source_path),
                "source_score_sha256": sha256_file(source_path),
                "latency_mean_ms": float(latency.mean()),
                "latency_p50_ms": float(latency.median()),
                "latency_p95_ms": float(latency.quantile(0.95)),
                "component_latency_status": "UNAVAILABLE_IN_REUSED_CASCADE_V1",
                "parquet_path": str(output_path),
                "parquet_sha256": sha256_file(output_path),
                "reuse_hard_validation": audit,
                "fresh_or_sealed_labels_read": False,
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        manifests[verifier] = manifest
    return {"audit": audit, "manifests": manifests}


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def gpu_metadata() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }
        return {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


class BM25Top1Scorer:
    """Single frozen cheap score from router_feature_learnability.py."""

    score_max = None
    model_id = "internal:router_feature_learnability.BM25_top1"

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = "cpu"
        self.indices: dict[str, Any] = {}

    def reset_runtime_state(self) -> None:
        self.indices.clear()

    def score_batch(
        self,
        docs: list[str],
        claims: list[str],
        *,
        doc_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        from verifier_wrappers.router_feature_learnability import (
            _build_document_index,
            _retrieval_features,
        )

        if len(docs) != 1 or len(claims) != 1:
            raise ValueError("cheap_lexical requires strict episode batch=1")
        key = str((doc_keys or [sha256_text(docs[0])])[0])
        setup_ms = 0.0
        if key not in self.indices:
            self.indices[key] = _build_document_index(str(docs[0]))
            setup_ms = float(self.indices[key].setup_ms)
        started = time.perf_counter()
        features = _retrieval_features(str(claims[0]), self.indices[key])
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        return [
            {
                "score": float(features["bm25_top1"]),
                "aux": {
                    "native_output_type": "scalar_unbounded_nonnegative",
                    "score_direction": "higher_more_factual",
                    "latency_preprocessing_ms": setup_ms,
                    "latency_retrieval_ms": retrieval_ms,
                    "latency_inference_ms": 0.0,
                    "latency_aggregation_ms": 0.0,
                    "num_model_calls": 0,
                },
            }
        ]


class FactCCPinnedScorer:
    score_max = 1.0
    model_id = MODEL_REVISIONS["factcc"]
    label_space = ("CORRECT", "INCORRECT")

    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        repo = "manueldeprada/FactCC"
        revision = "c7b3148015d4ddc263f6e2acb2689e90ac061669"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(repo, revision=revision)
            .to(device)
            .eval()
        )
        id2label = {int(key): value.upper() for key, value in self.model.config.id2label.items()}
        self.label_space = tuple(id2label[index] for index in sorted(id2label))
        self.incorrect_index = next(
            (index for index, name in id2label.items() if "INCORRECT" in name), 1
        )
        self._torch = torch

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            docs,
            claims,
            truncation="longest_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits
        probs = self._torch.softmax(logits, dim=-1)
        outputs = []
        for row_logits, row_probs in zip(logits.cpu().tolist(), probs.cpu().tolist()):
            support = 1.0 - float(row_probs[self.incorrect_index])
            outputs.append(
                {
                    "score": support,
                    "aux": {
                        "native_output_type": "categorical",
                        "native_label_space": list(self.label_space),
                        "native_logits": [float(value) for value in row_logits],
                        "native_probs": [float(value) for value in row_probs],
                        "native_label": self.label_space[
                            int(max(range(len(row_probs)), key=row_probs.__getitem__))
                        ],
                        "label_source": "native_argmax",
                        "num_model_calls": 1,
                    },
                }
            )
        return outputs


class FactKBPinnedScorer:
    score_max = 1.0
    model_id = MODEL_REVISIONS["factkb"]
    label_space = ("nonfactual", "factual")

    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_revision = "d56496a37c37331ed7c76df8af9a0a510ea7b28b"
        tokenizer_revision = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", revision=tokenizer_revision
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                "bunsenfeng/FactKB", revision=model_revision, num_labels=2
            )
            .to(device)
            .eval()
        )
        self._torch = torch

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            claims,
            docs,
            truncation="longest_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits
        probs = self._torch.softmax(logits, dim=-1)
        outputs = []
        for row_logits, row_probs in zip(logits.cpu().tolist(), probs.cpu().tolist()):
            outputs.append(
                {
                    "score": float(row_probs[1]),
                    "aux": {
                        "native_output_type": "categorical",
                        "native_label_space": list(self.label_space),
                        "native_logits": [float(value) for value in row_logits],
                        "native_probs": [float(value) for value in row_probs],
                        "native_label": self.label_space[int(row_probs[1] >= row_probs[0])],
                        "label_source": "native_argmax",
                        "num_model_calls": 1,
                    },
                }
            )
        return outputs


class SummaCZSPinnedScorer:
    score_max = 1.0
    model_id = MODEL_REVISIONS["summac_zs"]
    max_doc_sentences = 200
    nli_microbatch = 128

    def __init__(self, *, device: str = "cuda") -> None:
        import nltk
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            nltk.data.find(resource)
        revision = "7296194b9009373def4f7c5dad292651e4b5cf4e"
        self.sent_tokenize = nltk.sent_tokenize
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-large-mnli", revision=revision
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli", revision=revision
            )
            .to(device)
            .eval()
        )
        id2label = {int(key): value.upper() for key, value in self.model.config.id2label.items()}
        self.entail_index = next(index for index, name in id2label.items() if "ENTAIL" in name)
        self.contradict_index = next(
            index for index, name in id2label.items() if "CONTRADICT" in name
        )
        self._torch = torch
        self._sentence_cache: dict[str, list[str]] = {}

    def reset_runtime_state(self) -> None:
        self._sentence_cache.clear()

    def score_batch(
        self,
        docs: list[str],
        claims: list[str],
        *,
        doc_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if len(docs) != 1 or len(claims) != 1:
            raise ValueError("summac_zs requires strict episode batch=1")
        key = str((doc_keys or [sha256_text(docs[0])])[0])
        if key not in self._sentence_cache:
            sentences = [value for value in self.sent_tokenize(str(docs[0])) if value.strip()]
            self._sentence_cache[key] = sentences[: self.max_doc_sentences]
        premises = self._sentence_cache[key]
        if not premises:
            return [{"score": 0.5, "aux": {"n_doc_sentences": 0, "num_model_calls": 0}}]
        best = -1.0
        calls = 0
        for start in range(0, len(premises), self.nli_microbatch):
            chunk = premises[start : start + self.nli_microbatch]
            encoded = self.tokenizer(
                chunk,
                [str(claims[0])] * len(chunk),
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
            probs = self._torch.softmax(logits, dim=-1)
            value = (
                probs[:, self.entail_index] - probs[:, self.contradict_index]
            ).max().item()
            best = max(best, float(value))
            calls += 1
        score = (best + 1.0) / 2.0
        return [
            {
                "score": score,
                "aux": {
                    "native_output_type": "scalar",
                    "native_label": "supported" if score >= 0.5 else "unsupported",
                    "label_source": "derived_threshold_0.5",
                    "n_doc_sentences": len(premises),
                    "raw_entail_minus_contradict": best,
                    "aggregation_rule": "max_entail_minus_contradict_over_document_sentences",
                    "num_model_calls": calls,
                },
            }
        ]


class Qwen30APIScorer:
    score_max = 1.0
    model_id = MODEL_REVISIONS["qwen30_judge"]

    def __init__(
        self,
        *,
        api_base: str,
        served_model: str,
        max_tokens: int = 400,
        device: str = "cuda",
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=api_base, api_key="EMPTY")
        served = {model.id for model in self.client.models.list().data}
        if served_model not in served:
            raise RuntimeError(f"{served_model} not served; got {sorted(served)}")
        self.served_model = served_model
        self.max_tokens = int(max_tokens)
        self.device = device

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        from verifier_wrappers.structured_high_judge import (
            SYSTEM_PROMPT,
            numbered_source,
            structured_response_format,
            validate_payload_for_score,
        )

        if len(docs) != 1 or len(claims) != 1:
            raise ValueError("qwen30_judge requires strict batch=1")
        preprocessing_started = time.perf_counter()
        numbered, sentence_count = numbered_source(str(docs[0]))
        claim = str(claims[0])
        user_prompt = (
            f"SOURCE (numbered sentences):\n{numbered}\n\n"
            f"SENTENCE:\n{claim}\n\nJSON:"
        )
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        payload = None
        raw_response = None
        parse_error = None
        usage = None
        attempts = 0
        inference_ms = 0.0
        aggregation_ms = 0.0
        while attempts < 2 and payload is None:
            attempts += 1
            call_started = time.perf_counter()
            call_recorded = False
            try:
                response = self.client.chat.completions.create(
                    model=self.served_model,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format=structured_response_format(sentence_count),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                inference_ms += (time.perf_counter() - call_started) * 1000.0
                call_recorded = True
                raw_response = response.choices[0].message.content or ""
                parse_started = time.perf_counter()
                payload = validate_payload_for_score(
                    json.loads(raw_response),
                    claim=claim,
                    n_source_sentences=sentence_count,
                )
                aggregation_ms += (time.perf_counter() - parse_started) * 1000.0
                usage = response.usage
            except Exception as exc:
                if not call_recorded:
                    inference_ms += (time.perf_counter() - call_started) * 1000.0
                parse_error = f"{type(exc).__name__}: {exc}"[:500]
        parse_ok = payload is not None
        return [
            {
                "score": payload["support_probability"] if parse_ok else None,
                "parse_ok": parse_ok,
                "parse_error": None if parse_ok else parse_error,
                "aux": {
                    "native_output_type": "structured",
                    "native_label": payload["label"] if parse_ok else None,
                    "label_source": "native_structured_label" if parse_ok else None,
                    "payload": payload,
                    "raw_response": raw_response,
                    "parse_attempts": attempts,
                    "latency_preprocessing_ms": preprocessing_ms,
                    "latency_inference_ms": inference_ms,
                    "latency_aggregation_ms": aggregation_ms,
                    "num_model_calls": attempts,
                    "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                },
            }
        ]


def build_unified_scorer(
    verifier: str,
    *,
    device: str,
    api_base: str = "http://127.0.0.1:8000/v1",
    served_model: str = "qwen3-30b-a3b-judge",
) -> Any:
    if verifier == "cheap_lexical":
        return BM25Top1Scorer(device="cpu")
    if verifier in {"factcg", "minicheck_dbta", "minicheck_ft5", "alignscore", "hhem"}:
        from verifier_wrappers.primary_scoring import build_primary_scorer

        return build_primary_scorer(verifier, device=device)
    if verifier == "qwen30_judge":
        return Qwen30APIScorer(
            api_base=api_base,
            served_model=served_model,
            device=device,
        )
    if verifier == "factcc":
        return FactCCPinnedScorer(device=device)
    if verifier == "factkb":
        return FactKBPinnedScorer(device=device)
    if verifier == "summac_zs":
        return SummaCZSPinnedScorer(device=device)
    if verifier == "lettuce_v2":
        from verifier_wrappers.extended_scorers import LettuceV2Scorer

        return LettuceV2Scorer(device=device)
    if verifier == "fenice":
        from verifier_wrappers.candidate_verifiers import FENICEScorer

        return FENICEScorer()
    raise ValueError(f"unsupported verifier: {verifier}")


def scorer_model_revision(scorer: Any, verifier: str) -> str:
    if verifier in {"alignscore", "lettuce_v2", "fenice"}:
        return MODEL_REVISIONS[verifier]
    return str(getattr(scorer, "model_id", MODEL_REVISIONS.get(verifier, verifier)))


def _call_scorer(scorer: Any, row: Any) -> dict[str, Any]:
    parameters = inspect.signature(scorer.score_batch).parameters
    kwargs = {}
    if "doc_keys" in parameters:
        kwargs["doc_keys"] = [str(row.doc_group_key)]
    outputs = scorer.score_batch(
        [str(row.source_document)],
        [str(row.candidate_sentence)],
        **kwargs,
    )
    if len(outputs) != 1:
        raise ValueError(f"scorer returned {len(outputs)} outputs for batch=1")
    return outputs[0]


def warm_up_scorer(scorer: Any, frame: pd.DataFrame, count: int = 5) -> None:
    if len(frame) < count:
        raise ValueError("not enough rows for warm-up")
    for row in frame.head(count).itertuples(index=False):
        cuda_sync()
        output = _call_scorer(scorer, row)
        cuda_sync()
        if output.get("score") is None and output.get("parse_ok", True):
            raise ValueError("warm-up returned no score")
    reset = getattr(scorer, "reset_runtime_state", None)
    if callable(reset):
        reset()


def select_smoke(frame: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("smoke count must be positive")
    selected = (
        frame.sort_values(
            ["source_token_count", "doc_group_key", "episode_id"],
            ascending=[False, True, True],
        )
        .drop_duplicates("doc_group_key")
        .head(count)
    )
    if len(selected) != count:
        raise ValueError(f"not enough source groups for {count}-row smoke")
    return selected[list(frame.columns)].copy()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    repair_tail = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                repair_tail = True
                break
            raise ValueError(f"invalid JSONL at {path}:{index + 1}") from exc
        forbidden = forbidden_gold_columns(item, allow_native_label=True)
        if forbidden:
            raise ValueError(f"scoring cache leaks supervision: {forbidden}")
        rows.append(item)
    if repair_tail:
        _write_jsonl(path.with_suffix(path.suffix + ".repair"), rows, append=False)
        os.replace(path.with_suffix(path.suffix + ".repair"), path)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
        handle.flush()


def _cache_by_episode(path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    by_id = {}
    for row in rows:
        episode_id = str(row.get("episode_id", ""))
        if not episode_id:
            raise ValueError(f"cache row missing episode_id: {path}")
        by_id[episode_id] = row
    if len(by_id) != len(rows):
        _write_jsonl(path.with_suffix(path.suffix + ".compact"), list(by_id.values()), append=False)
        os.replace(path.with_suffix(path.suffix + ".compact"), path)
    return by_id


def _drop_parse_failures(
    path: Path, cache: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    kept = {
        episode_id: dict(row)
        for episode_id, row in cache.items()
        if bool(row.get("parse_ok"))
    }
    if len(kept) != len(cache):
        _write_jsonl(path, list(kept.values()), append=False)
    return kept


def _component(aux: Mapping[str, Any], key: str) -> float | None:
    value = aux.get(key)
    return float(value) if value is not None else None


def _output_row(
    episode: Any,
    output: Mapping[str, Any],
    *,
    verifier: str,
    model_revision: str,
    protocol_version: str,
    device: str,
    total_ms: float,
    warmup_done: bool,
) -> dict[str, Any]:
    aux = dict(output.get("aux") or {})
    parse_ok = bool(output.get("parse_ok", True))
    score_value = output.get("score")
    if parse_ok:
        if score_value is None or not math.isfinite(float(score_value)):
            raise ValueError(f"{verifier} returned invalid score {score_value}")
    score = float(score_value) if score_value is not None else None
    preprocessing = _component(aux, "latency_preprocessing_ms")
    retrieval = _component(aux, "latency_retrieval_ms")
    inference = _component(aux, "latency_inference_ms")
    aggregation = _component(aux, "latency_aggregation_ms")
    components = (preprocessing, retrieval, inference, aggregation)
    unattributed = None if any(value is not None for value in components) else float(total_ms)
    calls = aux.get("num_model_calls")
    if calls is None:
        calls = aux.get("n_chunks", aux.get("chunk_count", 1))
    prompt_counts = aux.get("prompt_token_counts")
    input_tokens = aux.get("input_tokens")
    if input_tokens is None and prompt_counts:
        input_tokens = sum(int(value) for value in prompt_counts)
    return {
        "episode_id": str(episode.episode_id),
        "dataset": str(episode.dataset),
        "role": str(episode.role),
        "original_split": str(episode.original_split),
        "doc_group_key": str(episode.doc_group_key),
        "summary_model": str(episode.summary_model),
        "verifier": verifier,
        "model_revision": model_revision,
        "protocol_version": protocol_version,
        "score": score,
        "raw_score": score,
        "native_label": aux.get("native_label"),
        "native_logits_json": json_value(aux.get("native_logits")),
        "native_probs_json": json_value(aux.get("native_probs")),
        "native_aux_json": json_value(aux),
        "payload_json": json_value(aux.get("payload")),
        "raw_response": aux.get("raw_response"),
        "parse_ok": parse_ok,
        "parse_error": output.get("parse_error"),
        "latency_total_ms": float(total_ms),
        "latency_preprocessing_ms": preprocessing,
        "latency_retrieval_ms": retrieval,
        "latency_inference_ms": inference,
        "latency_aggregation_ms": aggregation,
        "latency_unattributed_ms": unattributed,
        "num_model_calls": int(calls),
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(aux["output_tokens"]) if aux.get("output_tokens") is not None else None,
        "source_token_count": int(episode.source_token_count),
        "source_char_count": int(episode.source_char_count),
        "device": device,
        "batch_size": 1,
        "warmup_done": bool(warmup_done),
        "reused_from_cascade_v1": False,
    }


def validate_score_frame(frame: pd.DataFrame, dataset: str, verifier: str) -> None:
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"score schema mismatch for {dataset}/{verifier}")
    _validate_protocol_version(
        frame, verifier, label=f"score output {dataset}/{verifier}"
    )
    forbidden = forbidden_gold_columns(frame.columns, allow_native_label=True)
    if forbidden:
        raise ValueError(f"score output leaks supervision: {forbidden}")
    expected = int(UNIFIED_DATASETS[dataset]["rows"])
    if len(frame) != expected:
        raise ValueError(f"{dataset}/{verifier} rows {len(frame)} != {expected}")
    if frame["episode_id"].astype(str).duplicated().any():
        raise ValueError(f"duplicate scores for {dataset}/{verifier}")
    if set(frame["dataset"].astype(str)) != {dataset}:
        raise ValueError(f"dataset mismatch for {dataset}/{verifier}")
    if set(frame["verifier"].astype(str)) != {verifier}:
        raise ValueError(f"verifier mismatch for {dataset}/{verifier}")
    valid = frame["parse_ok"].fillna(False).astype(bool)
    if frame.loc[valid, "score"].isna().any():
        raise ValueError(f"parse-ok scores contain nulls for {dataset}/{verifier}")
    if (frame.loc[valid, "score"].astype(float) < 0).any():
        raise ValueError(f"negative scores for {dataset}/{verifier}")
    if verifier != "cheap_lexical" and (frame.loc[valid, "score"].astype(float) > 1).any():
        raise ValueError(f"scores exceed one for {dataset}/{verifier}")
    if dataset == "tofueval" and set(frame["role"].astype(str)) != {"BURNED_DIAGNOSTIC"}:
        raise ValueError("TofuEval scores are not marked BURNED_DIAGNOSTIC")


def _validate_cache_identity(
    cache: Mapping[str, Mapping[str, Any]],
    *,
    verifier: str,
    protocol_version: str,
    model_revision: str,
    input_fingerprint: str,
) -> None:
    for row in cache.values():
        if (
            row.get("verifier") != verifier
            or row.get("protocol_version") != protocol_version
            or row.get("model_revision") != model_revision
            or row.get("input_fingerprint") != input_fingerprint
        ):
            raise ValueError(f"cache action/protocol/model/input mismatch for {verifier}")


def _score_pending_rows(
    frame: pd.DataFrame,
    *,
    scorer: Any,
    verifier: str,
    model_revision: str,
    protocol_version: str,
    device: str,
    input_fingerprint: str,
    cache_path: Path,
    already_done: set[str],
    progress_total: int,
) -> None:
    pending = frame.loc[~frame["episode_id"].astype(str).isin(already_done)]
    with cache_path.open("a", encoding="utf-8") as sink:
        for offset, episode in enumerate(pending.itertuples(index=False), start=1):
            cuda_sync()
            started = time.perf_counter()
            output = _call_scorer(scorer, episode)
            cuda_sync()
            total_ms = (time.perf_counter() - started) * 1000.0
            row = _output_row(
                episode,
                output,
                verifier=verifier,
                model_revision=model_revision,
                protocol_version=protocol_version,
                device=device,
                total_ms=total_ms,
                warmup_done=True,
            )
            row["input_fingerprint"] = input_fingerprint
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
            sink.flush()
            completed = len(already_done) + offset
            if completed % 100 == 0 or completed == progress_total:
                print(f"[{dataset_label(frame)}/{verifier}] {completed}/{progress_total}", flush=True)


def dataset_label(frame: pd.DataFrame) -> str:
    values = set(frame["dataset"].astype(str))
    return next(iter(values)) if len(values) == 1 else "mixed"


def _repeat_profile(
    scorer: Any,
    input_frame: pd.DataFrame,
    scored: pd.DataFrame,
    count: int = 200,
) -> dict[str, Any]:
    sample = input_frame.head(min(count, len(input_frame))).copy()
    reset = getattr(scorer, "reset_runtime_state", None)
    if callable(reset):
        reset()
    repeat = []
    for row in sample.itertuples(index=False):
        cuda_sync()
        started = time.perf_counter()
        _call_scorer(scorer, row)
        cuda_sync()
        repeat.append((time.perf_counter() - started) * 1000.0)
    original = scored.set_index("episode_id").loc[sample["episode_id"], "latency_total_ms"].astype(float)
    first_mean = float(original.mean())
    second_mean = float(sum(repeat) / len(repeat))
    relative = abs(second_mean - first_mean) / max(first_mean, 1e-12)
    return {
        "rows": len(sample),
        "first_round_mean_ms": first_mean,
        "second_round_mean_ms": second_mean,
        "relative_mean_difference": relative,
        "unstable_over_10_percent": relative > 0.10,
    }


def score_dataset(
    *,
    scorer: Any,
    verifier: str,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    device: str,
    model_load_ms: float,
    restart_count: int,
    smoke_count: int = 20,
    repeat_count: int = 200,
    required_parse_rate: float = 0.99,
) -> dict[str, Any]:
    input_frame, input_manifest, input_path = read_input(input_dir, dataset)
    protocol_version = PROTOCOLS[verifier]
    model_revision = scorer_model_revision(scorer, verifier)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset}__{verifier}"
    cache_path = output_dir / f"{stem}.jsonl"
    parquet_path = output_dir / f"{stem}.parquet"
    manifest_path = output_dir / f"{stem}.manifest.json"
    if parquet_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(parquet_path)
        validate_score_frame(frame, dataset, verifier)
        if manifest.get("protocol_version") != protocol_version:
            raise ValueError(
                f"existing manifest protocol mismatch for {dataset}/{verifier}: "
                f"expected={protocol_version!r} "
                f"actual={manifest.get('protocol_version')!r}"
            )
        if (
            manifest.get("status") == "SCORED"
            and manifest.get("input_sha256") == input_manifest["parquet_sha256"]
            and manifest.get("parquet_sha256") == sha256_file(parquet_path)
        ):
            print(f"PASS existing {dataset}/{verifier}", flush=True)
            return manifest

    input_fingerprint = str(input_manifest["parquet_sha256"])
    cache = _cache_by_episode(cache_path)
    if verifier == "qwen30_judge":
        cache = _drop_parse_failures(cache_path, cache)
    _validate_cache_identity(
        cache,
        verifier=verifier,
        protocol_version=protocol_version,
        model_revision=model_revision,
        input_fingerprint=input_fingerprint,
    )
    input_ids = set(input_frame["episode_id"].astype(str))
    extra = set(cache) - input_ids
    if extra:
        raise ValueError(f"cache contains {len(extra)} unexpected episodes for {dataset}/{verifier}")

    smoke = select_smoke(input_frame, smoke_count)
    _score_pending_rows(
        smoke,
        scorer=scorer,
        verifier=verifier,
        model_revision=model_revision,
        protocol_version=protocol_version,
        device=device,
        input_fingerprint=input_fingerprint,
        cache_path=cache_path,
        already_done=set(cache),
        progress_total=smoke_count,
    )
    cache = _cache_by_episode(cache_path)
    smoke_rows = [cache[str(value)] for value in smoke["episode_id"]]
    smoke_parse_rate = sum(bool(row["parse_ok"]) for row in smoke_rows) / smoke_count
    if smoke_parse_rate < 0.95:
        raise ValueError(f"{dataset}/{verifier} smoke parse rate {smoke_parse_rate:.4f} < 0.95")
    print(f"PASS smoke {dataset}/{verifier}: rows={smoke_count}", flush=True)

    if verifier == "qwen30_judge":
        cache = _drop_parse_failures(cache_path, cache)

    _score_pending_rows(
        input_frame,
        scorer=scorer,
        verifier=verifier,
        model_revision=model_revision,
        protocol_version=protocol_version,
        device=device,
        input_fingerprint=input_fingerprint,
        cache_path=cache_path,
        already_done=set(cache),
        progress_total=len(input_frame),
    )
    cache = _cache_by_episode(cache_path)
    missing = input_ids - set(cache)
    if missing:
        raise ValueError(f"{dataset}/{verifier} cache missing {len(missing)} rows")
    rows = [cache[str(value)] for value in input_frame["episode_id"]]
    for row in rows:
        row.pop("input_fingerprint", None)
    scored = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    validate_score_frame(scored, dataset, verifier)
    scored.to_parquet(parquet_path, index=False)
    repeat = _repeat_profile(scorer, input_frame, scored, count=repeat_count)
    latency = scored["latency_total_ms"].astype(float)
    valid = scored["parse_ok"].fillna(False).astype(bool)
    components = {
        column: bool(scored[column].notna().any())
        for column in (
            "latency_preprocessing_ms",
            "latency_retrieval_ms",
            "latency_inference_ms",
            "latency_aggregation_ms",
            "latency_unattributed_ms",
        )
    }
    parse_ok_rate = float(valid.mean())
    validation_blocker = (
        None
        if parse_ok_rate >= float(required_parse_rate)
        else f"parse_ok_rate {parse_ok_rate:.6f} < {required_parse_rate:.6f}"
    )
    manifest = {
        "schema_version": "unified_v1_verifier_score_manifest_v1",
        "created_at_utc": utc_now(),
        "status": "SCORED" if validation_blocker is None else "FAILED",
        "blocker": validation_blocker,
        "dataset": dataset,
        "role": UNIFIED_DATASETS[dataset]["role"],
        "verifier": verifier,
        "model_revision": model_revision,
        "protocol_version": protocol_version,
        "rows_requested": len(input_frame),
        "rows": len(scored),
        "coverage": len(scored) / len(input_frame),
        "parse_ok_rate": parse_ok_rate,
        "score_complete_rate": float(scored.loc[valid, "score"].notna().mean()),
        "strict_batch1": True,
        "batch_size": 1,
        "warmup_forwards": 5,
        "warmup_done": True,
        "model_load_ms": float(model_load_ms),
        "restart_count": int(restart_count),
        "reused_from_cascade_v1": False,
        "input_path": str(input_path),
        "input_sha256": input_fingerprint,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "parquet_path": str(parquet_path),
        "parquet_sha256": sha256_file(parquet_path),
        "latency_mean_ms": float(latency.mean()),
        "latency_p50_ms": float(latency.median()),
        "latency_p95_ms": float(latency.quantile(0.95)),
        "latency_components_available": components,
        "component_latency_note": (
            "Opaque existing encoder adapters expose total pipeline latency only; "
            "unavailable components remain null and total is recorded as unattributed."
        ),
        "repeat_profile": repeat,
        "runtime": gpu_metadata(),
        "fresh_or_sealed_labels_read": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if validation_blocker is not None:
        raise ValueError(f"{dataset}/{verifier}: {validation_blocker}")
    return manifest


def write_state_manifest(
    output_dir: Path,
    *,
    dataset: str,
    verifier: str,
    status: str,
    blocker: str | None = None,
) -> None:
    path = Path(output_dir) / f"{dataset}__{verifier}.manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") == "SCORED":
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "unified_v1_verifier_score_manifest_v1",
                "created_at_utc": utc_now(),
                "status": status,
                "dataset": dataset,
                "role": UNIFIED_DATASETS[dataset]["role"],
                "verifier": verifier,
                "blocker": blocker,
                "rows": 0,
                "coverage": 0.0,
                "parse_ok_rate": None,
                "fresh_or_sealed_labels_read": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def update_scoring_status(output_dir: Path, results_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for verifier in ALL_VERIFIERS:
        manifests = []
        for dataset in UNIFIED_DATASETS:
            path = output_dir / f"{dataset}__{verifier}.manifest.json"
            manifests.append(
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {
                    "dataset": dataset,
                    "status": "PENDING",
                    "rows": 0,
                    "coverage": 0.0,
                    "parse_ok_rate": None,
                    "blocker": None,
                }
            )
        states = [str(item.get("status")) for item in manifests]
        if all(state == "SCORED" for state in states):
            state = "SCORED"
        elif all(state == "SKIPPED" for state in states):
            state = "SKIPPED"
        elif "RUNNING" in states:
            state = "RUNNING"
        elif "SCORED" in states:
            state = "PARTIAL"
        elif "FAILED" in states:
            state = "FAILED"
        else:
            state = "PENDING"
        expected = sum(int(UNIFIED_DATASETS[name]["rows"]) for name in UNIFIED_DATASETS)
        scored_rows = sum(int(item.get("rows") or 0) for item in manifests)
        parse_values = [
            (float(item["parse_ok_rate"]), int(item.get("rows") or 0))
            for item in manifests
            if item.get("parse_ok_rate") is not None and int(item.get("rows") or 0) > 0
        ]
        parse_rate = (
            sum(rate * count for rate, count in parse_values) / sum(count for _, count in parse_values)
            if parse_values
            else None
        )
        blockers = sorted({str(item["blocker"]) for item in manifests if item.get("blocker")})
        rows.append(
            {
                "verifier": verifier,
                "state": state,
                "rows": scored_rows,
                "expected_rows": expected,
                "coverage": scored_rows / expected,
                "parse_ok_rate": parse_rate,
                "datasets": {item["dataset"]: item.get("status") for item in manifests},
                "blocker": "; ".join(blockers) if blockers else None,
            }
        )
    payload = {"updated_at_utc": utc_now(), "verifiers": rows}
    (results_dir / "SCORING_STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Unified-v1 Scoring Status",
        "",
        f"Updated: `{payload['updated_at_utc']}`",
        "",
        "TofuEval is `BURNED_DIAGNOSTIC` in every artifact and is excluded from clean headline claims.",
        "",
        "| verifier | state | rows | coverage | parse rate | ragtruth / frank / tofueval | blocker |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        parse = "n/a" if row["parse_ok_rate"] is None else f"{row['parse_ok_rate']:.4f}"
        datasets = " / ".join(row["datasets"][name] for name in UNIFIED_DATASETS)
        lines.append(
            f"| {row['verifier']} | {row['state']} | {row['rows']}/{row['expected_rows']} "
            f"| {row['coverage']:.4f} | {parse} | {datasets} | {row['blocker'] or ''} |"
        )
    (results_dir / "SCORING_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
