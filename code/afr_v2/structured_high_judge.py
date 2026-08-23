from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Any

from afr_v2.global_gamma_calibration import _sentence_offsets


LABELS = {"SUPPORTED", "UNSUPPORTED"}
ERROR_TYPES = {
    "NONE",
    "UNSUPPORTED_EXTRINSIC",
    "CONTRADICTED_INTRINSIC",
    "MIXED",
}
SPAN_GROUNDING_ERROR = "unsupported_spans must be exact claim substrings"
LABEL_ERROR_TYPE_ERROR = "label and error_type are inconsistent"

SYSTEM_PROMPT = """You are a strict factual-consistency judge for summarization.
Judge the SENTENCE only against the numbered SOURCE sentences. Do not use outside knowledge.
Respond with exactly one JSON object and nothing else, with these fields:
"label": "SUPPORTED" or "UNSUPPORTED". Use "UNSUPPORTED" if ANY part of the sentence is not fully supported by the source or conflicts with it.
"support_probability": a number in [0,1], your probability that the sentence is fully supported.
"evidence_ids": a list of at most 3 integers, the indices of the source sentences most relevant to your decision (supporting or refuting). Use [] only if no source sentence is relevant.
"unsupported_spans": a list of at most 3 strings, each an EXACT substring of the sentence that is unsupported or contradicted. Use [] if the sentence is supported.
"error_type": "NONE" if supported; "UNSUPPORTED_EXTRINSIC" if the sentence adds information absent from the source; "CONTRADICTED_INTRINSIC" if it conflicts with the source; "MIXED" if both.
Do not explain. Do not rewrite or repair the sentence."""


def structured_response_format(n_source_sentences: int) -> dict[str, Any]:
    if n_source_sentences < 1:
        raise ValueError("n_source_sentences must be positive")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "factual_consistency_judgment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": sorted(LABELS),
                    },
                    "support_probability": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": n_source_sentences - 1,
                        },
                        "maxItems": 3,
                    },
                    "unsupported_spans": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                    },
                    "error_type": {
                        "type": "string",
                        "enum": sorted(ERROR_TYPES),
                    },
                },
                "required": [
                    "label",
                    "support_probability",
                    "evidence_ids",
                    "unsupported_spans",
                    "error_type",
                ],
                "additionalProperties": False,
            },
        },
    }


def numbered_source(source: str) -> tuple[str, int]:
    offsets = _sentence_offsets(source)
    lines = [
        f"[{index}] {source[start:end].strip()}"
        for index, (start, end) in enumerate(offsets)
    ]
    return "\n".join(lines), len(offsets)


def _ground_claim_span(claim: str, text: str) -> tuple[str, int, int] | None:
    start = claim.find(text) if text else -1
    if start >= 0:
        return text, start, start + len(text)

    def project(value: str) -> tuple[str, list[int]]:
        characters: list[str] = []
        offsets: list[int] = []
        for index, character in enumerate(value):
            for normalized in unicodedata.normalize("NFKC", character).casefold():
                if normalized.isalnum():
                    characters.append(normalized)
                    offsets.append(index)
        return "".join(characters), offsets

    projected_claim, claim_offsets = project(claim)
    projected_text, _ = project(text)
    if not projected_text:
        return None
    projected_start = projected_claim.find(projected_text)
    if projected_start < 0:
        return None
    if projected_claim.find(projected_text, projected_start + 1) >= 0:
        return None
    start = claim_offsets[projected_start]
    end = claim_offsets[projected_start + len(projected_text) - 1] + 1
    return claim[start:end], start, end


def validate_payload(
    payload: dict[str, Any],
    *,
    claim: str,
    n_source_sentences: int,
    require_grounded_spans: bool = True,
    require_label_error_type_consistency: bool = True,
) -> dict[str, Any]:
    label = str(payload.get("label", ""))
    if label not in LABELS:
        raise ValueError(f"bad label: {label!r}")
    probability = float(payload.get("support_probability"))
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"bad support_probability: {probability}")

    raw_evidence = payload.get("evidence_ids", [])
    if not isinstance(raw_evidence, list):
        raise ValueError("bad evidence_ids")
    evidence: list[int] = []
    evidence_dropped = 0
    for value in raw_evidence:
        try:
            evidence.append(int(value))
        except (TypeError, ValueError):
            evidence_dropped += 1
    evidence_clipped = len(evidence) > 3
    evidence = evidence[:3]
    evidence_in_range = all(0 <= value < n_source_sentences for value in evidence)
    if evidence_dropped or evidence_clipped or not evidence_in_range:
        raise ValueError("invalid evidence_ids payload")

    raw_spans = payload.get("unsupported_spans", [])
    if not isinstance(raw_spans, list):
        raise ValueError("bad unsupported_spans")
    spans_clipped = len(raw_spans) > 3
    spans = []
    for item in raw_spans[:3]:
        text = str(item) if not isinstance(item, dict) else str(item.get("text", ""))
        grounded = _ground_claim_span(claim, text)
        spans.append(
            {
                "text": grounded[0] if grounded else text,
                "start": grounded[1] if grounded else None,
                "end": grounded[2] if grounded else None,
                "grounded_in_claim": grounded is not None,
            }
        )
    spans_grounded = all(span["grounded_in_claim"] for span in spans)
    if spans_clipped or (require_grounded_spans and not spans_grounded):
        raise ValueError(SPAN_GROUNDING_ERROR)

    error_type = str(payload.get("error_type", ""))
    if error_type not in ERROR_TYPES:
        raise ValueError(f"bad error_type: {error_type!r}")
    label_consistent = (label == "SUPPORTED") == (error_type == "NONE")
    if require_label_error_type_consistency and not label_consistent:
        raise ValueError(LABEL_ERROR_TYPE_ERROR)
    if label == "SUPPORTED" and spans:
        raise ValueError("supported payload must not contain unsupported spans")
    if label == "UNSUPPORTED" and not spans:
        raise ValueError("unsupported payload must identify at least one exact span")
    return {
        "label": label,
        "support_probability": probability,
        "evidence_ids": evidence,
        "evidence_ids_in_range": evidence_in_range,
        "evidence_clipped": evidence_clipped,
        "evidence_dropped": evidence_dropped,
        "unsupported_spans": spans,
        "spans_clipped": spans_clipped,
        "spans_grounded": spans_grounded,
        "error_type": error_type,
        "label_error_type_consistent": label_consistent,
    }


def validate_payload_for_score(
    payload: dict[str, Any], *, claim: str, n_source_sentences: int
) -> dict[str, Any]:
    """Keep a valid primary judgment when only auxiliary span grounding fails."""
    try:
        validated = validate_payload(
            payload,
            claim=claim,
            n_source_sentences=n_source_sentences,
        )
        validated["span_validation_fallback"] = False
        validated["label_error_type_validation_fallback"] = False
        return validated
    except ValueError as exc:
        if str(exc) != SPAN_GROUNDING_ERROR:
            raise

    try:
        validated = validate_payload(
            payload,
            claim=claim,
            n_source_sentences=n_source_sentences,
            require_grounded_spans=False,
        )
        label_error_type_fallback = False
    except ValueError as exc:
        exact_empty_span_signature = (
            str(payload.get("label", "")) == "UNSUPPORTED"
            and payload.get("support_probability") == 0.0
            and payload.get("evidence_ids") == []
            and payload.get("unsupported_spans") == [""]
            and str(payload.get("error_type", "")) == "NONE"
        )
        if str(exc) != LABEL_ERROR_TYPE_ERROR or not exact_empty_span_signature:
            raise
        validated = validate_payload(
            payload,
            claim=claim,
            n_source_sentences=n_source_sentences,
            require_grounded_spans=False,
            require_label_error_type_consistency=False,
        )
        label_error_type_fallback = True
    validated["span_validation_fallback"] = True
    validated["label_error_type_validation_fallback"] = label_error_type_fallback
    return validated


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not Path(path).exists():
        return rows
    repair_truncated_tail = False
    with Path(path).open(encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if line_number == len(lines) and not line.endswith("\n"):
                    repair_truncated_tail = True
                    break
                raise ValueError(f"invalid high cache JSON at {path}:{line_number}") from exc
            forbidden = [name for name in row if name.lower().startswith("label") or "gold" in name.lower()]
            if forbidden:
                raise ValueError(f"high cache contains supervision: {forbidden}")
            rows[str(row["episode_id"])] = row
    if repair_truncated_tail:
        temporary = Path(path).with_suffix(Path(path).suffix + ".tail-repair.tmp")
        with temporary.open("w", encoding="utf-8") as sink:
            for row in rows.values():
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    return rows
