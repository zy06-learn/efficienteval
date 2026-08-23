from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pysbd


REPLAY_COLUMNS = {
    "episode_id",
    "doc_group_key",
    "summary_id",
    "sentence_id",
    "generator_id",
    "split",
    "is_official_test",
    "label_supported",
    "router_confidence",
    "base_accept",
}


def _sentence_offsets(response: str) -> list[tuple[int, int]]:
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    offsets = []
    for span in segmenter.segment(response):
        raw = response[int(span.start) : int(span.end)]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = int(span.start) + leading
        end = int(span.end) - trailing
        if start < end:
            offsets.append((start, end))
    return offsets


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", "", "none", "null"}:
            return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_ragtruth_spans(raw: Any, *, response: str) -> list[dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        payload: Any = []
    elif isinstance(raw, str):
        payload = json.loads(raw) if raw.strip() else []
    else:
        payload = raw
    if not isinstance(payload, list):
        raise ValueError("RAGTruth hallucination spans must be a list")

    result = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"RAGTruth span {index} must be an object")
        if _as_bool(item.get("implicit_true", False)):
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"RAGTruth span {index} has invalid offsets") from error
        if not 0 <= start < end <= len(response):
            raise ValueError(f"RAGTruth span {index} lies outside the response")
        text = str(item.get("text", ""))
        if text and response[start:end] != text:
            raise ValueError(
                f"RAGTruth span {index} text does not match response offsets"
            )
        result.append(
            {
                "start": start,
                "end": end,
                "text": response[start:end],
                "label_type": str(item.get("label_type", "unknown")),
                "implicit_true": False,
            }
        )
    return sorted(result, key=lambda item: (item["start"], item["end"]))


def segment_ragtruth_response(
    *,
    response_id: str,
    doc_group_key: str,
    source_document: str,
    response: str,
    generator_id: str,
    spans: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    response = str(response)
    checked_spans = parse_ragtruth_spans(list(spans), response=response)
    rows = []
    summary_id = f"ragtruth:{response_id}"
    for index, (start, end) in enumerate(_sentence_offsets(response)):
        overlapping = [
            item
            for item in checked_spans
            if int(item["start"]) < end and int(item["end"]) > start
        ]
        sentence_id = f"{summary_id}:s{index}"
        rows.append(
            {
                "episode_id": sentence_id,
                "doc_group_key": str(doc_group_key),
                "summary_id": summary_id,
                "sentence_id": sentence_id,
                "response_id": str(response_id),
                "generator_id": str(generator_id),
                "dataset": "RAGTruth-Summary",
                "source_document": str(source_document),
                "candidate_sentence": response[start:end],
                "sentence_start": start,
                "sentence_end": end,
                "label_supported": int(not overlapping),
                "unsupported_spans_json": json.dumps(
                    overlapping, ensure_ascii=True, sort_keys=True
                ),
                "split": "fresh_crc_calibration",
                "is_official_test": False,
            }
        )
    if not rows:
        raise ValueError(f"RAGTruth response {response_id} has no sentences")
    return rows


def segment_unlabeled_response(
    *,
    response_id: str,
    doc_group_key: str,
    source_document: str,
    response: str,
    generator_id: str,
) -> list[dict[str, Any]]:
    response = str(response)
    summary_id = f"ragtruth:{response_id}"
    rows = []
    for index, (start, end) in enumerate(_sentence_offsets(response)):
        sentence_id = f"{summary_id}:s{index}"
        rows.append(
            {
                "episode_id": sentence_id,
                "doc_group_key": str(doc_group_key),
                "summary_id": summary_id,
                "sentence_id": sentence_id,
                "response_id": str(response_id),
                "generator_id": str(generator_id),
                "dataset": "RAGTruth-Summary",
                "source_document": str(source_document),
                "candidate_sentence": response[start:end],
                "sentence_start": start,
                "sentence_end": end,
                "split": "fresh_crc_calibration",
                "is_official_test": False,
            }
        )
    if not rows:
        raise ValueError(f"RAGTruth response {response_id} has no sentences")
    return rows


def validate_global_gamma_replay(frame: pd.DataFrame) -> dict[str, int]:
    missing = sorted(REPLAY_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"global-gamma replay missing columns: {missing}")
    if frame.empty:
        raise ValueError("global-gamma replay is empty")
    if not frame["split"].astype(str).eq("fresh_crc_calibration").all():
        raise ValueError("global-gamma replay contains a sealed or non-calibration split")
    if frame["is_official_test"].fillna(False).astype(bool).any():
        raise ValueError("global-gamma replay contains sealed official-test rows")
    if frame["sentence_id"].astype(str).duplicated().any():
        raise ValueError("global-gamma replay contains duplicate sentence ids")
    labels = pd.to_numeric(frame["label_supported"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("global-gamma labels must be binary")
    confidence = pd.to_numeric(frame["router_confidence"], errors="raise").astype(float)
    if not np.isfinite(confidence).all() or not confidence.between(0.0, 1.0).all():
        raise ValueError("router confidence must be finite and in [0, 1]")
    frame["base_accept"].map(_as_bool)

    summary_mapping = frame.groupby("summary_id").agg(
        documents=("doc_group_key", "nunique"),
        generators=("generator_id", "nunique"),
    )
    if not summary_mapping["documents"].eq(1).all():
        raise ValueError("one summary maps to multiple source documents")
    if not summary_mapping["generators"].eq(1).all():
        raise ValueError("one summary maps to multiple generators")
    return {
        "sentences": int(len(frame)),
        "summaries": int(frame["summary_id"].nunique()),
        "documents": int(frame["doc_group_key"].nunique()),
        "generators": int(frame["generator_id"].nunique()),
    }


def evaluate_global_gamma(frame: pd.DataFrame, *, gamma: float) -> dict[str, Any]:
    counts = validate_global_gamma_replay(frame)
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    confidence = pd.to_numeric(frame["router_confidence"], errors="raise").astype(float)
    labels = pd.to_numeric(frame["label_supported"], errors="raise").astype(int)
    base_accept = frame["base_accept"].map(_as_bool).astype(bool)
    activated = confidence > gamma
    accepted = activated & base_accept
    wrong = accepted & labels.eq(0)

    sentence = frame[["doc_group_key", "summary_id", "generator_id"]].copy()
    sentence["accepted"] = accepted.to_numpy(dtype=bool)
    sentence["wrong"] = wrong.to_numpy(dtype=bool)
    summary = sentence.groupby(
        ["doc_group_key", "summary_id", "generator_id"], as_index=False
    ).agg(any_accept=("accepted", "any"), wrong_accept=("wrong", "any"))
    document = summary.groupby("doc_group_key").agg(
        mean_summary_loss=("wrong_accept", "mean"),
        max_summary_loss=("wrong_accept", "max"),
    )
    document_risk = float(document["mean_summary_loss"].mean())
    document_n = int(len(document))
    generator_risk = {
        str(generator): float(group["wrong_accept"].mean())
        for generator, group in summary.groupby("generator_id", sort=True)
    }
    return {
        **counts,
        "gamma": gamma,
        "confidence_comparison": "strictly_greater_than",
        "sentence_activate_n": int(activated.sum()),
        "sentence_accept_n": int(accepted.sum()),
        "sentence_wrong_accept_n": int(wrong.sum()),
        "sentence_accept_coverage": float(accepted.mean()),
        "sentence_wrong_accept_risk": float(wrong.mean()),
        "summary_accept_coverage": float(summary["any_accept"].mean()),
        "summary_wrong_accept_n": int(summary["wrong_accept"].sum()),
        "summary_wrong_accept_risk": float(summary["wrong_accept"].mean()),
        "document_mean_summary_loss": document_risk,
        "empirical_document_risk": document_risk,
        "document_max_wrong_rate": float(document["max_summary_loss"].mean()),
        "generator_summary_wrong_risk": generator_risk,
        "crc_empirical_plus_one": float(
            (document_n * document_risk + 1.0) / (document_n + 1.0)
        ),
    }


def calibrate_global_gamma_crc(
    frame: pd.DataFrame,
    *,
    alpha: float,
    gamma_floor: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    validate_global_gamma_replay(frame)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    gamma_floor = float(gamma_floor)
    if not 0.0 <= gamma_floor < 1.0:
        raise ValueError("gamma_floor must lie in [0, 1)")
    confidence = pd.to_numeric(frame["router_confidence"], errors="raise").astype(float)
    candidates = sorted(
        {gamma_floor, 1.0}
        | {float(value) for value in confidence if float(value) >= gamma_floor}
    )
    rows = [evaluate_global_gamma(frame, gamma=gamma) for gamma in candidates]
    curve = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)
    for column in ("document_mean_summary_loss", "sentence_accept_coverage"):
        values = curve[column].to_numpy(dtype=np.float64)
        if (np.diff(values) > 1e-12).any():
            raise AssertionError(f"global-gamma curve is not monotone: {column}")

    feasible = curve.loc[curve["crc_empirical_plus_one"] <= float(alpha)]
    if feasible.empty or int(feasible.iloc[0]["sentence_accept_n"]) == 0:
        selected = curve.loc[curve["gamma"].eq(1.0)].iloc[0].to_dict()
        selected.update(
            {
                "status": "FALLBACK_NO_ACCEPT",
                "alpha": float(alpha),
                "gamma_floor": gamma_floor,
                "crc_expected_risk_bound": 0.0,
                "bound_basis": "structural no-accept loss at gamma_max",
            }
        )
    else:
        selected = feasible.iloc[0].to_dict()
        selected.update(
            {
                "status": "CALIBRATED",
                "alpha": float(alpha),
                "gamma_floor": gamma_floor,
                "empirical_document_risk": float(
                    selected["document_mean_summary_loss"]
                ),
                "crc_expected_risk_bound": float(
                    selected["crc_empirical_plus_one"]
                ),
                "bound_basis": "Conformal Risk Control Theorem 1 expected risk",
            }
        )
    selected["method"] = "GLOBAL_GAMMA_EXPECTED_CRC"
    selected["loss_upper_bound"] = 1.0
    selected["exchangeability_unit"] = "source_document"
    return selected, curve
