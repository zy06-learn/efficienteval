from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from afr_v2.cascade_primary_assets import PRIMARY_SPLIT, SCORING_COLUMNS
from afr_v2.research_freeze import sha256_file


PROTOCOLS = {
    "hhem": "hhem_2_1_open_fullsource_tokenwindow512_overlap64_max_batch1_v2",
    "alignscore": "alignscore_nli_sp_fullsource_persistent_batch1_rawscore_v1",
    "factcg": "factcg_deberta_v3_large_fullsource_chunked_promptbatch1_nativeaux_v2",
    "minicheck_dbta": "minicheck_deberta_v3_large_fullsource_chunked_nativeaux_batch1_v1",
    "minicheck_ft5": "minicheck_flan_t5_large_fullsource_chunked_predictprefix_nativeaux_batch1_v2",
}


def build_hhem_token_windows(
    tokenizer: Any,
    *,
    prompt: str,
    document: str,
    claim: str,
    max_length: int = 512,
    overlap_tokens: int = 64,
    safety_margin: int = 8,
) -> dict[str, Any]:
    """Cover the full source with prompt-safe token windows and no gaps."""

    empty_prompt = prompt.format(text1="", text2=str(claim))
    overhead = len(
        tokenizer(empty_prompt, add_special_tokens=True, truncation=False)[
            "input_ids"
        ]
    )
    budget = int(max_length) - int(overhead) - int(safety_margin)
    if budget <= int(overlap_tokens) + 8:
        raise ValueError(
            f"HHEM claim leaves no usable source budget: overhead={overhead}"
        )
    original_tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if original_tokenizer_limit is not None:
        tokenizer.model_max_length = max(int(original_tokenizer_limit), 1_000_000)
    try:
        source_ids = tokenizer(
            str(document), add_special_tokens=False, truncation=False
        )["input_ids"]
    finally:
        if original_tokenizer_limit is not None:
            tokenizer.model_max_length = original_tokenizer_limit
    if source_ids and isinstance(source_ids[0], list):
        source_ids = source_ids[0]
    source_ids = list(source_ids)
    if not source_ids:
        source_ids = []

    chunks: list[str] = []
    prompt_lengths: list[int] = []
    start = 0
    while start < max(len(source_ids), 1):
        if source_ids:
            window_ids = source_ids[start : start + budget]
            if not window_ids:
                break
            chunk = tokenizer.decode(window_ids, skip_special_tokens=True)
        else:
            window_ids = []
            chunk = ""
        full_prompt = prompt.format(text1=chunk, text2=str(claim))
        prompt_length = len(
            tokenizer(full_prompt, add_special_tokens=True, truncation=False)[
                "input_ids"
            ]
        )
        while prompt_length > max_length and len(window_ids) > 1:
            overflow = prompt_length - int(max_length)
            window_ids = window_ids[: max(1, len(window_ids) - overflow - 1)]
            chunk = tokenizer.decode(window_ids, skip_special_tokens=True)
            full_prompt = prompt.format(text1=chunk, text2=str(claim))
            prompt_length = len(
                tokenizer(full_prompt, add_special_tokens=True, truncation=False)[
                    "input_ids"
                ]
            )
        if prompt_length > max_length:
            raise ValueError(
                f"HHEM prompt cannot fit max_length={max_length}: {prompt_length}"
            )
        chunks.append(chunk)
        prompt_lengths.append(int(prompt_length))
        if not source_ids or start + len(window_ids) >= len(source_ids):
            break
        advance = len(window_ids) - int(overlap_tokens)
        if advance <= 0:
            raise ValueError("HHEM chunk advance must be positive")
        start += advance

    return {
        "chunks": chunks,
        "prompt_token_counts": prompt_lengths,
        "source_token_count": len(source_ids),
        "claim_prompt_overhead_tokens": int(overhead),
        "chunk_token_budget": int(budget),
        "chunk_overlap_tokens": int(overlap_tokens),
        "max_length": int(max_length),
    }


class HHEMChunkedScorer:
    """HHEM 2.1 over full-source token windows, one window per forward pass."""

    max_length = 512
    overlap_tokens = 64

    def __init__(self, *, device: str = "cuda") -> None:
        from afr_v2.candidate_verifiers import HHEMScorer

        base = HHEMScorer(device=device)
        self.model = base.model
        self.tokenizer = self.model.tokenzier
        self.prompt = str(self.model.prompt)
        self.model_id = f"{base.model_id}|tokenwindow512_overlap64_max_v2"

    def score_batch(
        self, docs: list[str], claims: list[str]
    ) -> list[dict[str, Any]]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must have the same length")
        results = []
        for document, claim in zip(docs, claims):
            packed = build_hhem_token_windows(
                self.tokenizer,
                prompt=self.prompt,
                document=str(document),
                claim=str(claim),
                max_length=self.max_length,
                overlap_tokens=self.overlap_tokens,
            )
            scores = [
                float(self.model.predict([(chunk, str(claim))])[0])
                for chunk in packed["chunks"]
            ]
            best = int(max(range(len(scores)), key=lambda index: scores[index]))
            results.append(
                {
                    "score": scores[best],
                    "aux": {
                        "native_output_type": "scalar_chunked",
                        "n_chunks": len(scores),
                        "best_chunk_index": best,
                        "support_prob_per_chunk": [
                            round(value, 6) for value in scores
                        ],
                        "aggregation_rule": "max_support_prob_over_token_windows",
                        "chunk_prompt_batch_size": 1,
                        "max_prompt_tokens": max(packed["prompt_token_counts"]),
                        "prompt_token_counts": packed["prompt_token_counts"],
                        "source_token_count": packed["source_token_count"],
                        "claim_prompt_overhead_tokens": packed[
                            "claim_prompt_overhead_tokens"
                        ],
                        "chunk_token_budget": packed["chunk_token_budget"],
                        "chunk_overlap_tokens": packed["chunk_overlap_tokens"],
                        "adapter_prompt_limit": packed["max_length"],
                    },
                }
            )
        return results


def build_primary_scorer(action: str, *, device: str = "cuda") -> Any:
    if action == "hhem":
        return HHEMChunkedScorer(device=device)
    if action == "alignscore":
        from afr_v2.candidate_verifiers import AlignScorePersistentScorer

        return AlignScorePersistentScorer(device=device, batch_size=1)
    if action == "factcg":
        from afr_v2.native_scorers import FactCGNativeScorer

        class _BoundedFactCGNativeScorer(FactCGNativeScorer):
            """Native FactCG with bounded chunk micro-batches for long USB sources."""

            def score_batch(
                self, docs: list[str], claims: list[str]
            ) -> list[dict[str, Any]]:
                if len(docs) != len(claims):
                    raise ValueError("docs and claims must have the same length")
                results = []
                for document, claim in zip(docs, claims):
                    chunks = self._chunks(str(document))
                    prompts = [
                        self._prompt.format(document=chunk, claim=str(claim))
                        for chunk in chunks
                    ]
                    all_logits: list[list[float]] = []
                    support: list[float] = []
                    for prompt in prompts:
                        encoded = self.tokenizer(
                            [prompt],
                            max_length=2048,
                            truncation="only_first",
                            padding="longest",
                            return_tensors="pt",
                        ).to(self.device)
                        with self._torch.inference_mode():
                            logits = self.model(**encoded).logits
                        probs = self._torch.softmax(logits, dim=-1)
                        all_logits.append([float(value) for value in logits[0].cpu().tolist()])
                        support.append(float(probs[0, 1].cpu()))
                    best = int(max(range(len(support)), key=lambda index: support[index]))
                    results.append(
                        {
                            "score": support[best],
                            "aux": {
                                "n_chunks": len(chunks),
                                "best_chunk_index": best,
                                "native_output_type": "categorical",
                                "native_label_space": list(self.label_space),
                                "chunk_logits": [
                                    [round(value, 6) for value in row]
                                    for row in all_logits
                                ],
                                "support_prob_per_chunk": [
                                    round(value, 6) for value in support
                                ],
                                "chunk_argmax_labels": [
                                    self.label_space[int(value >= 0.5)]
                                    for value in support
                                ],
                                "native_label": self.label_space[int(support[best] >= 0.5)],
                                "label_source": "native_argmax",
                                "aggregation_rule": "max_support_prob_over_chunks",
                                "chunk_prompt_batch_size": 1,
                            },
                        }
                    )
                return results

        return _BoundedFactCGNativeScorer(device=device)
    if action == "minicheck_dbta":
        from afr_v2.minicheck_scorers import MiniCheckDebertaScorer

        return MiniCheckDebertaScorer(device=device)
    if action == "minicheck_ft5":
        from afr_v2.minicheck_scorers import MiniCheckFT5Scorer

        return MiniCheckFT5Scorer(device=device)
    raise ValueError(f"unsupported primary action: {action}")


def validate_primary_input_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("split") != PRIMARY_SPLIT:
        raise ValueError("primary scorer manifest split mismatch")
    if manifest.get("calibration_labels_read") is not False:
        raise ValueError("manifest does not prove fresh labels unread")
    if int(manifest.get("official_test_rows_read", -1)) != 0:
        raise ValueError("manifest reports official-test access")
    if int(manifest.get("sealed_rows", -1)) != 0:
        raise ValueError("manifest reports sealed rows")
    if manifest.get("scoring_input_has_gold") is not False:
        raise ValueError("manifest does not prove gold-free scoring input")
    if manifest.get("status") != "READY_FOR_UNLABELED_SCORING":
        raise ValueError("primary assets are not ready for scoring")
    return manifest


def validate_primary_scoring_frame(
    frame: pd.DataFrame, *, manifest: dict[str, Any], input_path: Path
) -> None:
    if list(frame.columns) != SCORING_COLUMNS:
        raise ValueError(
            f"primary scoring columns differ from frozen gold-free schema: {list(frame.columns)}"
        )
    if int(manifest["inputs"]["rows"]) != len(frame):
        raise ValueError("input row count differs from manifest")
    expected_hash = str(manifest["inputs"]["sha256"])
    actual_hash = sha256_file(Path(input_path))
    if actual_hash != expected_hash:
        raise ValueError("input parquet hash differs from manifest")


def select_balanced_smoke(frame: pd.DataFrame, count: int = 20) -> pd.DataFrame:
    """Canary both datasets and their longest documents, one row per source."""

    if count < 4 or count % 2:
        raise ValueError("balanced smoke count must be an even integer >=4")
    per_dataset = count // 2
    selected = []
    for dataset in ("RAGTruth-Summary", "USB-full-source"):
        part = frame.loc[frame["dataset"].eq(dataset)].copy()
        part["_source_length"] = part["source_document"].astype(str).str.len()
        part = (
            part.sort_values(["_source_length", "doc_group_key"], ascending=[False, True])
            .drop_duplicates("doc_group_key")
            .head(per_dataset)
            .drop(columns="_source_length")
        )
        if len(part) != per_dataset:
            raise ValueError(f"not enough distinct {dataset} sources for smoke")
        selected.append(part)
    return pd.concat(selected, ignore_index=True)[list(frame.columns)]


def _json_value(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def expand_native_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose common native fields while retaining the lossless aux_json payload."""

    result = frame.copy()
    aux = result["aux_json"].map(lambda value: json.loads(value or "{}"))
    scalar_fields = [
        "native_output_type",
        "native_label",
        "label_source",
        "aggregation_rule",
        "best_chunk_index",
        "n_chunks",
        "chunk_count",
        "max_encoded_tokens",
        "worker_pid",
        "persistent_worker",
        "max_prompt_tokens",
        "source_token_count",
        "claim_prompt_overhead_tokens",
        "chunk_token_budget",
        "chunk_overlap_tokens",
        "adapter_prompt_limit",
    ]
    json_fields = [
        "native_label_space",
        "native_logits",
        "native_probs",
        "chunk_logits",
        "chunk_argmax_labels",
        "support_prob_per_chunk",
        "prompt_token_counts",
    ]
    for field in scalar_fields:
        result[field] = aux.map(lambda item, name=field: item.get(name))
    for field in json_fields:
        result[f"{field}_json"] = aux.map(
            lambda item, name=field: _json_value(item.get(name))
        )
    result["parse_ok"] = True
    result["parse_error"] = None
    result["total_latency_ms"] = result["latency_ms"].astype(float)
    return result
