"""Extended-pool scorer adapters (doc 33 s1.2): LettuceDetect-v2,
Granite-Guardian-3.2-3b-a800m, AttrScore-Flan-T5-Large. All run under the main
venv (transformers 5.x; recorded in protocol). Native outputs preserved per
doc 33 s2; scores in [0,1] with higher = supported for harness compatibility."""
from __future__ import annotations

from pathlib import Path
from typing import Any

LETTUCE_MODEL_ID = "KRLabsOrg/lettucedect-v2-mmbert-base"
LETTUCE_TAXONOMY_ID = "KRLabsOrg/lettucedect-v2-taxonomy-head"
LETTUCE_PASSAGE_WINDOW_TOKENS = 512
LETTUCE_PASSAGE_OVERLAP_TOKENS = 64
LETTUCE_DETECTOR_MAX_LENGTH = 4096
LETTUCE_TAXONOMY_MAX_LENGTH = 1024
GRANITE_MODEL_ID = "ibm-granite/granite-guardian-3.2-3b-a800m"
ATTRSCORE_MODEL_ID = "osunlp/attrscore-flan-t5-large"

ATTR_PROMPT = (
    "Verify whether a given reference can support the claim. Options: Attributable, "
    "Extrapolatory or Contradictory. Attributable means the reference fully supports the "
    "claim, Extrapolatory means the reference lacks sufficient information to validate the "
    "claim, and Contradictory means the claim contradicts the information presented in the "
    "reference.\nClaim: {claim}\nReference: {reference}"
)
ATTR_LABELS = ("Attributable", "Extrapolatory", "Contradictory")


def resolve_cached_snapshot(repo_id: str, *, resolver=None) -> tuple[str, str]:
    """Resolve an already-cached HF snapshot without allowing network access."""
    if resolver is None:
        from huggingface_hub import snapshot_download

        resolver = snapshot_download
    path = str(resolver(repo_id, local_files_only=True))
    return path, Path(path).name


def split_tokenizer_windows(
    text: str,
    tokenizer: Any,
    *,
    window_tokens: int = LETTUCE_PASSAGE_WINDOW_TOKENS,
    overlap_tokens: int = LETTUCE_PASSAGE_OVERLAP_TOKENS,
) -> tuple[list[str], int]:
    """Split source text with the detector tokenizer before official passage grouping."""
    if window_tokens <= 0:
        raise ValueError("window_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= window_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < window")
    token_ids = tokenizer.encode(str(text), add_special_tokens=False, verbose=False)
    if not token_ids:
        return [""], 0
    passages = []
    start = 0
    while start < len(token_ids):
        end = min(start + window_tokens, len(token_ids))
        while True:
            passage = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            decoded_count = len(tokenizer.encode(passage, add_special_tokens=False))
            if decoded_count <= window_tokens:
                break
            # Tokenizers can add one or two tokens when a decoded subwindow is
            # re-encoded without its original left/right context. Contract the
            # right boundary until the independently scored passage fits.
            end -= max(1, decoded_count - window_tokens)
            if end <= start:
                raise ValueError("unable to construct a non-empty detector-tokenizer window")
        passages.append(passage)
        if end == len(token_ids):
            break
        if end - start <= overlap_tokens:
            raise ValueError("source passage is too short to preserve the requested overlap")
        start = end - overlap_tokens
    return passages, len(token_ids)


class LettuceV2Scorer:
    """LettuceDetect-v2 mmBERT-base span detector via the official
    `lettucedetect` package. score = 1 - max(span confidence); no spans => 1.0."""

    label_space = ("hallucinated", "clean")

    def __init__(self, *, device: str = "cuda") -> None:
        from lettucedetect.models.inference import HallucinationDetector

        model_path, model_revision = resolve_cached_snapshot(LETTUCE_MODEL_ID)
        taxonomy_path, taxonomy_revision = resolve_cached_snapshot(LETTUCE_TAXONOMY_ID)
        self.detector = HallucinationDetector(
            method="transformer",
            model_path=model_path,
            taxonomy_head=taxonomy_path,
            device=device,
            max_length=LETTUCE_DETECTOR_MAX_LENGTH,
        )
        self.model_id = (
            f"{LETTUCE_MODEL_ID}@{model_revision}+"
            f"{LETTUCE_TAXONOMY_ID}@{taxonomy_revision}"
        )
        self.device = device

    def _passages_and_groups(
        self, doc: str, claim: str
    ) -> tuple[list[str], int, list[list[str]]]:
        passages, source_token_count = split_tokenizer_windows(
            doc,
            self.detector.detector.tokenizer,
            window_tokens=LETTUCE_PASSAGE_WINDOW_TOKENS,
            overlap_tokens=LETTUCE_PASSAGE_OVERLAP_TOKENS,
        )
        implementation = self.detector.detector
        groups = implementation._group_passages_into_chunks(passages, None, str(claim))

        # The official grouping code does not split an individual passage. Fail
        # closed if an unexpectedly long answer leaves too little room for even
        # one pre-windowed passage, instead of allowing tokenizer truncation.
        from lettucedetect.detectors.prompt_utils import PromptUtils

        for group in groups:
            prompt = PromptUtils.format_context(group, None, implementation.lang)
            prompt_tokens = len(
                implementation.tokenizer.encode(prompt, add_special_tokens=False)
            )
            answer_tokens = len(
                implementation.tokenizer.encode(str(claim), add_special_tokens=False)
            )
            if prompt_tokens + answer_tokens + 3 > implementation.max_length:
                raise ValueError(
                    "Lettuce detector input would exceed max_length after source passage windowing"
                )
        return passages, source_token_count, groups

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        results = []
        for doc, claim in zip(docs, claims):
            passages, source_token_count, groups = self._passages_and_groups(
                str(doc), str(claim)
            )
            spans = self.detector.predict(
                context=passages, question=None, answer=str(claim), output_format="spans"
            ) or []
            confidences = [float(s.get("confidence", 0.0)) for s in spans]
            top = max(confidences) if confidences else 0.0
            results.append(
                {
                    "score": float(1.0 - top),
                    "aux": {
                        "native_output_type": "structured_units",
                        "native_label_space": list(self.label_space),
                        "native_label": "hallucinated" if spans else "clean",
                        "label_source": "derived_from_spans(any span => hallucinated)",
                        "n_spans": len(spans),
                        "unit_outputs": [dict(span) for span in spans],
                        "aggregation_rule": "one_minus_max_span_confidence",
                        "source_passage_count": len(passages),
                        "source_model_token_count": source_token_count,
                        "passage_window_tokens": LETTUCE_PASSAGE_WINDOW_TOKENS,
                        "passage_overlap_tokens": LETTUCE_PASSAGE_OVERLAP_TOKENS,
                        "detector_max_length": LETTUCE_DETECTOR_MAX_LENGTH,
                        "detector_chunk_count": len(groups),
                        "num_model_calls": len(groups),
                        "source_chunking_rule": (
                            "detector_tokenizer_windows_512_overlap64_no_special_tokens"
                        ),
                        "detector_grouping_rule": (
                            "official_greedy_passage_grouping_reserve_answer_plus_3_special_tokens"
                        ),
                        "cross_chunk_aggregation_rule": (
                            "official_max_hallucination_probability_per_answer_token"
                        ),
                        "taxonomy_max_length": LETTUCE_TAXONOMY_MAX_LENGTH,
                        "taxonomy_context_rule": (
                            "answer_plus_joined_source_passages_truncated_by_official_typing_head"
                        ),
                    },
                }
            )
        return results


class GraniteGuardianScorer:
    """Granite-Guardian 3.2 3b-a800m groundedness check per official model card:
    messages [{role: context}, {role: assistant}], guardian_config
    {'risk_name': 'groundedness'}; Yes = risk (ungrounded). support = P(No)."""

    label_space = ("ungrounded_yes", "grounded_no")

    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(GRANITE_MODEL_ID)
        self.model = (
            AutoModelForCausalLM.from_pretrained(GRANITE_MODEL_ID, torch_dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        self.device = device
        self.model_id = GRANITE_MODEL_ID
        self._torch = torch
        self._yes_ids = {self.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")}
        self._no_ids = {self.tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")}

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        results = []
        for doc, claim in zip(docs, claims):
            messages = [
                {"role": "context", "content": str(doc)},
                {"role": "assistant", "content": str(claim)},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                guardian_config={"risk_name": "groundedness"},
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
            with self._torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=20,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            first = self._torch.softmax(output.scores[0][0], dim=-1)
            p_yes = float(sum(first[i] for i in self._yes_ids))
            p_no = float(sum(first[i] for i in self._no_ids))
            z = max(p_yes + p_no, 1e-9)
            text = self.tokenizer.decode(
                output.sequences[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            label = "ungrounded_yes" if p_yes >= p_no else "grounded_no"
            results.append(
                {
                    "score": float(p_no / z),
                    "aux": {
                        "native_output_type": "categorical",
                        "native_label_space": list(self.label_space),
                        "native_probs": [round(p_yes / z, 6), round(p_no / z, 6)],
                        "native_probs_unnormalized": [round(p_yes, 6), round(p_no, 6)],
                        "native_label": label,
                        "label_source": "native_argmax(first-token yes/no mass)",
                        "generated_text": text[:120],
                        "aggregation_rule": None,
                    },
                }
            )
        return results


class AttrScoreFT5Scorer:
    """AttrScore-Flan-T5-Large: 3-way Attributable/Extrapolatory/Contradictory.
    Source chunked to fit the 512-token window; per-chunk 3-way probs from
    first-token logits; sentence score = max chunk P(Attributable)."""

    label_space = ATTR_LABELS
    chunk_tokens = 380  # leave room for prompt + claim within 512

    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ATTRSCORE_MODEL_ID)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(ATTRSCORE_MODEL_ID).to(device).eval()
        )
        self.device = device
        self.model_id = ATTRSCORE_MODEL_ID
        self._torch = torch
        self._label_first_ids = [
            self.tokenizer.encode(label, add_special_tokens=False)[0] for label in ATTR_LABELS
        ]

    def _chunks(self, text: str) -> list[str]:
        ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if not ids:
            return [""]
        return [
            self.tokenizer.decode(ids[i : i + self.chunk_tokens])
            for i in range(0, len(ids), self.chunk_tokens)
        ]

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        results = []
        for doc, claim in zip(docs, claims):
            chunks = self._chunks(doc)
            prompts = [ATTR_PROMPT.format(claim=str(claim), reference=chunk) for chunk in chunks]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            decoder_input_ids = self._torch.zeros(
                (inputs["input_ids"].size(0), 1), dtype=self._torch.long, device=self.device
            )
            with self._torch.no_grad():
                logits = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids,
                ).logits.squeeze(1)
            label_logits = logits[:, self._torch.tensor(self._label_first_ids, device=logits.device)]
            probs = self._torch.softmax(label_logits, dim=-1).cpu().tolist()
            attributable = [row[0] for row in probs]
            best = int(max(range(len(attributable)), key=lambda i: attributable[i]))
            results.append(
                {
                    "score": float(attributable[best]),
                    "aux": {
                        "native_output_type": "categorical",
                        "native_label_space": list(ATTR_LABELS),
                        "n_chunks": len(chunks),
                        "best_chunk_index": best,
                        "chunk_probs_3way": [[round(float(v), 6) for v in row] for row in probs],
                        "chunk_logits_3way": [
                            [round(float(v), 6) for v in row] for row in label_logits.cpu().tolist()
                        ],
                        "native_label": ATTR_LABELS[
                            int(max(range(3), key=lambda i: probs[best][i]))
                        ],
                        "label_source": "native_argmax(first-token 3-way, best chunk)",
                        "aggregation_rule": "max_attributable_prob_over_chunks",
                        "repair_cue": {
                            "Contradictory": "replace", "Extrapolatory": "delete", "Attributable": "keep",
                        },
                    },
                }
            )
        return results
