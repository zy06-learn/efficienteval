"""Native-output scorer variants (doc 33 s1-s2): identical scoring protocols to
the frozen screening adapters, but additionally capture the models' native
class logits/probabilities and argmax labels. Scores must match the original
runs bit-for-bit (same model revisions, same inputs, same aggregation)."""
from __future__ import annotations

from typing import Any


class FactKBNativeScorer:
    """FactKB (bunsenfeng/FactKB): RoBERTa 2-class head, input pair
    (summary, article), truncation 512, label index 1 = factual.
    Protocol identical to the legacy FactKBScorer."""

    model_id = "bunsenfeng/FactKB"
    label_space = ("nonfactual", "factual")

    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=2)
            .to(device)
            .eval()
        )
        self._torch = torch

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            claims, docs, truncation="longest_first", max_length=512,
            padding=True, return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**encoded).logits
        probs = self._torch.softmax(logits, dim=-1)
        results = []
        for row_logits, row_probs in zip(logits.cpu().tolist(), probs.cpu().tolist()):
            argmax = int(row_probs[1] >= row_probs[0])
            results.append(
                {
                    "score": float(row_probs[1]),
                    "aux": {
                        "native_output_type": "categorical",
                        "native_label_space": list(self.label_space),
                        "native_logits": [round(float(v), 6) for v in row_logits],
                        "native_probs": [round(float(v), 6) for v in row_probs],
                        "native_label": self.label_space[argmax],
                        "label_source": "native_argmax",
                        "aggregation_rule": None,
                    },
                }
            )
        return results


class FactCGNativeScorer:
    """FactCG: identical chunking/prompt/protocol to candidate_verifiers.
    FactCGScorer (same revision, 2048 only_first truncation, per-chunk 2-class
    softmax, sentence score = max over chunks), plus per-chunk logits/argmax."""

    label_space = ("unsupported", "supported")

    def __init__(self, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        from verifier_wrappers.candidate_verifiers import (
            FACTCG_MODEL_ID,
            FACTCG_PROMPT,
            FACTCG_REVISION,
            factcg_chunks,
        )

        config = AutoConfig.from_pretrained(
            FACTCG_MODEL_ID, revision=FACTCG_REVISION,
            num_labels=2, finetuning_task="text-classification",
        )
        config.problem_type = "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(
            FACTCG_MODEL_ID, revision=FACTCG_REVISION, use_fast=True
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                FACTCG_MODEL_ID, revision=FACTCG_REVISION, config=config
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.model_id = f"{FACTCG_MODEL_ID}@{FACTCG_REVISION}"
        self._torch = torch
        self._chunks = factcg_chunks
        self._prompt = FACTCG_PROMPT

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        results = []
        for document, claim in zip(docs, claims):
            chunks = self._chunks(str(document))
            prompts = [self._prompt.format(document=chunk, claim=str(claim)) for chunk in chunks]
            encoded = self.tokenizer(
                prompts, max_length=2048, truncation="only_first",
                padding="longest", return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
            probs = self._torch.softmax(logits, dim=-1)
            support = probs[:, 1].cpu().tolist()
            best = int(max(range(len(support)), key=lambda i: support[i]))
            results.append(
                {
                    "score": float(support[best]),
                    "aux": {
                        "n_chunks": len(chunks),
                        "best_chunk_index": best,
                        "native_output_type": "categorical",
                        "native_label_space": list(self.label_space),
                        "chunk_logits": [[round(float(v), 6) for v in row] for row in logits.cpu().tolist()],
                        "support_prob_per_chunk": [round(float(p), 6) for p in support],
                        "chunk_argmax_labels": [self.label_space[int(p >= 0.5)] for p in support],
                        "native_label": self.label_space[int(support[best] >= 0.5)],
                        "label_source": "native_argmax",
                        "aggregation_rule": "max_support_prob_over_chunks",
                    },
                }
            )
        return results
