"""MiniCheck scorers aligned with the official EMNLP 2024 inference code.

Both variants preserve newlines, split with NLTK, chunk the source (Flan-T5:
500 words; DeBERTa: 400 tokens), join each chunk and claim with ``eos_token``,
and max-aggregate per-chunk support. The official Flan-T5 path additionally
prefixes every joined input with ``"predict: "`` before tokenization. Flan-T5
reads token logits [3, 209] from a zero decoder start; DeBERTa uses its native
two-class softmax.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize

FT5_MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
FT5_REVISION = "96eafd01cee2d16cf81aaa2fb226b14f422a37b3"
FT5_MAX_LEN = 2048
FT5_CHUNK_WORDS = 500

DBTA_MODEL_ID = "lytang/MiniCheck-DeBERTa-v3-Large"
DBTA_REVISION = "2f2d01a54fa022a7ffadb76260e1ea8bc88c82bb"
DBTA_MAX_LEN = 2048
DBTA_CHUNK_TOKENS = 400


def _sent_tokenize_with_newlines(text: str) -> list[str]:
    blocks = str(text).split("\n")
    tokenized: list[str] = []
    for block in blocks:
        tokenized.extend(sent_tokenize(block))
        tokenized.append("\n")
    return tokenized[:-1]


class _MiniCheckBase:
    input_prefix = ""

    def _doc_chunks(self, doc: str) -> list[str]:
        sentences = _sent_tokenize_with_newlines(doc) or [""]
        chunks: list[str] = []
        current: list[str] = []
        count = 0
        for sentence in sentences:
            size = self._sentence_size(sentence)
            if count + size > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = [sentence]
                count = size
            else:
                current.append(sentence)
                count += size
        if current:
            chunks.append(" ".join(current))
        chunks = [chunk.replace(" \n ", "\n").strip() for chunk in chunks]
        return [chunk for chunk in chunks if chunk] or [""]

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must have the same length")
        results = []
        for doc, claim in zip(docs, claims):
            chunks = self._doc_chunks(str(doc))
            texts = [
                self.input_prefix + self.tokenizer.eos_token.join([chunk, str(claim)])
                for chunk in chunks
            ]
            probs, logits = self._support_probs_with_logits(texts)
            best = int(max(range(len(probs)), key=lambda i: probs[i]))
            results.append(
                {
                    "score": float(probs[best]),
                    "aux": {
                        "n_chunks": len(chunks),
                        "best_chunk_index": best,
                        "support_prob_per_chunk": [round(float(p), 6) for p in probs],
                        "native_output_type": "categorical",
                        "native_label_space": list(self.label_space),
                        "chunk_logits": [[round(float(v), 6) for v in row] for row in logits],
                        "chunk_argmax_labels": [
                            self.label_space[int(p >= 0.5)] for p in probs
                        ],
                        "native_label": self.label_space[int(probs[best] >= 0.5)],
                        "label_source": "native_argmax",
                        "aggregation_rule": "max_support_prob_over_chunks",
                    },
                }
            )
        return results

    def _support_probs(self, texts: list[str]) -> list[float]:
        return self._support_probs_with_logits(texts)[0]


class MiniCheckFT5Scorer(_MiniCheckBase):
    chunk_size = FT5_CHUNK_WORDS
    label_space = ("unsupported", "supported")  # token ids [3, 209]
    input_prefix = "predict: "

    def __init__(self, *, device: str = "cuda") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(FT5_MODEL_ID, revision=FT5_REVISION)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(FT5_MODEL_ID, revision=FT5_REVISION)
            .to(device)
            .eval()
        )
        self.device = device
        self.model_id = f"{FT5_MODEL_ID}@{FT5_REVISION}"

    def _sentence_size(self, sentence: str) -> int:
        return len(sentence.split())

    def _support_probs_with_logits(self, texts: list[str]) -> tuple[list[float], list[list[float]]]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=FT5_MAX_LEN
        ).to(self.device)
        decoder_input_ids = torch.zeros(
            (inputs["input_ids"].size(0), 1), dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
        logits = outputs.logits.squeeze(1)
        # official: token id 3 = no support, 209 = support
        label_logits = logits[:, torch.tensor([3, 209], device=logits.device)]
        probs = F.softmax(label_logits, dim=-1)[:, 1].detach().cpu().tolist()
        return probs, label_logits.detach().cpu().tolist()


class MiniCheckDebertaScorer(_MiniCheckBase):
    chunk_size = DBTA_CHUNK_TOKENS
    label_space = ("unsupported", "supported")  # 2-class head, index 1 = supported

    def __init__(self, *, device: str = "cuda") -> None:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        config = AutoConfig.from_pretrained(
            DBTA_MODEL_ID,
            revision=DBTA_REVISION,
            num_labels=2,
            finetuning_task="text-classification",
        )
        config.problem_type = "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(
            DBTA_MODEL_ID, revision=DBTA_REVISION, use_fast=True
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                DBTA_MODEL_ID, revision=DBTA_REVISION, config=config
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.model_id = f"{DBTA_MODEL_ID}@{DBTA_REVISION}"

    def _sentence_size(self, sentence: str) -> int:
        return len(
            self.tokenizer(
                sentence, padding=False, add_special_tokens=False,
                max_length=DBTA_MAX_LEN, truncation=True,
            )["input_ids"]
        )

    def _support_probs_with_logits(self, texts: list[str]) -> tuple[list[float], list[list[float]]]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=DBTA_MAX_LEN
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[:, 1].detach().cpu().tolist()
        return probs, outputs.logits.detach().cpu().tolist()
