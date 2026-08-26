from __future__ import annotations

import gc
import os
import json
import math
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


FACTCG_MODEL_ID = "yaxili96/FactCG-DeBERTa-v3-Large"
FACTCG_REVISION = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
HHEM_MODEL_ID = "vectara/hallucination_evaluation_model"
HHEM_REVISION = "8e4a2e6e96c708cc76c2344f7e4757df2515292c"
DEFAULT_EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "verifiers"
# AlignScore needs its own environment: its pinned dependencies conflict with the ones the
# rest of the pool runs under. Both the interpreter and the checkpoint are therefore
# external to this repository and are named by environment variable.
DEFAULT_ALIGNSCORE_PYTHON = Path(
    os.environ.get("ALIGNSCORE_PYTHON", sys.executable))
DEFAULT_ALIGNSCORE_CHECKPOINT = Path(
    os.environ.get("ALIGNSCORE_CHECKPOINT",
                   Path.home() / "ckpts" / "alignscore" / "AlignScore-large.ckpt"))
DEFAULT_ALIGNSCORE_WORKER = (
    Path(__file__).resolve().parents[1] / "scripts" / "alignscore_persistent_worker.py"
)
# Full-summary requests can legitimately exceed one minute. Keep the full input
# and fail closed after a bounded wait instead of truncating it.
DEFAULT_ALIGNSCORE_REQUEST_TIMEOUT_SECONDS = 600.0

FACTCG_PROMPT = (
    "{document}\n\nChoose your answer: based on the paragraph above can we conclude "
    'that "{claim}"?\n\nOPTIONS:\n- Yes\n- No\nI think the answer is '
)


def factcg_chunks(document: str, max_chunk_words: int = 550) -> list[str]:
    from nltk.tokenize import sent_tokenize, word_tokenize

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sent_tokenize(str(document)):
        sentence_words = len(word_tokenize(sentence))
        if current and current_words + sentence_words > max_chunk_words:
            chunks.append("\n".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append("\n".join(current))
    return chunks or [str(document)]


class FactCGScorer:
    def __init__(self, *, device: str = "cuda", prompt_batch_size: int = 8) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if prompt_batch_size <= 0:
            raise ValueError("prompt_batch_size must be positive")
        config = AutoConfig.from_pretrained(
            FACTCG_MODEL_ID,
            revision=FACTCG_REVISION,
            num_labels=2,
            finetuning_task="text-classification",
        )
        config.problem_type = "single_label_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(
            FACTCG_MODEL_ID,
            revision=FACTCG_REVISION,
            use_fast=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            FACTCG_MODEL_ID,
            revision=FACTCG_REVISION,
            config=config,
        ).to(device).eval()
        self.device = device
        self.prompt_batch_size = int(prompt_batch_size)
        self.model_id = f"{FACTCG_MODEL_ID}@{FACTCG_REVISION}"
        self._torch = torch

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must have the same length")
        prompts: list[str] = []
        owners: list[int] = []
        chunks_by_doc: list[list[str]] = []
        for index, (document, claim) in enumerate(zip(docs, claims)):
            chunks = factcg_chunks(document)
            chunks_by_doc.append(chunks)
            prompts.extend(
                FACTCG_PROMPT.format(document=chunk, claim=claim) for chunk in chunks
            )
            owners.extend([index] * len(chunks))

        scores_by_doc: list[list[float]] = [[] for _ in docs]
        max_tokens_by_doc = [0 for _ in docs]
        for start in range(0, len(prompts), self.prompt_batch_size):
            batch_prompts = prompts[start : start + self.prompt_batch_size]
            batch_owners = owners[start : start + self.prompt_batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                max_length=2048,
                truncation="only_first",
                padding="longest",
                return_tensors="pt",
            ).to(self.device)
            with self._torch.inference_mode():
                batch_scores = self._torch.softmax(
                    self.model(**encoded).logits,
                    dim=-1,
                )[:, 1]
            token_counts = encoded.attention_mask.sum(dim=1).detach().cpu().tolist()
            for owner, score, token_count in zip(
                batch_owners,
                batch_scores.detach().cpu().tolist(),
                token_counts,
            ):
                scores_by_doc[owner].append(float(score))
                max_tokens_by_doc[owner] = max(
                    max_tokens_by_doc[owner],
                    int(token_count),
                )

        outputs = []
        for index, scores in enumerate(scores_by_doc):
            best_index = max(range(len(scores)), key=scores.__getitem__)
            outputs.append(
                {
                    "score": float(scores[best_index]),
                    "aux": {
                        "chunk_count": len(chunks_by_doc[index]),
                        "best_chunk_index": int(best_index),
                        "max_encoded_tokens": int(max_tokens_by_doc[index]),
                    },
                }
            )
        return outputs


class HHEMScorer:
    def __init__(self, *, device: str = "cuda") -> None:
        from transformers import AutoModelForSequenceClassification

        self.model = AutoModelForSequenceClassification.from_pretrained(
            HHEM_MODEL_ID,
            revision=HHEM_REVISION,
            trust_remote_code=True,
        ).to(device).eval()
        self.model_id = f"{HHEM_MODEL_ID}@{HHEM_REVISION}"

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must have the same length")
        scores = self.model.predict(list(zip(docs, claims))).detach().cpu().tolist()
        return [{"score": float(score), "aux": {}} for score in scores]


class AlignScorePersistentScorer:
    model_id = "AlignScore-large (yzha/AlignScore)"

    def __init__(
        self,
        *,
        device: str = "cuda",
        batch_size: int = 1,
        worker_command: Sequence[str] | None = None,
        ready_timeout_seconds: float = 180.0,
        request_timeout_seconds: float = DEFAULT_ALIGNSCORE_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if worker_command is None:
            for path in (
                DEFAULT_ALIGNSCORE_PYTHON,
                DEFAULT_ALIGNSCORE_CHECKPOINT,
                DEFAULT_ALIGNSCORE_WORKER,
            ):
                if not path.exists():
                    raise FileNotFoundError(path)
            worker_command = [
                str(DEFAULT_ALIGNSCORE_PYTHON),
                str(DEFAULT_ALIGNSCORE_WORKER),
                "--ckpt",
                str(DEFAULT_ALIGNSCORE_CHECKPOINT),
                "--device",
                str(device),
                "--batch-size",
                str(batch_size),
            ]
        started = time.perf_counter()
        self.process = subprocess.Popen(
            list(worker_command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.request_count = 0
        self.returncode: int | None = None
        ready = self._read_payload(float(ready_timeout_seconds))
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"AlignScore worker did not become ready: {ready}")
        self.worker_pid = int(ready.get("pid", self.process.pid))
        self.startup_ms = (time.perf_counter() - started) * 1000.0

    def _read_payload(self, timeout_seconds: float) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("AlignScore worker stdout is unavailable")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("AlignScore worker response timed out")
            readable, _, _ = select.select(
                [self.process.stdout.fileno()], [], [], remaining
            )
            if not readable:
                raise TimeoutError("AlignScore worker response timed out")
            line = self.process.stdout.readline()
            if line == "":
                returncode = self.process.poll()
                raise RuntimeError(
                    f"AlignScore worker exited before responding: returncode={returncode}"
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must have the same length")
        if not docs:
            return []
        if self.process.poll() is not None:
            raise RuntimeError(
                f"AlignScore worker is not running: returncode={self.process.returncode}"
            )
        if self.process.stdin is None:
            raise RuntimeError("AlignScore worker stdin is unavailable")
        self.request_count += 1
        request_id = f"{self.worker_pid}:{self.request_count}"
        self.process.stdin.write(
            json.dumps(
                {
                    "request_id": request_id,
                    "docs": [str(value) for value in docs],
                    "claims": [str(value) for value in claims],
                }
            )
            + "\n"
        )
        self.process.stdin.flush()
        response = self._read_payload(self.request_timeout_seconds)
        if response.get("request_id") != request_id:
            raise RuntimeError(
                f"AlignScore worker response ID mismatch: {response.get('request_id')}"
            )
        if response.get("error"):
            raise RuntimeError(f"AlignScore worker error: {response['error']}")
        scores = response.get("scores")
        if not isinstance(scores, list) or len(scores) != len(docs):
            raise RuntimeError("AlignScore worker returned a malformed score list")
        output = []
        for score in scores:
            value = float(score)
            if not math.isfinite(value):
                raise RuntimeError("AlignScore worker returned a non-finite score")
            output.append(
                {
                    "score": value,
                    "aux": {
                        "worker_pid": self.worker_pid,
                        "request_id": request_id,
                        "persistent_worker": True,
                    },
                }
            )
        return output

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
        self.returncode = process.returncode

    def __enter__(self) -> "AlignScorePersistentScorer":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def normalize_fenice_score(raw_score: float) -> float:
    raw_score = float(raw_score)
    if not math.isfinite(raw_score):
        raise ValueError(f"FENICE returned non-finite score {raw_score}")
    return (min(1.0, max(-1.0, raw_score)) + 1.0) / 2.0


class FENICEScorer:
    def __init__(
        self,
        *,
        external_root: Path = DEFAULT_EXTERNAL_ROOT,
        claim_extractor_batch_size: int = 32,
        nli_batch_size: int = 32,
    ) -> None:
        source = Path(external_root) / "fenice" / "src"
        if not source.exists():
            raise FileNotFoundError(source)
        sys.path.insert(0, str(source))
        self.claim_extractor_batch_size = int(claim_extractor_batch_size)
        self.nli_batch_size = int(nli_batch_size)
        self.model_id = "FENICE official pipeline"
        from metric.FENICE import FENICE

        self.metric = FENICE(
            use_coref=False,
            claim_extractor_batch_size=self.claim_extractor_batch_size,
            nli_batch_size=self.nli_batch_size,
            nli_max_length=1024,
        )

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        if len(docs) != len(claims):
            raise ValueError("docs and claims must have the same length")
        raw_outputs = self.metric.score_batch(
            [
                {"document": str(document), "summary": str(claim)}
                for document, claim in zip(docs, claims)
            ]
        )
        outputs = []
        for raw in raw_outputs:
            raw_score = float(raw["score"])
            alignments = raw.get("alignments", [])
            outputs.append(
                {
                    "score": normalize_fenice_score(raw_score),
                    "aux": {
                        "fenice_raw_score": raw_score,
                        "score_transform": "clip((raw+1)/2,0,1)",
                        "atomic_units": alignments,
                        "claim_count": len(alignments),
                        "coreference_enabled": False,
                    },
                }
            )
        return outputs

    def close(self) -> None:
        metric = getattr(self, "metric", None)
        if metric is None:
            return
        self.metric = None
        del metric
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
