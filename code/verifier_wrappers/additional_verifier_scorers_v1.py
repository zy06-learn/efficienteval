"""Pinned scorers for the additional summary-sentence verifier benchmark."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping


MODEL_REVISIONS = {
    "wecheck": "nightdessert/WeCheck@71199c2d1ac86ad4195e0545a584613eaa8ae787",
    "granite_guardian_3_2_factuality": (
        "ibm-granite/granite-guardian-3.2-8b-factuality-detection@"
        "de0c27b0ed657529269b573e106a3c72d18f85f9"
    ),
    "granite_guardian_4_1_groundedness": (
        "ibm-granite/granite-guardian-4.1-8b@"
        "69820a3f3c8f265e2fe61b5a8fcea2146c2fcb16"
    ),
    "gpt_oss_20b_judge": (
        "openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee"
    ),
    "qwen3_next_80b_judge": (
        "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8@"
        "c5f5f263bdd5cc134092897864e8905d8fe7b928"
    ),
}

PROTOCOLS = {
    "wecheck": "wecheck_official_sigmoid_logit0_source_claim512_batch1_v1",
    "granite_guardian_3_2_factuality": (
        "granite_guardian_3_2_official_context_assistant_yesrisk_logprobs_nothink_batch1_v1"
    ),
    "granite_guardian_4_1_groundedness": (
        "granite_guardian_4_1_official_groundedness_nothink_yesrisk_logprobs_batch1_v1"
    ),
    "gpt_oss_20b_judge": "gpt_oss_20b_structured_judge_no_cot_batch1_v1",
    "qwen3_next_80b_judge": "qwen3_next_80b_fp8_structured_judge_no_cot_batch1_v1",
}


def _strict_single(docs: list[str], claims: list[str], verifier: str) -> None:
    if len(docs) != 1 or len(claims) != 1:
        raise ValueError(f"{verifier} requires strict batch=1")


def _normalized_binary_token(value: Any) -> str:
    return re.sub(r"[^a-z]", "", str(value).casefold())


def support_probability_from_top_logprobs(
    top_logprobs: Any,
    *,
    supported_token: str = "no",
    unsupported_token: str = "yes",
) -> float | None:
    """Normalize official yes/no token mass into P(supported)."""
    mass = {supported_token: 0.0, unsupported_token: 0.0}
    for position in top_logprobs or []:
        if not isinstance(position, Mapping):
            continue
        for token, logprob in position.items():
            normalized = _normalized_binary_token(token)
            if normalized in mass:
                mass[normalized] += math.exp(float(logprob))
    denominator = mass[supported_token] + mass[unsupported_token]
    if denominator <= 0.0:
        return None
    return float(mass[supported_token] / denominator)


class WeCheckScorer:
    score_max = 1.0
    model_id = MODEL_REVISIONS["wecheck"]

    def __init__(self, *, model_path: str | Path, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                str(model_path), local_files_only=True
            )
            .to(device)
            .eval()
        )

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        _strict_single(docs, claims, "wecheck")
        encoded = self.tokenizer(
            docs,
            claims,
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits
        support = self._torch.sigmoid(logits[:, 0])
        outputs = []
        for row_logits, probability in zip(logits.cpu().tolist(), support.cpu().tolist()):
            score = float(probability)
            outputs.append(
                {
                    "score": score,
                    "aux": {
                        "native_output_type": "official_sigmoid_logit0",
                        "native_label": "supported" if score >= 0.5 else "unsupported",
                        "label_source": "official_score_threshold_0.5",
                        "native_logits": [float(value) for value in row_logits],
                        "native_probs": {"support_sigmoid_logit0": score},
                        "num_model_calls": 1,
                    },
                }
            )
        return outputs


class _CompletionRiskScorer:
    score_max = 1.0

    def __init__(
        self,
        *,
        verifier: str,
        model_id: str,
        tokenizer_path: str | Path,
        api_base: str,
        served_model: str,
        device: str,
        max_tokens: int,
    ) -> None:
        from openai import OpenAI
        from transformers import AutoTokenizer

        self.verifier = verifier
        self.model_id = model_id
        self.device = device
        self.max_tokens = int(max_tokens)
        self.client = OpenAI(base_url=api_base, api_key="EMPTY", timeout=300.0)
        served = {model.id for model in self.client.models.list().data}
        if served_model not in served:
            raise RuntimeError(f"{served_model} not served; got {sorted(served)}")
        self.served_model = served_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True
        )

    def _prompt(self, document: str, claim: str) -> str:
        raise NotImplementedError

    def _parse_label(self, text: str) -> str | None:
        raise NotImplementedError

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        _strict_single(docs, claims, self.verifier)
        preprocessing_started = time.perf_counter()
        prompt = self._prompt(str(docs[0]), str(claims[0]))
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        raw_response = None
        parse_error = None
        transport_error = None
        attempts = 0
        inference_ms = 0.0
        support_probability = None
        label = None
        while attempts < 2 and label is None:
            attempts += 1
            call_started = time.perf_counter()
            try:
                response = self.client.completions.create(
                    model=self.served_model,
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    logprobs=20,
                )
                inference_ms += (time.perf_counter() - call_started) * 1000.0
                choice = response.choices[0]
                raw_response = choice.text or ""
                label = self._parse_label(raw_response)
                top_logprobs = getattr(getattr(choice, "logprobs", None), "top_logprobs", None)
                support_probability = support_probability_from_top_logprobs(top_logprobs)
                if label is None:
                    parse_error = f"unparseable response: {raw_response[:200]!r}"
            except Exception as exc:
                inference_ms += (time.perf_counter() - call_started) * 1000.0
                parse_error = f"{type(exc).__name__}: {exc}"[:500]
                transport_error = exc
        if label is None and transport_error is not None and raw_response is None:
            raise RuntimeError(parse_error) from transport_error
        parse_ok = label in {"supported", "unsupported"}
        if parse_ok and support_probability is None:
            support_probability = 1.0 if label == "supported" else 0.0
        return [
            {
                "score": support_probability if parse_ok else None,
                "parse_ok": parse_ok,
                "parse_error": None if parse_ok else parse_error,
                "aux": {
                    "native_output_type": "official_binary_risk_generation",
                    "native_label": label,
                    "label_source": "official_yes_no_generation",
                    "native_probs": (
                        {"supported": support_probability, "unsupported": 1.0 - support_probability}
                        if support_probability is not None
                        else None
                    ),
                    "raw_response": raw_response,
                    "parse_attempts": attempts,
                    "latency_preprocessing_ms": preprocessing_ms,
                    "latency_inference_ms": inference_ms,
                    "latency_aggregation_ms": 0.0,
                    "num_model_calls": attempts,
                },
            }
        ]


class GraniteGuardian32FactualityScorer(_CompletionRiskScorer):
    model_id = MODEL_REVISIONS["granite_guardian_3_2_factuality"]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            verifier="granite_guardian_3_2_factuality",
            model_id=self.model_id,
            max_tokens=16,
            **kwargs,
        )

    def _prompt(self, document: str, claim: str) -> str:
        messages = [
            {"role": "context", "content": document},
            {"role": "assistant", "content": claim},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _parse_label(self, text: str) -> str | None:
        match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
        if not match:
            return None
        return "unsupported" if match.group(1).casefold() == "yes" else "supported"


class GraniteGuardian41GroundednessScorer(_CompletionRiskScorer):
    model_id = MODEL_REVISIONS["granite_guardian_4_1_groundedness"]
    CRITERIA = (
        "A text is considered ungrounded or unfaithful if it includes information "
        "lacking support from, or directly contradicting, the provided document(s). "
        "This risk arises when the text fabricates details, misinterprets the content, "
        "or makes unsupported extrapolations beyond what is explicitly stated in the "
        "document(s)."
    )
    NO_THINK = (
        "<no-think>As a judge agent, assess whether the provided text meets the given "
        "judging criteria using all available information, including conversations, "
        "documents, and tools. Provide your score immediately without explanation. "
        "Output empty <think>\n</think> tags followed by your score in <score></score> tags."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            verifier="granite_guardian_4_1_groundedness",
            model_id=self.model_id,
            max_tokens=64,
            **kwargs,
        )

    def _guardian_block(self) -> str:
        return (
            f"<guardian>{self.NO_THINK}\n\n"
            f"### Criteria: {self.CRITERIA}\n\n"
            "### Scoring Schema: If the last assistant's text meets the criteria, "
            "return 'yes'; otherwise, return 'no'."
        )

    def _prompt(self, document: str, claim: str) -> str:
        messages = [
            {"role": "assistant", "content": claim},
            {"role": "user", "content": self._guardian_block()},
        ]
        documents = [{"doc_id": "0", "text": document}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            documents=documents,
        )

    def _parse_label(self, text: str) -> str | None:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        matches = re.findall(r"<score>\s*(yes|no)\s*</score>", cleaned, flags=re.IGNORECASE)
        if not matches:
            return None
        return "unsupported" if matches[-1].casefold() == "yes" else "supported"


class StructuredJudgeAPIScorer:
    score_max = 1.0

    def __init__(
        self,
        *,
        verifier: str,
        api_base: str,
        served_model: str,
        device: str,
        max_tokens: int = 400,
    ) -> None:
        from openai import OpenAI

        self.verifier = verifier
        self.model_id = MODEL_REVISIONS[verifier]
        self.device = device
        self.max_tokens = int(max_tokens)
        self.client = OpenAI(base_url=api_base, api_key="EMPTY", timeout=300.0)
        served = {model.id for model in self.client.models.list().data}
        if served_model not in served:
            raise RuntimeError(f"{served_model} not served; got {sorted(served)}")
        self.served_model = served_model

    def score_batch(self, docs: list[str], claims: list[str]) -> list[dict[str, Any]]:
        from verifier_wrappers.structured_high_judge import (
            SYSTEM_PROMPT,
            numbered_source,
            structured_response_format,
            validate_payload,
        )

        _strict_single(docs, claims, self.verifier)
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
        transport_error = None
        usage = None
        attempts = 0
        inference_ms = 0.0
        aggregation_ms = 0.0
        while attempts < 2 and payload is None:
            attempts += 1
            call_started = time.perf_counter()
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
                raw_response = response.choices[0].message.content or ""
                parse_started = time.perf_counter()
                payload = validate_payload(
                    json.loads(raw_response),
                    claim=claim,
                    n_source_sentences=sentence_count,
                )
                aggregation_ms += (time.perf_counter() - parse_started) * 1000.0
                usage = response.usage
            except Exception as exc:
                inference_ms += (time.perf_counter() - call_started) * 1000.0
                parse_error = f"{type(exc).__name__}: {exc}"[:500]
                transport_error = exc
        if payload is None and transport_error is not None and raw_response is None:
            raise RuntimeError(parse_error) from transport_error
        parse_ok = payload is not None
        return [
            {
                "score": payload["support_probability"] if parse_ok else None,
                "parse_ok": parse_ok,
                "parse_error": None if parse_ok else parse_error,
                "aux": {
                    "native_output_type": "structured",
                    "native_label": payload["label"].casefold() if parse_ok else None,
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


def build_additional_scorer(
    verifier: str,
    *,
    device: str,
    model_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    api_base: str = "http://127.0.0.1:8001/v1",
    served_model: str = "additional-verifier",
) -> Any:
    if verifier == "wecheck":
        if model_path is None:
            raise ValueError("wecheck requires model_path")
        return WeCheckScorer(model_path=model_path, device=device)
    if verifier == "granite_guardian_3_2_factuality":
        if tokenizer_path is None:
            raise ValueError(f"{verifier} requires tokenizer_path")
        return GraniteGuardian32FactualityScorer(
            tokenizer_path=tokenizer_path,
            api_base=api_base,
            served_model=served_model,
            device=device,
        )
    if verifier == "granite_guardian_4_1_groundedness":
        if tokenizer_path is None:
            raise ValueError(f"{verifier} requires tokenizer_path")
        return GraniteGuardian41GroundednessScorer(
            tokenizer_path=tokenizer_path,
            api_base=api_base,
            served_model=served_model,
            device=device,
        )
    if verifier in {"gpt_oss_20b_judge", "qwen3_next_80b_judge"}:
        return StructuredJudgeAPIScorer(
            verifier=verifier,
            api_base=api_base,
            served_model=served_model,
            device=device,
        )
    raise ValueError(f"unsupported additional verifier: {verifier}")
