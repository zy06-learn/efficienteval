#!/usr/bin/env python3
"""P1: score the test-side cohort, bypassing prepare_inputs.

The stock harness entry point validates that the cohort carries `label_supported`, because in
the v1 flow scoring and evaluation shared one frame. Here the cohort is deliberately the
label-free projection, so that path cannot be used: satisfying it would mean putting a label
column next to the test rows during scoring, which is exactly what the protocol forbids. This
driver calls score_frame directly instead, so no label ever enters the scoring stage.

Per-verifier results are cached by the harness itself (jsonl keyed on episode_key plus an
identity hash), so re-running resumes rather than repeats.
"""
from __future__ import annotations
import os
import sys, gc, json, hashlib, time
from pathlib import Path

import pandas as pd

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ingest_and_scoring"))
from verifier_wrappers.unified_summary_verifiers_v1 import build_scorer, score_frame, API_VERIFIERS

V2 = ROOT / "ingest_and_scoring"
OUT = V2 / "results" / "p1_scoring"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "status").mkdir(exist_ok=True)

COHORT = V2 / "data" / "P1_SCORING_COHORT.parquet"
frame = pd.read_parquet(COHORT)
assert not [c for c in frame.columns if "label" in c.lower()], "cohort must be label-free"
input_sha = hashlib.sha256(COHORT.read_bytes()).hexdigest()
print(f"cohort {len(frame)} rows | sha256 {input_sha[:16]}...", flush=True)

wanted = sys.argv[1].split(",") if len(sys.argv) > 1 else []
api_base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8001/v1"

import glob
_ws = sorted(glob.glob(os.environ.get("HF_HUB", "/home/zeyu/.cache/huggingface/hub")
                       + "/models--nightdessert--WeCheck/snapshots/*"))
MODEL_PATHS = {"wecheck": Path(_ws[-1])} if _ws else {}
# vLLM-served verifiers need the tokenizer that matches the served weights; the driver
# script exports TOKENIZER_PATH alongside each model it serves.
_TP = os.environ.get("TOKENIZER_PATH")
TOKENIZER_HINT = {}

# HHEM ships custom modelling code; newer transformers expects every PreTrainedModel to expose
# `all_tied_weights_keys`, which that vendored class predates. Supplying the attribute as a class
# default restores loading without touching the vendored file or the model weights.
try:
    import transformers
    if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
        transformers.PreTrainedModel.all_tied_weights_keys = None
except Exception as _e:
    print("transformers compat shim skipped:", _e)

for verifier in wanted:
    tag = OUT / "status" / f"{verifier}.done"
    if tag.exists():
        print(f"=== {verifier} already done, skip ===", flush=True)
        continue
    print(f"=== {verifier} start {time.strftime('%H:%M:%S')} ===", flush=True)
    t0 = time.time()
    try:
        kwargs = {"device": "cuda"}
        if verifier in MODEL_PATHS:
            kwargs["model_path"] = MODEL_PATHS[verifier]
        if verifier in API_VERIFIERS:
            kwargs["api_base"] = api_base
            kwargs["served_model"] = "unified-summary-verifier"
            kwargs["tokenizer_path"] = Path(TOKENIZER_HINT.get(verifier) or _TP)
        scorer = build_scorer(verifier, **kwargs)
        # Protocol deviation (recorded in PROTOCOL_DEVIATIONS.md): a row whose generated text
        # cannot be parsed into a label is recorded as UNAVAILABLE for this verifier rather than
        # aborting the run. This matches how context-overflow rows are already handled, and the
        # router supports unavailable actions natively. Aborting instead would let one unparseable
        # row remove an entire verifier from the study.
        from verifier_wrappers.unified_summary_verifiers_v1 import unavailable_output as _unavail
        _orig_sb = scorer.score_batch
        _pf = {"n": 0}

        def _safe(docs, claims, __o=_orig_sb, __pf=_pf):
            out = __o(docs, claims)
            o = out[0]
            aux = dict(o.get("aux") or {})
            bad = aux.get("available", True) and (
                not o.get("parse_ok", True) or o.get("score") is None)
            if bad:
                __pf["n"] += 1
                u = _unavail("parse_failure:" + str(o.get("parse_error"))[:100])
                u["aux"]["parse_failure"] = True
                return [u]
            return out

        scorer.score_batch = _safe
        result = score_frame(scorer=scorer, verifier=verifier, frame=frame,
                             input_sha256=input_sha, output_dir=OUT, warmup=3)
        close = getattr(scorer, "close", None)
        if callable(close):
            close()
        del scorer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        tag.write_text(json.dumps({"ok": True, "seconds": time.time() - t0}, default=str))
        print(f"=== {verifier} OK in {time.time()-t0:.0f}s parse_failures={_pf[chr(34)+chr(110)+chr(34)] if False else _pf['n']} ===", flush=True)
    except Exception as exc:
        print(f"=== {verifier} FAILED after {time.time()-t0:.0f}s: "
              f"{type(exc).__name__}: {exc} ===", flush=True)
        import traceback
        traceback.print_exc()
print("=== BATCH COMPLETE ===", flush=True)
