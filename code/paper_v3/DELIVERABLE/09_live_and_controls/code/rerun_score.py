#!/usr/bin/env python3
"""Independent re-run of the three pool verifiers on a sample of the frozen scoring cohort.

Writes to a fresh output_dir so the harness cache cannot serve stale rows: this is a genuine
re-execution, not a resume. Nothing under paper_v2/results is touched.
"""
from __future__ import annotations
import os
import sys, gc, json, hashlib, time, os
from pathlib import Path
import pandas as pd

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_v2"))
from afr_v2.unified_summary_verifiers_v1 import (build_scorer, score_frame, API_VERIFIERS,
                                                 unavailable_output as _unavail)

N        = int(os.environ.get("RERUN_N", "240"))
OUT      = Path(os.environ["RERUN_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "status").mkdir(exist_ok=True)
COHORT   = ROOT / "paper_v2" / "data" / "P1_SCORING_COHORT.parquet"
api_base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8001/v1"
wanted   = sys.argv[1].split(",")

full = pd.read_parquet(COHORT)
assert not [c for c in full.columns if "label" in c.lower()], "cohort must be label-free"
# stratified by corpus, fixed seed so the same rows are used across phases
key = "dataset_key" if "dataset_key" in full.columns else full.columns[0]
per = max(1, N // full[key].nunique())
parts = [full[full[key] == v].sample(min((full[key] == v).sum(), per), random_state=20260819)
         for v in sorted(full[key].unique())]
frame = pd.concat(parts, ignore_index=True)
frame.to_parquet(OUT / "RERUN_COHORT.parquet")
input_sha = hashlib.sha256((OUT / "RERUN_COHORT.parquet").read_bytes()).hexdigest()
print(f"re-run cohort {len(frame)} rows over {frame[key].nunique()} corpora | "
      f"sha {input_sha[:16]}", flush=True)

_TP = os.environ.get("TOKENIZER_PATH")
for verifier in wanted:
    tag = OUT / "status" / f"{verifier}.done"
    if tag.exists(): print(f"=== {verifier} done, skip ==="); continue
    print(f"=== {verifier} start {time.strftime('%H:%M:%S')} ===", flush=True)
    t0 = time.time()
    kwargs = {"device": "cuda"}
    if verifier in API_VERIFIERS:
        kwargs.update(api_base=api_base, served_model="unified-summary-verifier",
                      tokenizer_path=Path(_TP))
    scorer = build_scorer(verifier, **kwargs)
    _orig, _pf = scorer.score_batch, {"n": 0}
    def _safe(docs, claims, __o=_orig, __pf=_pf):
        out = __o(docs, claims); o = out[0]
        if (o.get("aux") or {}).get("available", True) and (
                not o.get("parse_ok", True) or o.get("score") is None):
            __pf["n"] += 1
            u = _unavail("parse_failure"); u["aux"]["parse_failure"] = True; return [u]
        return out
    scorer.score_batch = _safe
    score_frame(scorer=scorer, verifier=verifier, frame=frame,
                input_sha256=input_sha, output_dir=OUT, warmup=3)
    if callable(getattr(scorer, "close", None)): scorer.close()
    del scorer; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    tag.write_text(json.dumps({"ok": True, "seconds": time.time() - t0}))
    print(f"=== {verifier} OK in {time.time()-t0:.0f}s parse_fail={_pf['n']} ===", flush=True)
