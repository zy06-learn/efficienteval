#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afr_v2.unified_summary_verifiers_v1 import (  # noqa: E402
    API_VERIFIERS,
    CANARY_NAME,
    DATASET_NAME,
    VERIFIERS,
    audit_canary,
    build_scorer,
    finalize_results,
    prepare_inputs,
    score_frame,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--cohort", type=Path, required=True)
    prepare.add_argument("--result-dir", type=Path, required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--verifier", choices=VERIFIERS, required=True)
    score.add_argument("--result-dir", type=Path, required=True)
    score.add_argument("--canary", action="store_true")
    score.add_argument("--device", default="cuda")
    score.add_argument("--warmup", type=int, default=5)
    score.add_argument("--model-path", type=Path)
    score.add_argument("--tokenizer-path", type=Path)
    score.add_argument("--api-base", default="http://127.0.0.1:8001/v1")
    score.add_argument("--served-model", default="unified-summary-verifier")
    score.add_argument("--max-context", type=int, default=16384)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--result-dir", type=Path, required=True)

    canary_audit = subparsers.add_parser("audit-canary")
    canary_audit.add_argument("--result-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_inputs(args.cohort, args.result_dir / "inputs")
    elif args.command == "score":
        dataset = CANARY_NAME if args.canary else DATASET_NAME
        input_path = args.result_dir / "inputs" / f"{dataset}.parquet"
        frame = __import__("pandas").read_parquet(input_path)
        warmup_frame = None
        if args.canary:
            warmup_frame = __import__("pandas").read_parquet(
                args.result_dir / "inputs" / f"{DATASET_NAME}.parquet"
            )
        scorer = None
        try:
            scorer = build_scorer(
                args.verifier,
                device=args.device,
                model_path=args.model_path,
                tokenizer_path=args.tokenizer_path,
                api_base=args.api_base,
                served_model=args.served_model,
                max_context=args.max_context,
            )
            output_dir = args.result_dir / ("canary_scores" if args.canary else "scores")
            result = score_frame(
                scorer=scorer,
                verifier=args.verifier,
                frame=frame,
                input_sha256=sha256_file(input_path),
                output_dir=output_dir,
                warmup=args.warmup,
                warmup_frame=warmup_frame,
            )
        finally:
            if scorer is not None:
                close = getattr(scorer, "close", None)
                if callable(close):
                    close()
                del scorer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    elif args.command == "finalize":
        result = finalize_results(
            args.result_dir / "inputs",
            args.result_dir / "scores",
            args.result_dir,
        )
    elif args.command == "audit-canary":
        result = audit_canary(
            args.result_dir / "inputs",
            args.result_dir / "canary_scores",
            args.result_dir,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
