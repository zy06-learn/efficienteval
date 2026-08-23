#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent JSONL worker for the isolated AlignScore environment."
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-mode", default="nli_sp")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        from alignscore import AlignScore

        scorer = AlignScore(
            model="roberta-large",
            batch_size=args.batch_size,
            device=args.device,
            ckpt_path=args.ckpt,
            evaluation_mode=args.eval_mode,
            verbose=False,
        )
    print(
        json.dumps(
            {
                "status": "ready",
                "pid": os.getpid(),
                "batch_size": args.batch_size,
                "device": args.device,
                "evaluation_mode": args.eval_mode,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id")
            docs = [str(value) for value in request["docs"]]
            claims = [str(value) for value in request["claims"]]
            if len(docs) != len(claims):
                raise ValueError("docs and claims must have the same length")
            with contextlib.redirect_stdout(sys.stderr):
                scores = scorer.score(contexts=docs, claims=claims)
            response = {
                "request_id": request_id,
                "pid": os.getpid(),
                "scores": [float(score) for score in scores],
            }
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "request_id": request_id,
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
