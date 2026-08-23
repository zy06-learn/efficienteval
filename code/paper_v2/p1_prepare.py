#!/usr/bin/env python3
"""P1 step 0: build the scoring cohort from the sealed test set.

Only rows that are missing at least one verifier score are scored. FRANK test already carries
all 15 scores from the frozen run that also produced TRAIN, so re-scoring it would replace
frozen numbers with numbers from a different run and break comparability with the training side.

The cohort is built from TEST_SCORING.parquet, which is the label-free projection: it carries
no label column, so this step cannot read test labels. Verified explicitly below.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import pandas as pd

import os

# This file lives in paper_v2, so its own directory is the default. AFR_ROOT overrides it
# for callers that run the stage from somewhere else.
V2 = Path(os.environ["AFR_ROOT"]) / "paper_v2" if os.environ.get("AFR_ROOT") \
    else Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2.parent))
import config_v2 as C

t = pd.read_parquet(V2 / "data" / "TEST_SCORING.parquet")

label_cols = [c for c in t.columns if "label" in c.lower() or "supported" in c.lower()]
assert not label_cols, f"scoring projection must not carry labels, found {label_cols}"
print(f"TEST_SCORING rows={len(t)}  label columns={label_cols} (must be empty) OK")

verifiers = list(C.VERIFIERS)
score_cols = [f"score__{v}" for v in verifiers]
missing_any = t[score_cols].isna().any(axis=1)
cohort = t.loc[missing_any].copy()

print(f"\nrows needing at least one score: {len(cohort)}")
print(cohort["dataset_key"].value_counts().to_string())
print(f"\nrows fully scored already (kept frozen): {int((~missing_any).sum())}")
print(t.loc[~missing_any, "dataset_key"].value_counts().to_string())

required = ["episode_key", "episode_id", "dataset_key", "role", "doc_group_key", "group_id",
            "source_document", "candidate_summary", "source_token_count",
            "summary_token_count", "semantic_input_tokens"]
have = [c for c in required if c in cohort.columns]
miss = [c for c in required if c not in cohort.columns]
print(f"\nrequired columns present: {len(have)}/{len(required)}")
if miss:
    print("MISSING:", miss)
    for c in miss:
        if c == "summary_token_count" and "claim_token_count" in cohort.columns:
            cohort["summary_token_count"] = cohort["claim_token_count"]
            print("  filled summary_token_count from claim_token_count")
        elif c == "semantic_input_tokens":
            cohort["semantic_input_tokens"] = (
                cohort["source_token_count"].astype(int) + cohort["summary_token_count"].astype(int))
            print("  filled semantic_input_tokens = source + summary tokens")
    miss = [c for c in required if c not in cohort.columns]
    assert not miss, f"still missing {miss}"

out = V2 / "data" / "P1_SCORING_COHORT.parquet"
cohort.to_parquet(out, index=False)
print(f"\nwrote {out}  rows={len(cohort)}")
print("sha256 will be recorded by the scoring harness manifest")
