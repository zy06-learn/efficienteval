"""Frozen protocol inputs for the EfficientEval v2 clean rebuild.

Selection outputs remain ``None`` until P3.  P0-P2 code must fail closed if it
tries to obtain a test label through the gold-free scoring asset.
"""
from __future__ import annotations
import os
import sys

from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_ROOT.parent
DATA = PAPER_ROOT / "data"
LOGS = PAPER_ROOT / "logs"
RESULTS = PAPER_ROOT / "results"
STATUS = PAPER_ROOT / "status"
CHECKPOINTS = PAPER_ROOT / "checkpoints"
RUN_RECORDS = PAPER_ROOT / "run_records"

PYTHON = Path(os.environ.get("AFR_PYTHON", sys.executable))
ALIGNSCORE_PYTHON = Path(os.environ.get("ALIGNSCORE_PYTHON", sys.executable))

SEEDS = (17, 29, 43, 7, 11, 23, 31, 37, 53, 61)
INNER_FOLDS = 5
INNER_PROTOCOL = "source-group-disjoint train4/validation1"
BOOTSTRAP_DRAWS = 2_000
POOL_SCREEN_TREES = 200
POOL_SCREEN_SEEDS = SEEDS[:3]
POOL_FINALIST_COUNT = 10
S4_RANDOM_TRIALS = 16
QUALITY_TOLERANCE = 0.005

VERIFIERS = (
    "factcc",
    "factkb",
    "hhem",
    "wecheck",
    "factcg",
    "minicheck_dbta",
    "minicheck_ft5",
    "alignscore",
    "lettuce_v2",
    "granite_guardian_3_1_2b",
    "granite_guardian_3_2_3b_a800m",
    "granite_guardian_3_2_8b_factuality",
    "granite_guardian_4_1_3b_factuality_lora",
    "qwen30_fast",
    "qwen30_judge",
)

# Deterministic leakage exclusion, declared before P1 and before any TRAIN/TEST file existed.
# One CNN/DailyMail article is shared by RAGTruth official train (source_id=12047) and CoGenSumm
# official test (document 515b5abb01fa23aa87ae475d0cde3fe8529605ca). A full pairwise audit over
# all 28 corpus-split pairs found this to be the only canonical-document overlap. The test side
# is preserved intact -- it is the confirmatory set and a published benchmark -- and the five
# RAGTruth TRAIN rows carrying the same article are excluded. This is a pre-declared rule keyed
# on the canonical content hash, not a result-driven choice.
TRAIN_EXCLUDED_CONTENT_DOC_KEYS = frozenset({
    "7f8caaade47aea0ddcdaa9bf44b025f3ee2772edd66639f267b35f543b98691a",
})

EXPECTED = {
    "train_rows": 5_276,
    "train_groups": 890,
    "test_rows": 3_236,
    "test_groups": 645,
    "train_by_dataset": {
        "ragtruth_train": (2_983, 499),
        "frank_valid": (669, 149),
        "unisumeval_train": (1_089, 135),
        "cogensumm_val": (535, 107),
    },
    "test_by_dataset": {
        "ragtruth_test": (900, 150),
        "frank_test": (1_569, 350),
        "unisumeval_dev": (367, 45),
        "cogensumm_test": (400, 100),
    },
}

SOURCE_SHA256 = {
    "results/unified_summary_verifiers_v1/ROUTER_TRAINING_MATRIX.parquet":
        "ce851c5536b653c1504f90eed9e083c76fc719afda1231cc7dcb13654a4d2c92",
    "results/summary_router_compact16_direct_v1/COMPACT16_FEATURES.parquet":
        "8572cdacb38613c1a61eaa80b209ed3a3172e667080224be82669381aa07f413",
    "results/mixed_dataset_v1/MIXED_COHORT.parquet":
        "c34052894b74b78ce1cbcf8e6387490e70be95d49d264a37b0f6dc53f14feaab",
    "data/unified_v1/frank/processed.jsonl":
        "fcc3bc7d6e0dec491073b8edf45d6e495d1d029e6eb45cb9453a49107d0465ce",
    "data/external/RAGTruth/dataset/source_info.jsonl":
        "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    "data/external/RAGTruth/dataset/response.jsonl":
        "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "data/summary_level_raw_v1/cogensumm/official_archive/summary-correctness-v1.0.zip":
        "9cbcae866faad48a208b798864d289708e1a8f0ff67bfca8dbc0089370e33140",
    "data/summary_level_raw_v1/source_corpora/cnndm_3.0.0/cnndm-test-0000.parquet":
        "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e",
    "data/cross_granularity_eval_v1/scoring_inputs/unisumeval_dev_summary.parquet":
        "48c1b7b46f6824d25b47d37205c230138c8afa7d0ff1dda06adf5776fc501146",
    "data/cross_granularity_eval_v1/router_features/unisumeval_dev_summary.parquet":
        "03e055dec7af369dca7d932fa2bb7f26c1a2ac009fdf4898d7f9c4c45b39c199",
    "data/cross_granularity_eval_v1/gold/unisumeval_dev_summary.parquet":
        "bae0fada9af086d6ec410c5e7f7f883478047bd59d2627384b8973a373ca530e",
}

# Populated only after S1-S7, immediately before PREREGISTRATION.md is frozen.
FROZEN_SELECTION = {
    "features": None,
    "target": None,
    "learner": None,
    "criterion": None,
    "hyperparameters": None,
    "pool": None,
    "cost_form": None,
    "beta_rule": None,
    "beta": None,
    "calibration": None,
    "final_calibration_group_fraction": None,
    "final_calibration_assignment": None,
}

CONFIRMED_TRAIN = {
    ("hhem", "ragtruth_train"),
    ("factkb", "frank_valid"),
}

