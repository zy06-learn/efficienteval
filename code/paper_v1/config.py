#!/usr/bin/env python3
"""The single source of truth for the paper's configuration.

Every experiment script imports the locked Router values from here; ablation grids remain local
to their diagnostic scripts. The previous
round of experiments drifted precisely because four scripts called fit_heads() without passing
hp=, silently falling back to a different RandomForest than the one the paper claims to use, and
two of those also used a beta rule we had already shown to be broken. Centralising removes the
possibility rather than relying on remembering.

Provenance of each locked value -- what chose it and on what evidence:

  pool          exhaustive validation screen of every k=2 and k=3 subset of the 15 verifiers,
                run under THIS configuration (expC1_poolscreen.py). Rank 1 of the 445 feasible
                k=3 pools with no cost constraint, by +.0105 over rank 2 (4.2 seed SD), and the
                terminal point of the quality/cost Pareto frontier: no pool at any cost scores
                higher. The 20 infeasible pools are exactly the subsets of the five 512-token
                verifiers, which jointly fail to cover one row whose summary is 1444 tokens.
                Greedy forward selection does NOT find this pool -- it reaches only rank 6/445
                at k=3 (19 SD short) and is still behind at k=6 -- because each of the three
                members is individually weak (.53/.62/.75 AUROC) so every prefix looks bad.
                No fourth member helps: 0 of 12 supersets gain more than one seed SD.
                An earlier note here credited 'rank 1 of 1060 pools' without qualification; that
                ranking came from results/day2_v1, a faster screen that scores this pool .7734
                against .8201 here, and its ordering does not transfer.
  features      relevance filter + correlation dedup + exhaustive k<=3 + forward addition
  learner       8-learner sweep on the final configuration; top few within noise, RF chosen
  target        6 targets x 2 learners; the only design axis with a real effect (4.7 points)
  criterion     3 RandomForest criteria; spread .0009, squared_error selected by min val loss
  hyperparams   16-trial random search selected by MINIMISING validation loss (advisor's rule)
  beta rule     budget-constrained; replaces a product rule that collapsed under cost perturbation
  recalibration isotonic on the router's own output, fit on the validation fold

RECALIBRATION is deliberately reported both ways in the main comparison. These are two explicit
evaluation views, not permission to inspect test and retroactively call one preregistered.
Whatever is reported must be applied to EVERY comparable system, not just ours. Risk-control
Table 2 is stage-1 only so isotonic fitting cannot reuse its conformal-calibration labels.
"""
from __future__ import annotations
import os
from pathlib import Path

# ------------------------------------------------------------------ paths
# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SRC_ROOT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT))
PKG_ROOT = SRC_ROOT / "paper_v1"
RESULTS = PKG_ROOT / "results"
NOT_REPORTED = RESULTS / "_not_reported"
STATUS = PKG_ROOT / "status"
LOGS = PKG_ROOT / "logs"

# ------------------------------------------------------------------ locked configuration
POOL = ("factcc", "lettuce_v2", "granite_guardian_3_1_2b")

FEATURES = [
    "claim_token_count",            # summary length in tokens
    "claim_source_length_ratio",    # compression ratio, summary / source
    "fact_mention_density",         # (entity mentions + value mentions) / normaliser
    "tfidf_top1",                   # TF-IDF similarity to the best-matching source sentence
    "entity_coverage",              # fraction of summary entities found in the source
    "rougeL_top1",                  # ROUGE-L against the best-matching source sentence
    "structured_source_line_ratio", # fraction of source lines that are bullets/headers
    "evidence_span_normalized",     # (max - min) best-evidence sentence index / (n_sent - 1)
    "summary_sentence_count",       # number of sentences in the summary
]

LEARNER = "rf"
TARGET = "regret"                   # T[i,j] = 1 + (Q[i,j] - max_j Q[i,j])
CRITERION = "squared_error"
HP = {
    "n_estimators": 800,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": 10,
}

BETA_RULE = "budget"                # min cost s.t. val AUROC >= best - EPS
EPS = 0.005

# ------------------------------------------------------------------ protocol
N_FOLDS = 10                        # pooled protocol: 8 train / 1 val / 1 test, group-disjoint
PERDATASET_FOLDS = 5                # per-corpus protocol: 3 train / 1 val / 1 test
SEEDS = (17, 29, 43, 7, 11, 23, 31, 37, 53, 61)
SEEDS_FAST = SEEDS[:5]
SEEDS_30 = SEEDS + (101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                    151, 157, 163, 167, 173, 179, 181, 191, 193, 197)
BOOTSTRAP = 2000

# ------------------------------------------------------------------ evaluation
DELTAS = (0.10, 0.20, 0.30, 0.40)   # conformal: guarantee catching >= 1-delta of hallucinations
COVERAGE = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ECE_BINS = 15

# Balance anchors, fixed so the metric is comparable across every experiment.
# Quality spans the cheapest-to-strongest fixed verifier; cost is logarithmic because the
# verifiers span 27x, and a linear axis would compress every cheap system into one point.
BALANCE_Q_FLOOR, BALANCE_Q_CEIL = 0.6876, 0.7931
BALANCE_C_MIN, BALANCE_C_MAX = 42.6, 1170.2

# ------------------------------------------------------------------ contamination record
# from results/one_shot_v1/DATASET_VERIFIER_CONTAMINATION_MATRIX.csv
CONFIRMED_TRAIN = {("hhem", "ragtruth_train"), ("factkb", "frank_train")}
DEV_EXPOSED = {"factcg", "minicheck_ft5", "minicheck_dbta", "alignscore"}

# ------------------------------------------------------------------ red lines
# storysumm rows: 0 | sealed/official test rows: 0 | verifier inference calls: 0
# Every verifier score is read from the frozen 6850 x 15 matrix.
VERIFIER_INFERENCE_ALLOWED = False


def banner() -> str:
    return (f"pool={POOL} | learner={LEARNER} | target={TARGET} | criterion={CRITERION}\n"
            f"hp={HP}\nbeta={BETA_RULE}(eps={EPS}) | features={len(FEATURES)} | seeds={len(SEEDS)}")
