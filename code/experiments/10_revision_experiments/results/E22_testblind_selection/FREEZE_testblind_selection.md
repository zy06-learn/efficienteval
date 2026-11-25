# Frozen selection rule: test-blind exhaustive pool selection (Protocol B)

Written 2026-09-29, before the Balance ranking of the finalists was inspected and before any
test-split row is evaluated. The selection-stage screening and budget picks were already computed
(validation folds inside TRAIN only); nothing from the official test split has been read.

## Data

- Selection data: the 5,276 rows of `paper_v2/data/TRAIN.parquet`, obtained from the historical
  6,850-row selection matrix by removing every row whose source document appears in
  `paper_v2/data/TEST.parquet` (1,569 FRANK-test rows and 5 RAGTruth-train rows). The script
  asserts that the remaining rows equal TRAIN exactly.
- Inner protocol: the historical source-grouped 8/1/1 rotation, run inside TRAIN.
- This is the historical procedure re-run on test-excluded data, not an exact reproduction:
  the partition differs from the original because the test-side rows are gone.

## Stage 1: budget picks (historical R6, unchanged)

Every 2- and 3-member pool of the 15 verifiers is screened (seed 17, 4 rotations, HGB heads,
brier target, compact16 features). For each budget in {100, 150, 200, 250, 300, 400, 600} ms and
each pool size in {2, 3}, among pools with validation latency <= budget, the pick is ordered by:

1. higher validation AUROC
2. lower validation latency
3. fewer members
4. lexicographic pool name

## Stage 2: confirmation (historical R6 finals, unchanged)

Each distinct budget pick is re-run with seeds {17, 29, 43} over all ten rotations, RF heads,
brier target, under two feature configurations (compact16 and dataset-id-only). Each
(pool, feature configuration) is summarised by the mean held-out AUROC and mean latency over the
three seeds. Only budget picks are eligible; the paper's historical pool is confirmed as well, but
it is eligible only if some budget picked it.

## Stage 3: the historical normalized quality-cost criterion (R9 Balance, unchanged)

    q~ = (AUROC - Q_floor) / (Q_ceil - Q_floor)
    c~ = (ln ms - ln C_min) / (ln C_max - ln C_min)
    Balance = 1 - sqrt((1 - q~)^2 + c~^2) / sqrt(2)

Anchors, computed on the 5,276 selection rows exactly as r9_run.py does: Q_floor = AUROC of the
cheapest single verifier, Q_ceil = AUROC of the best single verifier, C_min / C_max = lowest and
highest single-verifier mean latency.

The selected pool is the pool of the eligible (pool, feature configuration) row ordered by:

1. higher Balance
2. higher mean validation AUROC
3. lower mean latency
4. fewer members
5. lexicographic pool name

Seed stability is reported by repeating Stage 3 on each seed separately and counting how often
each pool comes first; this is descriptive and does not change the selection.

## Stage 4: final evaluation (once)

The selected pool is frozen. The final paper system (the six unified cheap features, relative
quality target, random-forest heads, multiplicative cost discount, stage-1 Platt, stage-2
isotonic, MCC threshold) is fitted on TRAIN; hyperparameters are chosen on TRAIN by minimum
validation head loss as in part1c; beta, the isotonic map and the threshold are chosen on the
validation part of TRAIN. It is evaluated once on the official test split and reported whatever
the result. If the selected pool is the historical pool, this system is the part1c system and
its existing Protocol B result is the final evaluation.
