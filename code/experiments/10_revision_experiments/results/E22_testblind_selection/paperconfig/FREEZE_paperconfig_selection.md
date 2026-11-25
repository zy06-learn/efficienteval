# Frozen selection rule: test-blind exhaustive pool selection under the paper's router (Protocol B)

Written 2026-09-29, before this run was started. It is the companion of
`FREEZE_testblind_selection.md`, which re-runs the historical selection-stage router. This run
keeps the same search structure (inventory, budgets, budget picks, confirmation with three seeds
and two feature configurations, normalized quality-cost criterion, tie-breaker) and replaces the
selection-stage router with the router the paper reports.

Already seen before this freeze, disclosed here: `pool_rescreen_v1` (2026-08-17) screened the 455
three-member pools under the paper's configuration with a single 80/20 fit/validation split and
without stage-2 or the full latency accounting; on 2026-09-29 its numbers were ranked by the
budget and Balance rule. This run differs from it in the pool inventory (pairs added), the inner
protocol (8/1/1 rotations), the latency definition (end to end), the output scored (after
stage-2), and the confirmation stage.

## Data

- Selection data: the 5,276 rows of `paper_v2/data/TRAIN.parquet` only. The test frame is loaded
  label-free and never used; the script asserts it carries no label column.
- Inner protocol: source-grouped ten folds inside TRAIN, stratified by corpus and majority label
  (`v3core.folds_stratified`), with the paper's 8/1/1 rotations (`v3core.rotations`). In each
  rotation, eight folds fit, one fold validates, one fold is held out.

## The router being selected for (the paper's final system, unchanged)

Per rotation and pool:

- stage-1 Platt calibrators, regret targets, random-forest heads (one per member), and verifier
  costs are fitted on the eight fit folds;
- features: the six unified cheap features;
- hyperparameters: part1c's Protocol B selection (`HP_SELECTED.json`, `hyperparameters_B`:
  200 trees, min leaf 10, max features 0.5, max depth 6), chosen on TRAIN only;
- beta: the paper's grid and rule (cheapest beta within 0.005 of the best validation AUROC),
  chosen on the validation fold;
- stage-2 isotonic map fitted on the validation fold;
- AUROC is computed on the held-out fold after stage-2, pooled over the ten held-out folds of a
  seed (the paper's Protocol A estimator, restricted to TRAIN);
- latency is the paper's end-to-end definition: verifier call, both cheap-feature extractors, head
  inference, routing arithmetic, both calibration stages. A fixed verifier is charged its call
  and its calibration.

A pool is evaluable only if every TRAIN row has at least one available member.

## Stage 1: budget picks

Every 2- and 3-member pool of the 15 verifiers is run with seed 17 over all ten rotations. For each
budget in {100, 150, 200, 250, 300, 400, 600} ms and each pool size in {2, 3}, among evaluable
pools whose mean end-to-end latency is <= budget, the pick is ordered by:

1. higher inner held-out AUROC
2. lower latency
3. fewer members
4. lexicographic pool name

## Stage 2: confirmation

Each distinct budget pick is re-run with seeds {17, 29, 43} over all ten rotations under two
feature configurations: the six cheap features, and the corpus one-hot only (the paper's
dataset-identity control, `ds_only.py`). The feature configuration changes only what the heads
read; both configurations are charged the same end-to-end latency. Each (pool, configuration)
is summarised by the mean over the three seeds of the held-out AUROC and of the latency. The
paper's pool is confirmed as a reference; it is eligible only if some budget picked it.

## Stage 3: normalized quality-cost criterion (Balance)

    q~ = (AUROC - Q_floor) / (Q_ceil - Q_floor)
    c~ = (ln ms - ln C_min) / (ln C_max - ln C_min)
    Balance = 1 - sqrt((1 - q~)^2 + c~^2) / sqrt(2)

Anchors from the 15 single verifiers run through the same protocol (seeds 17, 29, 43, ten
rotations, Platt on fit folds, isotonic on the validation fold, held-out AUROC, end-to-end
latency): Q_floor = AUROC of the cheapest single verifier, Q_ceil = the highest single-verifier
AUROC, C_min / C_max = the lowest / highest single-verifier latency.

The selected pool is the pool of the eligible (pool, configuration) row ordered by:

1. higher Balance
2. higher mean held-out AUROC
3. lower mean latency
4. fewer members
5. lexicographic pool name

## Stability (descriptive, does not change the selection)

- Stage 3 repeated on each seed separately; count how often each pool comes first.
- Stage 1 repeated with seeds 29 and 43; count how often each pool is the pick per budget.

## Stage 4: final evaluation (once)

The selected pool is frozen. The final paper system is fitted on TRAIN with part1c's Protocol B
procedure (ten seeds, 80/20 fit/validation split inside TRAIN, beta, isotonic and MCC threshold on
the validation part) and evaluated once on the official test split, reported whatever the
result. If the selected pool is the paper's pool, that system is the part1c system and its
existing Protocol B result is the final evaluation; the test split is not read again.
