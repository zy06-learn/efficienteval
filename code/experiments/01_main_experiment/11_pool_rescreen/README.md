# Pool provenance: the gap, and what closing it showed

## The gap

The verifier pool in use was selected in shared, not in part1. Its evidence is
`shared/results/POOL_SCREEN_LOCKEDCFG.csv`, where
`factcc + granite_guardian_3_1_2b + lettuce_v2` ranks **1 of 540** at validation AUROC
`0.82005 @ 90.60 ms`.

That screen ran on shared's 6,850-row matrix. Measured against the current splits, that
matrix contains:

- **0** of Protocol B's TEST rows at the episode level
- **351 of 645 = 54.4%** of Protocol B's TEST **document groups**

The project's own splitting criterion is `content_doc_key`, precisely because one source
document carries several candidate summaries and row-level splitting would leak a shared source.
By that criterion the pool selection was not independent of Protocol B's test set. Features and
hyperparameters were later reselected on TRAIN alone; the pool never was.

## Closing it

`experiments/pool_rescreen_v1.py` re-runs the selection on data Protocol B's TEST never touched:

- all C(15,3) = 455 three-verifier pools
- TRAIN only, `with_test_labels=False`, grouped fit/validation split, five seeds
- the six frozen features, regret target, random forest, part1c's Protocol B hyperparameters,
  the frozen beta grid and tolerance
- score: pre-stage-2 routed validation AUROC, the convention `fixedpool_select_v1` used
- eligibility: the rule `freeze_v3.py` declared before consulting any screen — k = 3, not
  dominated by a single verifier on TRAIN, every member receiving at least 5% of routed traffic
- confirmation: the top twenty eligible pools plus the pool in use, re-evaluated under A-style
  TRAIN-only rotations

## What it showed

455 pools screened, 90 eligible. The pool in use:

| | |
|---|---|
| Validation AUROC | 0.75930 (SD 0.00333) |
| Validation latency | 89.92 ms |
| Minimum member share | 13.6% |
| Dominated by a single verifier | no |
| Rank by AUROC alone | **109 / 455** overall, **28 / 90** among eligible |
| A-style confirmation | rank 11 / 21, joint rank 18 / 21 |
| **Quality-cost Pareto frontier** | **on it**, in both the all-pools view (12 frontier pools) and the eligible-only view (7 frontier pools) |

The rank-1 status does **not** survive re-selection on clean data. What survives is the
frontier position, and the margins around it are large:

- Only **7 of 455** pools are cheaper. The best validation AUROC among all of them is
  **0.70748**, i.e. 0.052 below the pool in use.
- **108** pools have higher AUROC. The cheapest of those costs **251.95 ms, 2.80x** the pool in
  use; the ones that lead the AUROC ranking cost 356-541 ms, four to six times more.
- Among the seven eligible frontier pools it has the **lowest seed SD** (0.00333 against
  0.00601-0.01956) and the **highest minimum member share** (13.6%).

## How this must be described

Not "the best pool". The defensible claim, now established on data the confirmatory test set
never touched, is:

> The pool sits on the quality-cost Pareto frontier of an exhaustive TRAIN-only screen of all
> 455 three-verifier pools. Ranked on validation AUROC alone it is 109th of 455, but every pool
> that scores higher costs at least 2.8x as much, and every cheaper pool gives up at least
> 0.052 AUROC. It is one of seven frontier pools that also satisfy the pre-declared eligibility
> rule, and the most seed-stable of them.

The original rank-1 result should be reported as what it is: a selection made on a matrix that
shared 54.4% of the test documents, superseded by this screen.

## Files

- `POOL_RESCREEN_B.csv` — all 455 pools: AUROC, SD, latency, per-member shares, minimum share,
  single-verifier domination count, eligibility, rank
- `POOL_RESCREEN_A_CONFIRM.csv` — A-style confirmation of the top twenty plus the pool in use
- `POOL_FIXED_REFERENCE.csv` — per-verifier TRAIN validation AUROC and latency, the basis of the
  domination test
- `rescreen.log` — complete run log
