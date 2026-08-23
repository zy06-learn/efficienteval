# Part 1c — Main experiment: reselected hyperparameters, end-to-end latency

## Status

Authoritative archive for the Part 1 main experiment. Supersedes `part1b_main_clean_v1`
and `part1_main_pooled_v1`; both are preserved untouched.

Immutable source run: `paper_v3/runs/part1c_main_full_v1`. Wall clock 40 min 39 s, CPU only.

## What is inherited and what is selected here

Read verbatim from `part1_main_pooled_v1/00_contract/FROZEN_v3.json` and asserted field by
field before anything is fitted, never re-searched:

| | |
|---|---|
| Verifier pool | `factcc`, `lettuce_v2`, `granite_guardian_3_1_2b` |
| Cheap features | `structured_source_line_ratio`, `bm25_mean3`, `entity_coverage`, `entity_value_colocation`, `year_count`, `conflicting_value_rate` |
| Supervision target | `regret` |
| Learner | random forest, one independent regression head per verifier |
| Cost rule | `U = T_hat * exp(-beta * c / c_max)`, cheapest beta within 0.005 of best validation AUROC |
| Calibration | stage-1 Platt per verifier on fit; stage-2 isotonic per fitted instance on validation |
| Seeds | the ten declared seeds; splits grouped by `content_doc_key` |

Selected by this run, on TRAIN only, never touching TEST:

| | Protocol A | Protocol B |
|---|---|---|
| `n_estimators` | 800 | 200 |
| `min_samples_leaf` | 5 | 10 |
| `max_features` | 0.5 | 0.5 |
| `max_depth` | 6 | 6 |

Rule: sixteen randomly initialised candidates (`default_rng(0)` over trees {200,400,800},
leaf {1,2,5,10}, max_features {sqrt,log2,0.5}, depth {6,10,12,16,24}, deduplicated), each
scored under its protocol's own cross-validation shape over five seeds, chosen by **minimum
validation head loss** — mean squared error of the head predictions against the regret target.
part1's two values are included as reference points. Evidence:
`05_hyperparameter_selection/HP_SELECTION.csv`.

The selection quantity is deliberately disjoint from the metric the paper reports. Its cost is
recorded rather than assumed: on this feature set head loss and routing AUROC are **negatively**
rank-correlated (Spearman A `-0.541`, B `-0.312`), so the loss rule is aligned with quality
here, and it forgoes only `0.00052` (A) and `0.00379` (B) validation AUROC against selecting on
AUROC directly. An earlier round observed the opposite sign on a different feature set; that
concern does not hold for the current six features, and both numbers are in the contract.

## Drift sentinel

Changing the hyperparameters changes the router, so this run cannot be checked against part1
directly. `OURS_legacy_hp` carries part1's frozen hyperparameters through the identical new
harness and must reproduce part1's numbers exactly. Both protocols pass at
`delta = 0.00e+00` on AUROC and on part1-basis latency:

| Protocol | AUROC | Latency (part1 basis) |
|---|---:|---:|
| A | 0.7899055002 | 87.886519 ms |
| B | 0.8166451196 | 85.444745 ms |

This is the proof that splits, Platt, the regret target, head fitting, routing, beta selection
and isotonic are unchanged, and that the only moving part is the hyperparameters. See
`06_gates/`.

## End-to-end latency

Every millisecond the deployed system spends is charged. The breakdown reconciles with the
reported mean to `1e-13` ms, asserted at report time
(`02_latency/{A,B}_LATENCY_BREAKDOWN.csv`):

| Component | Protocol A | Protocol B |
|---|---:|---:|
| Feature extraction, both extractors | 33.221 | 27.293 |
| Random-forest head inference | 0.464 | 0.031 |
| Routing arithmetic | 0.0003 | 0.0001 |
| Stage-1 Platt application | 0.0002 | 0.0001 |
| Stage-2 isotonic application | 0.0000 | 0.0000 |
| Selected verifier call | 78.855 | 78.820 |
| **End to end** | **112.541** | **106.144** |
| part1 basis, for comparison | 88.189 | 86.485 |
| Uncharged by part1 | 24.352 | 19.658 |

Feature extraction spans two extractors and both are charged in full. The base extractor is
itself an exact sum of `feature_query_latency_ms` and `feature_document_setup_ms`, asserted in
preflight.

A fixed verifier computes no features and runs no heads, so its end-to-end cost is its call
plus the calibration applied to it — the tables report exactly that. This is correct accounting
rather than a handicap: the router pays for the features because it needs them to choose.

## Main results

Protocol A — 8,512 rows / 1,535 groups, ten 8/1/1 rotations over ten seeds:

| | AUROC | SD | End-to-end | ECE | Brier | AURC | BAcc | MCC | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **OURS** | **0.79330** | 0.00173 | **112.54 ms** | 0.02161 | 0.17262 | **0.14905** | 0.71951 | 0.45888 | 0.67083 |
| AlignScore | 0.78113 | 0.00131 | 677.29 ms | 0.02154 | 0.18128 | 0.15431 | 0.68446 | 0.40741 | 0.72390 |
| Qwen30-fast | 0.76893 | 0.00319 | 273.35 ms | 0.01440 | 0.17332 | 0.15661 | 0.71863 | 0.49619 | 0.70767 |

OURS is first of sixteen by AUROC and first by AURC. It is separable from fourteen of the
fifteen fixed verifiers. The exception is AlignScore: `+0.01217`, nominally
`[0.00038, 0.02426]`, but the table makes fifteen comparisons at once and the
Bonferroni-corrected interval is `[-0.00721, 0.02964]`, which contains zero. The honest reading
is that OURS holds the highest absolute AUROC and is indistinguishable from the strongest fixed
verifier under multiplicity correction, at a 6.02x latency ratio. Correction levels for every
comparison are in `02_main_tables/publication/A_PAIRED.csv`; Part 2's ablation tables report
both levels side by side.

Protocol B — TRAIN 5,276 / 890 to TEST 3,236 / 645, ten seeds:

| | AUROC | SD | End-to-end | ECE | Brier | AURC | BAcc | MCC | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen30-fast | 0.83384 | 0.00066 | 211.65 ms | 0.02506 | 0.15688 | 0.12914 | 0.77251 | 0.56311 | 0.73504 |
| **OURS** | **0.82256** | 0.00474 | **106.14 ms** | 0.03883 | 0.16855 | 0.14039 | 0.74153 | 0.50613 | 0.68465 |
| Qwen30-judge | 0.81500 | 0.00158 | 600.87 ms | 0.05597 | 0.16377 | 0.14312 | 0.75918 | 0.55993 | 0.76755 |
| AlignScore | 0.81167 | 0.00108 | 561.65 ms | 0.04345 | 0.17279 | 0.13772 | 0.73058 | 0.46608 | 0.74940 |
| Granite-3.1-2b | 0.79583 | 0.00316 | 131.81 ms | 0.05570 | 0.18276 | 0.17090 | 0.72866 | 0.48515 | 0.73198 |

Paired intervals for OURS minus the arm, source-group cluster bootstrap, 2,000 draws:

- Granite-3.1-2b, the strongest pool member: `+0.02672 [0.00916, 0.04315]`, **separable**,
  1.24x lower latency.
- AlignScore: `+0.01089 [-0.00625, 0.02753]`, indistinguishable, 5.29x faster.
- Qwen30-judge: `+0.00755 [-0.01044, 0.02477]`, indistinguishable, 5.66x faster.
- Qwen30-fast: `-0.01128 [-0.02875, 0.00552]`, **indistinguishable**, at 1.99x the latency.
- The remaining eleven are all separably worse.

In neither protocol is any fixed verifier both more accurate and cheaper than OURS, so the
operating point is not dominated. Under end-to-end accounting the only verifiers cheaper than
OURS are FactKB, FactCC and Lettuce-v2, all separably worse in quality.

## Calibration and thresholds

`03_threshold_and_risk/`. Every system, the router and all fifteen baselines alike, receives the
identical two-stage calibration, so ECE, MCE, Brier and thresholded metrics are comparable.
Both stage-1 and stage-2 probabilities are stored.

Isotonic regression is monotone but not order-preserving: PAVA pools the score regions where
the empirical label rate runs backwards, and pairs inside a pooled block become ties worth 0.5,
so a block contributing less than 0.5 gains. Stage 2 therefore moves AUROC in both directions
and moves it most for the weakest verifiers — HHEM gains `+0.01846` in A and `+0.03609` in B.
For the router it costs `-0.00353` in B, gains `+0.00248` in A where it also has to put ten
rotations on a common scale, and buys `-0.075` to `-0.080` ECE.

OURS does **not** have the lowest calibration error once the baselines get the same treatment:
Lettuce-v2 (0.01403) and Qwen30-fast (0.01440) are below it in A, Qwen30-fast (0.02506) in B.

Four threshold rules are selected on validation alone and applied identically to all systems.
`mcc` is primary — it hands each baseline its own validation-optimal operating point rather than
forcing a common cut, and it is what part1 used. For OURS the untuned 0.5 cut is nonetheless the
best operating point in both protocols:

| Protocol | Rule | Mean tau | Accuracy | BAcc | MCC |
|---|---|---:|---:|---:|---:|
| A | fixed05 | 0.5000 | **0.7407** | 0.7231 | **0.4644** |
| A | mcc (primary) | 0.5410 | 0.7379 | 0.7195 | 0.4589 |
| A | youden | 0.6570 | 0.7364 | **0.7331** | 0.4644 |
| A | conformal | 0.7982 | 0.7286 | 0.7335 | 0.4621 |
| B | fixed05 | 0.5000 | **0.7413** | **0.7469** | **0.5125** |
| B | mcc (primary) | 0.5449 | 0.7365 | 0.7415 | 0.5061 |
| B | youden | 0.7322 | 0.7325 | 0.7289 | 0.4694 |
| B | conformal | 0.7992 | 0.7202 | 0.7138 | 0.4499 |

That is what a calibrated output should do: under equal costs the Bayes threshold is 0.5, and
tuning only fits the validation prior, which shifts from a 62.98% positive rate on TRAIN to
47.81% on TEST. The primary rule stays `mcc` so that no operating point is reselected after
TEST was read; the comparison is reported as a result.

## Routing behaviour

| Protocol | FactCC | Lettuce-v2 | Granite-3.1-2b |
|---|---:|---:|---:|
| A | 28.61% | 52.41% | 18.98% |
| B | 34.34% | 34.74% | 30.91% |

No pool slot is dead. Beta mode is 0.1 in both protocols; per-fit grids are in
`04_beta_evidence/`. The per-corpus split stays extreme and is reported as fitted behaviour,
not as evidence of domain-invariant routing.

## Removed from the experiment

`OURS_current_pool` is gone. It evaluated a pool chosen by an earlier automatic search using
features selected for a different pool, which handicapped it. If the comparison is wanted, the
fair numbers are that search's own frozen run in `paper_v3/results/`: `0.78118 @ 538.99 ms` in
A and `0.80628 @ 465.15 ms` in B, against which the current pool is both more accurate and
about six times cheaper.

## Stale-content audit

Preflight asserts, and the log records, that: the inherited contract matches the expected pool,
features, target, learner and beta grid; the verifier set is exactly the fifteen declared in
`config_v2.VERIFIERS` and contains all pool members; paper_v1's default pool equals the current
pool while its nine-feature list and its `800/1/sqrt/10` default are unreachable because every
call site passes features, hyperparameters, target and actions explicitly; the inherited
contract's `comparison_pool` and `secondary_pool` fields are ignored. The two declared
training-contamination pairs, `(factkb, frank_valid)` and `(hhem, ragtruth_train)`, are out of
pool and are logged so table notes can cite them.

## Directory guide

- `00_contract/` — inherited frozen configuration, and the hyperparameters selected here
- `01_main_tables/publication/` — 16-row Protocol A/B tables and paired intervals
- `01_main_tables/with_sentinel/` — 17-row tables including `OURS_legacy_hp`
- `01_main_tables/calls/` — routing shares and latency, overall and per corpus
- `02_latency/` — end-to-end breakdown and the raw per-fit components
- `03_threshold_and_risk/` — threshold sensitivity, risk-coverage curves, stage-2 effect
- `04_beta_evidence/` — per-fit beta grids and selected values
- `05_hyperparameter_selection/` — the sixteen-candidate sweep with all four rankings
- `06_gates/` — drift sentinel results
- `07_provenance/` — run metadata, git state, data and code hashes, PID, DONE
- `../08_scripts/` — every executed source file
- `09_logs/` — complete stage logs
- `10_row_level/` — per-row stage-1 and stage-2 probabilities, both latency bases, per-row
  thresholds under four rules, and the router's selected action
- `MANIFEST.sha256`, `REPORT_zh.md`

## Limitations carried forward

- **Multiplicity.** Each main table makes fifteen comparisons against OURS. Under Bonferroni
  correction Protocol A holds fourteen of fifteen (AlignScore becomes indistinguishable) and
  Protocol B holds twelve of fifteen. The three that were already reported as indistinguishable
  in B — Qwen30-fast, Qwen30-judge, AlignScore — remain so at both levels, so no conclusion
  reverses. Cite the corrected level.
- **Pool provenance.** The pool was selected in paper_v1 on a 6,850-row matrix that shares 54.4%
  of Protocol B's TEST document groups; features and hyperparameters were later reselected on
  TRAIN alone but the pool never was. `pool_rescreen_v1` closes this by re-running the selection
  on TRAIN only, over all 455 three-verifier subsets, under the current features and
  hyperparameters. The result: the pool sits **on the quality-cost Pareto frontier** (twelve
  frontier pools of 455; seven among the ninety that satisfy the project's declared eligibility
  rule), it is the lowest-variance and highest-minimum-share pool on that frontier, but it ranks
  only **109 of 455** on validation AUROC alone. Every pool scoring higher costs at least 2.80x
  as much; every cheaper pool gives up at least 0.052 AUROC. The earlier "rank 1 of 540" is
  superseded and must not be quoted.
- **Two latency regimes for Granite-3.1-2b.** Its recorded latency differs by 1.5–1.7x between
  two measurement campaigns covering different corpora, and it carries 30.9% of Protocol B
  traffic. Normalising every row to the slower campaign moves OURS from 106.14 to 108.86 ms and
  Granite from 131.81 to 164.86 ms, i.e. the latency advantage over the strongest pool member
  widens from 1.24x to 1.51x. The tables report the as-measured figure, which is the
  conservative end. No system dominates OURS under either accounting.
- Protocol A's features were selected on the TRAIN subset of its own pooled frame, so A is a
  cross-validated estimate, not an independent confirmation. Protocol B is confirmatory.
- Feature selection scored candidates on pre-stage-2 routed AUROC and did not fit an isotonic
  layer per candidate. Features were not reselected for the new hyperparameters.
- Per-corpus, routing is not uniformly cheaper than a fixed policy: on UniSumEval OURS averages
  278.20 ms against 147.60 ms for always-Granite, because that corpus's long sources make
  FactCC's sliding-window count explode.
- `macro_auroc` is a diagnostic column throughout; pooled AUROC is the primary metric.
- Neither protocol establishes transfer to an unseen corpus family, and the cheap features
  carry a corpus fingerprint.

## Downstream

Part 2's ablation (`artifacts/part2_ablation_v1`) uses this archive as its reference system and
reproduces its OURS row exactly in both protocols. Three of its findings bear on how this
archive's tables should be read: stage-2 isotonic costs Protocol B `0.00353` AUROC while cutting
its ECE threefold, three of the six features are not individually separable from noise, and the
cost term's quality contribution is separable only in Protocol A.
