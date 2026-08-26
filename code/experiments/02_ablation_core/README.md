# Part 2 — Formal ablation

## Status

Authoritative ablation archive for the frozen Part 1 system. Source run:
`experiments/runs/part2_ablation_v1`. Reference system: `part1c_main_full_v1`.

Twenty-five arms, both protocols, ten seeds, plus one free arm recovered from
part1c's stored stage-1 probabilities. Every arm changes exactly one thing and
shares the reference system's splits, seeds, pool, features, supervision target,
learner, hyperparameters, beta grid, calibration and threshold rule.

Wall clock: 1 h 56 m on eight CPU cores.

## Attribution gate

Each arm's delta is only interpretable if the reference arm reproduces the main
experiment. It does, exactly, in both protocols:

| Protocol | AUROC | delta | Deterministic latency | delta |
|---|---:|---:|---:|---:|
| A | 0.7933003647 | `0.00e+00` | 88.189008 ms | `0.00e+00` |
| B | 0.8225560999 | `0.00e+00` | 86.485309 ms | `1.42e-14` |

Only the deterministic part of the latency is gated — the selected verifier's
recorded latency plus the recorded feature-extraction cost. Head inference, the
routing arithmetic and the calibration application are wall-clock measurements
that differ between runs by microseconds and cannot be bit-identical; they are
reported in the end-to-end total but not gated. See `02_gates/`.

## Multiplicity

Twenty-five comparisons against the reference system, so both the nominal 95%
interval and the Bonferroni-corrected interval (alpha = 0.002, i.e. the 99.8%
interval) are reported for every arm. Both come from a single cluster bootstrap
pass resampled by `content_doc_key`, 2,000 draws, same seed as the main tables.
Separable at 95% / under Bonferroni: 20 and 17 of 25 in Protocol B, 19 and 18 of
25 in Protocol A.

Claims in the paper should cite the Bonferroni column.

## What the ablation establishes

Deltas are `arm minus reference`, so negative means the arm is worse.

### The routing decision is the dominant effect

| Arm | B | A |
|---|---:|---:|
| Random routing, serving only | −0.16609 | −0.14979 |
| Random routing, everything refitted on random routing | −0.16025 | −0.14325 |

Both are Bonferroni-separable and are by far the largest effects in the table.
The two variants matter jointly: `random_routing_serving` keeps beta, the
isotonic layer and the threshold fitted on *real* routing and only randomises
the action at serving time, which leaves open whether the loss is really about
action choice or about calibration built on a different distribution.
`random_routing_refit` refits beta, isotonic and the threshold on the randomised
routing as well, and recovers only 0.006. The effect is the assignment of
instances to verifiers, not a calibration artefact.

### Every pool member is load-bearing, and one is irreplaceable

| Member | B serving | B retrain | A serving | A retrain |
|---|---:|---:|---:|---:|
| Granite-3.1-2b | −0.17629 | −0.16425 | −0.10244 | −0.09893 |
| Lettuce-v2 | −0.08604 | −0.02255 | −0.09566 | −0.05030 |
| FactCC | −0.05377 | −0.01840 | −0.06703 | −0.03843 |

The two removal semantics are reported separately and answer different
questions. `drop_serving` masks the member at serving time with the heads still
trained on the full pool, which is the production question of a verifier going
down. `drop_retrain` removes it from the pool and refits the heads, beta and
both calibration stages on the remainder, which is the design question of
whether the member earns its slot.

Refitting recovers most of the loss for FactCC and Lettuce-v2 — they are partly
substitutable — and almost none of it for Granite-3.1-2b. No pool slot is dead.

### Verifiers and stage-1 calibration are both necessary

`no_verifier`, where the cheap features predict the label directly and no
verifier is called at all, costs −0.02589 (B) and −0.04405 (A) while dropping
latency to 27.30 ms and 33.37 ms. `no_stage1`, routing on raw verifier scores
with no per-verifier Platt map, costs −0.04368 (B) and −0.03232 (A). Both are
Bonferroni-separable.

## Three negative results

### 1. Stage-2 isotonic costs ranking quality in Protocol B

| Protocol | Arm | AUROC delta | ECE |
|---|---|---:|---:|
| B | `no_stage2` | **+0.00353**, Bonferroni-separable | 0.11393 vs 0.03883 |
| A | `no_stage2` | −0.00248, separable at 95% only | 0.10196 vs 0.02161 |

Removing the second calibration stage *improves* Protocol B's AUROC by a
Bonferroni-separable margin, and triples its calibration error. Isotonic
regression is monotone but not order-preserving: PAVA pools the score regions
where the empirical label rate runs backwards, and every pair inside a pooled
block becomes a tie worth 0.5, which can move AUROC in either direction. In
Protocol A the stage also has to place ten rotations on a common scale, and
there it helps.

Stage 2 must therefore be described as a calibration component that costs a
small, measurable amount of ranking quality — not as a component that improves
the system.

### 2. Only one to three of the six features are individually separable

Bonferroni-separable single-feature removals:

| Feature | B | A |
|---|---:|---:|
| `structured_source_line_ratio` | −0.03025 | −0.01387 |
| `bm25_mean3` | −0.01489 (95% only) | −0.02616 |
| `entity_coverage` | −0.00053 (not separable) | −0.00525 |
| `entity_value_colocation` | −0.00081 (not separable) | −0.00114 (not separable) |
| `year_count` | +0.00104 (not separable) | −0.00042 (not separable) |
| `conflicting_value_rate` | −0.00010 (not separable) | +0.00063 (not separable) |

Three of the six are not separable from noise in either protocol, and two of
them are nominally positive when removed in Protocol B.

The group arms keep the feature set defensible: dropping either extractor's
contribution is separable in both protocols — `keep_base3_only` costs −0.03696
(B) and −0.02862 (A), `keep_compact16_3_only` costs −0.02638 (B) and −0.02743
(A). The correct claim is that the feature set is not reducible to a single
extractor, while individual attribution beyond one to three features is not
supported.

### 3. The cost term's quality contribution is established only in Protocol A

| Protocol | `no_cost` AUROC delta | Separable | Latency |
|---|---:|---|---:|
| A | −0.03588 | Bonferroni | 189.59 vs 112.54 ms (+68%) |
| B | −0.01352 | 95% only, **not** Bonferroni | 142.74 vs 106.15 ms (+34%) |

Pinning beta to zero is a large latency regression in both protocols and a
separable quality regression only in Protocol A. The cost term is primarily a
budget control that also regularises over-selection of the expensive action; its
quality contribution should not be claimed as established in Protocol B.

## Supervision target

| Target | B | A |
|---|---:|---:|
| `regret` (reference) | 0 | 0 |
| `brier` | −0.00466 (not separable) | −0.00286 (not separable) |
| `prob_quality` | −0.00825 (Bonferroni) | −0.00078 (not separable) |
| `pairwise_rank` | −0.01095 (Bonferroni) | **+0.00106** (not separable) |
| `is_best` | −0.03467 (Bonferroni) | −0.01525 (Bonferroni) |
| `correctness` | −0.05699 (Bonferroni) | −0.03118 (Bonferroni) |

Continuous relative-quality supervision beats binary supervision by 0.015 to
0.057, separably in both protocols, which is the claim the target design
supports. Among the continuous variants, `regret` is separably best only in
Protocol B; in Protocol A `pairwise_rank` is nominally higher and `brier` and
`prob_quality` are indistinguishable. Optimality of `regret` within the
continuous family is not established.

`correctness` requires per-verifier hard-decision correctness, which the frozen
matrix does not carry, so it is derived from the recorded `decision__*` columns
against the label. This is the only arm whose inputs are computed rather than
read.

## Directory guide

- `00_contract/` — the reference system's frozen configuration and selected hyperparameters
- `01_ablation_tables/` — per-protocol arm tables and paired intervals with both correction levels
- `02_gates/` — reference-arm reproduction results
- `03_provenance/` — launch record, hashes, PID, DONE
- `../08_routing_code/` — every executed source file
- `05_logs/` — complete stage logs
- `06_row_level/` — per-arm, per-seed probabilities, latencies and thresholds
- `MANIFEST.sha256` — archive integrity hashes
- `REPORT_zh.md` — Chinese report

## Scope

Arms not covered here, and not claimed: cascade and second-call families,
learner sensitivity, feature-noise robustness, sample efficiency, oracle upper
bounds, and corpus-transfer. The earlier notes contain such tables produced
under a superseded configuration; none of their numbers apply to the current
contract.
