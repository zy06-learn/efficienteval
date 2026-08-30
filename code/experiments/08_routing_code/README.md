# Stage 3 — the routing code

This is the stage the work is about, and the only one that runs from this repository alone:
stages 1 and 2 are shipped frozen in [`../00_inputs/`](../00_inputs/) with all source and
summary text removed. Everything in `../01_main_experiment/` through
`../05_pool_provenance/` was produced by the twelve files here.

Read them in this order. The first three are 800 lines together and everything else assumes
them; the five experiment scripts can then be read in any order, because none of them imports
another.

```
  config.py      what is locked, and why locking it was necessary
  v3core.py      the data boundary, and one implementation of routing
  core.py        one implementation of every other step
        │
        ├── part1c_main_full_v1.py         the main tables            ─┐
        ├── part2_ablation_v1.py           25 one-thing-changed arms   │  each asserts the
        ├── part3_extended_v1.py           per-corpus, lattice, curves │  frozen contract,
        ├── part3_percorpus_selected_v1.py the deployable competitor   │  each must reproduce
        ├── part4_cascade_v1.py            13 competitor policies      │  part1c's OURS
        └── pool_rescreen_v1.py            pool provenance audit      ─┘
```

---

## The shared layer

### `config.py` — 116 lines

The single source of truth for the locked configuration: pool, the six features, target,
learner, hyperparameters, the β grid and its tolerance, the seeds. Experiment scripts import
these; ablation grids stay local to the script that sweeps them.

Its docstring records why this file exists rather than each script carrying its own defaults.
In an earlier round four scripts called `fit_heads()` without passing `hp=` and silently fell
back to a different random forest than the paper claimed, and two of those also used a β rule
that had already been shown broken. Centralising removes the possibility instead of relying on
reviewer attention.

### `v3core.py` — 203 lines

**Read this first if you read only one file.** Its docstring states the data boundary the whole
study rests on, and the module enforces it rather than leaving it to convention:

```
SELECT      TRAIN only, 5,276 rows / 890 document groups. Features, target, learner,
            hyperparameters, pool, the beta rule and both calibration stages are chosen
            on this and nothing else.
Protocol A  pooled 8/1/1 rotation over TRAIN + TEST, 8,512 rows / 1,535 groups.
Protocol B  fit on TRAIN, read TEST once. Confirmatory.
```

| Function | What it does |
|---|---|
| `load(with_test_labels)` | Attaches the stage-2 scores to the label-free test frame and returns `TRAIN`, `TEST`, `ALL`, `verifiers`. The flag is the gate on the confirmatory read — `hpselect` calls it with `False` and asserts `label_supported` is absent. |
| `stratified_group_split(frame, seed, fraction)` | The fit/validation split. Group-disjoint on `content_doc_key`, stratified by corpus and by majority label. Returns a boolean mask where `True` is the held-out part. |
| `folds_stratified(frame, seed, n_folds)` | The same construction as ten folds, for Protocol A. |
| `rotations(n_folds)` | Yields `(test fold, validation fold, training folds)` — the 8/1/1 contract, ten times, so every row gets exactly one out-of-fold prediction. |
| `route(frame, heads, cal, beta, actions, cost_vec, mask)` | `U_ij = T̂_ij · exp(−β·c_j/c_max)`, argmax over available actions, unavailable set to `−∞`. Returns the selection, the selected calibrated probability, and the mask. |
| `choose_beta(val, heads, cal, actions, cost_vec, eps)` | The cheapest β on the grid whose validation AUROC is within `eps = 0.005` of the best. |

### `core.py` — 503 lines

Every other step, once. Fitting a head, choosing β, computing a metric — there is exactly one
place each of those happens, because the failure mode this replaced is invisible in any single
script's output and only surfaces when two tables disagree.

| Group | Functions | Notes |
|---|---|---|
| Cost | `fold_costs`, `verifier_cost` | Mean action cost estimated **only from the current training rows**. The evaluation side never contributes a cost estimate. |
| Splits | `pooled_folds`, `pooled_rotations`, `dataset_folds`, `dataset_rotations` | Group-disjoint throughout. The `dataset_*` pair is the 3/1/1 within-corpus protocol Part 3 uses. |
| Calibration | `platt`, `apply_platt`, `isotonic` | Stage 1 is per-verifier Platt fitted on the fit partition; stage 2 is one isotonic layer fitted on validation and applied identically to the router and to all fifteen baselines. The router emits whichever member it selected, so its raw output is a mixture of three separately-calibrated scales — that is what stage 2 is for. |
| Supervision | `targets(frame, cal, actions, kind)` | `Y[:, j]` is what head `j` should learn. `regret` is the locked choice; `prob_quality`, `brier`, `pairwise_rank`, `is_best` and `correctness` are here so Part 2 can ablate the choice rather than assert it. |
| Heads | `make_regressor`, `make_classifier`, `fit_heads` | One head per action, fitted only on rows where that action was available. `make_classifier` **raises** on an unsupported learner name instead of silently substituting a forest. |
| Routing | `choose_beta`, `route` | |
| Accounting | `make_frame(..., charge_features)` | `charge_features` must be `False` for anything that does not compute the cheap features. A fixed verifier and an ensemble do not, and folding the router's 8.6 ms into their cost would flatter the router. |
| Metrics | `ece`, `risk_coverage`, `metrics` | `metrics` returns every number the paper reports for one system on one seed. `risk_coverage` sorts by `|p − 0.5|` descending and integrates selective error over fifty coverage points. |
| Thresholds | `conformal_tau`, `group_conformal_tau` | The group version calibrates on one worst-case unsupported score per source group, so the calibration unit is the independent document rather than a correlated summary row. |
| Intervals | `paired_bootstrap`, `paired_cluster_bootstrap`, `cluster_bootstrap_auroc` | The cluster versions resample `content_doc_key`. Summaries of one source document are not independent, and resampling rows would treat them as if they were, understating the variance. |

### `tenfold_v1.py` — 131 lines

The 8/1/1 rotation contract as a standalone module: eight folds fit the heads, one fold fits
the Platt calibrators **and** selects β (never used to fit trees), one fold is evaluated once
with everything already frozen. Rotating ten times gives every summary exactly one test-fold
prediction. Imported by the frozen pipeline under this name; see
[Two duplicated modules](#two-duplicated-modules).

---

## The experiments

### `part1c_main_full_v1.py` — 886 lines — the main tables

Five stages, run in order by the launcher: `preflight | hpselect | protoB | protoA | report`.

**`preflight` (101 lines) — refuse to run on a drifted system.** Before anything is fitted it
checks, and raises rather than warns, that:

- every field of the inherited contract matches `EXPECTED_CONTRACT` — pool, features, target,
  learner, β grid, seeds, split construction;
- the shared module's *defaults* agree with the contract, and it logs the stale-default audit
  explicitly: the shared layer still carries a larger feature list and different hyperparameters
  from an earlier round, and the log records that every call site in this file passes
  `features=`, `hp=`, `target=` and `actions=` explicitly, so none of it is reachable;
- the verifier set has not drifted and every pool member is present;
- the six row and group counts match the declared numbers exactly (`5,276 / 890 / 3,236 / 645`
  and their sums);
- no document straddles TRAIN and TEST;
- the latency columns both feature extractors write are present in all three frames.

**`hpselect` (135 lines) — the only thing this run selects.** Sixteen randomly initialised
candidates (`n_estimators ∈ {200,400,800}`, `min_samples_leaf ∈ {1,2,5,10}`,
`max_features ∈ {sqrt,log2,0.5}`, `max_depth ∈ {6,10,12,16,24}`), deduplicated, plus part1's two
values as reference points. Each is scored on **TRAIN only** — `load(with_test_labels=False)`,
with an assertion that the test frame carries no label — under that protocol's own CV shape,
and the winner is the minimum **validation head loss**, not the reported metric.

The trade-off is written into the docstring rather than hidden: head loss and routing AUROC are
positively rank-correlated here, so the minimum-loss region is the shallow-forest region. Both
the loss *and* the AUROC of every candidate go into `HP_SELECTION.csv` so the cost of the rule
is visible. Selected: Protocol A `800 / leaf 5 / 0.5 / depth 6`, Protocol B `200 / leaf 10 /
0.5 / depth 6`.

**`protoB` / `protoA` — the runs.** `_run_seed` puts one `(fit, val, eval)` triple through every
system and returns per-system dictionaries; `proto_b` fits on TRAIN and reads TEST once,
`proto_a` walks the ten rotations. Row-level probabilities are stored, which is what lets
[`sig_main.py`](../09_live_and_controls/code/sig_main.py) compute the main tables' intervals
without refitting anything.

**Latency accounting** is the change this file's name refers to, and it is worth reading
closely, because it is the number the paper's claim rests on:

| Function | Charges |
|---|---|
| `_pre_call_feature_ms` | Both extractors in full: `feature_latency_ms` (itself the exact sum of `feature_query_latency_ms` and `feature_document_setup_ms`) plus `compact16_feature_latency_ms`. The six frozen features span two extractors, so both are charged. |
| `_verifier_ms` | The recorded latency of the action actually selected, per row. |
| `_stage2` | Fits isotonic on validation, applies it to the evaluation rows, and **measures** the per-row application cost with `perf_counter`. Also returns the four threshold rules computed on both stage-1 and stage-2 probabilities. |
| `_assemble` | The router pays `verifier + features + heads + routing + Platt + isotonic`. A fixed verifier computes no features and runs no heads, so it pays `verifier + calibration`. That is correct accounting, not a handicap. |

**`OURS_legacy_hp` — the drift sentinel.** The same router under part1's frozen hyperparameters,
which must reproduce part1's numbers exactly. Since hyperparameters are the only thing that
changed between part1 and part1c, this arm is the check that the primitives — splits, Platt,
the regret target, head fitting, routing, β selection, isotonic — have not moved.

**`report` (129 lines)** writes the publication tables, per-corpus breakdowns, risk–coverage
curves, the β evidence, and the provenance block (`_sha256` over the code, `_git` state).

### `part2_ablation_v1.py` — 478 lines — 25 arms, one thing changed each

Everything is inherited from part1c and asserted, and **`OURS` here must reproduce part1c
exactly — that equality is the gate.** `arm_list()` builds the arms; the published family is 25
comparisons against `OURS`:

| Group | Arms | What changes |
|---|---|---|
| cost | `no_cost` | β pinned to 0 |
| calibration | `no_stage1` | no per-verifier Platt; route on raw scores |
| | `no_stage2` | the free arm — read straight from part1c's stored stage-1 probabilities, no refit |
| routing | `random_routing_serving` | head outputs replaced by noise at serving only |
| | `random_routing_refit` | noise at validation too, so β, isotonic and the threshold are all fitted on random routing |
| verifiers | `no_verifier` | the cheap features predict the label directly; nothing is called |
| | `drop_retrain::X` ×3 | X removed, heads / β / calibration refitted |
| | `drop_serving::X` ×3 | X masked at serving only; the heads still know it |
| features | `drop_feature::F` ×6 | one feature removed, heads refitted, **no substitute feature searched for** |
| | `keep_base3_only`, `keep_compact16_3_only` | only one extractor's three features |
| target | `target::T` ×5 | only the supervision target changes |

`run_arm` (64 lines) is the dispatcher: it reads the spec dict, overrides exactly one of
`actions` / `features` / `target` / `platt` / `beta` / `mask` / `random`, and runs the rest of
the pipeline unchanged.

Two accounting rules the docstring states and the code enforces: arms using fewer features
**still pay the full extraction cost**, because the extractors are not per-feature; arms that
call no verifier pay no verifier cost.

### `part3_extended_v1.py` — 772 lines — per-corpus, lattice, convergence

Ten stages in two phases, because Protocol A is expensive:

```
PHASE1  prep  percorpus  latticeB  convergeB  declaredB  reportB
PHASE2                   latticeA  convergeA  declaredA  reportA
```

| Stage | What it computes |
|---|---|
| `prep` | Per-corpus label-prior shift and feasibility. Nothing is fitted. |
| `percorpus` | Within-corpus training and evaluation on the 3/1/1 protocol, against all fifteen fixed verifiers. `_pooled_reference` also restricts part1c's pooled Protocol B result to each corpus's evaluation rows, so the two settings are comparable. |
| `lattice` | All 2⁶ = 64 feature subsets, and the pool subsets. The lattice contains the reference **twice** — once as the full feature subset, once as the full pool subset — and both must reproduce part1c bit for bit. That is the attribution gate. |
| `_shapley_and_bestk` | Exact Shapley value over the full lattice, plus the best subset at each size. |
| `converge` | Data-size curve over `SIZE_FRACTIONS = 0.05 … 1.00` of the fit partition's document groups, and forest-size curve over `TREE_GRID = 1 … 800`. |
| `declared` | Six pre-declared arms: reference, two legacy feature sets (`A9`, `B5`), call-proportion-matched random routing, and two label-oracle heads. These **store row-level predictions** so paired intervals are meaningful. |
| `report` | Gate, Shapley, best-k, and paired intervals for the declared contrasts. |

**Why Shapley and not 64 significance tests**, from the docstring: the six features are
redundant, and Part 2 already showed only one to three survive a 25-comparison correction,
because removing a feature whose information another also carries produces no measurable loss.
Testing 64 subsets would need α = 0.00078 and would return "not separable" almost everywhere —
a property of the design, not a finding. The average marginal contribution over all subsets
containing a feature is the quantity that actually answers "what is this feature worth" under
redundancy, and 64 subsets is exactly the material to compute it exactly at n = 6. So the
lattice stores aggregates only, and paired intervals are reported for the small declared family
(five comparisons) where they mean something.

`route_and_score` (61 lines) is this file's end-to-end fitted router; `no_verifier_arm` is the
features-predict-the-label control; `_fit_heads_models` mirrors `core.fit_heads` but also
returns the fitted forests, which the Shapley pass needs.

### `part3_percorpus_selected_v1.py` — 166 lines — the deployable competitor

The per-corpus table reports, for each corpus, the best fixed verifier found by taking the
**maximum test AUROC** over fifteen candidates. That is an oracle: no deployable system knows in
advance which verifier will win on a corpus it has not evaluated. Left as the only comparison,
the table reads as "a fixed verifier beats the router", which is not what it shows.

This adds the honest version. For each corpus and seed, using exactly the fit/validation split
the per-corpus table used, it selects one fixed verifier by **validation** AUROC — even when the
candidates are restricted to the three pool actions — and evaluates that choice. It beats
within-corpus routing on all four corpora, by `0.05965 / 0.01172 / 0.01532 / 0.02624`, with
highly consistent selection across the ten seeds. The paper reports that result.

### `part4_cascade_v1.py` — 560 lines — 13 competitor policies

The comparisons the ablation left open, in three families. From the docstring: the obvious
question about a router that chooses before paying is why not start with the cheapest verifier
and escalate when unsure — that is the cascade family, and it is the comparison a reviewer
reaches for first.

| Family | Arms | Policy |
|---|---|---|
| cascade | `confidence` | run the cheapest verifier; if it is not confident, call the highest-head-value remaining one |
| | `disagreement` | run the two cheapest; if their hard decisions disagree, add the most expensive |
| | `learned_deferral` | run the cheapest; escalate when a model trained on the features predicts it will be wrong |
| second call | `verifier_confidence`, `raw_margin`, `discounted_margin` | keep the router's choice, then add the best remaining action when the corresponding signal is weak |
| learner | `extra_trees`, `hgb`, `gbr`, `ridge`, `tree`, `knn`, `mlp` | the head family changes, nothing else |

`Fit` (41 lines) is one fitted stack — Platt, heads, cost vector, β — that every arm is built on
top of, so the arms differ only in the policy layered over it. `_pick_tau` selects the threshold
with the highest validation AUROC **among those whose validation latency respects
`LATENCY_CEILING = 1.5×` the reference**, which is what keeps a cascade from buying accuracy
with unbounded cost.

The gate here is tighter than elsewhere: `TOL_AUROC = 1e-9`, `TOL_MS = 1e-6` against part1c's
`OURS`, compared on AUROC and on `ms_det` — the reproducible part of the latency (the selected
verifier's recorded latency plus recorded feature extraction), because head inference, the
routing arithmetic and the calibration application are wall-clock measurements that differ by
microseconds between runs.

Part 4 runs in its own directory and writes nowhere Parts 2 or 3 write; the only thing it reads
from elsewhere is the frozen reference contract.

### `pool_rescreen_v1.py` — 233 lines — pool provenance

The one disclosed exception to the TRAIN-only boundary. The pool in use was selected in an
earlier round on a 6,850-row matrix that shares **54.4%** of Protocol B's test document groups.
Features and hyperparameters were later reselected on TRAIN alone; the pool never was, so under
the project's own grouping criterion the pool choice is not independent of the confirmatory test
set.

This script re-screens **every** three-verifier pool under the current frozen configuration,
with selection restricted to TRAIN, and reports where the frozen pool ranks. It is an audit: it
does not feed back into the frozen configuration, and the paper says so.

---

## The launchers

| Script | Usage |
|---|---|
| `run_part1c_full_v1.sh` | `run_part1c_full_v1.sh <run_dir> [smoke]` |
| `run_part2_ablation_v1.sh` | `run_part2_ablation_v1.sh <run_dir> [smoke]` |
| `run_part3_v1.sh` | `run_part3_v1.sh <run_dir> <phase1\|phase2> [smoke]` — the phase is mandatory, there is no default |
| `chain_part4.sh` | Waits for Part 3 phase 2's completion marker under `$AFR_ROOT/experiments/runs`, then starts Part 4 in its own directory |

All four are reachable through [`../../../reproduce.sh`](../../../reproduce.sh), which sets
`AFR_ROOT` and `AFR_INPUTS` from its own location and writes runs where `chain_part4.sh` looks
for them. `tests/test_reproduction.py::test_reproduce_sh_matches_the_launchers` checks that the
entry point and these scripts still agree on arguments and paths.

Every script takes an optional `smoke` flag that shortens the run — two rotations instead of
ten, three features in the lattice instead of six — for checking that a stage runs end to end
before committing hours to it.

## Two duplicated modules

`summary_router_compact16_direct_v1.py` (1907 lines, the larger feature extractor and the
grouped-fold builder), `pool_gate_sweep_v1.py` (433) and `tenfold_v1.py` (131) exist as
**byte-identical copies** in both this directory and [`../../verifier_wrappers/`](../../verifier_wrappers/).
The two trees import them under different module names, and the frozen pipeline resolves one
name from each tree. Removing either copy would change which module loads, so both stay.

## What every experiment here has in common

1. **Inherit and assert.** Read the frozen contract, compare it field by field, raise on drift.
2. **Reproduce the reference first.** Every script carries part1c's `OURS` as an arm and gates
   on reproducing it before its own results are believed.
3. **Change one thing.** Arms differ from the reference in exactly one respect, and the shared
   layer is what makes that guarantee checkable rather than aspirational.
4. **Charge everything.** End-to-end latency on the same basis for every arm, with
   `charge_features` deciding who pays for the extractors.
5. **Write the provenance.** Code hashes, git state, the launch command and a completion marker,
   next to the tables.
