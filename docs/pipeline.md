# The pipeline, end to end

Four stages. Each one writes a frozen output that the next reads, and every frozen output
this release depends on is shipped, so stage 3 runs on its own.

```
   corpora                     stage 1                 stage 2                stage 3            stage 4
 ┌──────────┐   ingest   ┌────────────────┐  score  ┌──────────────┐  route ┌──────────┐  live ┌──────────┐
 │ CoGenSumm│──────────► │ TRAIN 5,276    │───────► │ 15 verifiers │──────► │  router  │─────► │ controls │
 │ FRANK    │            │ TEST  3,236    │         │ score matrix │        │  tables  │       │   sig    │
 │ RAGTruth │            │ grouped by     │         │ + latency    │        │ ablations│       │ few-shot │
 │UniSumEval│            │content_doc_key │         │ + protocols  │        │  cascade │       │ dataset  │
 └──────────┘            └────────────────┘         └──────────────┘        └──────────┘       └──────────┘
                                  │                        │                      ▲
                                  └────────────────────────┴──────────────────────┘
                                       shipped frozen in 00_inputs/ (text removed)
```

## Stage 1: ingest

`code/ingest_and_scoring/ingest/build_splits_v2.py`

Unifies four summary-factuality datasets into one summary-level binary classification task.
The grouping unit is `content_doc_key`, derived from the normalised source-document content,
so every summary of one source document lands in one group and no group is split across the
train and test sides.

Before any model was fitted, five RAGTruth training rows were removed under a pre-declared
content-hash rule because their canonical documents also occur in the CoGenSumm test split.
The test side is left intact.

Output: `TRAIN.parquet`, `TEST.parquet`, `TEST_SCORING.parquet`, `P1_SCORING_COHORT.parquet`.

Needs the four corpora. They are not redistributed here.

## Stage 2: verifier scoring

`code/ingest_and_scoring/p1_score_local.sh` (nine local verifiers)
`code/ingest_and_scoring/p1_api.sh` (six vLLM-served verifiers)
`code/ingest_and_scoring/p1_score.py`, `code/ingest_and_scoring/ingest/verifier_cli.py` (entry points)
`code/verifier_wrappers/` (the scorers themselves)

Every verifier is run once per row under its own declared protocol, recorded in
`PROTOCOLS` in `code/verifier_wrappers/unified_summary_verifiers_v1.py`. The protocol string fixes the
source-window size, the overlap, the aggregation rule, and whether the official prompt
prefix is applied. It is part of the measurement: two verifiers with different window caps
are not measured under the same context budget, and the string says so.

The two batches do not run concurrently. A vLLM server sized with
`--gpu-memory-utilization` cannot coexist with a local encoder that is already holding
memory on the same device, so `p1_api.sh` waits for the local batch to finish.

Output: one parquet per verifier, carrying the score, the decision, availability, the
measured latency, and the token accounting. Fifteen files.

Needs a GPU, the model weights, and vLLM. See [`verifiers.md`](verifiers.md).

## Stage 3: routing

`code/experiments/08_routing_code/` — **[read that directory's README for the file-by-file
walkthrough](../code/experiments/08_routing_code/README.md).** This section is the summary; that
one explains what the code in each file does.

This is the stage the work is about, and the only stage that runs from this repository alone,
because stages 1 and 2 are shipped frozen in `code/experiments/00_inputs/` with all source and
summary text removed.

Twelve files: three shared modules and five experiment scripts, plus four launchers.

### The three shared modules, in reading order

`config.py` locks the configuration — pool, features, target, learner, hyperparameters, the β
grid and its tolerance, the seeds — in one place. Its docstring records why: in an earlier round
four scripts called `fit_heads()` without `hp=` and silently fell back to a different forest
than the paper claimed.

`v3core.py` states and enforces the data boundary. `load(with_test_labels)` is the gate on the
confirmatory read; `stratified_group_split` and `folds_stratified` are group-disjoint on
`content_doc_key` and stratified by corpus and majority label; `rotations` is the 8/1/1
contract; `route` and `choose_beta` are the routing implementation every stage shares.

`core.py` is one implementation of every other step: training-only cost estimation, both
calibration stages, every supervision target the ablation compares, head fitting, the metrics,
the threshold rules, and the three bootstraps. Two details matter downstream —
`make_classifier` raises on an unknown learner name rather than substituting a forest, and
`make_frame(..., charge_features)` decides who pays for the feature extractors, which is what
keeps the latency comparison honest.

### The five experiment scripts

| Script | Stages | Produces |
|---|---|---|
| `part1c_main_full_v1.py` | `preflight`, `hpselect`, `protoB`, `protoA`, `report` | The main tables. Refuses to run on a drifted contract, selects hyperparameters on TRAIN only by minimum validation head loss, runs both protocols over ten seeds, charges end-to-end latency, applies four threshold rules, stores row-level probabilities. |
| `part2_ablation_v1.py` | `protoB`, `protoA`, `report` | 25 arms, each changing exactly one thing: β, either calibration stage, the routing signal, a verifier, a feature, the target. |
| `part3_extended_v1.py` | ten, in two phases | Per-corpus training, the 64-subset feature lattice with exact Shapley, data-size and forest-size convergence curves, and six pre-declared arms with row-level output. |
| `part3_percorpus_selected_v1.py` | one | The deployable per-corpus competitor: pick one fixed verifier by *validation* AUROC rather than by the oracle maximum over test AUROC. |
| `part4_cascade_v1.py` | `protoB`, `protoA`, `report` | 13 competitor policies — three cascades, three second-call rules, seven alternative learners — under a 1.5x latency ceiling. |
| `pool_rescreen_v1.py` | one | Pool provenance: re-screens all three-verifier pools on TRAIN alone, as an audit that does not feed back. |

### What they all share

Every script inherits the frozen contract and asserts it field by field before fitting anything;
every script carries part1c's `OURS` as an arm and must reproduce it before its own numbers are
believed; every arm differs from the reference in exactly one respect; and every run writes code
hashes, git state, the launch command and a completion marker next to its tables.

## Stage 4: live re-run and controls

`code/experiments/09_live_and_controls/code/`

| Script | Question it answers |
|---|---|
| `live_main.py`, `live_pipeline.py` | Is the reported test result read from a pre-computed matrix? Runs Protocol B with every verifier called for real, per row and per seed. |
| `dataset_control.py`, `ds_only.py` | Does the router receive dataset identity? Four arms, including one that hands it the corpus explicitly. |
| `fewshot_frac.py` | How much of its own pool does a held-out corpus need? Sweeps fractions, not counts, because the four pools differ by 5.6x. This is the curve the paper plots. |
| `fewshot.py`, `fewshot_k0.py` | The superseded absolute-count sweep and its k=0 point, kept because the archived curve came from them. |
| `sig_main.py` | Paired cluster bootstrap over `content_doc_key`, 2,000 draws, Bonferroni over fifteen comparisons. |
| `sig_controls.py` | The same test applied to the three controls, with family sizes 1, 3 and 4. Refits every arm, because the original runs saved summaries but not row-level probabilities. |
| `rerun_score.py`, `cmp.py`, `diag_live.py` | Re-scoring diagnostics used to trace the live-versus-matrix residual. |
| `_contract.py` | The guard every script here inherits. It refuses to run if the frozen configuration has drifted, and it is what the release gate calls. |

`_contract.py` is the mechanism that makes the rest of this trustworthy. Any arm that is
nominally the main system must reproduce `FROZEN_B_AUROC` exactly, or it raises rather than
reporting numbers for a different system.
