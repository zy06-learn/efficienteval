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

`code/paper_v2/ingest/build_splits_v2.py`

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

`code/paper_v2/p1_score_local.sh` (nine local verifiers)
`code/paper_v2/p1_api.sh` (six vLLM-served verifiers)
`code/paper_v2/p1_score.py`, `code/paper_v2/ingest/verifier_cli.py` (entry points)
`code/afr_v2/` (the scorers themselves)

Every verifier is run once per row under its own declared protocol, recorded in
`PROTOCOLS` in `code/afr_v2/unified_summary_verifiers_v1.py`. The protocol string fixes the
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

`code/paper_v3/DELIVERABLE/08_scripts/`

This is the stage the work is about, and the only stage that runs from this repository
alone, because stages 1 and 2 are shipped frozen in
`code/paper_v3/DELIVERABLE/00_inputs/` with all source and summary text removed.

| Script | Produces |
|---|---|
| `part1c_main_full_v1.py` | the main tables, both protocols, latency, threshold and risk, β evidence, hyperparameter selection, gates |
| `part2_ablation_v1.py` | core ablations: feature subsets, target, learner, pool |
| `part3_extended_v1.py` | extended ablations and the feature lattice |
| `part3_percorpus_selected_v1.py` | per-corpus tables |
| `part4_cascade_v1.py` | cascade variants and alternative learners |
| `pool_rescreen_v1.py` | pool provenance: re-screening the k=2 and k=3 subsets |
| `v3core.py` | the shared layer: data boundary, β selection, routing, one implementation used by every stage above |

`v3core.py` is worth reading first. It states the data boundary the whole study rests on:
every choice is made on TRAIN, Protocol A is a cross-validated estimate over TRAIN and TEST
pooled, and Protocol B reads TEST once. The docstring also records what the previous round
got wrong, which is why the boundary is enforced in code rather than by convention.

## Stage 4: live re-run and controls

`code/paper_v3/DELIVERABLE/09_live_and_controls/code/`

| Script | Question it answers |
|---|---|
| `live_main.py`, `live_pipeline.py` | Is the reported test result read from a pre-computed matrix? Runs Protocol B with every verifier called for real, per row and per seed. |
| `dataset_control.py`, `ds_only.py` | Does the router receive dataset identity? Four arms, including one that hands it the corpus explicitly. |
| `fewshot.py`, `fewshot_k0.py` | How many in-corpus examples does a held-out corpus need before it improves? |
| `sig_main.py` | Paired cluster bootstrap over `content_doc_key`, 2,000 draws, Bonferroni over fifteen comparisons. |
| `rerun_score.py`, `cmp.py`, `diag_live.py` | Re-scoring diagnostics used to trace the live-versus-matrix residual. |
| `_contract.py` | The guard every script here inherits. It refuses to run if the frozen configuration has drifted, and it is what the release gate calls. |

`_contract.py` is the mechanism that makes the rest of this trustworthy. Any arm that is
nominally the main system must reproduce `FROZEN_B_AUROC` exactly, or it raises rather than
reporting numbers for a different system.
