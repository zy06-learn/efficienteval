# Cost-Aware Routing of Factuality Verifiers from Cheap Pre-Call Features

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](requirements.txt)
[![Reproduction gate](https://img.shields.io/badge/reproduction%20gate-passing-brightgreen.svg)](docs/reproducibility.md)

Code and results for **EfficientEval**, a router that picks *which* factuality verifier to
call for a given (source, summary) pair before any verifier has run, using six features
computed from the text alone. The write-up is in preparation; this repository is the
pipeline and the evidence it produced.

A summary-factuality evaluation pipeline usually fixes one verifier and pays its cost on
every instance. The cost spread across published verifiers is roughly six-fold in
end-to-end latency, and no single verifier is best on every instance. This release treats
verifier choice as a per-instance decision: a small regression head predicts each
verifier's regret from cheap pre-call features, a cost term discounts expensive verifiers,
and exactly one verifier is called.

```
U_ij = T̂_ij · exp(−β · c_j / c_max)          one verifier call per instance
T_ij = 1 + (Q_ij − max_k Q_ik)                regret supervision
```

## Highlights

- **Pre-call decision.** The router never sees a verifier score before choosing. The six
  features are document statistics computed from the source and the candidate summary.
- **One call per instance.** Not a cascade, not an ensemble. Reported latency is
  end-to-end and includes feature extraction, head inference, routing arithmetic, both
  calibration stages, and the verifier call itself.
- **Two protocols, one configuration.** Features, target, learner, hyperparameters, the β
  rule and both calibration stages are chosen on the TRAIN partition alone. The verifier
  pool is the exception: it was screened in an earlier round on a matrix that shares 54.4%
  of Protocol B's test document groups, and was not reselected afterwards. A TRAIN-only
  rescreen of all 455 three-verifier subsets is included as an audit and does not feed back
  into the frozen configuration. Protocol B reads TEST once and is the confirmatory result.
- **Text-free reproduction bundle.** `code/experiments/00_inputs/` (6.3 MB) carries
  the frozen features and verifier scores with all source and summary text removed. It
  reproduces every published table without redistributing the corpora.
- **Verified release.** The reference arm in this repository refits Protocol B from the
  frozen inputs and lands on the published AUROC. It is exact on the machine the frozen run
  executed on, and within `1.1e-05` on other aarch64 hosts, `1.07e-04` on x86_64, against a
  seed standard deviation of `0.00474` for the same arm. The gate enforces both levels; see
  [Reproduction](#reproduction).

## How the router works

One pass, in the order the code runs it. Every step names the module that implements it, so
this section doubles as a reading order.

**1. Read the pair.** A test instance is a source document and one candidate summary. Nothing
about the corpus it came from is part of the input; `dataset_key` exists in the frames for
bookkeeping and grouping and is never a feature. The control in
[`ds_only.py`](code/experiments/09_live_and_controls/code/ds_only.py) is what establishes that
this is a real constraint rather than an assertion.

**2. Compute six numbers from the text.** No verifier has run. Two extractors produce them,
which is why the reported latency charges both:

| Feature | What it measures | Computed in |
|---|---|---|
| `structured_source_line_ratio` | fraction of source lines that are tables, lists, or short all-caps/colon headers — how much of the document is not prose | `summary_router_compact16_direct_v1.py::_structured_line_ratio` |
| `entity_value_colocation` | of the (entity, number) pairs in the summary, the fraction whose entity and number both appear in that sentence's top-3 retrieved source sentences. `1.0` when the summary states no such pair | `…direct_v1.py::_row_features` |
| `conflicting_value_rate` | of the same pairs, the fraction where the evidence sentences carry the entity but a *different* number. `0.0` when there are no pairs | `…direct_v1.py::_row_features` |
| `bm25_mean3` | mean BM25 of each summary sentence against its three best-matching source sentences — how well the summary is grounded lexically | `router_feature_learnability.py::_retrieval_features` |
| `entity_coverage` | fraction of the summary's entities that occur in the source's entity set | `router_feature_learnability.py` |
| `year_count` | number of years stated in the summary | `router_feature_learnability.py` |

They are document statistics, so they carry corpus signal — a classifier reaches 89.6% corpus
accuracy from them alone. That is expected and is not a leak; see
[Controls](#controls) for the experiment that separates the two.

**3. Predict each verifier's regret.** One random-forest regression head per action, fitted only
on rows where that action was available:

```
T_ij = 1 + (Q_ij − max_k Q_ik)          in [0, 1], 1 iff j is the best action for i
```

`Q_ij` is the stage-1 calibrated probability verifier `j` assigns to the true class on instance
`i`. Regret rather than a binary is-best label because the binary throws away *how much* worse
a runner-up is, and on instances where two verifiers are near-tied it makes the head learn a
coin flip. `core.py::targets` implements every target the ablation compares; `config.py` records
why `regret` is the locked one.

**4. Discount by cost, take the argmax.**

```
U_ij = T̂_ij · exp(−β · c_j / c_max)
j*_i = argmax_j U_ij                     unavailable actions get −∞
```

`c_j` is verifier `j`'s mean cost estimated **from training rows only**
(`core.py::fold_costs`), never from the evaluation side. β is not a Lagrange multiplier solved
for a budget — it is an operating parameter chosen by a validation rule: *the cheapest value on
a fixed ten-point grid whose validation AUROC is within 0.005 of the best*
(`core.py::choose_beta`). The mode is 0.1 in both protocols. That rule has a failure mode the
paper reports: when the quality gap between actions lands near the 0.005 tolerance the rule is
bistable between two β values, and on one few-shot arm two seeds in ten flipped to β = 0.2,
whose discount excludes the only verifier that works on that corpus.

**5. Call exactly one verifier.** Not a cascade, not an ensemble, no second opinion. The other
two are never invoked for that row. This is a property of the policy class, not a conclusion of
the derivation.

**6. Put the answer on a common scale.** Two calibration stages, both fitted without touching
the evaluation rows:

- **Stage 1, per-verifier Platt** (`core.py::platt`), fitted on the fit partition. Three
  verifiers trained separately do not share a probability scale, and the regret target above is
  only meaningful once they do.
- **Stage 2, one isotonic layer** (`core.py::isotonic`), fitted on the validation partition and
  applied identically to the router *and to all fifteen fixed baselines*. Needed because the
  router emits whichever member it selected, so its output is a mixture of three scales.

Isotonic is monotone within a fold, but its PAVA solution is only *weakly* monotone: plateaus
collapse distinct probabilities to one value, creating ties, so pooled AUROC does move. Measured:
`+0.00248` on Protocol A and `−0.00353` on Protocol B. Every number in this repository is
post-stage-2.

**7. Decide.** For the metrics that need a hard call, the threshold is the one that maximises
MCC on the system's *own* validation partition — never 0.5, and never retuned on test. Three
alternatives (fixed 0.5, Youden, split-conformal) are reported alongside.

### The data boundary

Stated in [`v3core.py`](code/experiments/08_routing_code/v3core.py) and enforced there rather
than by convention, because the previous round of this project drifted exactly here:

```
SELECT      = TRAIN only, 5,276 rows / 890 document groups.
              Features, target, learner, hyperparameters, pool, the beta rule and both
              calibration stages are chosen on this and nothing else.
Protocol A  = pooled 8/1/1 rotation over TRAIN + TEST, 8,512 rows / 1,535 groups.
              A cross-validated estimate; every row gets exactly one out-of-fold prediction.
Protocol B  = fit on TRAIN, read TEST once. The confirmatory result.
```

The pool is the one documented exception, disclosed in [Highlights](#highlights).

### What keeps it honest

Three mechanisms, none of them optional:

- **One implementation of each step.** `core.py` is the only place a head is fitted, β is
  chosen, or a metric is computed. Its docstring records why: in an earlier round four scripts
  called `fit_heads()` without `hp=`, silently fell back to a different forest than the paper
  claimed, and two of them also used a β rule already shown to be broken. That class of bug is
  invisible in any one script's output and only surfaces when two tables disagree.
- **Inherit and assert.** Every experiment reads the frozen contract and compares it field by
  field before fitting anything (`part1c_main_full_v1.py::preflight`,
  `_contract.py::load`). A drifted pool or feature list raises; it does not warn.
- **Reproduce the reference arm first.** Any arm that is nominally the main system must land on
  `FROZEN_B_AUROC = 0.8225560999095635` before its own results are believed
  (`_contract.py::check_reference`). The ablation lattices contain the reference twice on
  purpose, once as the full feature subset and once as the full pool subset, and both must
  reproduce it.

## Results

### Protocol A: grouped ten-fold rotation, 8,512 rows / 1,535 groups, ten seeds

| System | AUROC | SD | End-to-end | AURC | Macro |
|---|---:|---:|---:|---:|---:|
| **EfficientEval** | **0.79330** | 0.00173 | **112.54 ms** | **0.14905** | 0.67083 |
| AlignScore | 0.78113 | 0.00131 | 677.29 ms | 0.15431 | 0.72390 |

First of sixteen systems by AUROC and by AURC. Separable from fourteen of the fifteen fixed
verifiers. AlignScore is statistically indistinguishable once the fifteen comparisons are
corrected for, at a 6.02x latency ratio.

### Protocol B: TRAIN 5,276 / 890 → TEST 3,236 / 645, ten seeds, confirmatory

| System | AUROC | SD | End-to-end | AURC | Macro |
|---|---:|---:|---:|---:|---:|
| Qwen30-fast | 0.83384 | 0.00066 | 211.65 ms | 0.12914 | 0.73504 |
| **EfficientEval** | **0.82256** | 0.00474 | **106.14 ms** | 0.14039 | 0.68465 |
| Qwen30-judge | 0.81500 | 0.00158 | 600.87 ms | 0.14312 | 0.76755 |
| AlignScore | 0.81167 | 0.00108 | 561.65 ms | 0.13772 | 0.74940 |
| Granite-3.1-2b | 0.79583 | 0.00316 | 131.81 ms | 0.17090 | 0.73198 |

Second of sixteen. Separably better than the strongest pool member, indistinguishable from
AlignScore and Qwen30-judge at five to six times lower latency. Qwen30-fast holds the higher
point estimate at twice the latency, but that difference is not separable either: the paired
interval is [-0.0283, 0.0062] and contains zero. In neither protocol is any fixed verifier both more
accurate and cheaper, so the operating point is not dominated. **Universal superiority is
not claimed.**

Significance is a paired cluster bootstrap over `content_doc_key`, 2,000 draws, Bonferroni
corrected over the fifteen comparisons. Per-comparison intervals:
[`A_SIGNIFICANCE.csv`](code/experiments/09_live_and_controls/results/A_SIGNIFICANCE.csv),
[`B_SIGNIFICANCE.csv`](code/experiments/09_live_and_controls/results/B_SIGNIFICANCE.csv).

### Controls

| Question | Answer | Evidence |
|---|---|---|
| Is the test result read from a pre-computed matrix? | No. Re-running Protocol B with every verifier called live for every (row, seed) pair gives 0.82205 against the frozen 0.82256, a delta of −0.00051 across 32,360 live calls, with a paired interval of [−0.0015, 0.0005] that contains zero and lies entirely within ±0.0015. The residual traces to vLLM prefix-caching non-determinism. | [`LIVE_MAIN_B.json`](code/experiments/09_live_and_controls/results/LIVE_MAIN_B.json) |
| Does the router receive dataset identity? | No. Four arms differing only in the head's input: constant only 0.79583, corpus one-hot only 0.80845, the six cheap features 0.82256, six features plus corpus one-hot 0.82121. Handing the head the corpus explicitly moves AUROC by −0.00135, an interval of [−0.0040, 0.0065] containing zero (p = 0.59), not separable under Bonferroni over the three comparisons. The six features do carry corpus signal (89.6% corpus accuracy against a 56.5% majority baseline), which is expected of document statistics and is not a label leak. | [`DATASET_ONLY_ARMS.json`](code/experiments/09_live_and_controls/results/DATASET_ONLY_ARMS.json), [`CONTROLS_SIGNIFICANCE.csv`](code/experiments/09_live_and_controls/results/CONTROLS_SIGNIFICANCE.csv) |
| How many in-corpus examples does a held-out corpus need? | Corpus dependent, and the sweep is over fractions of each corpus's own pool because the pools differ by 5.6x. Three of the four improve separably under Bonferroni over the four corpora: RAGTruth 0.543 → 0.638 (p < 0.001, saturating at 20–30% of its pool), UniSumEval 0.537 → 0.592 (p = 0.010, still rising at 100%), CoGenSumm 0.652 → 0.680 (p = 0.005). FRANK is flat and not separable (p = 0.40): a head fitted on the other three already transfers to it. | [`FEWSHOT_FRACTION_CURVE.csv`](code/experiments/09_live_and_controls/results/FEWSHOT_FRACTION_CURVE.csv), [`CONTROLS_SIGNIFICANCE.csv`](code/experiments/09_live_and_controls/results/CONTROLS_SIGNIFICANCE.csv) |

## The frozen system

| | |
|---|---|
| Verifier pool | `factcc`, `lettuce_v2`, `granite_guardian_3_1_2b` |
| Cheap features | `structured_source_line_ratio`, `bm25_mean3`, `entity_coverage`, `entity_value_colocation`, `year_count`, `conflicting_value_rate` |
| Supervision target | `regret`, `T_ij = 1 + (Q_ij − max_k Q_ik)` |
| Learner | random-forest regression, one head per action |
| Hyperparameters | Protocol A `800 / leaf 5 / 0.5 / depth 6`; Protocol B `200 / leaf 10 / 0.5 / depth 6`, both selected on TRAIN by minimum validation head loss over sixteen randomly initialised candidates |
| β | cheapest value whose validation AUROC is within 0.005 of the best, over a fixed ten-point grid; mode 0.1 in both protocols |
| Calibration | per-verifier Platt on the fit partition, then one isotonic layer on the validation partition, applied identically to the router and to all fifteen baselines |
| Decision threshold | validation MCC-optimal (primary); three alternatives reported |

## Data

| Dataset | TRAIN rows | TRAIN groups | TEST rows | TEST groups |
|---|---:|---:|---:|---:|
| CoGenSumm | 535 | 107 | 400 | 100 |
| FRANK | 669 | 149 | 1,569 | 350 |
| RAGTruth | 2,983 | 499 | 900 | 150 |
| UniSumEval | 1,089 | 135 | 367 | 45 |
| **Total** | **5,276** | **890** | **3,236** | **645** |

Groups are `content_doc_key`, derived from normalised source-document content, so all
summaries of one source document stay in one group. Five RAGTruth training rows were
removed under a pre-declared content-hash rule because their canonical documents also occur
in the CoGenSumm test split. The test side is left intact.

## Quickstart

```bash
git clone https://github.com/zy06-learn/efficienteval.git
cd efficienteval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh verify
```

`reproduce.sh verify` checks the artifact manifest and re-fits the Protocol B router from
the frozen inputs. It must print a delta inside the tolerance in force. It takes under a minute on CPU and
needs no GPU, no model downloads, and no corpus access.

## Repository layout

```
.
├── reproduce.sh                single entry point for every stage
├── tests/test_reproduction.py  manifest check and the reference-arm gate
├── docs/
│   ├── pipeline.md             the four stages end to end
│   ├── reproducibility.md      what runs what, and what each stage costs
│   └── verifiers.md            the fifteen verifiers, protocols, serving setup
├── figures/                    the paper's three figures and the script that draws them
└── code/                       AFR_ROOT
    ├── ingest_and_scoring/     stage 1 (corpus ingest, TRAIN/TEST split) and the stage-2 driver
    ├── verifier_wrappers/      the fifteen verifier implementations stage 2 calls
    ├── shared/                 two modules (core.py, config.py) every later stage imports
    ├── results/                the two sha256-pinned matrices stages 1 and 2 produced
    └── experiments/            stages 3 and 4: the code and every published result
        ├── 00_inputs/              text-free reproduction bundle -- start here
        ├── 01_main_experiment/     protocols A and B, the paper's main tables
        ├── 02_ablation_core/       the core ablations
        ├── 03_ablation_extended/   per-corpus tables, Shapley, convergence curves
        ├── 04_cascade_and_learners/  cascade and alternative-learner comparisons
        ├── 05_pool_provenance/     how the three-verifier pool was selected, and its TRAIN-only rescreen
        ├── 06_verifier_registry/   upstream repository and revision of every verifier
        ├── 07_data_contract/       the split and grouping rules
        ├── 08_routing_code/        stage 3: the routing experiments themselves
        ├── 09_live_and_controls/   stage 4: live re-run and the three controls, code and results
        └── cross_stage_contract/   the frozen configuration every stage reads
```

Every directory carries a `README.md`, which GitHub renders when you open the directory, so
the tree can be read by browsing rather than by consulting this page.

**The five experiment directories share one shape.** Learn it once and `01_` through `05_`
all read the same way:

| Inside `0X_…/` | What it holds |
|---|---|
| `00_contract/` | the frozen configuration this run used |
| `01_…_tables/` | the results, as CSV -- this is where the paper's numbers live |
| `02_gates/` | the run's own check that it landed on the frozen values |
| `03_provenance/` | code hashes, git state, launch command, completion marker |
| `05_logs/` | the console output of the run |
| `06_row_level/` | per-row, per-seed predictions (`.npz`) |
| `MANIFEST.sha256` | a digest for every file in the directory |
| `README.md`, `REPORT_zh.md` | what the experiment is, and its results read out |

That chain is the point: any number in `01_…_tables/` can be traced down to the per-row
predictions that produced it and back up to the code revision that ran.

**Two things in the tree are deliberately not tidy.** The provenance files under
`03_provenance/` and `07_provenance/` record absolute paths on the machine the experiments
ran on (`/home/zeyu/projects/adaptive-faithfulness-router-v2/…`), under that tree's own
directory names. They are a historical record of what was executed and are left exactly as
written, so they do not match the layout above. And three modules
(`summary_router_compact16_direct_v1.py`, `pool_gate_sweep_v1.py`, `tenfold_v1.py`) exist as
byte-identical copies in both `verifier_wrappers/` and `experiments/08_routing_code/`, because the
two trees import them under different module names; removing either copy would change which
module the frozen pipeline loads.

## Every file, and what it does

63 source files, 22.9k lines. Grouped by the stage that runs them. The two not in a table
below are `reproduce.sh` at the root and `figures/make_figures.py`, which draws the paper's
three figures from the published CSVs. Anything else in the tree is data, a manifest, or a
per-directory `README.md`.

### Stage 1 — ingest: four corpora to one task

`code/ingest_and_scoring/`

| File | Lines | What it does |
|---|---:|---|
| `ingest/build_splits_v2.py` | 781 | The only program allowed to read official-test gold. Unifies the four corpora into summary-level binary classification, groups by `content_doc_key` (normalised source content), applies the pre-declared content-hash removal of five RAGTruth rows, and emits `TRAIN` / `TEST` / `TEST_SCORING` / `P1_SCORING_COHORT`. `TEST_SCORING` is a fail-closed projection with every gold-derived field stripped — stage 2 reads that one and cannot reach a label. |
| `ingest/verifier_cli.py` | 119 | Command-line entry for scoring one verifier over one cohort. |
| `config_v2.py` | 128 | Frozen protocol inputs for stage 1. Selection outputs stay `None` until stage 3, so stage-1 and stage-2 code fails closed if it tries to obtain a test label through the gold-free asset. |
| `core_v2.py` | 503 | The stage-1/2 sibling of `shared/core.py`: folds, calibration, head fitting, metrics. |
| `p1_prepare.py` | 65 | Builds the scoring cohort from the sealed test set, and only for rows missing at least one score. FRANK test already carries all fifteen from the frozen run that produced TRAIN; rescoring it would swap frozen numbers for numbers from a different run. |
| `p1_score.py` | 120 | Scores the test-side cohort, deliberately bypassing the stock entry point — that one validates the presence of `label_supported`, which would mean putting a label column next to test rows during scoring. |
| `p1_score_local.sh` | 32 | Stage 2a: the nine locally-hosted verifiers. Each writes `status/<name>.done`, so an interrupted run resumes instead of rescoring. |
| `p1_api.sh` | 94 | Stage 2b: the six vLLM-served verifiers, one model loaded at a time (`qwen30_fast` and `qwen30_judge` share weights and are scored back to back). The serving flags are part of the measurement: `--max-num-seqs 1` makes the reported latency strict batch-1, `--max-model-len 16384` fixes the context budget, and `--enable-prefix-caching` is the source of the non-determinism the live control traces. |

### Stage 2 — the fifteen verifiers

`code/verifier_wrappers/`. Each verifier runs once per row under a declared protocol string
recorded in `PROTOCOLS`; the string fixes window size, overlap, aggregation, and whether the
official prompt prefix applies. Two verifiers with different window caps are not measured under
the same context budget, and the protocol says so.

| File | Lines | What it does |
|---|---:|---|
| `unified_summary_verifiers_v1.py` | 1342 | The scoring harness: `PROTOCOLS`, input preparation, source windowing that preserves the summary, the scorer factory, per-frame scoring, out-of-fold threshold selection, the canary audit, and result finalisation. |
| `unified_scoring.py` | 1682 | Gold-free, resumable scoring utilities: validates that no forbidden gold column reached the input, hashes and verifies the file inventory, handles reuse of frozen RAGTruth scores, and records GPU metadata and peak memory alongside every score. |
| `summary_router_compact16_direct_v1.py` | 1907 | The larger of the two feature extractors: tokenisation, sentence splitting, entity and numeric-value spans, a local BM25/IDF index, and the seventeen compact features — three of which are in the frozen six. Also builds and validates the grouped folds. |
| `router_feature_learnability.py` | 1028 | The other feature extractor (`bm25_mean3`, `entity_coverage`, `year_count` come from here) plus the learnability diagnostics: ROUGE-L, retrieval features, selective targets, direct and sequential policy evaluation, operating-point selection, and cross-fitted LR/HGB probes. |
| `primary_scoring.py` | 357 | HHEM's chunked scorer, the primary scorer factory, input-manifest and frame validation, balanced smoke selection, native-column expansion. |
| `candidate_verifiers.py` | 405 | FactCG, HHEM, AlignScore (through a persistent worker), FENICE. |
| `additional_verifier_scorers_v1.py` | 449 | WeCheck, Granite-Guardian-3.2 factuality, Granite-Guardian-4.1 groundedness, and the structured judge API, with support probability read from top logprobs. |
| `extended_scorers.py` | 332 | LettuceDetect-v2, Granite-Guardian-3.2-3b-a800m, AttrScore-Flan-T5, with tokenizer-window splitting and native outputs preserved. |
| `minicheck_scorers.py` | 179 | Both MiniCheck variants, aligned with the official EMNLP 2024 inference code: newlines preserved, NLTK splitting, 500-word (Flan-T5) or 400-token (DeBERTa) chunks, `eos_token` join, max-aggregation, and the `"predict: "` prefix on the Flan-T5 path. |
| `native_scorers.py` | 129 | FactKB and FactCG variants that additionally capture native class logits and argmax labels. Scores must match the frozen runs bit-for-bit. |
| `structured_high_judge.py` | 274 | Response format, numbered source rendering, claim-span grounding, and payload validation for the structured judge. |
| `cascade_primary_assets.py` | 303 | Materialises the UniSumEval and RAGTruth annotations, builds scoring inputs and features, and audits reference overlap. |
| `global_gamma_calibration.py` | 316 | RAGTruth span parsing and response segmentation, and the CRC calibration of the global gamma. |
| `research_freeze.py` | 178 | The freeze validator: dataset ledger, frozen method, and evidence-file digests. |
| `tenfold_v1.py` | 131 | The 8/1/1 rotation contract: eight folds fit the heads, one fits the Platt calibrators *and* selects β, one is evaluated once with everything frozen. Rotating ten times gives every summary exactly one test-fold prediction. |
| `pool_gate_sweep_v1.py` | 433 | Earlier-round pool × gate sweep: does the learned router survive a change of action pool? Sweeps β over a grid and reports the whole curve rather than selecting on held-out labels. |
| `pool_gate_sweep_v2.py` | 149 | Fixes v1's matched-random flaw: v1 drew the budget-matched random policy once per configuration, and a single draw carries sd ≈ 0.02–0.03 AUROC, the same order as the effect. v2 uses an independent control bank. |
| `summary_router_compact16_targetmix_v1.py` | 870 | Earlier-round target-mix router: pairwise and reward bundles, nested fits, price calibration, reselection. |
| `summary_router_compact16_targetmix_lodo_v1.py` | 1293 | The leave-one-dataset-out variant of the above, with its own baselines, paired bootstrap, and Pareto frontier. |
| `scripts/alignscore_persistent_worker.py` | 77 | Keeps AlignScore resident across calls instead of paying its load time per row. |

### The shared layer

`code/shared/` — imported by every later stage. `08_routing_code/` carries byte-identical copies
under the module names the frozen pipeline resolves.

| File | Lines | What it does |
|---|---:|---|
| `core.py` | 508 | **The one place each step happens.** `fold_costs` and `verifier_cost` (training-only cost estimates), `pooled_folds` / `dataset_folds` and their rotations, `platt` / `apply_platt` / `isotonic` (both calibration stages), `targets` (every supervision target the ablation compares), `make_regressor` / `make_classifier` / `fit_heads`, `choose_beta`, `route`, `make_frame`, `ece` / `risk_coverage` / `metrics`, `conformal_tau` and `group_conformal_tau`, and the three bootstraps — `paired_bootstrap`, `paired_cluster_bootstrap`, `cluster_bootstrap_auroc`. Two details worth knowing: `make_classifier` refuses an unsupported learner name instead of silently substituting a forest, and `make_frame` takes `charge_features`, which must be `False` for any arm that does not compute the cheap features — folding the router's 8.6 ms into a fixed verifier's cost would flatter the router. |
| `config.py` | 116 | The single source of truth for the locked configuration, and the record of why centralising it was necessary. |
| `08_routing_code/v3core.py` | 203 | The data boundary above, in code: `load(with_test_labels)` gates the confirmatory read, `stratified_group_split` and `folds_stratified` are group-disjoint and stratified by corpus and majority label, `rotations` is the 8/1/1 contract, and `route` / `choose_beta` are the routing implementation every stage shares. |

### Stage 3 — routing

`code/experiments/08_routing_code/`. The only stage that runs from this repository alone.

| File | Lines | What it produces |
|---|---:|---|
| `part1c_main_full_v1.py` | 886 | **The main tables.** Asserts the inherited contract field by field, selects hyperparameters on TRAIN from sixteen randomly initialised candidates by minimum validation head loss, runs both protocols across ten seeds, charges end-to-end latency (`_pre_call_feature_ms` + `_verifier_ms` + heads + routing + both calibration stages), applies four threshold rules, and writes the gates and provenance. |
| `part2_ablation_v1.py` | 478 | Core ablations, both protocols. Each arm changes exactly one thing — feature subset, target, learner, pool — and `OURS` here must reproduce `part1c` exactly. That equality is the gate. |
| `part3_extended_v1.py` | 772 | Extended ablations: per-corpus training, feature and pool subset lattices, Shapley attribution, best-k, and the data-size and forest-size convergence curves. The lattices contain the reference twice, and both copies must reproduce `part1c` bit for bit. |
| `part3_percorpus_selected_v1.py` | 166 | The deployable per-corpus competitor. The per-corpus table's "best fixed verifier" is an oracle — it takes the maximum *test* AUROC over fifteen candidates, which no deployable system knows in advance. This adds the honest version: pick one fixed verifier by **validation** AUROC on the same split the router used. It beats within-corpus routing on all four corpora, and the paper reports that. |
| `part4_cascade_v1.py` | 560 | The competitors the ablation left open: confidence, disagreement and learned-deferral cascades; second calls chosen by confidence, raw margin, or discounted margin; and alternative learners. Runs in its own directory and writes nowhere the other parts write. |
| `pool_rescreen_v1.py` | 233 | Pool provenance. The pool was screened in an earlier round on a matrix sharing 54.4% of Protocol B's test groups. This re-screens all 455 three-verifier subsets on TRAIN alone and reports where the frozen pool ranks. It is an audit; it does not feed back. |
| `run_part1c_full_v1.sh` · `run_part2_ablation_v1.sh` · `run_part3_v1.sh` | 49 · 41 · 49 | Stage launchers. `run_part3_v1.sh` demands an explicit `phase1`/`phase2`; there is no default. |
| `chain_part4.sh` | 63 | Waits for Part 3 phase 2's completion marker under `$AFR_ROOT/experiments/runs`, then starts Part 4. |

### Stage 4 — live re-run and the controls

`code/experiments/09_live_and_controls/code/`

| File | Lines | Question it answers, and how |
|---|---:|---|
| `_contract.py` | 87 | The guard everything here inherits. Compares the frozen contract field by field, and `check_reference` requires any nominally-main arm to land on `FROZEN_B_AUROC`. Carries the two-level tolerance and the measured cross-platform deltas that justify it. |
| `live_main.py` | 140 | *Is the test result read from a pre-computed matrix?* Re-runs Protocol B with the verifier actually called for every (row, seed) pair — 32,360 real calls. Scores are never reused across seeds: a row two seeds both select is called twice. Staged `plan → local → api → finish`. |
| `live_pipeline.py` | 158 | The end-to-end deployed path for the same question: route from cheap features, call exactly one verifier, and only afterwards read the other two columns to compare against the matrix. |
| `rerun_score.py` | 70 | Independently re-runs the three pool verifiers on a sample, writing to a fresh directory so the harness cache cannot serve stale rows. This is what isolated the residual to vLLM prefix caching. |
| `cmp.py` | 25 | Diffs re-run scores against the frozen matrix per verifier: max \|Δ\|, exact-equality flag, correlation, and old-versus-new latency. |
| `diag_live.py` | 35 | The row-level version: for each live-selected action, max \|Δ\| against the stored score and the first differing rows. |
| `dataset_control.py` | 96 | *Does the router receive dataset identity?* Three answers in increasing strength: (a) the frozen feature list contains no dataset field; (b) how much corpus signal the six features carry anyway, by grouped 5-fold corpus classification — an upper bound, reported honestly; (c) the control that matters — add the corpus one-hot and re-run the whole pipeline. |
| `ds_only.py` | 79 | The four-arm version of (c): constant-only, corpus-one-hot-only, the six features, and both. The reading rule is written in the docstring *before* the numbers: if the one-hot arm lands near constant-only, corpus identity buys nothing; if it lands near the six features, the router is a corpus classifier in disguise. |
| `fewshot_frac.py` | 116 | *How much of its own data does a held-out corpus need?* Fits on the other three corpora and adds a **fraction** of the target's own pool. Fractions, not counts, because the pools differ by 5.6× — `k=512` is 96% of CoGenSumm's pool and 17% of RAGTruth's. Keeps row-level output at the 0% and 100% ends so the endpoints can be tested. |
| `fewshot.py` · `fewshot_k0.py` | 91 · 57 | The superseded absolute-count sweep and its `k=0` point, kept because the archived curve came from them. |
| `sig_main.py` | 55 | The main tables' significance. Resamples `content_doc_key`, not rows. Generates all 2,000 index sets **once** and reuses them across the fifteen comparisons — that is what makes the test paired. Inside each draw it takes the per-seed AUROC difference and averages over seeds. |
| `sig_controls.py` | 250 | The same test for the three controls, whose original runs saved summaries but not row-level probabilities — so every arm is refitted here, which is exactly why `check_reference` runs first. Families: live (1), arms (3), few-shot (4). AUROC is computed from the rank identity with averaged ranks, because isotonic creates ties. |

### The gate

| File | Lines | What it checks |
|---|---:|---|
| `tests/test_reproduction.py` | 283 | Six tests. `test_manifest`: every file the manifest names still hashes to the recorded digest. `test_reference_arm`: Protocol B refitted from the frozen inputs lands on `FROZEN_B_AUROC` within the tolerance in force. `test_bundle_carries_every_column_the_pipeline_needs`: the trimmed bundle satisfies the stage-3 preflight contract. `test_launchers_resolve`: every stage-3 launcher points at a script that exists. `test_reproduce_sh_matches_the_launchers`: `reproduce.sh` passes arguments the launchers accept and writes where they look. `test_control_scripts_resolve_their_imports`: every directory a stage-4 script puts on `sys.path` exists and carries the modules it imports. |

## Reproduction

Four stages, in order. Stage 3 alone reproduces every published table, because
stages 1 and 2 are shipped frozen in `00_inputs/`.

| Stage | What it does | Needs |
|---|---|---|
| 1. Ingest | Builds the TRAIN/TEST split from the four corpora | corpus access |
| 2. Verifier scoring | Runs all fifteen verifiers, producing the score matrix | GPU, model weights, vLLM |
| 3. Routing | Fits the heads, routes, calibrates, evaluates | CPU only |
| 4. Live and controls | Re-runs Protocol B with live verifier calls; the three controls | GPU for the live arm |

See [`docs/reproducibility.md`](docs/reproducibility.md) for the exact commands and for what
each stage costs.

## Status

The write-up is in preparation. This repository is the code and the results it produced, and
it is released ahead of the paper so that the numbers below can be checked against a working
pipeline rather than against prose. Author list and citation details will be added here when
the paper exists; until then there is nothing to cite but this repository.

## License

Apache-2.0. See [LICENSE](LICENSE).

Third-party verifiers are not redistributed here. Each carries its own licence;
[`code/experiments/06_verifier_registry/REGISTRY.md`](code/experiments/06_verifier_registry/REGISTRY.md)
records the upstream repository and revision of every one.
