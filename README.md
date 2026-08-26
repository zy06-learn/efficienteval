# Cost-Aware Routing of Factuality Verifiers from Cheap Pre-Call Features

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg)](requirements.txt)
[![Reproduction](https://img.shields.io/badge/main%20result-reproduces%20bit--for--bit-brightgreen.svg)](docs/reproducibility.md)

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
- **Verified release.** The reference arm in this repository reproduces the frozen main
  table at `delta = 0.000e+00`. See [Reproduction](#reproduction).

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
the frozen inputs. It must print `delta 0.000e+00`. It takes under a minute on CPU and
needs no GPU, no model downloads, and no corpus access.

## Repository layout

```
.
├── reproduce.sh                single entry point for every stage
├── tests/test_reproduction.py  manifest check and the bit-for-bit gate
├── docs/
│   ├── pipeline.md             the four stages end to end
│   ├── reproducibility.md      what runs what, and what each stage costs
│   └── verifiers.md            the fifteen verifiers, protocols, serving setup
├── figures/                    the paper's three figures and the script that draws them
└── code/                       AFR_ROOT
    ├── ingest_and_scoring/     stage 1 (corpus ingest, TRAIN/TEST split) and the stage-2 driver
    ├── verifiers/              the fifteen verifier implementations stage 2 calls
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
