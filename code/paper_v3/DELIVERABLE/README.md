# EfficientEval — main experiment, ablations, and the code that produced them

Self-contained. Nothing here depends on the historical directories `paper_v3/results/`,
`results_round1/`, `results_round2/`, the smoke runs, or the aborted run, and no number here was
produced under a superseded configuration. The source archives under `paper_v3/artifacts/` and
`paper_v3/runs/` remain untouched; this tree is a copy organised for reading and for writing the
paper from.

## The frozen system

| | |
|---|---|
| Verifier pool | `factcc`, `lettuce_v2`, `granite_guardian_3_1_2b` |
| Cheap features, unified across both protocols | `structured_source_line_ratio`, `bm25_mean3`, `entity_coverage`, `entity_value_colocation`, `year_count`, `conflicting_value_rate` |
| Supervision target | `regret`, i.e. `T_ij = 1 + (Q_ij - max_k Q_ik)` |
| Learner | random forest regression, one head per action |
| Hyperparameters | Protocol A `800 / leaf 5 / 0.5 / depth 6`, Protocol B `200 / leaf 10 / 0.5 / depth 6`, both selected on TRAIN by minimum validation head loss over sixteen randomly initialised candidates |
| Routing | `U_ij = T_hat_ij * exp(-beta * c_j / c_max)`, one verifier called per instance |
| Beta | cheapest value whose validation AUROC is within 0.005 of the best, over a fixed ten-point grid; mode 0.1 in both protocols |
| Calibration | per-verifier Platt on the fit part, then one isotonic layer on the validation part, applied identically to the router and to all fifteen baselines |
| Decision threshold | validation MCC-optimal, primary; three alternatives reported |
| Latency | end to end: feature extraction, head inference, routing arithmetic, both calibration stages, and the verifier call |

## Headline numbers

Protocol A, 8,512 rows / 1,535 document groups, ten 8/1/1 rotations over ten seeds:

| | AUROC | SD | End-to-end | AURC | Macro |
|---|---:|---:|---:|---:|---:|
| **OURS** | **0.79330** | 0.00173 | **112.54 ms** | **0.14905** | 0.67083 |
| AlignScore | 0.78113 | 0.00131 | 677.29 ms | 0.15431 | 0.72390 |

First of sixteen by AUROC and by AURC. Separable from fourteen of the fifteen fixed verifiers;
AlignScore is indistinguishable once the fifteen comparisons are corrected for, at a 6.02x
latency ratio.

Protocol B, TRAIN 5,276 / 890 to TEST 3,236 / 645, ten seeds, confirmatory:

| | AUROC | SD | End-to-end | AURC | Macro |
|---|---:|---:|---:|---:|---:|
| Qwen30-fast | 0.83384 | 0.00066 | 211.65 ms | 0.12914 | 0.73504 |
| **OURS** | **0.82256** | 0.00474 | **106.14 ms** | 0.14039 | 0.68465 |
| Qwen30-judge | 0.81500 | 0.00158 | 600.87 ms | 0.14312 | 0.76755 |
| AlignScore | 0.81167 | 0.00108 | 561.65 ms | 0.13772 | 0.74940 |
| Granite-3.1-2b | 0.79583 | 0.00316 | 131.81 ms | 0.17090 | 0.73198 |

Second of sixteen. Separably better than the strongest pool member, indistinguishable from
AlignScore and Qwen30-judge at five to six times lower latency. Qwen30-fast holds the higher
point estimate at twice the latency, but that difference is not separable either: the paired
interval [-0.0283, 0.0062] contains zero. In neither protocol is any fixed verifier both more accurate and cheaper, so
the operating point is not dominated. Universal superiority is not claimed.

## What each directory is for

### `00_inputs/`

The text-free reproduction bundle, 6.3 MB. Keys, labels, the six cheap features and every
verifier's score / availability / latency, with the third-party source and summary text removed.
Point `AFR_INPUTS` at it and the whole study retrains from scratch; verified to reproduce the
frozen Protocol B result bit for bit (`0.8225560999095635`, delta `0.000e+00`). See its own
`README.md` for what the boundary between "reproducible from this bundle" and "needs the original
corpora" is.

### `01_main_experiment/`

The confirmatory run. Read its `README.md` first. It contains the sixteen-row publication tables
for both protocols, paired intervals, the end-to-end latency breakdown reconciled to the reported
means, threshold sensitivity under four rules, risk–coverage curves, the stage-2 calibration
effect, per-fit beta grids, the hyperparameter selection sweep, and per-row probabilities and
latencies so any downstream analysis can be redone without refitting a model.

Its reproduction gate is the thing to check if anything is ever in doubt: a legacy-hyperparameter
arm reproduces the first frozen run at `delta = 0.00e+00` on both AUROC and deterministic latency
in both protocols, which is what licenses attributing later deltas to the changes that were made.

### `02_ablation_core/`

Twenty-five single-factor arms, both protocols, ten seeds, referenced to the main experiment and
reproducing its OURS row exactly. Reports both the nominal 95% interval and the
Bonferroni-corrected interval for every arm; cite the corrected one.

Establishes the routing decision as the dominant effect (random routing costs 0.166 in B and
0.150 in A), every pool member as load-bearing with Granite irreplaceable, and both the verifier
call and stage-1 calibration as necessary. Three honest negatives are documented there: stage-2
isotonic costs Protocol B 0.00353 AUROC while cutting its ECE threefold, three of the six
features are not individually separable from noise, and the cost term's quality contribution is
separable only in Protocol A.

### `03_ablation_extended/`

Per-corpus training, the full subset lattices with Shapley attribution, convergence curves, the
feature-set unification contrast, and two tight controls. Its tables carry their own headers;
there is no separate README for this directory.

The methodological point worth carrying into the paper: the six features are redundant, so
leave-one-out testing under-reports every one of them and exhaustive per-subset significance
testing would return "not separable" almost everywhere as a property of the design. The average
marginal contribution over all 64 subsets is the quantity that answers what a feature is worth,
and it ranks the features differently from both leave-one-out and solo evaluation.

### `04_cascade_and_learners/`

The three families that answer the obvious objections. Cascades start from the cheapest verifier
and escalate; second-call arms keep the router's choice and add a conditional second verifier;
the learner sweep replaces the random forest with seven alternatives. Fourteen arms, both
protocols, ten seeds, referenced to the main experiment and reproducing it exactly. Escalation
thresholds are chosen on validation under a 1.5x latency ceiling, and every arm pays for every
call it makes.

All three cascades are both worse and more expensive than routing before calling. No second-call
trigger produces a correction-surviving gain in either protocol. All seven alternative learners
are separably worse than the random forest.

### `05_pool_provenance/`

The pool was selected in paper_v1 on a matrix sharing 54.4% of Protocol B's test document groups,
and unlike the features and hyperparameters it was never reselected on TRAIN alone. This is the
re-selection: all 455 three-verifier subsets, TRAIN only, under the current features and
hyperparameters.

The pool sits on the quality–cost Pareto frontier and is the lowest-variance, highest-minimum-share
pool on it, but ranks 109 of 455 on validation AUROC alone. Every higher-scoring pool costs at
least 2.80x as much; every cheaper pool gives up at least 0.052 AUROC. The earlier "rank 1 of 540"
was measured on the contaminated matrix and must not be quoted.

### `06_verifier_code/`

The complete executed scoring path, for the time-complexity analysis. `COMPLEXITY.md` is the
document to read: it records the dispatch table, the exact windowing formulas as executed, which
of the three forward-count columns is trustworthy for which verifier group, the generation caps,
two corrections to the earlier notes, and precisely which measured quantities still need
recomputing from the frozen scoring parquets before a complexity section can be written.

No verifier needs to be re-run for that; the scoring outputs are frozen and hashed.

### `07_data_contract/`

The protocol constants, the expected row and group counts, the pre-declared leakage exclusion, the
frozen configuration, and the selected hyperparameters.

### `08_scripts/`

Every experiment script and its launcher, plus the shared layer they all import. Each script takes
a run directory and a stage name, writes its own provenance, and refuses to proceed if the
inherited contract does not match field by field.

## Reproducing

Each experiment is a launcher plus a run directory. From `paper_v3/`:

```bash
./run_part1c_full_v1.sh $PWD/runs/rerun_main 0
```

```bash
./run_part2_ablation_v1.sh $PWD/runs/rerun_ablation 0
```

```bash
./run_part3_v1.sh $PWD/runs/rerun_extended phase1 0
```

Stages are checkpointed with `.done` markers, so an interrupted run resumes at the stage it
stopped in. Every run asserts the inherited contract before fitting anything and every run's
reference arm must reproduce the main experiment exactly, so a silent drift cannot pass.

The interpreter is whatever `AFR_PYTHON` names, defaulting to `python3`; from the repository
root, `./reproduce.sh` sets it and the other variables for you. Everything is CPU only. No GPU
is needed to reproduce any table here, because the verifier scores and latencies are frozen
inputs.

### `09_live_and_controls/`

Four experiments added after the main study was frozen: the main result re-executed live with one
real verifier call per instance (32,360 calls, AUROC 0.82205 against the frozen 0.82256), the
dataset-identity control, the few-shot adaptation curve that replaces leave-one-dataset-out, and
the paired-bootstrap significance columns for the main tables. See its own `README.md`.

## Running this outside the original machine

Every script resolves its paths from environment variables, falling back to the author's layout:

| variable | what it points at | default |
|---|---|---|
| `AFR_ROOT` | repository root | `/home/zeyu/projects/adaptive-faithfulness-router-v2` |
| `AFR_PYTHON` | interpreter used by the shell launchers | the venv on the original machine |
| `HF_HUB` | Hugging Face snapshot cache | `~/.cache/huggingface/hub` |
| `ALIGNSCORE_PYTHON` | AlignScore's separate environment | the venv on the original machine |

```bash
export AFR_ROOT=/path/to/this/repository
```

`requirements.txt` pins the versions this study ran on. Sections 01-05 and 07-09 need only numpy,
pandas, scikit-learn, scipy and pyarrow; torch, transformers, openai and vllm are needed only to
re-run verifiers rather than read the frozen scores.

## What is deliberately not here

Feature-noise robustness. The earlier LaTeX notes contain a table for it, produced under a
configuration this work replaced; none of its numbers apply and none are quoted anywhere in this
tree. Cross-corpus transfer is now covered by `09_live_and_controls/` rather than being absent.

`MANIFEST.sha256` covers every file. Verify with `sha256sum -c --quiet MANIFEST.sha256` from this
directory.
