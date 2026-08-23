# The verifier pool

Fifteen verifiers are scored. Three of them form the frozen routing pool; the other twelve
are the fixed-verifier baselines the router is compared against.

| | |
|---|---|
| Frozen pool | `factcc`, `lettuce_v2`, `granite_guardian_3_1_2b` |
| Locally hosted | `factkb`, `factcc`, `lettuce_v2`, `hhem`, `factcg`, `minicheck_dbta`, `minicheck_ft5`, `alignscore`, `wecheck` |
| vLLM served | `granite_guardian_3_2_8b_factuality`, `qwen30_judge`, `granite_guardian_4_1_3b_factuality_lora`, `granite_guardian_3_1_2b`, `granite_guardian_3_2_3b_a800m`, `qwen30_fast` |

The pool was not chosen by hand. It is rank 1 of the 445 feasible k=3 subsets under an
exhaustive validation screen, and the terminal point of the quality/cost Pareto frontier.
The screen is in `code/paper_v3/DELIVERABLE/05_pool_provenance/`.

## Protocols

Each verifier runs under a declared protocol string, listed in `PROTOCOLS` in
`code/afr_v2/unified_summary_verifiers_v1.py`. The string is not a label. It fixes the
source-window size, the overlap, the aggregation rule, and whether the official prompt
prefix is applied, and two verifiers with different window caps are not measured under the
same context budget.

| Verifier | Protocol |
|---|---|
| `factkb`, `factcc`, `hhem`, `wecheck` | `summary_once_fullclaim_sourcewin512_overlap64_max_oof_v1` |
| `lettuce_v2` | `summary_once_official_sourcewin512_overlap64_span_oof_v1` |
| `factcg`, `minicheck_dbta`, `minicheck_ft5` | `summary_once_fullclaim_sourcewin2048_overlap64_max_oof_v1` |
| `alignscore` | `summary_once_native_nli_sp_fullsource_oof_v1` |
| `granite_guardian_3_2_8b_factuality` | `summary_once_official_factuality_evidence_select_oof_v1` |
| `granite_guardian_3_1_2b`, `granite_guardian_3_2_3b_a800m` | `summary_once_official_groundedness_evidence_select_oof_v1` |
| `granite_guardian_4_1_3b_factuality_lora` | `summary_once_official_lora_json_evidence_select_oof_v1` |
| `qwen30_judge` | `summary_once_json_prefix_label_probability_no_cot_oof_v2` |
| `qwen30_fast` | `summary_once_binary_token_logprob_no_cot_oof_v1` |

`sourcewin512` against `sourcewin2048` is the difference that matters most when reading the
latency column: a verifier capped at 512 tokens sees fewer windows per document than one
capped at 2048, and pays proportionally less.

## Serving setup

The six served verifiers run behind vLLM 0.20.0 on the V1 engine, one model loaded at a
time. The flags are part of the measurement:

| Flag | Why |
|---|---|
| `--max-num-seqs 1` | reported latency is strict batch-1, so it is comparable to the locally-hosted encoders |
| `--max-model-len 16384` | fixes the context budget every served verifier is measured under |
| `--enable-prefix-caching` | on, and the reason re-runs are not bit-identical (see below) |
| `temperature=0.0`, `logprobs=20` | scoring reads the label token probability rather than sampling text |

`--gpu-memory-utilization` varies by model: 0.90 for the Qwen3-30B FP8 checkpoint, 0.85 for
the 8B Granite, 0.60 for the smaller Granite models.

Model revisions are pinned in `MODEL_REVISIONS` in
`code/afr_v2/unified_summary_verifiers_v1.py`. Every entry carries a commit hash, not a tag.

### Prefix caching is why re-runs differ

With `--enable-prefix-caching`, a row can hit a warm prefix and return a score that differs
in the last digits from a cold run. This is the whole residual between the live Protocol B
re-run and the frozen matrix result: −0.00051 across 32,360 calls. The locally-hosted
encoders re-run bit-identically, which is how the residual was localised to the serving
stack rather than to the router.

## Two findings that change how a number should be read

**`granite_guardian_4_1_3b_factuality_lora` ran on base weights.** No `--enable-lora` flag
is passed in `p1_api.sh`, and none was passed in the study, so no adapter was ever attached.
This arm is base `granite-4.1-3b`, which is why its AUROC of 0.47487 sits below chance and
why the results report it as Granite-4.1-3B rather than as a factuality LoRA. Four
independent checks confirmed it. The verifier key keeps its original spelling because the
frozen score files are named after it; renaming it would detach the arm from its own scores.

**`minicheck_ft5` v1 scores are superseded.** The official MiniCheck-Flan-T5 inference
prepends `predict: ` to the model input. An earlier adapter omitted it. The current protocol
is `predictprefix` and the code rejects parquet files written under the old one. The v1
artifacts are retained and labelled rather than deleted.

Both are recorded in `code/paper_v3/DELIVERABLE/06_verifier_code/REGISTRY.md`, along with
the upstream repository, the audited state, and the principal boundary of every verifier.

## Complexity accounting

`code/paper_v3/DELIVERABLE/06_verifier_code/COMPLEXITY.md` gives the per-verifier cost model
in full: the parameter counts, the attention and feed-forward widths, and for the
mixture-of-experts models the activated width rather than the total. The complexity table is
derived from it, written out per architecture rather than as a single substituted formula.

## Licences

Third-party verifiers are not redistributed in this repository. Each carries its own
licence, and `REGISTRY.md` records the upstream repository and revision of every one so that
a reader can obtain them directly. FENICE in particular is non-commercial, and is an
optional verifier outside the frozen pool.
