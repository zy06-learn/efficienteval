# Verifier code — executed path and the facts a complexity analysis needs

This directory exists so that a time-complexity analysis can be written against the code this
project actually ran, not against model cards or upstream defaults. Four upstream chunking
defaults are overridden by wrapper classes here, and two of the fifteen verifiers report their
forward counts through columns that mean different things, so a complexity claim derived from the
wrong source is wrong by a large factor rather than a small one.

Everything below was read from the source in `../../../afr_v2/` during this audit. Where a claim is carried
over from the earlier notes rather than re-verified, that is stated.

## The executed path

`../../../paper_v2/p1_score.py` is the driver. It calls `build_scorer` and `score_frame` from
`../../../afr_v2/unified_summary_verifiers_v1.py`. `build_scorer` dispatches each verifier to one of five
wrappers, and this dispatch is the thing to read first, because the wrapper determines the
execution pattern, not the model:

| Wrapper | Verifiers | Windowing lives in | Forward accounting |
|---|---|---|---|
| `PairWindowScorer` | `factcc`, `factkb`, `wecheck` | `source_windows_preserving_summary`, this file | `model_forward_calls = forward_items = C` |
| `SummaryPreservingMiniCheck` | `minicheck_dbta`, `minicheck_ft5` | same function, this file | `model_forward_calls = 1`, `forward_items = C` |
| `SummaryPreservingFactCG` | `factcg` | same function, this file | as written by the class |
| `MetadataScorer` | `hhem`, `alignscore`, `lettuce_v2` | inside the wrapped scorer | derived from the wrapped scorer's `n_chunks` |
| `CompletionAPIScorer` | the six Granite and Qwen models | no windowing; one prompt | `1 + g` |

Two corrections to the earlier notes follow from this table.

1. **HHEM and AlignScore are not `PairWindowScorer`s.** They are `MetadataScorer`s wrapping
   scorers built by `build_primary_scorer` in `../../../afr_v2/primary_scoring.py`. HHEM's windowing is
   `build_hhem_token_windows` in that file, which is a separate implementation from the one the
   other windowed verifiers use. The earlier notes described HHEM's windowing correctly but
   filed it under the wrong wrapper, so a reader following the dispatch would not find it.
2. **`MetadataScorer` derives its counts rather than measuring them.** It reads
   `aux["n_chunks"]` and falls back to `source_window_count`, then to 1. A wrapped scorer that
   never writes `n_chunks` therefore reports `forward_items = 1` regardless of how many forward
   passes it made. This is the mechanism behind the Lettuce-v2 counting caveat in the notes, and
   it is a property of this wrapper, not of Lettuce.

Every scorer asserts `len(docs) == 1 and len(claims) == 1`. **Batch size is one everywhere**, by
construction, so per-instance latency is measurable but throughput numbers from this harness are
not batch-optimal.

## Windowing: the exact executed formula

`source_windows_preserving_summary`, used by FactCC, FactKB, WeCheck, FactCG and both MiniCheck
variants:

```
E(b)   = length of the encoded pair with an empty source        # empty_length
budget = max(1, T_max - E(b))
o      = min(64, max(0, budget // 4))                            # overlap shrinks with budget
advance per step = len(window_ids) - o
```

giving the closed form the notes report:

```
C = ceil( (m - o) / (T_max - E(b) - o) )
```

**One correction.** The closed form is a lower bound, not an exact count. After slicing `budget`
token ids, the function re-encodes the *decoded* chunk as a pair and, while that length exceeds
`T_max`, shrinks the window and re-encodes:

```python
while length > max_length and len(ids) > 1:
    ids = ids[: max(1, len(ids) - (length - max_length) - 1)]
```

Decoding and re-encoding is not length-preserving, so the effective window can be smaller than
`budget` and the true `C` can exceed the closed form. The earlier notes presented the formula as
exact. For asymptotics this changes nothing; for predicting a specific instance's window count it
matters, and `source_window_count` in the scoring output is the ground truth.

HHEM uses `build_hhem_token_windows` in `../../../afr_v2/primary_scoring.py` instead, which differs in two
ways the notes had right:

```
budget = T_max - overhead - safety_margin      # safety_margin = 8
o      = 64, fixed, never shrunk by budget
C      = ceil( (m - o) / (T_max - E(b) - 8 - o) )
```

## The counting columns, and which one to trust

Three columns in `paper_v2/results/p1_scoring/*.parquet` describe forward work, and they disagree
on purpose:

| Column | Meaning | Trustworthy for |
|---|---|---|
| `model_forward_calls` | number of times the model was invoked | the windowed group; **always 1** for MiniCheck |
| `forward_items` | number of inputs actually pushed through | the windowed and batched groups; **degenerates to 1** for Lettuce-v2 |
| `source_window_count` | windows produced by the packer | the windowed group |
| `output_tokens` | generated tokens | the six API models, where cost is `1 + g` |

Computing complexity from `model_forward_calls` for MiniCheck yields the conclusion that its cost
is independent of source length, which is false; computing it from `forward_items` for Lettuce-v2
yields a constant, which is also false. **A complexity analysis must select the column per
verifier group**, and the mapping is the table above.

## Generation caps for the API models

Read from `CompletionAPIScorer.__init__`:

```python
self.max_tokens = 64 if verifier == "qwen30_judge" else 20
if verifier == "qwen30_fast":
    self.max_tokens = 4
```

So `qwen30_judge` may emit up to 64 tokens, `qwen30_fast` exactly 4, and every Granite model up
to 20. The measured medians in the earlier notes (Granite-3.1-2b `g = 2`, Granite-3.2-3b `g = 10`,
Granite-3.2-8b median 10 max 20, Granite-4.1-3b `g = 7`, `qwen30_fast` always 4, `qwen30_judge`
median 18 max 64) are consistent with these caps but were measured on a 6,850-row matrix and are
carried over, not re-measured here.

Prompt construction differs per model family and affects the prefill length but not the
asymptotics: Granite 3.1/3.2-3b use a `guardian_config` chat template with a `context` role,
Granite 3.2-8b uses the plain template, and Granite 4.1-3b puts the summary in an `assistant`
turn followed by a long instruction in a `user` turn. The earlier notes did not record the 4.1
layout.

`_select_source` is a label-free sentence-selection fallback for prompts that exceed
`max_context = 16384`. The earlier notes report that `source_selected` is false on every row of
all six models, so this branch never fired; that observation is carried over.

## What still needs measuring before a complexity section can be written

The symbolic derivations in the earlier notes (its Tables C, D, E and A: forward counts, per-pass
cost, parameter-substituted cost, and model structure) depend only on model architecture and the
executed code path. This audit confirms the code path they assume, with the two corrections
above, so those tables remain usable.

Every **measured** column in them does not transfer, because it was taken from the 6,850-row
Protocol A matrix that the current contract replaced:

- `F` quantiles (median, p99, max window or forward counts) per verifier
- median latency per verifier
- generated-token distributions
- the `corr(t, m)` and `corr(t, b)` correlation table
- the claim that the longest summary is 1,444 tokens; the current pooled frame contains a
  2,155-token summary, so windowed verifiers reach larger counts than those quantiles indicate

All of these can be recomputed from the frozen `p1_scoring` parquets, which are unchanged and
whose hashes are recorded in `01_main_experiment/07_provenance/DATA_INPUTS.sha256`. No verifier
needs to be re-run.

## Latency measurement caveat that affects the main tables

`granite_guardian_3_1_2b` carries two measurement regimes that differ by 1.5 to 1.7x between
scoring campaigns covering different corpora, and it takes 30.9% of Protocol B traffic. The main
tables report the as-measured figure, which is the conservative end; normalising to the slower
campaign widens the router's latency advantage over the strongest pool member from 1.24x to
1.51x. This is recorded in `01_main_experiment/README.md` and is a property of the recorded
latencies, not of the code in this directory.

## File map

| File | Lines | What to read it for |
|---|---:|---|
| `../../../paper_v2/p1_score.py` | 115 | the driver, and how scoring outputs are written |
| `../../../afr_v2/unified_summary_verifiers_v1.py` | 1342 | dispatch, windowing, four of the five wrappers, `score_frame` |
| `../../../afr_v2/primary_scoring.py` | 357 | HHEM windowing, and `build_primary_scorer` for hhem / alignscore / factcg / minicheck |
| `../../../afr_v2/unified_scoring.py` | 1682 | FactCC and FactKB pinned scorers |
| `../../../afr_v2/extended_scorers.py` | 332 | Lettuce-v2 scorer |
| `../../../afr_v2/additional_verifier_scorers_v1.py` | 449 | WeCheck scorer |
| `REGISTRY.md` | — | the verifier registry as maintained in `verifiers/` |

AlignScore runs in a separate subprocess with its own environment
(`ALIGNSCORE_PYTHON` in `config_v2.py`), so the external counters see one call for it regardless
of the `B * n_b` pairs it computes internally. Its cost structure is the one genuinely
multiplicative case among the fifteen and has to be read from the upstream
`inference_per_example`, not from this repository.
