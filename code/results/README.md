# `results/` — the two pinned matrices stage 3 reads

**The name is the pipeline's output convention, not a description of the contents.** Every
module in this repository writes under a `results/` directory relative to its own root
(`OUTPUT_RELATIVE = Path("results/...")`), and these two files were written that way by
stages 1 and 2. To stage 3 they are *inputs*. The directory keeps the name because the paths
are baked into the frozen modules that read them, and renaming it here would leave the
pipeline writing to `results/` while reading from somewhere else.

| File | What it is |
|---|---|
| `unified_summary_verifiers_v1/ROUTER_TRAINING_MATRIX.parquet` | every verifier's score for every one of the 6,850 rows: what the router is trained and evaluated against |
| `summary_router_compact16_direct_v1/COMPACT16_FEATURES.parquet` | the cheap pre-call features, computed from the source and candidate summary alone |

`verifier_wrappers/pool_gate_sweep_v1.load_matrix` checks the sha256 of both against
`EXPECTED_MATRIX_SHA256` and `EXPECTED_FEATURE_SHA256`, and the row count against 6,850,
before returning. `shared/core.py` calls it at import, so a drifted matrix fails immediately
instead of silently changing a result.

These two files still carry source and summary text. The **text-free** bundle that reproduces
every published table without redistributing the corpora is
[`../experiments/00_inputs/`](../experiments/00_inputs/) -- use that one unless you are
re-running stage 2.
