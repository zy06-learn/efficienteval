# `ingest_and_scoring/` — stages 1 and 2

Stage 1 builds the TRAIN/TEST split from the four corpora. Stage 2 drives the fifteen
verifiers over it and writes the score matrix.

| File | What it does |
|---|---|
| `ingest/build_splits_v2.py` | stage 1: normalises the four corpora, derives `content_doc_key` from source-document content, and cuts the grouped TRAIN/TEST split |
| `ingest/verifier_cli.py` | the command-line front end for scoring one verifier |
| `p1_prepare.py` | assembles the scoring worklist |
| `p1_score.py`, `p1_score_local.sh`, `p1_api.sh` | stage 2: local and served scoring runs |
| `config_v2.py` | the frozen protocol inputs and paths for both stages. Stage 1 and 2 share `../shared/core.py` for folds, calibration, head fitting and metrics. |

**Neither stage runs from this repository alone.** Stage 1 needs the four corpora, which are
not redistributed here; stage 2 needs GPU weights and a vLLM server. Their output is shipped
frozen and text-free in [`../experiments/00_inputs/`](../experiments/00_inputs/), which is
what every published table is actually computed from. See
[`../../docs/reproducibility.md`](../../docs/reproducibility.md) for what each stage costs.
