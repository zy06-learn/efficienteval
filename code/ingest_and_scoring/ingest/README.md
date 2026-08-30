# Stage 1 — ingest

| File | What it does |
|---|---|
| `build_splits_v2.py` | The only program allowed to read official-test gold. Unifies CoGenSumm, FRANK, RAGTruth and UniSumEval into one summary-level binary task, groups by `content_doc_key` (a hash of the normalised source document, so every summary of one document stays together), applies the pre-declared content-hash removal of five RAGTruth training rows whose documents also occur in the CoGenSumm test split, validates the contract, and writes `TRAIN.parquet`, `TEST.parquet`, `TEST_SCORING.parquet` and `P1_SCORING_COHORT.parquet`. |
| `verifier_cli.py` | Command-line entry for scoring one verifier over one cohort. |

`TEST_SCORING.parquet` is the point of the design: it is a projection of the test set with every
gold-derived field removed, and stage 2 reads *that*, never `TEST.parquet`. Scoring therefore
cannot reach a label even by accident, and `config_v2.py` fails closed if it tries.

This stage needs the four corpora, which are not redistributed here. Its frozen output is
shipped instead, with all text removed, in
[`../../experiments/00_inputs/`](../../experiments/00_inputs/).
