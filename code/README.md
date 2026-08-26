# `code/` — AFR_ROOT

Everything the pipeline runs from. `reproduce.sh` exports `AFR_ROOT` to this directory, and
every module resolves its paths from it, so the repository can live anywhere on disk.

| Directory | Stage | What it is |
|---|---|---|
| [`ingest_and_scoring/`](ingest_and_scoring/) | 1, 2 | Builds the TRAIN/TEST split from the four corpora, then drives the fifteen verifiers over it. |
| [`verifier_wrappers/`](verifier_wrappers/) | 2 | The wrappers stage 2 calls, one per verifier. The upstream checkouts they wrap are not redistributed; they belong in a sibling `verifiers/` directory. |
| [`shared/`](shared/) | all | Two modules, `core.py` and `config.py`, that every later stage imports. |
| [`results/`](results/) | 2 → 3 | The two sha256-pinned matrices stages 1 and 2 produced, which stage 3 reads as inputs. |
| [`experiments/`](experiments/) | 3, 4 | The routing experiments, the controls, and every published result. |
| [`scripts/`](scripts/) | 2 | One serving helper, called by a verifier wrapper. |

**Stage 3 alone reproduces every published table.** Stages 1 and 2 need the corpora, GPU
weights and a running vLLM server; their output is shipped frozen in
[`experiments/00_inputs/`](experiments/00_inputs/), so a fresh clone can go straight to
stage 3 on CPU. See [`../docs/pipeline.md`](../docs/pipeline.md).

These directories were renamed for the release; the working tree called them `paper_v1`,
`paper_v2`, `paper_v3`, `afr_v2` and `DELIVERABLE`. Files under `*_provenance/` still record
the original names, because they document what actually ran.
