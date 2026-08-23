# Reproduction

## What runs without anything else

Stage 3 reproduces every published table and ablation from this repository alone, on
CPU, with no model downloads and no corpus access. That is possible because stages 1 and 2
are shipped frozen in `code/paper_v3/DELIVERABLE/00_inputs/`: a 6.3 MB bundle carrying the
features and the fifteen verifiers' scores, with `source_document` and `candidate_summary`
removed and 59 of the original 256 columns kept.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./reproduce.sh verify
```

Expected output ends with:

```
reference arm = 0.8225560999095635
frozen        = 0.8225560999095635
delta         = 0.000e+00
  [gate] release-tree reference arm reproduces the frozen main table, delta 0.000e+00
```

`delta 0.000e+00` is the release gate, not a formality. Tolerance is zero. The only
randomness in the routing pipeline is the random forest's bootstrap and node-level feature
subsampling, both seeded, so any nonzero delta means something in the tree differs from what
produced the published results.

## Full stage 3

```bash
./reproduce.sh main        # main tables, both protocols
./reproduce.sh ablation    # core ablations
./reproduce.sh extended    # extended ablations and per-corpus tables
./reproduce.sh cascade     # cascade and alternative learners
```

Each writes to `runs/<stage>/` and is resumable: a stage that already wrote its `.done`
marker is skipped, so an interrupted run continues rather than restarting.

Pass `smoke` as a second argument to run two seeds instead of ten and a truncated feature
lattice, which is enough to check that the plumbing works.

## Stage 4

```bash
./reproduce.sh controls
```

Runs the dataset control, the few-shot curve, and the significance bootstrap. All CPU.

The live Protocol B re-run is separate because it calls every verifier for real: 32,360
calls across ten seeds, needing a GPU and a running vLLM server.

```bash
export AFR_ROOT="$PWD/code" LIVE_OUT="$PWD/runs/live"
export AFR_INPUTS="$AFR_ROOT/paper_v3/DELIVERABLE/00_inputs"
for stage in plan local api finish; do
  LIVE_STAGE=$stage python code/paper_v3/DELIVERABLE/09_live_and_controls/code/live_main.py
done
```

The four stages are split deliberately. Routing is computed once in `plan`, then the local
encoders and the served models run in separate phases, because both would otherwise contend
for the same GPU.

## Stages 1 and 2

These cannot run from this repository alone. Stage 1 needs the four corpora; stage 2 needs
the model weights and a vLLM server.

```bash
export AFR_ROOT="$PWD/code" AFR_PYTHON="$(command -v python)"
pip install -r requirements-verifiers.txt

python code/paper_v2/ingest/build_splits_v2.py     # stage 1
bash   code/paper_v2/p1_score_local.sh             # stage 2a, nine local verifiers
bash   code/paper_v2/p1_api.sh                     # stage 2b, six served verifiers
```

`p1_api.sh` brings up one vLLM server at a time and takes down each before the next.
`HF_HUB` must point at a hub cache that already holds the pinned snapshots; the revisions
are recorded in `MODEL_REVISIONS` in `code/afr_v2/unified_summary_verifiers_v1.py`.

## Environment

Recorded: dgxspark, NVIDIA GB10, CUDA 13.0, driver 580.142, Python 3.12, vLLM 0.20.0 on the
V1 engine.

Versions in `requirements.txt` are pinned rather than floated. The bit-for-bit gate is
sensitive to the scikit-learn build, so a floated dependency would turn a reproduction check
into a reproduction guess.

## Determinism, and where it stops

Three things are worth stating plainly, because a reader who reruns this will meet all of
them.

**The local encoders are bit-identical on re-run.** Re-scoring the same rows with the nine
locally-hosted verifiers returns the same values.

**The served verifiers are not.** vLLM prefix caching means a row can hit a warm prefix and
return a score differing in the last digits. This is the entire residual between the live
Protocol B re-run (0.822050) and the frozen matrix result (0.822556): a delta of −0.00051
over 32,360 calls. It is a property of the serving stack, not of the router, and it is why
`--enable-prefix-caching` is documented in `p1_api.sh` rather than quietly set.

**Row order is a randomization source.** Feeding the same rows in a different order moves
the result by about as much as changing the seed. This is not specific to this work, and it
is not usually reported. It is recorded here because a reader who reshuffles the input and
sees movement should know it is expected.

## Repository layout, and why the directories are named that way

`code/` is `AFR_ROOT`. Its internal layout is the layout the experiments actually ran under,
and the running code has not been renamed after the fact.

- `afr_v2/` holds the verifier implementations and scoring wrappers.
- `paper_v1/` is **not** a superseded copy of the experiments. It contributes exactly two
  modules, `core.py` and `config.py`, that the current pipeline still imports.
- `paper_v2/` is stage 1 and stage 2.
- `paper_v3/` is stage 3 and stage 4, plus all published evidence under `DELIVERABLE/`.
- `results/` holds the two sha256-pinned frozen matrices that `core.py` verifies at import.

Superseded rounds are not part of this release. Every code file appears exactly once: the
per-experiment code snapshots in the original working tree were byte-identical duplicates,
and one canonical copy is kept in their place.

The following were changed from the working tree so that the pipeline runs outside the
author's machine. Nothing else was touched, and the reference gate was rerun after every one
of them and still reports `delta 0.000e+00`.

1. `AFR_ROOT` now defaults to the repository code root, derived from each file's own location
   (21 call sites). It previously fell back to an absolute path in the author's home
   directory, so any script run without the variable set failed with a `FileNotFoundError`
   naming a machine the reader does not have.
2. The four stage-3 launchers point at `08_scripts/` and default to `python3` rather than an
   absolute interpreter path.
3. `config_v2.py` defaults its interpreter paths to `sys.executable`; `candidate_verifiers.py`
   reads `ALIGNSCORE_PYTHON` and `ALIGNSCORE_CHECKPOINT`; `p1_prepare.py` derives its own
   directory instead of naming one.
4. `p1_api.sh` and `p1_score_local.sh` take `AFR_ROOT`, `AFR_PYTHON`, `HF_HUB`, `VLLM_API` and
   `P1_SCORING_DIR` instead of absolute paths. `p1_api.sh` also gained the
   `granite_guardian_3_1_2b` entry: it is the third member of the frozen pool, but its scores
   came from an earlier round under a launcher that is not part of this release, so serving it
   was undocumented here.
5. The stage-2 CLI entry point was renamed from `169_unified_summary_verifiers_v1_reference.py`
   to `ingest/verifier_cli.py` and now resolves the `afr_v2` package itself rather than relying
   on an externally set `PYTHONPATH`.
6. Byte-identical code snapshots were removed in favour of one canonical copy each, and all
   three manifests were regenerated to match.

Run records under `*/0*_provenance/` keep their original absolute paths. They are evidence of
what ran, not configuration.
