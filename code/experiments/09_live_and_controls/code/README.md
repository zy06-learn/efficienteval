# Stage 4 — code

The scripts behind [`../`](../): the live re-execution of Protocol B, the three controls, and
the significance intervals for both the main tables and the controls.

[`../README.md`](../README.md) states what each experiment found.
[`../../../../docs/pipeline.md`](../../../../docs/pipeline.md) puts them in the pipeline.
The repository README has a one-line summary of every file
([Every file, and what it does](../../../../README.md#every-file-and-what-it-does)).

Read `_contract.py` first. Everything here inherits it, and it is why any of the rest can be
trusted: it compares the frozen configuration field by field and refuses to run on drift, and
`check_reference` requires any arm that is nominally the main system to land on
`FROZEN_B_AUROC = 0.8225560999095635` before its own numbers are believed. Several of these
scripts refit the main system from scratch, so that check is not a formality.

| File | Question |
|---|---|
| `_contract.py` | The guard, and the two-level reproduction tolerance with the measured cross-platform deltas that justify it. |
| `live_main.py` | Is the reported test result read from a pre-computed matrix? 32,360 real verifier calls say no. This is the source of `LIVE_MAIN_B.json`, the number the paper reports. |
| `live_pipeline.py` | The single end-to-end pass of the same check, superseded by `live_main.py`. Kept because `LIVE_PIPELINE.json` came from it. |
| `rerun_score.py`, `cmp.py`, `diag_live.py` | Diagnostics; no published artifact. Where does the live-versus-matrix residual come from? It traces to vLLM prefix caching, not to routing. |
| `ds_only.py` | Does the router receive dataset identity? Four arms, one of which hands it the corpus explicitly. The version the paper reports. |
| `dataset_control.py` | The same question in three parts. Parts (a) and (b) are unique to it; part (c) is superseded by `ds_only.py`, and the two differ by about 0.0016 because they build the corpus one-hot in a different column order. |
| `fewshot_frac.py` | How much of its own pool does a held-out corpus need? Fractions, not counts. |
| `fewshot.py`, `fewshot_k0.py` | Superseded by `fewshot_frac.py`. Kept because `FEWSHOT_CURVE.csv` came from them. |
| `sig_main.py` | Paired cluster bootstrap for the main tables, Bonferroni over fifteen comparisons. |
| `sig_controls.py` | The same test for the controls, with family sizes 1, 3 and 4. |

Every script derives `AFR_ROOT` from its own location, so a fresh clone runs without any
environment set up. `tests/test_reproduction.py::test_control_scripts_resolve_their_imports`
checks that the directories each one puts on `sys.path` exist and carry the modules it imports.
