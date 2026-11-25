# Revision experiments (September–October 2026)

The experiments added after the 2026-08-30 release, while the paper was being revised. They use
the same frozen system, the same TRAIN/TEST boundary and the same score matrix as `01_` to `09_`;
none of them changes the configuration. Several only add metrics to a result that was already
published, and those re-derive the published numbers first and stop if any cell differs.

## Layout

| | |
|---|---|
| `code/` | Every script that produced a result below, byte-identical to what ran. |
| `results/<run>/` | The outputs of one run: CSV/JSON tables, its reproduction gate, logs. |

**About paths.** These scripts ran in the original working tree, before the directories were
renamed for this release (`paper_v3` → `experiments`, `paper_v2` → `ingest_and_scoring`,
`paper_v1` → `shared`, `afr_v2` → `verifier_wrappers`; see `../../README.md`). They are kept
exactly as executed so that the code hashes in the run records stay valid, which means their
path constants refer to the working tree and need `AFR_ROOT`-style rewiring to run from a fresh
clone. `../11_rebuttal_x1/` shows that rewiring for two scripts and verifies it bit for bit.

## What each run is, and where it appears in the paper

| Run | What it answers | Script(s) | Paper |
|---|---|---|---|
| `E08_p0_rules_poolsize_mixture_cascadesweep` | TRAIN-only greedy pool path to k=15; alternative selection rules; full cascade-family sweep (3 triggers × 9 configurations × 20 thresholds); Dirichlet mixture robustness. Both protocols. | `p0_exp_v1.py`, `p0_exp_a_v1.py`, `p0_cascade_v1.py`, `p0_cascade_a_v1.py` | routing baselines, pool sensitivity, mixture |
| `E12_live_metrics_p4` | All metrics for the live re-execution, recomputed from its saved per-row scores after first reproducing the main table from the frozen matrix. | `p4_live_metrics.py` | `tab:live` |
| `E13_aligned_front_ensemble_p5` | Cascade frontier and equal-budget cheap ensembles on the main table's metric set; 550 configurations per protocol re-derived before new columns are written. | `p5_front_metrics.py`, `p5_ensemble_metrics.py` | routing baselines |
| `E16_percorpus_full_p7` | Single-corpus training and evaluation, 4 corpora × 18 arms × 2 protocols. | `p6_percorpus_gate.py`, `p6_table7_complete.py`, `run_p6_table7_complete.sh`, `p7_percorpus_full.py` | `tab:pc-*` |
| `E18_fewshot_metrics_p9` | Transfer and adaptation sweep (leave one corpus out, then add 0–100% of its pool) with the full metric set; the 80 published AUROC/ms cells are reproduced first. | `p9_fewshot_metrics.py` | `tab:fewshot-curve` |
| `E19_poolsize_metrics_p10` | Pool-size sweep with the full metric set, on the TRAIN-only greedy order from E08. | `p10_poolsize_metrics.py` | `tab:pool-sensitivity` |
| `E20_matched_quality_p11` | Latency each system needs to reach a given quality target. | `p11_matched_quality.py` | routing baselines |
| `E21_mixture_fix_p8b` | Natural, equal and Dirichlet corpus mixtures for all sixteen systems. Supersedes `p8_mixture_metrics.py`, which used one seed's thresholds for every Protocol A resample; both scripts are kept. | `p8_mixture_metrics.py`, `p8b_mixture_fix.py` | `tab:mixture-full` |
| `E22_testblind_selection` | Test-blind exhaustive pool selection, with the selection rule frozen in writing before any test row was read (`FREEZE_*.md`). Re-runs the historical selection-stage router, the paper's final router, and other screening seeds. | `r6r9_trainside_20260929.py`, `apply_frozen_selection_20260929.py`, `sens_runnerup_20260929.py`, `r6_reseed_20260929.py`, `gate_codedrift_20260929.py`, `day2_run.py`, `exp1_paperconfig_20260929.py`, `run_exp1_paperconfig_20260929.sh`, `exp1_altpools_final_20260929.py` | pool provenance |
| `E23_percorpus_frozen_exp2` | The deployed pooled-trained router read one corpus at a time; within-corpus bootstrap, permutation controls, corpus-only router, per-corpus call shares, and the within/cross-corpus decomposition of pooled AUROC. | `exp2_percorpus_20260929.py` | `tab:percorpus-frozen` |
| `E24_sig_cheap` | Significance of the cheap-features-only row, on the same bootstrap draws as the main table (Bonferroni family 15 → 16). | `sig_cheap_20260929.py` | `tab:significance` |
| `E25_adr_p12` | ADR-v1, the adaptive-detector-routing baseline, adapted to this task, with its significance test. | `p12_adr_baseline.py`, `p12b_adr_significance.py` | routing baselines |

## Gates

Each run that re-derives published numbers writes its check next to its outputs
(`*_GATE.json`, `*_REPRODUCTION.csv`, or the reproduction section of its log). The tolerance is
1e-9 on quality; where a latency column is measured live rather than read from the matrix, the
log records the difference and the historical value is kept. One log in `E22` is 219 MB and is
shipped as a first/last-200-line excerpt.
