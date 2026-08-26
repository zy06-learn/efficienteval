# `03_ablation_extended/`

The ablations that did not fit in `02_ablation_core/`: how the router behaves inside a single
corpus, which features carry the signal, and whether the fit has converged.

| Table | What it answers |
|---|---|
| `01_tables/E1_PERCORPUS*.csv` | trained and evaluated inside one corpus, against all fifteen fixed verifiers. This is where the router's limit shows: within a single corpus it does not lead. |
| `01_tables/E1_PRIOR_SHIFT.csv` | how much of the pooled result is label-prior shift between corpora |
| `01_tables/E2_LATTICE_*.csv`, `E2_BEST_K_*.csv` | every feature subset, and the best subset at each size |
| `01_tables/E2_SHAPLEY_*.csv` | per-feature Shapley attribution |
| `01_tables/E3_LEARNING_CURVE_*.csv` | AUROC against the fraction of fit-partition document groups |
| `01_tables/E3_TREE_CURVE_*.csv` | AUROC against forest size; the paper's convergence figure is drawn from this |
| `01_tables/E4E5_DECLARED_*.csv` | the pre-declared secondary arms and their paired intervals |

`REPORT_zh.md` reads the results out. The directory shape (`02_gates/`, `03_provenance/`,
`05_logs/`, `06_row_level/`) is the same as every other experiment directory; see
[`../README.md`](../README.md).

Code: [`../08_routing_code/part3_extended_v1.py`](../08_routing_code/part3_extended_v1.py)
and `part3_percorpus_selected_v1.py`. Run with `./reproduce.sh extended phase1` then
`phase2`.
