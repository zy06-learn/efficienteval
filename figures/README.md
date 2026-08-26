# `figures/`

The paper's three figures and the script that draws them. Both PDF (for the paper) and PNG
(for reading on GitHub) are shipped.

| Figure | What it shows |
|---|---|
| `quality_cost.*` | AUROC against end-to-end latency for the router and all fifteen fixed verifiers |
| `convergence.*` | head loss and routing AUROC against fit-partition size and forest size |
| `fewshot.*` | cross-corpus transfer: AUROC against the fraction of a held-out corpus's own training pool that is added back |

`make_figures.py` draws all three. It fits and evaluates nothing: every value it plots is a
literal in the script, transcribed from a published table. Each block names its source in a
comment -- `01_main_experiment/01_main_tables/publication/{A,B}_MAIN.csv`,
`03_ablation_extended/01_tables/E3_{LEARNING,TREE}_CURVE_{A,B}.csv`, and
`09_live_and_controls/results/FEWSHOT_FRACTION_CURVE.csv`.

**The transcription is not checked.** Because the numbers are copied rather than read from
the CSVs, a regenerated table does not change the figure until someone edits this script. If
you re-run an experiment, re-check the corresponding block here against its table.
