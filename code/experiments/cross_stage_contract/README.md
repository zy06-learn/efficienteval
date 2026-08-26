# `cross_stage_contract/`

The frozen configuration the stage-4 control scripts resolve at run time.
`09_live_and_controls/code/_contract.py` reads `part1c_main_full_v1/00_contract/` from here,
so the controls are guaranteed to use the same pool, features and hyperparameters as the main
table rather than a re-derived copy.

| Entry | What it is |
|---|---|
| `part1c_main_full_v1` | a symlink to [`../01_main_experiment`](../01_main_experiment), the run whose contract is authoritative |
| `part1_main_pooled_v1/` | the earlier pooled run, kept for its contract and its main tables |

Nothing here is generated during reproduction; it is the pointer that keeps the stages
consistent.
