# `05_pool_provenance/`

How the three-verifier pool (`factcc`, `lettuce_v2`, `granite_guardian_3_1_2b`) was chosen,
and the audit of that choice.

The pool is the one part of the frozen configuration **not** selected on TRAIN alone. It was
screened in an earlier round on a 6,850-row matrix that shares 54.4% of Protocol B's test
document groups, and was never reselected afterwards. This directory exists so that the
reader can see this rather than infer it.

| File | What it is |
|---|---|
| `results/POOL_FIXED_REFERENCE.csv` | the pool as frozen, and the screen it came from |
| `results/POOL_RESCREEN_B.csv` | all 455 three-verifier subsets rescreened on TRAIN alone, under the current six features |
| `results/POOL_RESCREEN_A_CONFIRM.csv` | the same rescreen under Protocol A |
| `rescreen.log` | the run |

The rescreen is an audit. It does not feed back into the frozen configuration, and no result
in the paper was reselected using it.

Code: [`../08_routing_code/pool_rescreen_v1.py`](../08_routing_code/pool_rescreen_v1.py).
