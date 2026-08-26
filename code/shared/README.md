# `shared/` — the two modules every stage imports

Not an old version of anything, and not superseded. This directory contributes exactly two
files, and the current pipeline imports both.

| File | What it holds |
|---|---|
| `core.py` | Platt and isotonic calibration, head fitting, cost folding, routing, the paired cluster bootstrap, and the metric definitions. The reference gate calls `core.platt`, `core.apply_platt`, `core.fit_heads`, `core.fold_costs` and `core.isotonic`. |
| `config.py` | the frozen configuration: the verifier pool, the six cheap features, the action space, and the path roots derived from `AFR_ROOT`. |

At import, `core.py` calls `verifier_wrappers/pool_gate_sweep_v1.load_matrix`, which checks the
sha256 of both matrices in [`../results/`](../results/) and the 6,850-row
count before returning. A drifted input fails loudly rather than producing a different
number quietly.

The working tree called this directory `paper_v1`, which read as "the superseded first
version". It never was.
