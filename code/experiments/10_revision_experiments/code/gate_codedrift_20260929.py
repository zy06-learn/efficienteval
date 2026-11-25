#!/usr/bin/env python3
"""Gate: does today's code reproduce the historical (Aug 7) R6 screening numbers on the full
6,850-row matrix? If yes, the test-excluded re-run used the historical procedure unchanged."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ.setdefault("OMP_NUM_THREADS", "10")
ROOT = "/home/zeyu/projects/adaptive-faithfulness-router-v2"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from pathlib import Path
import numpy as np, pandas as pd
from afr_v2 import tenfold_v1 as core, pool_gate_sweep_v1 as v1
import day2_run as D
m = v1.load_matrix(Path(ROOT)); assert len(m) == 6850
hist = pd.read_csv(Path(ROOT) / "results/day2_v1/R6_SCREEN.csv").set_index("pool")
def screen(pool, seed=17, n_rot=4):
    actions = v1.order_by_cost(m, pool); costs = v1.pool_costs(m, pool)
    folds = core.build_folds(m, seed); va, vm = [], []
    for test_f, val_f, train_f in core.rotations()[:n_rot]:
        tr = m.loc[np.isin(folds, train_f)].reset_index(drop=True)
        val = m.loc[folds == val_f].reset_index(drop=True)
        cals = core.fit_calibrators_on_val(val, actions)
        c_tr = core.apply_calibrators(tr, actions, cals); c_va = core.apply_calibrators(val, actions, cals)
        hv = D.fit_heads(tr, [val], D.FEATS, actions, seed, "brier", c_tr, "hgb")[0]
        beta = core.select_beta(val, actions, hv, c_va, costs)[0]
        p = v1.router_prediction(val, actions, hv, c_va, costs, beta)
        va.append(v1.scope_auroc(p, "pooled")); vm.append(float(p["end_to_end_latency_ms"].mean()))
    return float(np.mean(va)), float(np.mean(vm))
for pool in [("factcc", "lettuce_v2", "granite_guardian_3_1_2b"), ("factkb", "lettuce_v2"),
             ("qwen30_fast", "factcg", "alignscore")]:
    name = "+".join(v1.order_by_cost(m, pool))
    a, ms = screen(pool)
    h = hist.loc[name]
    print(f"{name:45s} now {a:.10f} {ms:.6f} | Aug7 {h.val_auroc:.10f} {h.val_ms:.6f} | "
          f"dAUROC {abs(a-h.val_auroc):.1e} dms {abs(ms-h.val_ms):.1e}")
