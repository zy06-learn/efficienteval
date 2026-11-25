#!/usr/bin/env python3
"""Protocol A replication of the cascade-frontier sweep of p0_cascade_v1.py.

One seed produces one full-length out-of-fold prediction vector, assembled from the ten 8/1/1
rotations. Everything else -- the trigger set, the order and target enumeration, the threshold
grid, the router's own beta grid -- is identical to the Protocol B sweep.
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3core as V           # noqa: E402
import core                  # noqa: E402
import p0_cascade_v1 as CB    # noqa: E402
from p0_cascade_v1 import Fitted, frontier, NICE, POOL, GRID   # noqa: E402

# Fitted reads the module-global hyperparameters of the Protocol B sweep. Protocol A selected its
# own set on TRAIN, so swap them in before anything is fitted; otherwise the router row here would
# not match the Protocol A column of the main table.
CB.HP_B = dict(json.loads(
    (HERE / "artifacts" / "part1c_main_full_v1" / "00_contract" / "HP_SELECTED.json").read_text()
)["hyperparameters_A"])
print("Protocol A hyperparameters:", CB.HP_B)

SEEDS = list(V.SEEDS)
OUT = HERE / "results_p0exp"; OUT.mkdir(exist_ok=True)


def au(y, p):
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def main():
    _tr, _te, ALL, _v = V.load(with_test_labels=True)
    ALL = ALL.reset_index(drop=True)
    y = ALL["label_supported"].to_numpy(int)
    n = len(ALL)

    blocks = []          # (seed_index, positions, Fitted)
    for si, seed in enumerate(SEEDS):
        fold = V.folds_stratified(ALL, seed)
        for test_fold, val_fold, train_folds in V.rotations():
            m_fit, m_val, m_ev = np.isin(fold, train_folds), fold == val_fold, fold == test_fold
            blocks.append((si, np.flatnonzero(m_ev),
                           Fitted(ALL.loc[m_fit].reset_index(drop=True),
                                  ALL.loc[m_val].reset_index(drop=True),
                                  ALL.loc[m_ev].reset_index(drop=True), seed)))
        print(f"  fitted seed {seed} ({len(blocks)} blocks)", flush=True)

    def agg(fn):
        P = np.zeros((n, len(SEEDS)))
        MS = np.zeros((n, len(SEEDS)))
        CA = []
        for si, pos, F in blocks:
            r = fn(F)
            if r is None:
                return None
            q, ms, ca = r
            P[pos, si] = q
            MS[pos, si] = ms
            CA.append(ca)
        return (float(np.nanmean([au(y, P[:, i]) for i in range(P.shape[1])])),
                float(MS.mean()), float(np.mean(CA)))

    rows = []
    for beta in GRID:
        a, m, c = agg(lambda F, b=beta: F.router(b))
        rows.append({"family": "router (pre-call)", "config": f"beta={beta}",
                     "auroc": a, "ms": m, "calls": c})
        print(f"  router beta={beta}: {a:.4f} @ {m:.1f} ms", flush=True)

    TAU_Q = np.linspace(0.0, 0.95, 20)
    pairs = list(itertools.permutations(range(len(POOL)), 2)) + [(f, -1) for f in range(len(POOL))]
    for signal in ("confidence", "learned", "learned+score"):
        for first, second in pairs:
            for qi in TAU_Q:
                qs = [F.signals(first, signal) for _si, _p, F in blocks]
                if any(q is None for q in qs):
                    continue
                tau = float(np.mean([np.quantile(q[0], qi) for q in qs]))
                r = agg(lambda F, f=first, s=second, t=tau, g=signal: F.cascade(f, s, t, g))
                if r is None:
                    continue
                a, m, c = r
                rows.append({"family": f"cascade ({signal})",
                             "config": (f"{NICE[POOL[first]]}->"
                                        f"{'best remaining' if second < 0 else NICE[POOL[second]]}"
                                        f" q={qi:.2f}"),
                             "auroc": a, "ms": m, "calls": c})
        print(f"  swept {signal}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "E5_cascade_sweep_A.csv", index=False)
    print("\n===== best at or below each budget (Protocol A) =====")
    for fam in df["family"].unique():
        sub = df[df.family == fam]
        line = [fam]
        for budget in (100.0, 112.1, 130.0, 1e9):
            ok = sub[sub.ms <= budget]
            line.append(f"{ok.auroc.max():.4f}" if len(ok) else "  --  ")
        best = sub.loc[sub.auroc.idxmax()]
        print(f"  {line[0]:34s} " + "  ".join(line[1:]) +
              f"   best={best.auroc:.4f} @ {best.ms:.1f} ({best.config})")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"\n[cascade-A] {time.time()-t0:.0f}s")
