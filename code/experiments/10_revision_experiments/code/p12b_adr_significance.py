#!/usr/bin/env python3
"""EfficientEval minus each candidate ADR variant (p12), on sig_main.py's own replicates.

Replays E03's generator exactly as sig_cheap_20260929.py does (seed 20260821, 2,000
document-level replicates per protocol, Protocol A first, then B), checks the replay against
all stored E03 comparisons, and adds three ADR variants per protocol:
  nobudget  -- ADR with no latency budget (its highest-AUROC configuration);
  matched   -- the frontier point whose TEST latency is closest to EfficientEval's
               (the choice uses test latency, not labels; stated in the output);
  main      -- budget = EfficientEval's own validation latency at its selected beta.
Intervals: 95%, and Bonferroni at m = 17 (E03's 15 + the no-verifier arm + one ADR row,
i.e. the family if exactly one ADR row enters the paper's significance table).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3")
R = ROOT / "artifacts" / "part1c_main_full_v1" / "10_row_level"
E03 = ROOT / "runs" / "sig_main_v1"
P12 = ROOT / "runs" / "p12_adr_20260930"
OUT = P12 / "significance"
OUT.mkdir(parents=True, exist_ok=True)
B, RNG, M = 2000, np.random.default_rng(20260821), 17


def pct(d, m):
    a = 0.05 / m
    return np.percentile(d, [100 * a / 2, 100 * (1 - a / 2)])


gate, rows, picks = {}, [], {}
for proto in ("A", "B"):
    Z = np.load(R / f"probs_{proto}.npz", allow_pickle=True)
    D = np.load(P12 / proto / f"adr_rows_{proto}.npz", allow_pickle=True)
    assert (D["episode_key"].astype(str) == Z["episode_key"].astype(str)).all()
    y, groups = Z["y"].astype(int), Z["groups"].astype(str)
    ours = Z["prob__OURS"]
    n, S = ours.shape
    uniq, ginv = np.unique(groups, return_inverse=True)
    by_g = [np.where(ginv == k)[0] for k in range(len(uniq))]
    draws = [np.concatenate([by_g[k] for k in RNG.integers(0, len(uniq), len(uniq))])
             for _ in range(B)]

    def diffs(other):
        d_obs = float(np.mean([roc_auc_score(y, ours[:, j]) - roc_auc_score(y, other[:, j])
                               for j in range(S)]))
        ds = np.empty(B)
        for b, idx in enumerate(draws):
            yy = y[idx]
            if yy.min() == yy.max():
                ds[b] = np.nan
                continue
            ds[b] = np.mean([roc_auc_score(yy, ours[idx, j]) - roc_auc_score(yy, other[idx, j])
                             for j in range(S)])
        return d_obs, ds[np.isfinite(ds)]

    # gate: the replay reproduces every stored E03 comparison
    ref = pd.read_csv(E03 / f"{proto}_SIGNIFICANCE.csv").set_index("system")
    worst = 0.0
    for v, r in ref.iterrows():
        d_obs, ds = diffs(Z["prob__fixed__" + v])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        lob, hib = pct(ds, 15)
        worst = max(worst, abs(d_obs - r.d_auroc), abs(lo - r.ci95_lo), abs(hi - r.ci95_hi),
                    abs(lob - r.bonf_lo), abs(hib - r.bonf_hi))
    gate[proto] = {"max_abs_diff_vs_E03": float(worst), "pass": bool(worst < 1e-12)}
    assert worst < 1e-12, (proto, worst)

    ours_ms = float(Z["ms_full__OURS"].mean())
    front = [k.split("__", 1)[1] for k in D.files
             if k.startswith("ms__b") and np.isfinite(D[k]).all()]
    matched = min(front, key=lambda v: abs(float(D["ms__" + v].mean()) - ours_ms))
    picks[proto] = {"ours_ms": ours_ms, "matched_variant": matched,
                    "matched_ms": float(D["ms__" + matched].mean())}
    for label, v in (("nobudget", "nobudget"), ("matched", matched), ("main", "main")):
        d_obs, ds = diffs(D["prob__" + v])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        lb, hb = pct(ds, M)
        rows.append({"protocol": proto, "variant": label, "source_variant": v,
                     "d_auroc": d_obs, "ci95_lo": lo, "ci95_hi": hi,
                     "sig95": bool(lo > 0 or hi < 0), f"bonf{M}_lo": lb, f"bonf{M}_hi": hb,
                     f"sig_bonf_m{M}": bool(lb > 0 or hb < 0), "replicates": int(len(ds))})
        print(proto, rows[-1], flush=True)

pd.DataFrame(rows).to_csv(OUT / "ADR_SIGNIFICANCE.csv", index=False)
(OUT / "GATE.json").write_text(json.dumps({"replay_vs_E03": gate, "matched_pick": picks},
                                          indent=2))
print(json.dumps({"gate": gate, "picks": picks}, indent=2))
