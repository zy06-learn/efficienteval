#!/usr/bin/env python3
"""The cheap-features-only arm in the significance table, on sig_main.py's own replicates.

sig_main.py (E03) draws one set of 2,000 document-level replicates per protocol from a single
generator (seed 20260821, Protocol A first, then B) and reuses it for every comparison. This script
replays that generator, checks the replay against E03 on all fifteen stored comparisons, and adds
EfficientEval minus the no-verifier arm (part2_ablation_v1) on the same replicates. Bonferroni
intervals are written for both m = 15 (E03's family) and m = 16 (family including this arm), and
every E03 row is re-marked at m = 16 so any change of mark is visible.
"""
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/zeyu/projects/adaptive-faithfulness-router-v2/paper_v3")
R = ROOT / "artifacts" / "part1c_main_full_v1" / "10_row_level"
AB = ROOT / "runs" / "part2_ablation_v1" / "results"
E03 = ROOT / "runs" / "sig_main_v1"
OUT = ROOT / "runs" / "sig_cheap_20260929"
OUT.mkdir(parents=True, exist_ok=True)
B, RNG = 2000, np.random.default_rng(20260821)


def pct(d, m):
    a = 0.05 / m
    return np.percentile(d, [100 * a / 2, 100 * (1 - a / 2)])


gate, rows, remark = {}, [], []
for proto in ("A", "B"):
    Z = np.load(R / f"probs_{proto}.npz", allow_pickle=True)
    A = np.load(AB / f"ablation_{proto}.npz", allow_pickle=True)
    assert (A["episode_key"].astype(str) == Z["episode_key"].astype(str)).all()
    y, groups = Z["y"].astype(int), Z["groups"].astype(str)
    ours = Z["prob__OURS"]
    assert np.array_equal(A["prob__OURS"], ours)
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

    ref = pd.read_csv(E03 / f"{proto}_SIGNIFICANCE.csv").set_index("system")
    worst = 0.0
    for v, r in ref.iterrows():
        d_obs, ds = diffs(Z["prob__fixed__" + v])
        lo, hi = np.percentile(ds, [2.5, 97.5])
        lob, hib = pct(ds, 15)
        worst = max(worst, abs(d_obs - r.d_auroc), abs(lo - r.ci95_lo), abs(hi - r.ci95_hi),
                    abs(lob - r.bonf_lo), abs(hib - r.bonf_hi))
        l16, h16 = pct(ds, 16)
        remark.append({"protocol": proto, "system": v, "sig_bonf_m15": bool(r.sig_bonf),
                       "sig_bonf_m16": bool(l16 > 0 or h16 < 0)})
    gate[proto] = {"max_abs_diff_vs_E03": float(worst), "pass": bool(worst < 1e-12)}
    assert worst < 1e-12, (proto, worst)

    d_obs, ds = diffs(A["prob__no_verifier"])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    l15, h15 = pct(ds, 15)
    l16, h16 = pct(ds, 16)
    rows.append({"protocol": proto, "system": "no_verifier", "d_auroc": d_obs,
                 "ci95_lo": lo, "ci95_hi": hi, "sig95": bool(lo > 0 or hi < 0),
                 "bonf15_lo": l15, "bonf15_hi": h15, "sig_bonf_m15": bool(l15 > 0 or h15 < 0),
                 "bonf16_lo": l16, "bonf16_hi": h16, "sig_bonf_m16": bool(l16 > 0 or h16 < 0)})
    print(proto, rows[-1], flush=True)

pd.DataFrame(rows).to_csv(OUT / "CHEAP_SIGNIFICANCE.csv", index=False)
RM = pd.DataFrame(remark)
RM.to_csv(OUT / "E03_REMARK_AT_M16.csv", index=False)
gate["marks_changed_at_m16"] = RM[RM.sig_bonf_m15 != RM.sig_bonf_m16].to_dict("records")
(OUT / "GATE.json").write_text(json.dumps(gate, indent=2, default=lambda o: o.item()))
print(json.dumps(gate, indent=2, default=lambda o: o.item()))
