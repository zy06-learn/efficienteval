import os
#!/usr/bin/env python3
"""Paired cluster bootstrap for the main tables, with Bonferroni correction over the 15
fixed-verifier comparisons. Resamples content_doc_key, not rows, because summaries of one
source document are not independent."""
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.metrics import roc_auc_score

# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
R = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT) + "/paper_v3/artifacts/"
         "part1c_main_full_v1/10_row_level")
OUT = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT) + "/paper_v3/runs/sig_main_v1")
OUT.mkdir(parents=True, exist_ok=True)
B, RNG = 2000, np.random.default_rng(20260821)

for proto in ("A", "B"):
    Z = np.load(R / f"probs_{proto}.npz", allow_pickle=True)
    y, groups = Z["y"].astype(int), Z["groups"].astype(str)
    systems = [s for s in Z["systems"].astype(str) if s.startswith("fixed::")]
    ours = Z["prob__OURS"]                       # (n, S)
    n, S = ours.shape
    uniq, ginv = np.unique(groups, return_inverse=True)
    by_g = [np.where(ginv == k)[0] for k in range(len(uniq))]
    # one bootstrap index set reused for every comparison -> paired
    draws = [np.concatenate([by_g[k] for k in RNG.integers(0, len(uniq), len(uniq))])
             for _ in range(B)]
    rows = []
    for s in systems:
        other = Z["prob__" + s.replace("::", "__")]
        d_obs = float(np.mean([roc_auc_score(y, ours[:, j]) - roc_auc_score(y, other[:, j])
                               for j in range(S)]))
        ds = np.empty(B)
        for b, idx in enumerate(draws):
            yy = y[idx]
            if yy.min() == yy.max(): ds[b] = np.nan; continue
            ds[b] = np.mean([roc_auc_score(yy, ours[idx, j]) - roc_auc_score(yy, other[idx, j])
                             for j in range(S)])
        ds = ds[np.isfinite(ds)]
        lo95, hi95 = np.percentile(ds, [2.5, 97.5])
        a = 0.05 / len(systems)
        lob, hib = np.percentile(ds, [100 * a / 2, 100 * (1 - a / 2)])
        rows.append(dict(system=s.replace("fixed::", ""), d_auroc=d_obs,
                         ci95_lo=lo95, ci95_hi=hi95, sig95=bool(lo95 > 0 or hi95 < 0),
                         bonf_lo=lob, bonf_hi=hib, sig_bonf=bool(lob > 0 or hib < 0),
                         n_comparisons=len(systems)))
        print(f"{proto} {rows[-1]['system']:34s} d={d_obs:+.5f} "
              f"95%[{lo95:+.5f},{hi95:+.5f}] bonf[{lob:+.5f},{hib:+.5f}] "
              f"{'SIG' if rows[-1]['sig_bonf'] else 'ns'}", flush=True)
    df = pd.DataFrame(rows).sort_values("d_auroc", ascending=False)
    df.to_csv(OUT / f"{proto}_SIGNIFICANCE.csv", index=False)
    print(f"-> {proto}: {int(df.sig_bonf.sum())}/{len(df)} Bonferroni-separable\n", flush=True)
