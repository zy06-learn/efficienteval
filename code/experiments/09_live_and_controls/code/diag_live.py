import os
import numpy as np, pandas as pd, sys
from pathlib import Path
# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
O = Path(os.environ.get("AFR_ROOT", _AFR_DEFAULT) + "/experiments/runs/rerun_check_v1")
sys.path.insert(0, os.environ.get("AFR_ROOT", _AFR_DEFAULT))
sys.path.insert(0, os.environ.get("AFR_ROOT", _AFR_DEFAULT) + "/experiments")
os.environ["V3_RUN_DIR"] = str(O)
import v3core as V
POOL = ["factcc", "lettuce_v2", "granite_guardian_3_1_2b"]
sel = np.load(O / "live_sel.npy")
z   = np.load(O / "live_scores.npz"); raw = z["raw"]
keys = pd.read_parquet(O / "live_keys.parquet")["episode_key"].astype(str).to_numpy()
_TR, TEST, _A, _v = V.load(with_test_labels=True)
ev = TEST[TEST["episode_key"].astype(str).isin(set(keys))].reset_index(drop=True)
ev = ev.set_index(ev["episode_key"].astype(str)).loc[keys].reset_index(drop=True)
stored = np.column_stack([ev[f"score__{a}"].to_numpy(float) for a in POOL])
rows = np.arange(len(ev))
stored_sel = stored[rows, sel]
d = np.abs(raw - stored_sel)
print(f"{'verifier':26s} {'n':>4s} {'max|Δ raw|':>12s} {'#Δ>1e-9':>8s} {'中位|Δ|':>10s}")
for k, a in enumerate(POOL):
    m = sel == k
    print(f"{a:26s} {m.sum():4d} {d[m].max():12.3e} {int((d[m] > 1e-9).sum()):8d} "
          f"{np.median(d[m]):10.3e}")
bad = np.where(d > 1e-9)[0]
print(f"\n总差异行 {len(bad)}/{len(d)}")
if len(bad):
    print("前 5 个差异行：")
    for i in bad[:5]:
        print(f"  {POOL[sel[i]]:24s} live={raw[i]:.6f} stored={stored_sel[i]:.6f} "
              f"Δ={d[i]:.3e}")
