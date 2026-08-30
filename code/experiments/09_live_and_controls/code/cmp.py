"""Per-verifier diff of a re-run against the frozen score matrix.

DIAGNOSTIC. Writes no published artifact. Prints, for each verifier, max |delta|,
whether the scores are exactly equal, their correlation, and old versus new latency.
Used with rerun_score.py and diag_live.py to trace the live-versus-matrix residual.
"""
import os
import pandas as pd, numpy as np, sys, glob
# AFR_ROOT names the repository code root. The default is derived from this file's own
# location rather than hard-coded, so a fresh clone runs without any environment set up.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
NEW = os.environ.get("AFR_ROOT", _AFR_DEFAULT) + "/experiments/runs/rerun_check_v1"
OLD = os.environ.get("AFR_ROOT", _AFR_DEFAULT) + "/ingest_and_scoring/results/p1_scoring"
print(f"{'verifier':22s} {'n':>5s} {'score 最大|Δ|':>13s} {'完全相等':>8s} {'相关':>7s} "
      f"{'旧延迟':>8s} {'新延迟':>8s}")
for v in sys.argv[1:]:
    try:
        a = pd.read_parquet(f"{NEW}/{v}.parquet"); b = pd.read_parquet(f"{OLD}/{v}.parquet")
    except Exception as e:
        print(f"{v:22s} 读取失败 {e}"); continue
    m = a.merge(b, on="episode_key", suffixes=("_new", "_old"))
    sn, so = m["score_new"].astype(float), m["score_old"].astype(float)
    ok = sn.notna() & so.notna()
    d = (sn[ok] - so[ok]).abs()
    lat_o = m["latency_ms_old"].mean() if "latency_ms_old" in m else float("nan")
    lat_n = m["latency_ms_new"].mean() if "latency_ms_new" in m else float("nan")
    print(f"{v:22s} {ok.sum():5d} {d.max():13.3e} {str(bool((d==0).all())):>8s} "
          f"{np.corrcoef(sn[ok],so[ok])[0,1]:7.5f} {lat_o:8.2f} {lat_n:8.2f}")
    big = (d > 1e-6).sum()
    if big: print(f"    差异 >1e-6 的行数: {big}/{ok.sum()}  (中位|Δ|={d.median():.3e})")
