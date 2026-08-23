# Reproduction inputs (text-free)

Everything needed to retrain the router, route, calibrate and evaluate, in 6.3 MB.

| file | rows | what it carries |
|---|---:|---|
| `TRAIN.parquet` | 5,276 | keys, corpus, label, the six cheap features, per-verifier score / availability / latency |
| `TEST.parquet` | 3,236 | the same, plus the gold labels used only for evaluation |
| `TEST_SCORING.parquet` | 3,236 | the label-free projection the scoring stage saw |
| `p1_scoring/*.parquet` | 1,667 | the 15 verifiers' raw scoring outputs |

## What is deliberately absent

`source_document` and `candidate_summary` are dropped. That text belongs to CoGenSumm, FRANK,
RAGTruth and UniSumEval, and is not ours to redistribute. Every row keeps its `episode_key`, so
the text can be rejoined from the original corpora.

The consequence is a clean boundary:

- **Reproducible from this bundle alone**: stage-1 calibration, the relative-quality targets, the
  routing heads, beta selection, stage-2 calibration, thresholds, every table and every ablation.
- **Needs the original corpora**: re-running the verifiers themselves, and the live-execution
  experiment in `09_live_and_controls`, because those call models on the raw text.

## Verified

Training end to end from this bundle reproduces the frozen Protocol B main result
**bit for bit**: `0.8225560999095635`, delta `0.000e+00`.

```bash
export AFR_ROOT=/path/to/repo
export AFR_INPUTS=$AFR_ROOT/paper_v3/DELIVERABLE/00_inputs
python - <<'PY'
import sys, numpy as np
sys.path.insert(0, "08_scripts"); sys.path.insert(0, "09_live_and_controls/code")
import v3core as V, core, _contract as C
from sklearn.metrics import roc_auc_score
ct = C.load(); POOL, FE, HP = ct["pool"], ct["features"], ct["hp_B"]
TR, TE, _a, _v = V.load(with_test_labels=True)
aus = []
for seed in V.SEEDS:
    held = V.stratified_group_split(TR, seed)
    fit, val = TR.loc[~held].reset_index(drop=True), TR.loc[held].reset_index(drop=True)
    cals = core.platt(fit, actions=POOL)
    cf, cv = core.apply_platt(fit, cals, POOL), core.apply_platt(val, cals, POOL)
    ce = core.apply_platt(TE, cals, POOL)
    hv, he = core.fit_heads(fit, [val, TE], seed, cf, actions=POOL, features=FE, hp=HP,
                            target=ct["target"], learner=ct["learner"])
    cvec = np.array([core.fold_costs(fit, actions=POOL)[a] for a in POOL], float)
    b, _ = V.choose_beta(val, hv, cv, POOL, cvec)
    _s, pv, _m = V.route(val, hv, cv, b, POOL, cvec)
    _s2, pe, _m2 = V.route(TE, he, ce, b, POOL, cvec)
    q = core.isotonic(pv, val["label_supported"].to_numpy(int), pe)
    aus.append(roc_auc_score(TE["label_supported"].to_numpy(int), q))
C.check_reference(float(np.mean(aus)), 0.0, "end-to-end from the shipped bundle")
PY
```
