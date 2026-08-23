# Live execution and the three controls

Four experiments requested after the main study was frozen. None of them changes the frozen
configuration; all of them read the same TRAIN/TEST boundary and the same hyperparameters.

Set `AFR_ROOT` to the repository root before running anything here. Every script also honours
`AFR_PYTHON` and `HF_HUB` if those live elsewhere on your machine.

---

## 1. The main result, executed live (`code/live_main.py`)

The question was whether the reported test numbers depend on having scored the whole test set in
advance. They do not, and this run demonstrates it end to end.

Training still reads the frozen TRAIN scores, because that is what training needs. At test time
the router sees only the six cheap features, picks one verifier, and **that verifier is then
actually called**. The other two are never invoked for that row. Scores are not reused across
seeds: a row selected by two seeds is called twice.

| | AUROC | SD |
|---|---:|---:|
| Live, 32,360 real calls | **0.82205** | 0.00468 |
| Frozen matrix (main table) | 0.82256 | 0.00474 |
| Difference | **-0.00051** | |

3,236 test rows x 10 seeds = 32,360 rows, and exactly 32,360 verifier calls. Action shares are
0.3434 / 0.3474 / 0.3091, identical to the main table.

**Where the residual comes from.** A 240-row pilot (`code/live_pipeline.py`,
`results/LIVE_PIPELINE.json`) re-ran each pool member and compared against the frozen matrix
row by row:

| verifier | rows | max abs difference | rows differing |
|---|---:|---:|---:|
| FactCC | 94 | 0.000e+00 | 0 |
| Lettuce-v2 | 119 | 0.000e+00 | 0 |
| Granite-3.1-2B | 27 | 2.78e-02 | 26 |

The two locally executed encoders reproduce bit for bit, so for them the frozen matrix is pure
caching. The vLLM-served generative verifier does not: `--enable-prefix-caching` decides whether
a prefix is recomputed or reused, and the two paths differ in floating point. Typical deviation
is 1.2e-06, worst case 2.8e-02 where the top-20 logprob set changes membership. Propagated
through calibration this moves the headline AUROC by 5e-04.

## 2. Dataset identity is not an input (`code/dataset_control.py`)

Three answers of increasing strength, in `results/DATASET_CONTROL.json`.

**(a) By construction.** The frozen feature list is `structured_source_line_ratio`, `bm25_mean3`,
`entity_coverage`, `entity_value_colocation`, `year_count`, `conflicting_value_rate`. No dataset,
corpus, split or source-id field appears in it.

**(b) How much corpus signal the features carry anyway.** A classifier predicting the corpus from
the six features alone reaches **0.8959** accuracy against a 0.5654 majority baseline and 0.25
uniform chance. The features are document statistics, so this is expected and is reported rather
than hidden: corpora differ in how structured their documents are and how much lexical overlap
their summaries have.

**(c) The control that settles it.** Giving the router the corpus identity explicitly, as
one-hot features on top of the six, changes AUROC from **0.82256 to 0.81963**, a delta of
**-0.00293**. Explicit dataset identity buys nothing; if anything it dilutes the node-level
feature subsampling. Whatever corpus information matters is already reachable from `x`.

## 3. Few-shot adaptation curve (`code/fewshot.py`, `code/fewshot_k0.py`)

Leave-one-dataset-out measures a single point, `k = 0`, and cannot say how much in-domain data
would fix the gap. Here the router is trained on the other three corpora in full plus `k`
labelled rows from the held-out corpus, and evaluated on that corpus's own test split.

| corpus | k=0 (LODO) | k=all | pool | verdict |
|---|---:|---:|---:|---|
| FRANK | 0.83328 | 0.82520 | 669 | transfers already; in-domain data does not help |
| CoGenSumm | 0.65157 | 0.67983 | 535 | slow, noisy gain |
| UniSumEval | 0.53694 | 0.59231 | 1,089 | needs ~512 rows for most of the gain |
| RAGTruth | 0.54301 | 0.63796 | 2,983 | never saturates within the available pool |

There is no universal knee. FRANK is already covered by the other three corpora, while RAGTruth
starts near chance at `k = 0` and is still improving when its entire pool is consumed. The
practical reading is that cold-start cost is corpus-specific, and that the pooled setting works
because the corpora are complementary, not because transfer is easy.

## 4. Significance in the main tables (`code/sig_main.py`)

Paired cluster bootstrap over `content_doc_key`, 2,000 draws, one resample index shared by every
comparison so the systems stay paired. Both the nominal 95% interval and the Bonferroni interval
over the 15 fixed-verifier comparisons are reported; the corrected one governs.

- Protocol A: **14 of 15** separable. The exception is AlignScore, Bonferroni `[-0.0055, +0.0323]`.
- Protocol B: **12 of 15** separable. The exceptions are AlignScore, Qwen30-Fast and Qwen30-Judge.

`results/A_SIGNIFICANCE.csv` and `results/B_SIGNIFICANCE.csv` carry `d_auroc`, both intervals and
both verdicts per system, ready to paste as extra columns in the main tables.

---

## Reproducing

```bash
export AFR_ROOT=/path/to/this/repository
# 1. live main result (needs a GPU; ~35 min for the local phase, ~15 min for the vLLM phase)
LIVE_OUT=$AFR_ROOT/paper_v3/runs/live_main_v1 LIVE_STAGE=plan   python code/live_main.py
LIVE_OUT=$AFR_ROOT/paper_v3/runs/live_main_v1 LIVE_STAGE=local  python code/live_main.py
#   start vLLM with the flags in 06_verifier_code, then:
LIVE_OUT=$AFR_ROOT/paper_v3/runs/live_main_v1 LIVE_STAGE=api    python code/live_main.py
LIVE_OUT=$AFR_ROOT/paper_v3/runs/live_main_v1 LIVE_STAGE=finish python code/live_main.py
# 2-4 are CPU only
python code/dataset_control.py
python code/fewshot.py && python code/fewshot_k0.py
python code/sig_main.py
```

The local encoders and the vLLM server cannot share the single GB10, so `live_main.py` splits
execution into a local phase and an API phase. The routing decision is computed once, before
either phase, and is not revisited; the split therefore cannot change which verifier a row gets.
