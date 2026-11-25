# X1: what a new corpus needs before routing beats a fixed verifier

**Question.** The router transfers poorly to an unseen corpus (leave-one-corpus-out AUROC 0.543
on RAGTruth, 0.537 on UniSumEval). Given *k* labelled rows from the new corpus, is it better to
adapt the router with them, or to use the same rows to pick one fixed verifier and deploy that?

**Design.** Each corpus is held out in turn. The router is the published adaptation sweep: fitted
on the other three corpora plus *k* rows of the held-out corpus's own training pool, at
0, 5, 10, 20, 30, 40, 50, 70, 85 and 100% of the pool, ten seeds, Protocol B. Nothing about it is
new; both scripts reproduce the published sweep cell by cell before writing anything (80/80 cells,
`*_ROUTER_REPRODUCTION.csv`). The fixed-verifier policy gets exactly the same labelled data, the
same per-seed sample of *k* rows, and picks one verifier by AUROC in three scopes:

| Arm | Candidates |
|---|---|
| `all` | all fifteen verifiers |
| `budget` | verifiers whose mean latency on the router's validation split is no higher than the router's (label-free, no test data) |
| `pool` | the router's own three actions |

No verifier is re-run; everything is read from the frozen matrix.

Two evaluation sets:

| Script | Evaluated on | Selection data for the fixed verifier |
|---|---|---|
| `code/x1_bestfixed_at_k.py` → `results/single_corpus/` | the held-out corpus's test split only | the *k* target rows (source corpora if *k* = 0 or a class has fewer than five rows) |
| `code/x1_mixed_at_k.py` → `results/mixed_traffic/` | the four-corpus test mixture at natural proportions, i.e. the new corpus joins the stream | the three source corpora plus the *k* target rows |

## Results (means over ten seeds)

**Mixed traffic.** First *k* from which the router stays ahead at every larger *k*:

| Held-out corpus (pool) | Router at *k*=0 | Beats best pool member (Granite-3.1-2B, 0.799 / 132 ms) on AUROC and ms | Reaches best of all 15 (mostly AlignScore, 0.815 / 562 ms) at lower ms |
|---|---|---|---|
| CoGenSumm (535) | 0.817 / 88 ms | 0 | 0 |
| FRANK (669) | 0.802 / 119 ms | 134 | 134 |
| UniSumEval (1,089) | 0.754 / 92 ms | 327 | 1,089 (0.0002 below at 926) |
| RAGTruth (2,983) | 0.767 / 106 ms | 597 | 1,193 |

At the full pool the router is at 0.822–0.823 AUROC and 86–87 ms. The `budget` arm never exceeds
0.78. Seed standard deviations reach 0.06 at small *k*, so crossing points are approximate.

**Within the new corpus alone** the result reverses: a fixed verifier selected with the same rows
is as good as or better than the router at every *k* (e.g. RAGTruth, *k* = 298: 0.638 / 44 ms
against 0.613 / 101 ms). This is consistent with the paper's stated limitation that the gain is a
property of mixed traffic.

## Running it

```bash
export AFR_ROOT="$PWD/code" AFR_INPUTS="$PWD/code/experiments/00_inputs"
python code/experiments/11_rebuttal_x1/code/x1_mixed_at_k.py        # about 30 min on 8 CPU threads
X1_SMOKE=1 python code/experiments/11_rebuttal_x1/code/x1_bestfixed_at_k.py   # 2 points, 1-2 min
```

Outputs go to `11_rebuttal_x1/runs/<script>/` unless `V3_RUN_DIR` is set. The router cells are
checked against `inputs/FEWSHOT_FRAC_MS.csv`, the published sweep. With `AFR_STRICT_GATE=1` the
tolerance is 1e-9; that is what the recorded environment meets. Elsewhere it is 2e-3 (see
`docs/reproducibility.md` for why bit-for-bit equality is not portable). The smoke mode of both
scripts in this layout was re-run on the recorded environment and passed at 1e-9.

`code/as_run/` holds the two scripts and launchers exactly as they ran in the working tree; their
sha256 matches `results/*/RUN_INFO.txt`. `code/` holds the same scripts with only the path block
rewired to this repository.
