# `verifier_wrappers/` — the fifteen verifiers

Stage 2. The implementations and scoring wrappers that produce the score matrix stage 3
routes over. Nothing here is a third-party verifier's own source: each is a wrapper around an
upstream checkout, and
[`../experiments/06_verifier_registry/REGISTRY.md`](../experiments/06_verifier_registry/REGISTRY.md)
records the repository and revision of every one.

| File | What it does |
|---|---|
| `unified_scoring.py`, `unified_summary_verifiers_v1.py` | the scoring entry points; `build_scorer` dispatches to a verifier by name |
| `native_scorers.py`, `minicheck_scorers.py`, `extended_scorers.py`, `additional_verifier_scorers_v1.py` | per-family wrappers |
| `structured_high_judge.py` | the served LLM judges (Qwen30-fast, Qwen30-judge) |
| `candidate_verifiers.py` | the candidate registry, including which need a persistent worker |
| `pool_gate_sweep_v1.py`, `pool_gate_sweep_v2.py` | pool selection over verifier subsets |
| `summary_router_compact16_*.py` | the cheap pre-call feature extraction |
| `tenfold_v1.py` | the grouped ten-fold rotation Protocol A uses |
| `router_feature_learnability.py` | the feature-learnability analysis: ROUGE-L, sentence and entity coverage, and how much of the routing signal each feature carries |
| `global_gamma_calibration.py`, `primary_scoring.py`, `cascade_primary_assets.py`, `research_freeze.py` | calibration, primary scoring, and the freeze bookkeeping |

Three of these files (`summary_router_compact16_direct_v1.py`, `pool_gate_sweep_v1.py`,
`tenfold_v1.py`) also exist byte-identically under `../experiments/08_routing_code/`. The two
trees import them under different module names, so both copies are load-bearing.

**Where the third-party checkouts go.** `candidate_verifiers.py` resolves them to a sibling
directory, `code/verifiers/`, which is not shipped: each upstream verifier carries its own
licence and is not redistributed here. Clone them there, at the revisions
`../experiments/06_verifier_registry/REGISTRY.md` records. This package holds only the
wrappers, which is why it is named for them.

Running anything here needs `requirements-verifiers.txt`, a GPU, and for the served verifiers
a vLLM server. Reproducing the paper's tables does not — stage 3 reads the frozen scores.
