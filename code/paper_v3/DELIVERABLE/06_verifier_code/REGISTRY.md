# Verifier Registry

Audited state: 2026-07-22. Quality below is balanced accuracy from the
completed unified-v1 runtime audit; latency is strict batch-1 mean milliseconds
on RAGTruth/FRANK. TofuEval is burned diagnostic and is not displayed as a
headline result. `SCORED` means artifacts physically exist; consult protocol
status and fidelity before using them.

| name | role | state | fidelity | unified rows | BAcc R/F | mean ms R/F | principal boundary |
|---|---|---|---|---:|---:|---:|---|
| cheap_lexical | cheapest baseline | SCORED | CUSTOM | 26,880 | 0.5774 / 0.7015 | 1.60 / 1.38 | BM25 proxy, not a learned factuality metric |
| factcc | fast learned baseline | SCORED | COMPATIBLE_ADAPTER | 26,880 | 0.5719 / 0.6873 | 11.88 / 11.29 | binary HF conversion only; no FactCCX span heads |
| factkb | fast learned diagnostic | SCORED | COMPATIBLE_ADAPTER | 26,880 | 0.4930 / 0.8400 | 11.86 / 11.30 | FRANK confirmed training overlap through FactCollect |
| hhem | low-cost neural diagnostic | SCORED | MATERIAL_DEVIATION | 26,880 | 0.6978 / 0.7084 | 45.73 / 39.27 | project window-max metric; RAGTruth training overlap |
| alignscore | strong semantic anchor | SCORED | COMPATIBLE_ADAPTER | 26,880 | 0.7412 / 0.7497 | 120.74 / 101.40 | FRANK development exposure; checkpoint terms unpinned |
| factcg | medium/strong anchor | SCORED | COMPATIBLE_ADAPTER | 26,880 | 0.7650 / 0.7326 | 156.02 / 127.47 | RAGTruth and FRANK development exposure |
| minicheck_dbta | medium grounded checker | SCORED | COMPATIBLE_ADAPTER | 26,880 | 0.7528 / 0.7135 | 162.28 / 133.08 | RAGTruth and FRANK development exposure |
| minicheck_ft5 | medium grounded checker | SCORED, V1 SUPERSEDED | COMPATIBLE_ADAPTER (V2 CODE) | 26,880 legacy v1 | 0.7204 / 0.7161 legacy | 229.16 / 198.55 legacy | v1 omitted official `predict: `; v2 rescore required |
| summac_zs | static NLI baseline | SCORED | MATERIAL_DEVIATION | 26,880 | 0.5745 / 0.6248 | 280.72 / 244.13 | compatibility ID for SummaC-style DeBERTa NLI |
| qwen30_judge | expensive reference judge | SCORED | CUSTOM | 26,880 | 0.7412 / 0.7798 | 1026.04 / 1134.22 | project prompt/schema; training provenance unknown |
| summac_conv | archived baseline | DOWNLOAD_FAILED | UNRESOLVED | 0 | n/a | n/a | package/checkpoint and adapter absent |
| qafacteval | archived QA baseline | DOWNLOAD_FAILED | UNRESOLVED | 0 | n/a | n/a | package, component checkpoints and adapter absent |
| lettucedetect | archived localizer | SCORED outside unified-v1 | not audited here | 0 | n/a | n/a | RAGTruth overlap; localizer rather than scalar anchor |
| fenice | archived localizer | SCORED outside unified-v1 | not audited here | 0 | n/a | n/a | non-commercial license; full evaluation absent |
| genaudit | archived repair/localizer | SCORED outside unified-v1 | not audited here | 0 | n/a | n/a | USB-trained repair asset, not independent evaluator |
| attrscore_ft5 | archived screening | SCORED outside unified-v1 | not audited here | 0 | n/a | n/a | source-family exposure |
| granite_guardian_3b | archived screening | SCORED outside unified-v1 | not audited here | 0 | n/a | n/a | FRANK development exposure |

## State semantics

- `SCORED`: at least one score artifact exists. It does not imply current
  protocol, independent benchmark validity, or paper-exact implementation.
- `SCORED, V1 SUPERSEDED`: historical score coverage exists, but current code
  rejects it and a new-protocol rescore is required.
- `SCORED outside unified-v1`: only screening/smoke/repair artifacts exist.
- `INSTALLED_NOT_SCORED`: implementation/checkpoint evidence exists but no
  score artifact exists for the listed scope.
- `DOWNLOAD_FAILED`: a reproducible run is blocked by missing required assets.

The detailed paper/code comparison is
`the fidelity audit, which is internal and not part of this release`. Exact score
paths and protocol status remain machine-readable in each `STATUS.json` and in
`data/unified_v1/verifier_scores/*.manifest.json`.
