# `scripts/`

One helper, not a stage.

`alignscore_persistent_worker.py` keeps an AlignScore model resident across scoring calls,
because reloading it per instance dominates its measured latency. `verifiers/candidate_verifiers.py`
launches it by path. Nothing in stage 3 or stage 4 touches it.
