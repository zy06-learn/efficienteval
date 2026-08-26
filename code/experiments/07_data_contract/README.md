# `07_data_contract/`

The two JSON files stage 3 reads before it does anything, kept separately so that a run's
configuration can be diffed without reading the code.

| File | What it fixes |
|---|---|
| `INHERITED_FROZEN_v3.json` | the verifier pool, the six cheap features, the supervision target, the action space, and the seed list |
| `HP_SELECTED.json` | the hyperparameters chosen on TRAIN for each protocol: Protocol A `800 / leaf 5 / 0.5 / depth 6`, Protocol B `200 / leaf 10 / 0.5 / depth 6` |

Every experiment directory carries its own copy of both under `00_contract/`, so each run
records the configuration it actually used rather than pointing at a file that may move.
