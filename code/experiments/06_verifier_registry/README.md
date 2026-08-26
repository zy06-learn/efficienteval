# `06_verifier_registry/`

Two documents, no code — the name says registry for that reason.

| File | What it records |
|---|---|
| `REGISTRY.md` | every one of the fifteen verifiers: upstream repository, revision, licence, and how it is served. Third-party verifiers are not redistributed in this repository; this is how to obtain each one. |
| `COMPLEXITY.md` | the forward-work accounting behind the reported latencies, and where the three timing columns in the score matrix disagree |

The wrappers that call these verifiers are in [`../../verifiers/`](../../verifier_wrappers/).
