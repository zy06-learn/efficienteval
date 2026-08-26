"""Shared contract guard for the experiments in this directory.

Every script here inherits the frozen configuration of the main experiment. If any of it has
drifted the script must refuse to run rather than silently report numbers for a different
system, which is the same rule the frozen pipeline enforces.
"""
from __future__ import annotations
import json, os
from pathlib import Path

EXPECTED = {
    "pool": ["factcc", "lettuce_v2", "granite_guardian_3_1_2b"],
    "features": ["structured_source_line_ratio", "bm25_mean3", "entity_coverage",
                 "entity_value_colocation", "year_count", "conflicting_value_rate"],
    "target": "regret",
    "learner": "rf",
    "beta_grid": [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 2.4, 3.2],
}
# Protocol B OURS in the frozen main table, at full precision. Any arm of these experiments that
# is nominally the main system must land on this.
FROZEN_B_AUROC = 0.8225560999095635

# How close it has to land, and why that is not always zero.
#
# The frozen run executed on aarch64, Python 3.12.3, GCC 13.3.0; see
# 01_main_experiment/07_provenance/RUN_METADATA.txt. There the value reproduces bit-for-bit.
# Elsewhere it does not: the random forest is the only randomness in the pipeline, and its
# floating-point path differs with the CPU, the libm, and the scikit-learn wheel build. Measured
# against this release, from a clean install of requirements.txt:
#
#   aarch64, GitHub ubuntu-24.04-arm    0.822567198760909    delta 1.110e-05
#   x86_64,  GitHub ubuntu-latest       0.8226629359217386   delta 1.068e-04
#
# For scale, the seed standard deviation the main table reports for this arm is 0.00474, so the
# larger of the two is 2.3% of noise the paper already publishes. Neither changes any reported
# comparison. What they do change is that a zero tolerance is not portable, so the gate has two
# levels: PORTABLE_TOL by default, set above the largest measured cross-platform delta with room
# to spare, and zero when AFR_STRICT_GATE=1 asks for the recorded environment's own standard.
PORTABLE_TOL = 2e-4
STRICT_ENV = "AFR_STRICT_GATE"


def reference_tolerance() -> tuple[float, str]:
    """The tolerance in force, and a label for it."""
    if os.environ.get(STRICT_ENV) == "1":
        return 0.0, f"strict, bit-for-bit ({STRICT_ENV}=1)"
    return PORTABLE_TOL, "portable, cross-platform"

# AFR_ROOT names the repository code root. The default is derived from this file's own location
# rather than hard-coded, so a fresh clone runs without any environment set up first.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


def load(root: Path | str | None = None) -> dict:
    root = Path(root or os.environ.get(
        "AFR_ROOT", _AFR_DEFAULT))
    c = root / "experiments" / "cross_stage_contract" / "part1c_main_full_v1" / "00_contract"
    frozen = json.loads((c / "INHERITED_FROZEN_v3.json").read_text())
    hp = json.loads((c / "HP_SELECTED.json").read_text())
    bad = {k: (EXPECTED[k], frozen.get(k)) for k in EXPECTED if frozen.get(k) != EXPECTED[k]}
    if bad:
        raise AssertionError("inherited contract drifted: " + json.dumps(bad, ensure_ascii=False))
    return dict(pool=list(frozen["pool"]), features=list(frozen["features"]),
                target=frozen["target"], learner=frozen["learner"],
                beta_grid=list(frozen["beta_grid"]),
                hp_A=dict(hp["hyperparameters_A"]), hp_B=dict(hp["hyperparameters_B"]))


def check_reference(auroc: float, tol: float | None = None,
                    what: str = "reference arm") -> None:
    """A nominally-main-system arm must reproduce the frozen number.

    With tol left unset the tolerance comes from reference_tolerance(): zero on the recorded
    environment when AFR_STRICT_GATE=1, PORTABLE_TOL otherwise. Pass tol explicitly to override.
    """
    label = "explicit"
    if tol is None:
        tol, label = reference_tolerance()
    d = abs(float(auroc) - FROZEN_B_AUROC)
    if d > tol:
        raise AssertionError(
            f"{what} = {auroc!r} does not reproduce the frozen main table "
            f"{FROZEN_B_AUROC!r} (delta {d:.3e} > tol {tol:.3e}, {label})")
    exact = " bit-for-bit," if d == 0.0 else ""
    print(f"  [gate] {what} reproduces the frozen main table,{exact} "
          f"delta {d:.3e} (tol {tol:.3e}, {label})")
