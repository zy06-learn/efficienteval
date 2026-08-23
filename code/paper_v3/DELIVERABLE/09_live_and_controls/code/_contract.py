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
# is nominally the main system must land on this exactly.
FROZEN_B_AUROC = 0.8225560999095635

# AFR_ROOT names the repository code root. The default is derived from this file's own location
# rather than hard-coded, so a fresh clone runs without any environment set up first.
_AFR_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))


def load(root: Path | str | None = None) -> dict:
    root = Path(root or os.environ.get(
        "AFR_ROOT", _AFR_DEFAULT))
    c = root / "paper_v3" / "artifacts" / "part1c_main_full_v1" / "00_contract"
    frozen = json.loads((c / "INHERITED_FROZEN_v3.json").read_text())
    hp = json.loads((c / "HP_SELECTED.json").read_text())
    bad = {k: (EXPECTED[k], frozen.get(k)) for k in EXPECTED if frozen.get(k) != EXPECTED[k]}
    if bad:
        raise AssertionError("inherited contract drifted: " + json.dumps(bad, ensure_ascii=False))
    return dict(pool=list(frozen["pool"]), features=list(frozen["features"]),
                target=frozen["target"], learner=frozen["learner"],
                beta_grid=list(frozen["beta_grid"]),
                hp_A=dict(hp["hyperparameters_A"]), hp_B=dict(hp["hyperparameters_B"]))


def check_reference(auroc: float, tol: float = 0.0, what: str = "reference arm") -> None:
    """A nominally-main-system arm must reproduce the frozen number (tol=0 means bit-for-bit)."""
    d = abs(float(auroc) - FROZEN_B_AUROC)
    if d > tol:
        raise AssertionError(
            f"{what} = {auroc!r} does not reproduce the frozen main table "
            f"{FROZEN_B_AUROC!r} (delta {d:.3e} > tol {tol:.3e})")
    print(f"  [gate] {what} reproduces the frozen main table, delta {d:.3e}")
