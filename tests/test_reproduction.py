#!/usr/bin/env python3
"""The release gate: this repository must reproduce the frozen main table exactly.

Two checks, in increasing strength.

`test_manifest` confirms that every published evidence file still hashes to what the
manifest recorded, so a corrupted or silently edited artifact cannot pass unnoticed.

`test_reference_arm` re-fits the Protocol B router from the frozen inputs and requires the
resulting AUROC to land on `FROZEN_B_AUROC` within the tolerance `_contract` puts in force:
zero on the machine the frozen run executed on (`AFR_STRICT_GATE=1`), and `PORTABLE_TOL`
elsewhere, because the forest's floating-point path differs with the CPU and the
scikit-learn build. The randomness in the pipeline -- the bootstrap and the node-level
feature subsampling -- is seeded either way, so a delta above the tolerance means something
in the release tree differs from what produced the paper: a different module resolved on the
path, a different input bundle, a different scikit-learn. `_contract.PORTABLE_TOL` carries
the measured cross-platform values and how the level was chosen.

Run standalone (`python tests/test_reproduction.py`) or under pytest.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "code"
EXPERIMENTS = ROOT / "experiments"
CONTROLS = EXPERIMENTS / "09_live_and_controls" / "code"


def _environment() -> None:
    """The two variables the frozen pipeline reads, pointed at this checkout."""
    os.environ["AFR_ROOT"] = str(ROOT)
    os.environ["AFR_INPUTS"] = str(EXPERIMENTS / "00_inputs")
    os.environ.setdefault("V3_RUN_DIR", str(REPO / "runs" / "verify"))
    # The heads are fitted on CPU. Without this, importing the verifier wrappers can try to
    # claim a GPU that the reproduction does not need.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    Path(os.environ["V3_RUN_DIR"]).mkdir(parents=True, exist_ok=True)
    for path in (ROOT, ROOT / "ingest_and_scoring", ROOT / "experiments",
                 EXPERIMENTS / "08_routing_code", CONTROLS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def test_manifest() -> None:
    """Every file the manifest names must still hash to the recorded digest."""
    manifest = EXPERIMENTS / "MANIFEST.sha256"
    assert manifest.exists(), f"missing {manifest}"

    checked, missing, mismatched = 0, [], []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        target = EXPERIMENTS / relative.lstrip("./")
        if not target.exists():
            missing.append(relative)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != digest:
            mismatched.append(relative)
        checked += 1

    assert not missing, f"{len(missing)} manifest entries absent: {missing[:5]}"
    assert not mismatched, f"{len(mismatched)} digests drifted: {mismatched[:5]}"
    assert checked > 0, "manifest is empty"
    print(f"manifest: {checked} files verified")


def test_reference_arm() -> None:
    """Protocol B, main system, refitted here: must land on the frozen number exactly."""
    _environment()

    import numpy as np
    from sklearn.metrics import roc_auc_score

    import _contract
    import core
    import v3core as V

    contract = _contract.load()
    pool = list(contract["pool"])
    features = list(contract["features"])
    hp = dict(contract["hp_B"])

    train, test, _pooled, _validation = V.load(with_test_labels=True)
    assert train.shape[0] == 5_276 and train["content_doc_key"].nunique() == 890
    assert test.shape[0] == 3_236 and test["content_doc_key"].nunique() == 645

    per_seed = []
    for seed in V.SEEDS:
        held = V.stratified_group_split(train, seed)
        fit = train.loc[~held].reset_index(drop=True)
        validation = train.loc[held].reset_index(drop=True)

        calibrators = core.platt(fit, actions=pool)
        c_fit = core.apply_platt(fit, calibrators, pool)
        c_val = core.apply_platt(validation, calibrators, pool)
        c_test = core.apply_platt(test, calibrators, pool)

        h_val, h_test = core.fit_heads(
            fit, [validation, test], seed, c_fit, actions=pool,
            features=features, hp=hp, target="regret", learner="rf")

        costs = np.array([core.fold_costs(fit, actions=pool)[a] for a in pool], float)
        beta, _ = V.choose_beta(validation, h_val, c_val, pool, costs)
        _, p_val, _ = V.route(validation, h_val, c_val, beta, pool, costs)
        _, p_test, _ = V.route(test, h_test, c_test, beta, pool, costs)

        calibrated = core.isotonic(
            p_val, validation["label_supported"].to_numpy(int), p_test)
        per_seed.append(roc_auc_score(test["label_supported"].to_numpy(int), calibrated))

    auroc = float(np.mean(per_seed))
    print(f"reference arm = {auroc!r}")
    print(f"frozen        = {_contract.FROZEN_B_AUROC!r}")
    print(f"delta         = {abs(auroc - _contract.FROZEN_B_AUROC):.3e}")
    _contract.check_reference(auroc, what="release-tree reference arm")


def test_bundle_carries_every_column_the_pipeline_needs() -> None:
    """The trimmed input bundle must satisfy the stage-3 preflight contract.

    Kept in step with `preflight()` in 08_routing_code/part1c_main_full_v1.py: the six frozen
    features, the four latency columns, the identity that base feature latency is query plus
    document setup, and per-action availability with a finite latency wherever an action is
    available. The first release shipped without three of the latency columns and every stage-3
    script refused to start; the reproduction gate above did not notice, because it scores
    AUROC and never reads a latency.
    """
    import json

    import pandas as pd

    inputs = EXPERIMENTS / "00_inputs"
    contract = json.loads(
        (EXPERIMENTS / "01_main_experiment" / "00_contract"
         / "INHERITED_FROZEN_v3.json").read_text())
    features = list(contract["features"])
    pool = list(contract["pool"])
    latency = ["feature_latency_ms", "feature_query_latency_ms",
               "feature_document_setup_ms", "compact16_feature_latency_ms"]

    for name in ("TRAIN", "TEST"):
        frame = pd.read_parquet(inputs / f"{name}.parquet")

        missing = [c for c in features if c not in frame.columns]
        assert not missing, f"{name} is missing frozen features {missing}"

        missing = [c for c in latency if c not in frame.columns]
        assert not missing, f"{name} is missing latency columns {missing}"

        residual = (frame["feature_latency_ms"]
                    - frame["feature_query_latency_ms"]
                    - frame["feature_document_setup_ms"]).abs().max()
        assert residual <= 1e-6, f"{name}: base latency is not query + setup ({residual})"

        for action in pool:
            for column in (f"score__{action}", f"available__{action}",
                           f"latency_ms__{action}"):
                assert column in frame.columns, f"{name} is missing {column}"
            available = frame[f"available__{action}"].to_numpy(bool)
            latencies = frame[f"latency_ms__{action}"].to_numpy(float)
            assert not pd.isna(latencies[available]).any(), (
                f"{name}/{action}: an available row has no latency")

        for column in ("source_document", "candidate_summary"):
            assert column not in frame.columns, (
                f"{name} carries {column}; the bundle is meant to be text-free")

    print("bundle satisfies the stage-3 preflight contract")


def test_launchers_resolve() -> None:
    """Each stage-3 launcher must point at a script that exists in this checkout."""
    _environment()
    scripts = sorted((EXPERIMENTS / "08_routing_code").glob("*.sh"))
    assert scripts, "no launchers found"
    for launcher in scripts:
        probe = (
            f'set -u; AFR_ROOT="{ROOT}"; '
            + "; ".join(
                line for line in launcher.read_text().splitlines()
                if line.startswith(("SRC=", "BASE=", "SCRIPT="))
            )
            + "; echo \"$SCRIPT\""
        )
        target = subprocess.run(["bash", "-c", probe], capture_output=True,
                                text=True, check=True).stdout.strip()
        assert Path(target).is_file(), f"{launcher.name} points at a missing {target}"


def test_reproduce_sh_matches_the_launchers() -> None:
    """reproduce.sh must pass arguments the launchers accept, and write where they look.

    Three things the earlier existence check could not catch, each of which broke a documented
    command: a stage passing a placeholder where the launcher demands a named phase, a run
    directory that the cascade stage does not watch, and a controls list that omits the scripts
    whose outputs the paper reports.
    """
    reproduce = (REPO / "reproduce.sh").read_text()
    part3 = (EXPERIMENTS / "08_routing_code" / "run_part3_v1.sh").read_text()
    chain4 = (EXPERIMENTS / "08_routing_code" / "chain_part4.sh").read_text()

    # 1. Part 3 demands a phase, so reproduce.sh must require one too.
    assert 'PHASE="${2:?' in part3, "run_part3_v1.sh no longer demands a phase; revisit this test"
    assert "phase1|phase2" in reproduce, (
        "reproduce.sh does not validate the phase argument run_part3_v1.sh requires")

    # 2. cascade waits on Part 3's marker under AFR_ROOT/experiments/runs, so runs must go there.
    assert "BASE=${AFR_ROOT" in chain4 and "$BASE/runs/part3_extended_v1" in chain4
    assert 'RUNS="$AFR_ROOT/experiments/runs"' in reproduce, (
        "reproduce.sh writes runs where chain_part4.sh does not look for them")

    # 3. controls must run what the paper reports, and each named script must exist.
    controls = EXPERIMENTS / "09_live_and_controls" / "code"
    for name in ("fewshot_frac", "sig_controls", "sig_main", "dataset_control", "ds_only"):
        assert f" {name} " in reproduce or f" {name};" in reproduce, (
            f"reproduce.sh controls does not run {name}.py")
        assert (controls / f"{name}.py").is_file(), f"missing {name}.py"

    # Scripts that read a mandatory variable must have it exported by the entry point.
    if "os.environ[\"SIG_OUT\"]" in (controls / "sig_controls.py").read_text():
        assert "SIG_OUT=" in reproduce, "sig_controls.py needs SIG_OUT and reproduce.sh omits it"

    print("reproduce.sh honours the launcher contracts")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
