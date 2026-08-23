from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


REQUIRED_PARTITIONS = {
    "train",
    "development",
    "fresh_crc_calibration",
    "sealed_iid",
    "sealed_ood",
}
FORBIDDEN_EVIDENCE_MARKERS = ("iid_test", "official_test", "sealed_ood")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(mapping: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise ValueError(f"{context} missing keys: {missing}")


def validate_dataset_ledger(ledger: Mapping[str, Any]) -> dict[str, int]:
    _require(
        ledger,
        {"schema_version", "partition_registry", "datasets", "incidents"},
        "dataset ledger",
    )
    if ledger["schema_version"] != "dataset_ledger_v1":
        raise ValueError("unsupported dataset ledger schema")
    registry = ledger["partition_registry"]
    if set(registry) != REQUIRED_PARTITIONS:
        raise ValueError("dataset ledger must define the five-way partition contract")

    incidents = {str(item["id"]): item for item in ledger["incidents"]}
    incident_id = "2026-07-18_legacy_iid_aggregate_inventory"
    if incident_id not in incidents:
        raise ValueError("legacy IID aggregate inventory incident is not recorded")
    if not bool(incidents[incident_id].get("after_method_lock")):
        raise ValueError("incident timing relative to method lock is missing")

    legacy = ledger["datasets"].get("legacy_summarization_pool", {})
    legacy_iid = legacy.get("partitions", {}).get("legacy_iid", {})
    if legacy_iid.get("status") != "BURNED_BY_AGGREGATE_INVENTORY":
        raise ValueError("legacy IID must be burned after aggregate inventory access")
    if legacy_iid.get("headline_eligible") is not False:
        raise ValueError("burned legacy IID cannot be headline eligible")

    for dataset_id, dataset in ledger["datasets"].items():
        for partition_id, partition in dataset.get("partitions", {}).items():
            status = str(partition.get("status", ""))
            touched = bool(partition.get("content_touched", False))
            headline = bool(partition.get("headline_eligible", False))
            if status.startswith("SEALED") and touched:
                raise ValueError(
                    f"sealed partition marked touched: {dataset_id}/{partition_id}"
                )
            if "BURNED" in status and headline:
                raise ValueError(
                    f"burned partition marked headline eligible: {dataset_id}/{partition_id}"
                )

    return {
        "train_rows": int(registry["train"]["rows"]),
        "development_rows": int(registry["development"]["rows"]),
        "fresh_crc_rows": int(registry["fresh_crc_calibration"]["rows"]),
        "headline_test_rows": int(registry["sealed_iid"]["rows"])
        + int(registry["sealed_ood"]["rows"]),
    }


def validate_frozen_method(method: Mapping[str, Any]) -> dict[str, int]:
    _require(
        method,
        {
            "schema_version",
            "router",
            "cascade",
            "claim_boundary",
            "evidence",
        },
        "frozen method",
    )
    if method["schema_version"] != "selective_router_method_freeze_v1":
        raise ValueError("unsupported frozen method schema")

    router = method["router"]
    if router.get("model") != "two_stage_lr_gate":
        raise ValueError("primary router must remain two_stage_lr_gate")
    features = [str(name) for name in router.get("feature_columns", [])]
    forbidden_features = [
        name
        for name in features
        if "__score" in name
        or name.startswith("label")
        or name.startswith("gold")
        or name.startswith("future")
    ]
    if forbidden_features:
        raise ValueError(f"future action or gold feature leakage: {forbidden_features}")
    if len(features) != 27 or len(set(features)) != len(features):
        raise ValueError("frozen router must contain 27 unique features")
    threshold = float(router.get("activation_threshold", -1.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("router activation threshold must be a probability")

    cascade = method["cascade"]
    action_pool = set(cascade.get("action_pool", []))
    fixed_order = cascade.get("fixed_order", [])
    thresholds = cascade.get("thresholds", {})
    costs = cascade.get("warm_batch1_cost_ms", {})
    if not action_pool or not (
        action_pool == set(fixed_order) == set(thresholds) == set(costs)
    ):
        raise ValueError("cascade action set mismatch")
    if len(fixed_order) != len(action_pool):
        raise ValueError("cascade action set contains duplicates")
    for action, action_threshold in thresholds.items():
        if bool(action_threshold.get("low_enabled")):
            raise ValueError(f"unsupported-stop threshold unexpectedly enabled: {action}")
        if not bool(action_threshold.get("high_enabled")):
            raise ValueError(f"supported-stop threshold disabled: {action}")
        if not 0.0 <= float(action_threshold["tau_high"]) <= 1.0:
            raise ValueError(f"invalid high threshold: {action}")
        if float(costs[action]) <= 0.0:
            raise ValueError(f"invalid action cost: {action}")

    boundary = method["claim_boundary"]
    if boundary.get("diagnostic_only") is not True:
        raise ValueError("frozen result must remain diagnostic only")
    if boundary.get("headline_safe") is not False:
        raise ValueError("development freeze cannot be headline safe")
    if boundary.get("fresh_crc") is not False:
        raise ValueError("fresh CRC cannot be claimed before calibration")
    if boundary.get("legacy_iid_status") != "BURNED_BY_AGGREGATE_INVENTORY":
        raise ValueError("legacy IID disposition is not frozen conservatively")
    sealed_rows = int(boundary.get("experiment_sealed_rows", -1))
    if sealed_rows != 0:
        raise ValueError("the v5 experiment must have zero sealed rows")
    return {"feature_count": len(features), "experiment_sealed_rows": sealed_rows}


def validate_evidence_files(method: Mapping[str, Any], project_root: Path) -> int:
    root = Path(project_root).resolve()
    verified = 0
    for evidence_id, record in method.get("evidence", {}).items():
        path_text = str(record.get("path", ""))
        lowered = path_text.casefold()
        if any(marker in lowered for marker in FORBIDDEN_EVIDENCE_MARKERS):
            raise ValueError(
                f"forbidden sealed evidence path for {evidence_id}: {path_text}"
            )
        path = (root / path_text).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"evidence path escapes project root: {path_text}") from error
        if not path.is_file():
            raise ValueError(f"evidence file is missing: {path_text}")
        expected = str(record.get("sha256", ""))
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"evidence hash mismatch for {evidence_id}: {observed} != {expected}"
            )
        verified += 1
    if verified == 0:
        raise ValueError("frozen method contains no evidence files")
    return verified
