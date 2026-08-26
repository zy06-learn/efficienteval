from __future__ import annotations

import json
import math
import os
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from verifier_wrappers import summary_router_compact16_direct_v1 as base
from verifier_wrappers import summary_router_compact16_targetmix_v1 as targetmix


SCHEMA_VERSION = "summary_router_compact16_targetmix_lodo_v1"
OUTPUT_RELATIVE = Path("results/summary_router_compact16_targetmix_lodo_v1")
FEATURE_RELATIVE = Path("results/summary_router_compact16_direct_v1/COMPACT16_FEATURES.parquet")
EXPECTED_FEATURE_SHA256 = "8572cdacb38613c1a61eaa80b209ed3a3172e667080224be82669381aa07f413"

ACTIONS = base.ACTIONS
ACTION_COSTS_MS = base.ACTION_COSTS_MS
FEATURE_COLUMNS = base.FEATURE_COLUMNS
METHODS = ("HGB-S1", "HGB-S2", "MLP-S2")
BETAS = base.BETAS
SEEDS = base.SEEDS
INNER_FOLDS = 3
BOOTSTRAP_DRAWS = 2_000
TARGET_SHARE = targetmix.TARGET_SHARE
DATASETS = ("cogensumm_val", "frank_train", "ragtruth_train", "unisumeval_train")
EXPECTED_DATASET_ROWS = {
    "cogensumm_val": 535,
    "frank_train": 2_238,
    "ragtruth_train": 2_988,
    "unisumeval_train": 1_089,
}
EXPECTED_DATASET_GROUPS = {
    "cogensumm_val": 107,
    "frank_train": 499,
    "ragtruth_train": 500,
    "unisumeval_train": 135,
}
EXPECTED_PARTS = len(DATASETS) * len(METHODS) * len(SEEDS)
EXPECTED_PREDICTION_ROWS = 6_850 * len(METHODS) * len(SEEDS)
MAX_TARGET_SHARE_ERROR = 0.05


def _availability(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [frame[f"available__{action}"].astype(bool).to_numpy() for action in ACTIONS]
    )


def _candidate_grid(method: str, quick: bool) -> list[dict[str, Any]]:
    if method == "HGB-S1":
        return base._sklearn_grid(method, quick)
    if method == "HGB-S2":
        return targetmix._candidate_grid(method, quick)
    if method == "MLP-S2":
        return base._mlp_grid(quick)
    raise ValueError(f"unsupported LODO method: {method}")


def _fit_model(
    method: str,
    features: np.ndarray,
    correct: np.ndarray,
    availability: np.ndarray,
    parameters: Mapping[str, Any],
    beta: float,
    seed: int,
    *,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    fixed_epochs: int | None = None,
    quick: bool = False,
) -> Any:
    if method == "HGB-S1":
        return base._fit_sklearn_heads(
            method, features, correct, availability, parameters, seed
        )
    if method == "HGB-S2":
        return targetmix._fit_model(
            method, features, correct, availability, parameters, beta, seed
        )
    if method == "MLP-S2":
        return base._fit_mlp(
            features,
            correct,
            availability,
            parameters,
            beta,
            "S2",
            seed,
            validation=validation,
            fixed_epochs=fixed_epochs,
            quick=quick,
        )
    raise ValueError(method)


def _fit_calibrators(
    raw: np.ndarray, correct: np.ndarray, availability: np.ndarray
) -> list[Any]:
    return [
        base._Platt.fit(raw[availability[:, index], index], correct[availability[:, index], index])
        for index in range(len(ACTIONS))
    ]


def _calibrate(raw: np.ndarray, availability: np.ndarray, calibrators: Sequence[Any]) -> np.ndarray:
    values = np.column_stack(
        [calibrators[index].predict(raw[:, index]) for index in range(len(ACTIONS))]
    )
    return np.where(availability, values, 0.0)


def _log_scores(probability: np.ndarray, beta: float) -> np.ndarray:
    discounted = probability * base.cost_discounts(beta, ACTION_COSTS_MS)[None, :]
    return np.log(np.clip(discounted, 1e-12, 1.0))


def _select(
    probability: np.ndarray,
    availability: np.ndarray,
    beta: float,
    price: Mapping[str, float],
) -> np.ndarray:
    return targetmix._select_with_prices(
        _log_scores(probability, beta),
        availability,
        float(price["low_price"]),
        float(price["high_price"]),
    )


def _macro_dataset_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    rows = np.arange(len(frame))
    decisions = np.column_stack(
        [
            pd.to_numeric(frame[f"decision__{action}"], errors="coerce")
            .fillna(0)
            .to_numpy(int)
            for action in ACTIONS
        ]
    )
    verifier_scores = np.column_stack(
        [
            pd.to_numeric(frame[f"score__{action}"], errors="coerce")
            .fillna(0.5)
            .to_numpy(float)
            for action in ACTIONS
        ]
    )
    latency = np.column_stack(
        [
            pd.to_numeric(frame[f"latency_ms__{action}"], errors="coerce")
            .fillna(0)
            .to_numpy(float)
            for action in ACTIONS
        ]
    )
    labels = frame["label_supported"].to_numpy(int)
    hard = decisions[rows, selected]
    score = verifier_scores[rows, selected]
    dataset_rows: list[dict[str, float]] = []
    for dataset in sorted(frame["dataset_key"].astype(str).unique()):
        mask = frame["dataset_key"].astype(str).eq(dataset).to_numpy()
        dataset_rows.append(base._quality_metrics(labels[mask], hard[mask], score[mask]))
    shares = np.bincount(selected, minlength=len(ACTIONS)).astype(float) / len(selected)
    return {
        "macro_auroc": float(np.mean([row["auroc"] for row in dataset_rows])),
        "macro_balanced_accuracy": float(
            np.mean([row["balanced_accuracy"] for row in dataset_rows])
        ),
        "worst_dataset_auroc": float(np.min([row["auroc"] for row in dataset_rows])),
        "mean_verifier_latency_ms": float(latency[rows, selected].mean()),
        "rate_low": float(shares[0]),
        "rate_mid": float(shares[1]),
        "rate_high": float(shares[2]),
        "max_abs_share_error": float(np.max(np.abs(shares - TARGET_SHARE))),
    }


def choose_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no candidate rows")
    eligible = [row for row in rows if float(row["max_abs_share_error"]) <= MAX_TARGET_SHARE_ERROR]
    pool = eligible if eligible else list(rows)
    if not eligible:
        minimum = min(float(row["max_abs_share_error"]) for row in pool)
        pool = [row for row in pool if math.isclose(float(row["max_abs_share_error"]), minimum)]
    chosen = max(
        pool,
        key=lambda row: (
            float(row["macro_auroc"]),
            -float(row["mean_verifier_latency_ms"]),
            float(row["macro_balanced_accuracy"]),
            -len(str(row["parameters"])),
            str(row["parameters"]),
        ),
    )
    return dict(chosen)


def _oof_for_candidate(
    train: pd.DataFrame,
    values: np.ndarray,
    method: str,
    parameters: Mapping[str, Any],
    beta: float,
    seed: int,
    folds: np.ndarray,
    *,
    quick: bool,
) -> tuple[np.ndarray, list[int]]:
    correct, availability, _, _, _ = targetmix._matrices(train)
    raw = np.zeros_like(correct, dtype=float)
    epochs: list[int] = []
    for fold in sorted(np.unique(folds)):
        fit = folds != fold
        validation = folds == fold
        model_seed = base.stable_seed(seed, method, beta, json.dumps(parameters, sort_keys=True), int(fold))
        validation_data = None
        if method == "MLP-S2":
            validation_data = (values[validation], correct[validation], availability[validation])
        model = _fit_model(
            method,
            values[fit],
            correct[fit],
            availability[fit],
            parameters,
            beta,
            model_seed,
            validation=validation_data,
            quick=quick,
        )
        if method == "MLP-S2":
            epochs.append(int(model.best_epoch))
        raw[validation] = base._model_predict(model, values[validation], availability[validation])
    return raw, epochs


def fit_training_policy(
    train: pd.DataFrame,
    features: pd.DataFrame,
    method: str,
    seed: int,
    *,
    betas: Sequence[float] = BETAS,
    inner_folds: int = INNER_FOLDS,
    quick: bool = False,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(method)
    train_datasets = sorted(train["dataset_key"].astype(str).unique().tolist())
    if len(train_datasets) < 2:
        raise ValueError("LODO training requires multiple source datasets")
    values = features.loc[:, FEATURE_COLUMNS].to_numpy(np.float32)
    correct, availability, _, _, _ = targetmix._matrices(train)
    folds = base.build_group_folds(
        train, n_splits=int(inner_folds), seed=base.stable_seed(seed, method, "lodo-inner")
    )
    grids = _candidate_grid(method, quick)
    candidate_rows: list[dict[str, Any]] = []
    candidate_state: dict[tuple[int, float], tuple[np.ndarray, list[Any], list[int], dict[str, Any]]] = {}

    if method == "HGB-S1":
        beta_values = tuple(float(value) for value in betas)
        for candidate_index, parameters in enumerate(grids):
            raw, epochs = _oof_for_candidate(
                train, values, method, parameters, 0.0, seed, folds, quick=quick
            )
            calibrators = _fit_calibrators(raw, correct, availability)
            probability = _calibrate(raw, availability, calibrators)
            for beta in beta_values:
                price = targetmix.calibrate_prices(_log_scores(probability, beta), availability)
                selected = _select(probability, availability, beta, price)
                metrics = _macro_dataset_metrics(train, probability, selected)
                row = {
                    "candidate": int(candidate_index),
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "beta": float(beta),
                    "epoch": None,
                    **metrics,
                    "price_low": float(price["low_price"]),
                    "price_mid": 0.0,
                    "price_high": float(price["high_price"]),
                    "price_objective_l1": float(price["objective_l1"]),
                }
                candidate_rows.append(row)
                candidate_state[(candidate_index, float(beta))] = (raw, calibrators, epochs, dict(price))
                print(
                    f"[select] method={method} seed={seed} candidate={candidate_index + 1}/{len(grids)} "
                    f"beta={beta:g} macro_auroc={metrics['macro_auroc']:.6f} "
                    f"mix={metrics['rate_low']:.3f}/{metrics['rate_mid']:.3f}/{metrics['rate_high']:.3f}",
                    flush=True,
                )
    else:
        for beta in (float(value) for value in betas):
            for candidate_index, parameters in enumerate(grids):
                raw, epochs = _oof_for_candidate(
                    train, values, method, parameters, beta, seed, folds, quick=quick
                )
                calibrators = _fit_calibrators(raw, correct, availability)
                probability = _calibrate(raw, availability, calibrators)
                price = targetmix.calibrate_prices(_log_scores(probability, beta), availability)
                selected = _select(probability, availability, beta, price)
                metrics = _macro_dataset_metrics(train, probability, selected)
                row = {
                    "candidate": int(candidate_index),
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "beta": float(beta),
                    "epoch": int(round(np.median(epochs))) if epochs else None,
                    **metrics,
                    "price_low": float(price["low_price"]),
                    "price_mid": 0.0,
                    "price_high": float(price["high_price"]),
                    "price_objective_l1": float(price["objective_l1"]),
                }
                candidate_rows.append(row)
                candidate_state[(candidate_index, float(beta))] = (raw, calibrators, epochs, dict(price))
                print(
                    f"[select] method={method} seed={seed} candidate={candidate_index + 1}/{len(grids)} "
                    f"beta={beta:g} macro_auroc={metrics['macro_auroc']:.6f} "
                    f"mix={metrics['rate_low']:.3f}/{metrics['rate_mid']:.3f}/{metrics['rate_high']:.3f}",
                    flush=True,
                )

    chosen = choose_candidate(candidate_rows)
    key = (int(chosen["candidate"]), float(chosen["beta"]))
    _, calibrators, epochs, price = candidate_state[key]
    parameters = json.loads(str(chosen["parameters"]))
    final_seed = base.stable_seed(seed, method, float(chosen["beta"]), "lodo-final")
    fixed_epochs = int(chosen["epoch"]) if chosen.get("epoch") is not None else None
    model = _fit_model(
        method,
        values,
        correct,
        availability,
        parameters,
        float(chosen["beta"]),
        final_seed,
        fixed_epochs=fixed_epochs,
        quick=quick,
    )
    return {
        "model": model,
        "calibrators": calibrators,
        "price": price,
        "chosen": chosen,
        "all_candidates": candidate_rows,
        "train_datasets": train_datasets,
        "train_rows": int(len(train)),
        "train_groups": int(train["group_id"].nunique()),
        "inner_fold_rows": pd.Series(folds).value_counts().sort_index().to_dict(),
        "fixed_epochs": fixed_epochs,
    }


def split_lodo(frame: pd.DataFrame, heldout_dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if heldout_dataset not in DATASETS:
        raise ValueError(f"unknown heldout dataset: {heldout_dataset}")
    test = frame["dataset_key"].astype(str).eq(heldout_dataset).to_numpy()
    train = ~test
    if not train.any() or not test.any():
        raise ValueError("empty LODO train or heldout partition")
    train_datasets = set(frame.loc[train, "dataset_key"].astype(str))
    test_datasets = set(frame.loc[test, "dataset_key"].astype(str))
    if train_datasets & test_datasets or test_datasets != {heldout_dataset}:
        raise AssertionError("LODO dataset isolation failure")
    return train, test


def _artifact_paths(output_dir: Path, heldout: str, method: str, seed: int) -> tuple[Path, Path, Path]:
    slug = method.casefold().replace("-", "_")
    name = f"seed_{int(seed)}"
    return (
        output_dir / "parts" / heldout / slug / f"{name}.parquet",
        output_dir / "selections" / heldout / slug / f"{name}.json",
        output_dir / "models" / heldout / slug / f"{name}.joblib",
    )


def _atomic_joblib(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(dict(payload), temporary)
    os.replace(temporary, path)


def _prediction_part(
    test: pd.DataFrame,
    features: pd.DataFrame,
    policy: Mapping[str, Any],
    heldout: str,
    method: str,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    values = features.loc[:, FEATURE_COLUMNS].to_numpy(np.float32)
    availability = _availability(test)
    raw = base._model_predict(policy["model"], values, availability)
    probability = _calibrate(raw, availability, policy["calibrators"])
    beta = float(policy["chosen"]["beta"])
    selected = _select(probability, availability, beta, policy["price"])

    def callback(batch: np.ndarray, mask: np.ndarray) -> np.ndarray:
        batch_raw = base._model_predict(policy["model"], batch, mask)
        batch_probability = _calibrate(batch_raw, mask, policy["calibrators"])
        return _select(batch_probability, mask, beta, policy["price"])

    router_latency = base._measure_batch1_ms(callback, values[0], availability[0])
    rows = np.arange(len(test))
    decisions = np.column_stack(
        [
            pd.to_numeric(test[f"decision__{action}"], errors="coerce")
            .fillna(0)
            .to_numpy(int)
            for action in ACTIONS
        ]
    )
    verifier_scores = np.column_stack(
        [
            pd.to_numeric(test[f"score__{action}"], errors="coerce")
            .fillna(0.5)
            .to_numpy(float)
            for action in ACTIONS
        ]
    )
    latency = np.column_stack(
        [
            pd.to_numeric(test[f"latency_ms__{action}"], errors="coerce")
            .fillna(0)
            .to_numpy(float)
            for action in ACTIONS
        ]
    )
    output = test[["episode_key", "dataset_key", "group_id", "label_supported"]].copy()
    output["heldout_dataset"] = heldout
    output["method"] = method
    output["seed"] = int(seed)
    output["selected_beta"] = beta
    output["selected_action"] = np.asarray(ACTIONS, dtype=object)[selected]
    output["router_decision"] = decisions[rows, selected]
    output["probability_supported"] = verifier_scores[rows, selected]
    output["correct"] = (
        output["router_decision"].to_numpy(int) == output["label_supported"].to_numpy(int)
    ).astype(np.int8)
    output["verifier_latency_ms"] = latency[rows, selected]
    output["feature_latency_ms"] = features["feature_latency_ms"].to_numpy(float)
    output["router_latency_ms"] = float(router_latency)
    output["end_to_end_latency_ms"] = (
        output["verifier_latency_ms"]
        + output["feature_latency_ms"]
        + output["router_latency_ms"]
    )
    output["forced_upgrade"] = ~availability[:, 0]
    output["train_rate_low"] = float(policy["chosen"]["rate_low"])
    output["train_rate_mid"] = float(policy["chosen"]["rate_mid"])
    output["train_rate_high"] = float(policy["chosen"]["rate_high"])
    output["price_low"] = float(policy["price"]["low_price"])
    output["price_mid"] = 0.0
    output["price_high"] = float(policy["price"]["high_price"])
    for index, action in enumerate(ACTIONS):
        output[f"available__{action}"] = availability[:, index]
        output[f"correct__{action}"] = pd.to_numeric(
            test[f"correct__{action}"], errors="coerce"
        ).to_numpy(float)
        output[f"router_probability__{action}"] = probability[:, index]
        output[f"raw_router_probability__{action}"] = raw[:, index]
    return output.reset_index(drop=True), float(router_latency)


def run_lodo(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    *,
    heldouts: Sequence[str] = DATASETS,
    methods: Sequence[str] = METHODS,
    seeds: Sequence[int] = SEEDS,
    betas: Sequence[float] = BETAS,
    inner_folds: int = INNER_FOLDS,
    quick: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if len(frame) != len(features):
        raise ValueError("matrix/feature row mismatch")
    if not frame["episode_key"].astype(str).eq(features["episode_key"].astype(str)).all():
        raise ValueError("matrix/feature episode alignment mismatch")
    unknown_methods = sorted(set(methods) - set(METHODS))
    unknown_heldouts = sorted(set(heldouts) - set(DATASETS))
    if unknown_methods or unknown_heldouts:
        raise ValueError(f"unknown methods={unknown_methods}, heldouts={unknown_heldouts}")
    for heldout, method, seed in product(heldouts, methods, seeds):
        part_path, selection_path, model_path = _artifact_paths(
            output_dir, str(heldout), str(method), int(seed)
        )
        if part_path.is_file() and selection_path.is_file() and model_path.is_file():
            print(f"[lodo] reuse heldout={heldout} method={method} seed={seed}", flush=True)
            continue
        train_mask, test_mask = split_lodo(frame, str(heldout))
        train = frame.loc[train_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        train_features = features.loc[train_mask].reset_index(drop=True)
        test_features = features.loc[test_mask].reset_index(drop=True)
        print(
            f"[lodo] start heldout={heldout} method={method} seed={seed} "
            f"train_rows={len(train)} test_rows={len(test)}",
            flush=True,
        )
        policy = fit_training_policy(
            train,
            train_features,
            str(method),
            int(seed),
            betas=betas,
            inner_folds=int(inner_folds),
            quick=quick,
        )
        part, router_latency = _prediction_part(
            test, test_features, policy, str(heldout), str(method), int(seed)
        )
        test_shares = part["selected_action"].value_counts(normalize=True).reindex(ACTIONS, fill_value=0.0)
        selection = {
            "schema_version": SCHEMA_VERSION,
            "heldout_dataset": str(heldout),
            "method": str(method),
            "seed": int(seed),
            "train_datasets": policy["train_datasets"],
            "price_calibration_datasets": policy["train_datasets"],
            "test_datasets": [str(heldout)],
            "train_test_dataset_overlap": sorted(set(policy["train_datasets"]) & {str(heldout)}),
            "train_rows": int(policy["train_rows"]),
            "train_groups": int(policy["train_groups"]),
            "test_rows": int(len(test)),
            "test_groups": int(test["group_id"].nunique()),
            "inner_folds": int(inner_folds),
            "inner_fold_rows": policy["inner_fold_rows"],
            "selected": policy["chosen"],
            "all_candidates": policy["all_candidates"],
            "price": {
                "low": float(policy["price"]["low_price"]),
                "mid": 0.0,
                "high": float(policy["price"]["high_price"]),
                "objective_l1": float(policy["price"]["objective_l1"]),
            },
            "target_share": dict(zip(ACTIONS, TARGET_SHARE.tolist())),
            "heldout_share": {action: float(test_shares[action]) for action in ACTIONS},
            "heldout_share_drift": {
                action: float(test_shares[action] - policy["chosen"][f"rate_{tier}"])
                for action, tier in zip(ACTIONS, ("low", "mid", "high"))
            },
            "router_latency_ms": float(router_latency),
            "heldout_labels_used_for_training_selection_or_price": 0,
        }
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "heldout_dataset": str(heldout),
            "method": str(method),
            "seed": int(seed),
            "feature_columns": tuple(FEATURE_COLUMNS),
            "actions": tuple(ACTIONS),
            "beta": float(policy["chosen"]["beta"]),
            "parameters": json.loads(str(policy["chosen"]["parameters"])),
            "model": policy["model"],
            "calibrators": policy["calibrators"],
            "price": dict(policy["price"]),
        }
        base.write_json(selection_path, selection)
        _atomic_joblib(model_path, checkpoint)
        base.write_parquet(part_path, part)
        print(
            f"[lodo] complete heldout={heldout} method={method} seed={seed} "
            f"beta={float(policy['chosen']['beta']):g} "
            f"test_mix={test_shares.iloc[0]:.3f}/{test_shares.iloc[1]:.3f}/{test_shares.iloc[2]:.3f}",
            flush=True,
        )

    parts: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for heldout, method, seed in product(heldouts, methods, seeds):
        part_path, selection_path, model_path = _artifact_paths(
            output_dir, str(heldout), str(method), int(seed)
        )
        if not part_path.is_file() or not selection_path.is_file() or not model_path.is_file():
            raise AssertionError(f"missing LODO artifact for {heldout}/{method}/{seed}")
        parts.append(pd.read_parquet(part_path))
        selections.append(json.loads(selection_path.read_text()))
    predictions = pd.concat(parts, ignore_index=True)
    expected = sum(
        int(frame["dataset_key"].astype(str).eq(str(heldout)).sum()) for heldout in heldouts
    ) * len(methods) * len(seeds)
    keys = ["episode_key", "method", "seed"]
    if len(predictions) != expected or predictions.duplicated(keys).any():
        raise AssertionError("LODO prediction coverage failure")
    for selection in selections:
        if selection["train_test_dataset_overlap"]:
            raise AssertionError("LODO train/test dataset leakage")
    base.write_parquet(output_dir / "LODO_PREDICTIONS.parquet", predictions)
    return {"predictions": predictions, "selections": selections, "expected_rows": expected}


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, seed, heldout), part in predictions.groupby(
        ["method", "seed", "heldout_dataset"], sort=True
    ):
        metrics = base._prediction_metrics(
            part,
            {"method": method, "seed": int(seed), "heldout_dataset": heldout},
        )
        for tier, action in zip(("low", "mid", "high"), ACTIONS):
            metrics[f"train_rate_{tier}"] = float(part[f"train_rate_{tier}"].iloc[0])
            metrics[f"test_rate_{tier}"] = float(metrics[f"rate__{action}"])
            metrics[f"share_drift_{tier}"] = metrics[f"test_rate_{tier}"] - metrics[f"train_rate_{tier}"]
        metrics["selected_beta"] = float(part["selected_beta"].iloc[0])
        rows.append(metrics)
    return pd.DataFrame(rows)


def _dataset_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "heldout_dataset"]
    numeric = [
        column
        for column in metrics.columns
        if column not in {*keys, "seed"} and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    rows: list[dict[str, Any]] = []
    for (method, heldout), part in metrics.groupby(keys, sort=True):
        row: dict[str, Any] = {"method": method, "heldout_dataset": heldout, "seed_count": int(part["seed"].nunique())}
        for column in numeric:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _method_summary(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    seed_rows: list[dict[str, Any]] = []
    quality_columns = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "auroc",
        "supported_recall",
        "unsupported_recall",
        "point_biserial",
        "spearman",
        "kendall_tau_b",
    )
    for (method, seed), part in metrics.groupby(["method", "seed"], sort=True):
        pooled_part = predictions[
            predictions["method"].eq(method) & predictions["seed"].eq(seed)
        ]
        pooled = base._prediction_metrics(pooled_part)
        row: dict[str, Any] = {"method": method, "seed": int(seed)}
        for column in quality_columns:
            row[f"macro_{column}"] = float(part[column].mean())
            row[f"pooled_{column}"] = float(pooled[column])
        row["worst_dataset_auroc"] = float(part["auroc"].min())
        for column in (
            "mean_verifier_latency_ms",
            "mean_feature_latency_ms",
            "mean_router_latency_ms",
            "mean_end_to_end_latency_ms",
            "forced_upgrade_rate",
            *[f"rate__{action}" for action in ACTIONS],
            "share_drift_low",
            "share_drift_mid",
            "share_drift_high",
        ):
            row[column] = float(part[column].mean())
        seed_rows.append(row)
    seed_frame = pd.DataFrame(seed_rows)
    rows = []
    numeric = [column for column in seed_frame.columns if column not in {"method", "seed"}]
    for method, part in seed_frame.groupby("method", sort=True):
        row = {"method": method, "seed_count": int(part["seed"].nunique())}
        for column in numeric:
            row[f"{column}_mean"] = float(part[column].mean())
            row[f"{column}_std"] = float(part[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _fixed_baseline_part(frame: pd.DataFrame, selected: np.ndarray, baseline: str) -> pd.DataFrame:
    part = base._baseline_prediction(frame, pd.DataFrame(index=frame.index), selected)
    part["baseline"] = baseline
    part["heldout_dataset"] = part["dataset_key"].astype(str)
    part["reference_method"] = ""
    part["reference_seed"] = -1
    return part


def build_baseline_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    availability = _availability(frame)
    correct = np.column_stack(
        [
            pd.to_numeric(frame[f"correct__{action}"], errors="coerce")
            .fillna(0)
            .to_numpy(float)
            for action in ACTIONS
        ]
    )
    fixed: list[pd.DataFrame] = []
    for index, action in enumerate(ACTIONS):
        preference = np.zeros_like(availability, dtype=float)
        preference[:, index] = 1.0
        selected = base.masked_argmax(preference, availability)
        fixed.append(_fixed_baseline_part(frame, selected, f"always_{action}"))
    costs = np.asarray([ACTION_COSTS_MS[action] for action in ACTIONS])
    utility = correct + (costs.max() - costs)[None, :] * 1e-9
    oracle = base.masked_argmax(utility, availability)
    fixed.append(_fixed_baseline_part(frame, oracle, "cheapest_correct_oracle"))

    matched: list[pd.DataFrame] = []
    indexed = frame.set_index("episode_key", drop=False)
    for (method, seed, heldout), router in predictions.groupby(
        ["method", "seed", "heldout_dataset"], sort=True
    ):
        ordered = indexed.loc[router["episode_key"].astype(str)].reset_index(drop=True)
        available = _availability(ordered)
        shares = router["selected_action"].value_counts(normalize=True).reindex(ACTIONS, fill_value=0.0).to_numpy(float)
        weights = np.tile(shares, (len(ordered), 1)) * available
        weights /= weights.sum(axis=1, keepdims=True)
        generator = np.random.default_rng(base.stable_seed(int(seed), method, heldout, "lodo-matched-random"))
        draws = generator.random(len(ordered))
        selected = (draws[:, None] > np.cumsum(weights, axis=1)).sum(axis=1)
        part = base._baseline_prediction(
            ordered,
            pd.DataFrame(index=ordered.index),
            selected,
            feature_latency=router["feature_latency_ms"].to_numpy(float),
            router_latency=router["router_latency_ms"].to_numpy(float),
        )
        part["baseline"] = "matched_random"
        part["heldout_dataset"] = str(heldout)
        part["reference_method"] = str(method)
        part["reference_seed"] = int(seed)
        matched.append(part)
    return pd.concat([*fixed, *matched], ignore_index=True)


def _baseline_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fixed = predictions[~predictions["baseline"].eq("matched_random")]
    for (baseline, heldout), part in fixed.groupby(["baseline", "heldout_dataset"], sort=True):
        rows.append(base._prediction_metrics(part, {"scope": "dataset", "baseline": baseline, "heldout_dataset": heldout, "reference_method": "", "reference_seed": -1}))
    for baseline, part in fixed.groupby("baseline", sort=True):
        dataset_rows = [base._prediction_metrics(values) for _, values in part.groupby("heldout_dataset", sort=True)]
        macro = {key: float(np.mean([row[key] for row in dataset_rows])) for key in dataset_rows[0] if isinstance(dataset_rows[0][key], (int, float, np.number))}
        rows.append({"scope": "macro", "baseline": baseline, "heldout_dataset": "ALL", "reference_method": "", "reference_seed": -1, **macro})
        rows.append(base._prediction_metrics(part, {"scope": "pooled", "baseline": baseline, "heldout_dataset": "ALL", "reference_method": "", "reference_seed": -1}))
    matched = predictions[predictions["baseline"].eq("matched_random")]
    for (method, seed, heldout), part in matched.groupby(["reference_method", "reference_seed", "heldout_dataset"], sort=True):
        rows.append(base._prediction_metrics(part, {"scope": "dataset", "baseline": "matched_random", "heldout_dataset": heldout, "reference_method": method, "reference_seed": int(seed)}))
    for (method, seed), part in matched.groupby(["reference_method", "reference_seed"], sort=True):
        dataset_rows = [base._prediction_metrics(values) for _, values in part.groupby("heldout_dataset", sort=True)]
        macro = {key: float(np.mean([row[key] for row in dataset_rows])) for key in dataset_rows[0] if isinstance(dataset_rows[0][key], (int, float, np.number))}
        rows.append({"scope": "macro", "baseline": "matched_random", "heldout_dataset": "ALL", "reference_method": method, "reference_seed": int(seed), **macro})
        rows.append(base._prediction_metrics(part, {"scope": "pooled", "baseline": "matched_random", "heldout_dataset": "ALL", "reference_method": method, "reference_seed": int(seed)}))
    return pd.DataFrame(rows)


def _resampled_indices(part: pd.DataFrame, generator: np.random.Generator) -> np.ndarray:
    indices: list[np.ndarray] = []
    for dataset in DATASETS:
        dataset_part = part[part["heldout_dataset"].eq(dataset)]
        groups = sorted(dataset_part["group_id"].astype(str).unique())
        by_group = {
            group: dataset_part.index[dataset_part["group_id"].astype(str).eq(group)].to_numpy(int)
            for group in groups
        }
        sampled = generator.choice(groups, size=len(groups), replace=True)
        indices.extend(by_group[str(group)] for group in sampled)
    return np.concatenate(indices)


def _macro_auroc(part: pd.DataFrame) -> float:
    values = []
    for dataset in DATASETS:
        subset = part[part["heldout_dataset"].eq(dataset)]
        values.append(
            roc_auc_score(
                subset["label_supported"].to_numpy(int),
                subset["probability_supported"].to_numpy(float),
            )
        )
    return float(np.mean(values))


def bootstrap_paired(
    predictions: pd.DataFrame,
    baselines: pd.DataFrame,
    draws: int = BOOTSTRAP_DRAWS,
) -> pd.DataFrame:
    matched = baselines[baselines["baseline"].eq("matched_random")]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        router_method = predictions[predictions["method"].eq(method)].reset_index(drop=True)
        matched_method = matched[matched["reference_method"].eq(method)].reset_index(drop=True)
        router_values: list[float] = []
        matched_values: list[float] = []
        delta_values: list[float] = []
        latency_delta: list[float] = []
        generator = np.random.default_rng(base.stable_seed(2026, method, "lodo-bootstrap"))
        for _ in range(int(draws)):
            seed_router: list[float] = []
            seed_matched: list[float] = []
            seed_latency: list[float] = []
            for seed in SEEDS:
                router_seed = router_method[router_method["seed"].eq(seed)].reset_index(drop=True)
                matched_seed = matched_method[matched_method["reference_seed"].eq(seed)].reset_index(drop=True)
                if not router_seed["episode_key"].astype(str).eq(matched_seed["episode_key"].astype(str)).all():
                    raise AssertionError("paired bootstrap episode alignment failure")
                sampled = _resampled_indices(router_seed, generator)
                router_sample = router_seed.loc[sampled]
                matched_sample = matched_seed.loc[sampled]
                router_auroc = _macro_auroc(router_sample)
                matched_auroc = _macro_auroc(matched_sample)
                seed_router.append(router_auroc)
                seed_matched.append(matched_auroc)
                seed_latency.append(
                    float(router_sample["end_to_end_latency_ms"].mean() - matched_sample["end_to_end_latency_ms"].mean())
                )
            router_value = float(np.mean(seed_router))
            matched_value = float(np.mean(seed_matched))
            router_values.append(router_value)
            matched_values.append(matched_value)
            delta_values.append(router_value - matched_value)
            latency_delta.append(float(np.mean(seed_latency)))
        rows.append(
            {
                "method": method,
                "draws": int(draws),
                "router_macro_auroc_ci_low": float(np.quantile(router_values, 0.025)),
                "router_macro_auroc_ci_high": float(np.quantile(router_values, 0.975)),
                "matched_random_macro_auroc_ci_low": float(np.quantile(matched_values, 0.025)),
                "matched_random_macro_auroc_ci_high": float(np.quantile(matched_values, 0.975)),
                "delta_macro_auroc_vs_matched_random_ci_low": float(np.quantile(delta_values, 0.025)),
                "delta_macro_auroc_vs_matched_random_ci_high": float(np.quantile(delta_values, 0.975)),
                "delta_latency_ms_vs_matched_random_ci_low": float(np.quantile(latency_delta, 0.025)),
                "delta_latency_ms_vs_matched_random_ci_high": float(np.quantile(latency_delta, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _pareto_frontier(method_summary: pd.DataFrame, baseline_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in method_summary.iterrows():
        rows.append(
            {
                "name": str(row["method"]),
                "type": "router",
                "macro_auroc": float(row["macro_auroc_mean"]),
                "mean_end_to_end_latency_ms": float(row["mean_end_to_end_latency_ms_mean"]),
            }
        )
    fixed = baseline_metrics[
        baseline_metrics["scope"].eq("macro")
        & baseline_metrics["reference_method"].eq("")
        & ~baseline_metrics["baseline"].eq("cheapest_correct_oracle")
    ]
    for _, row in fixed.iterrows():
        rows.append(
            {
                "name": str(row["baseline"]),
                "type": "fixed_baseline",
                "macro_auroc": float(row["auroc"]),
                "mean_end_to_end_latency_ms": float(row["mean_end_to_end_latency_ms"]),
            }
        )
    work = pd.DataFrame(rows)
    keep: list[bool] = []
    for index, row in work.iterrows():
        dominated = (
            (work["macro_auroc"] >= float(row["macro_auroc"]))
            & (work["mean_end_to_end_latency_ms"] <= float(row["mean_end_to_end_latency_ms"]))
            & (
                (work["macro_auroc"] > float(row["macro_auroc"]))
                | (work["mean_end_to_end_latency_ms"] < float(row["mean_end_to_end_latency_ms"]))
            )
        )
        dominated.loc[index] = False
        keep.append(not bool(dominated.any()))
    work["pareto"] = keep
    return work.sort_values(["mean_end_to_end_latency_ms", "macro_auroc"], ascending=[True, False])


def _gate_results(
    method_summary: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    fixed = baseline_metrics[
        baseline_metrics["scope"].eq("macro")
        & baseline_metrics["reference_method"].eq("")
        & baseline_metrics["baseline"].isin([f"always_{action}" for action in ACTIONS])
    ].copy()
    factkb = fixed[fixed["baseline"].eq("always_factkb")].iloc[0]
    quality_best = fixed.sort_values("auroc", ascending=False).iloc[0]
    method_rows: list[dict[str, Any]] = []
    for _, row in method_summary.iterrows():
        method = str(row["method"])
        auroc = float(row["macro_auroc_mean"])
        latency = float(row["mean_end_to_end_latency_ms_mean"])
        dominated = bool(
            (
                (fixed["auroc"] >= auroc)
                & (fixed["mean_end_to_end_latency_ms"] <= latency)
                & (
                    (fixed["auroc"] > auroc)
                    | (fixed["mean_end_to_end_latency_ms"] < latency)
                )
            ).any()
        )
        ci = bootstrap[bootstrap["method"].eq(method)].iloc[0]
        gates = {
            "macro_auroc_above_always_factkb": auroc > float(factkb["auroc"]),
            "faster_than_best_quality_fixed": latency < float(quality_best["mean_end_to_end_latency_ms"]),
            "not_dominated_by_fixed_verifier": not dominated,
        }
        method_rows.append(
            {
                "method": method,
                "macro_auroc": auroc,
                "mean_end_to_end_latency_ms": latency,
                "quality_best_fixed": str(quality_best["baseline"]),
                "quality_best_fixed_macro_auroc": float(quality_best["auroc"]),
                "quality_best_fixed_latency_ms": float(quality_best["mean_end_to_end_latency_ms"]),
                "always_factkb_macro_auroc": float(factkb["auroc"]),
                "paired_delta_auroc_ci_low": float(ci["delta_macro_auroc_vs_matched_random_ci_low"]),
                "paired_delta_auroc_ci_high": float(ci["delta_macro_auroc_vs_matched_random_ci_high"]),
                **gates,
                "all_primary_gates_pass": all(gates.values()),
            }
        )
    passed = [row for row in method_rows if row["all_primary_gates_pass"]]
    candidate = max(passed, key=lambda row: (row["macro_auroc"], -row["mean_end_to_end_latency_ms"])) if passed else None
    return {
        "status": "CANDIDATE_FOUND" if candidate is not None else "NO_CANDIDATE_PASSES_ALL_GATES",
        "development_only_not_external_test": True,
        "methods": method_rows,
        "selected_development_candidate": candidate,
    }


def _write_report(
    output_dir: Path,
    method_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    gates: Mapping[str, Any],
) -> None:
    def markdown(frame: pd.DataFrame) -> str:
        return frame.to_markdown(index=False, floatfmt=".4f")

    method_view = method_summary[
        [
            "method",
            "macro_auroc_mean",
            "worst_dataset_auroc_mean",
            "pooled_balanced_accuracy_mean",
            "pooled_mcc_mean",
            "mean_end_to_end_latency_ms_mean",
            "rate__factkb_mean",
            "rate__granite_guardian_3_1_2b_mean",
            "rate__qwen30_fast_mean",
        ]
    ]
    baseline_view = baseline_metrics[
        baseline_metrics["scope"].eq("macro")
        & baseline_metrics["reference_method"].eq("")
    ][["baseline", "auroc", "balanced_accuracy", "mcc", "mean_end_to_end_latency_ms"]]
    dataset_view = dataset_summary[
        [
            "method",
            "heldout_dataset",
            "auroc_mean",
            "balanced_accuracy_mean",
            "mcc_mean",
            "mean_end_to_end_latency_ms_mean",
            "test_rate_low_mean",
            "test_rate_mid_mean",
            "test_rate_high_mean",
        ]
    ]
    lines = [
        "# Compact-16 TargetMix 严格 LODO 结果",
        "",
        "本实验只使用四个开发数据集执行 3 域训练、1 域完整留出。留出域标签未参与模型、超参数、β 或价格选择；本结果不是最终外部测试。",
        "",
        "## Router 四域等权结果",
        "",
        markdown(method_view),
        "",
        "## 固定 Verifier / Oracle 基线",
        "",
        markdown(baseline_view),
        "",
        "## 各留出域结果",
        "",
        markdown(dataset_view),
        "",
        "## 开发选择门",
        "",
        f"状态：`{gates['status']}`。即使找到候选，也只能用于后续冻结外部测试，不能视为外部泛化结论。",
        "",
    ]
    base.atomic_text(output_dir / "REPORT_ZH.md", "\n".join(lines))


def build_reports(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    metrics = _metric_rows(predictions)
    dataset_summary = _dataset_summary(metrics)
    method_summary = _method_summary(predictions, metrics)
    baseline_predictions = build_baseline_predictions(frame, predictions)
    baseline_metrics = _baseline_metrics(baseline_predictions)
    bootstrap = bootstrap_paired(predictions, baseline_predictions, draws=bootstrap_draws)
    pareto = _pareto_frontier(method_summary, baseline_metrics)
    gates = _gate_results(method_summary, baseline_metrics, bootstrap)
    outputs = {
        "LODO_METRICS_BY_SEED.csv": metrics,
        "LODO_DATASET_SUMMARY.csv": dataset_summary,
        "LODO_METHOD_SUMMARY.csv": method_summary,
        "BASELINE_METRICS.csv": baseline_metrics,
        "BOOTSTRAP_CI.csv": bootstrap,
        "COMBINED_PARETO_FRONTIER.csv": pareto,
    }
    for name, table in outputs.items():
        base.write_csv(output_dir / name, table)
    base.write_parquet(output_dir / "BASELINE_PREDICTIONS.parquet", baseline_predictions)
    base.write_json(output_dir / "DEVELOPMENT_GATE_RESULTS.json", gates)
    _write_report(output_dir, method_summary, dataset_summary, baseline_metrics, gates)
    return {**outputs, "BASELINE_PREDICTIONS.parquet": baseline_predictions, "gates": gates}


def validate_assets(project_root: Path) -> dict[str, Any]:
    matrix_hash = base.sha256_file(project_root / base.MATRIX_RELATIVE)
    input_hash = base.sha256_file(project_root / base.INPUT_RELATIVE)
    feature_hash = base.sha256_file(project_root / FEATURE_RELATIVE)
    actual = {
        "matrix_sha256": matrix_hash,
        "summary_input_sha256": input_hash,
        "compact16_feature_sha256": feature_hash,
    }
    expected = {
        "matrix_sha256": base.EXPECTED_MATRIX_SHA256,
        "summary_input_sha256": base.EXPECTED_INPUT_SHA256,
        "compact16_feature_sha256": EXPECTED_FEATURE_SHA256,
    }
    if actual != expected:
        raise ValueError(f"frozen LODO asset mismatch: {actual}")
    return {"status": "PASS", **actual}


def load_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assets = validate_assets(project_root)
    frame, _, input_audit = base.load_frozen_training_inputs(project_root)
    features = base.load_feature_asset(
        frame, project_root / "results/summary_router_compact16_direct_v1"
    )
    rows = frame["dataset_key"].value_counts().sort_index().to_dict()
    groups = frame.groupby("dataset_key")["group_id"].nunique().sort_index().to_dict()
    if rows != EXPECTED_DATASET_ROWS or groups != EXPECTED_DATASET_GROUPS:
        raise ValueError(f"LODO dataset inventory drift: rows={rows}, groups={groups}")
    audit = {
        "status": "PASS",
        "assets": assets,
        "rows": int(len(frame)),
        "groups": int(frame["group_id"].nunique()),
        "dataset_rows": rows,
        "dataset_groups": groups,
        "factkb_unavailable": int((~frame["available__factkb"].astype(bool)).sum()),
        "forced_upgrade_policy": "if_factkb_unavailable_select_best_available_mid_or_high",
        "storysumm_rows_read": 0,
        "official_or_sealed_test_rows_read": 0,
        "verifier_inference_calls": 0,
        "base_input_audit": input_audit,
    }
    return frame, features, audit


def preregistration_payload(project_root: Path, code_paths: Sequence[Path]) -> dict[str, Any]:
    assets = validate_assets(project_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": base.utc_now(),
        "scope": "strict_four_way_leave_one_dataset_out_development_evaluation",
        "assets": assets,
        "datasets": list(DATASETS),
        "dataset_rows": EXPECTED_DATASET_ROWS,
        "dataset_groups": EXPECTED_DATASET_GROUPS,
        "actions": list(ACTIONS),
        "action_costs_ms": ACTION_COSTS_MS,
        "methods": list(METHODS),
        "supervision": {
            "HGB-S1": "independent_correctness_probability_heads",
            "HGB-S2": "correctness_plus_utility_gap_weighted_pairwise_borda",
            "MLP-S2": "masked_correctness_bce_plus_utility_gap_pairwise_loss",
        },
        "betas": list(BETAS),
        "seeds": list(SEEDS),
        "inner_folds": INNER_FOLDS,
        "inner_split": "source_group_stratified_three_fold_on_training_datasets_only",
        "selection_primary": "equal_weight_training_dataset_macro_auroc",
        "selection_constraint": {"max_abs_target_share_error": MAX_TARGET_SHARE_ERROR},
        "selection_secondary": "lower_mean_verifier_latency_then_macro_balanced_accuracy",
        "target_share": dict(zip(ACTIONS, TARGET_SHARE.tolist())),
        "price_search": {
            "scope": "pooled_training_domain_oof_only",
            "mid_price": 0.0,
            "bounds": list(targetmix.PRICE_BOUNDS),
            "iterations": targetmix.PRICE_ITERATIONS,
            "objective": "l1_to_60_30_10",
        },
        "heldout_policy": "one_prediction_no_model_beta_price_or_unlabeled_target_distribution_adaptation",
        "expected_parts": EXPECTED_PARTS,
        "expected_prediction_rows": EXPECTED_PREDICTION_ROWS,
        "metrics": {
            "primary": ["equal_weight_macro_auroc", "worst_dataset_auroc", "mean_end_to_end_latency_ms"],
            "pooled": ["auroc", "balanced_accuracy", "mcc", "macro_f1", "supported_recall", "unsupported_recall", "point_biserial", "spearman", "kendall_tau_b"],
            "efficiency": ["verifier_latency_ms", "feature_latency_ms", "router_latency_ms", "end_to_end_latency_ms", "action_rates", "forced_upgrade_rate"],
        },
        "baselines": [
            "always_factkb",
            "always_granite_guardian_3_1_2b",
            "always_qwen30_fast",
            "matched_random_same_heldout_action_mix",
            "cheapest_correct_oracle",
        ],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "strata": "heldout_dataset",
            "unit": "source_group_with_replacement",
            "comparison": "router_vs_paired_matched_random_macro_auroc",
        },
        "success_gates": [
            "macro_auroc_above_always_factkb",
            "faster_than_best_quality_fixed_verifier",
            "not_dominated_by_any_fixed_verifier_in_macro_auroc_latency",
            "report_paired_95pct_ci_vs_matched_random",
            "report_every_failed_heldout_dataset",
        ],
        "development_only_not_external_test": True,
        "storysumm_rows_read": 0,
        "official_or_sealed_test_rows_read": 0,
        "verifier_inference_calls": 0,
        "code_hashes": {
            str(path.relative_to(project_root)): base.sha256_file(path) for path in code_paths
        },
    }


def audit_completed_run(project_root: Path, output_dir: Path) -> dict[str, Any]:
    assets = validate_assets(project_root)
    predictions = pd.read_parquet(output_dir / "LODO_PREDICTIONS.parquet")
    required = [
        "PREREG.json",
        "INPUT_AUDIT.json",
        "LODO_PREDICTIONS.parquet",
        "LODO_METRICS_BY_SEED.csv",
        "LODO_DATASET_SUMMARY.csv",
        "LODO_METHOD_SUMMARY.csv",
        "BASELINE_PREDICTIONS.parquet",
        "BASELINE_METRICS.csv",
        "BOOTSTRAP_CI.csv",
        "COMBINED_PARETO_FRONTIER.csv",
        "DEVELOPMENT_GATE_RESULTS.json",
        "REPORT_ZH.md",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    duplicate = int(predictions.duplicated(["episode_key", "method", "seed"]).sum())
    configs = predictions[["heldout_dataset", "method", "seed"]].drop_duplicates()
    part_files = list((output_dir / "parts").glob("**/*.parquet"))
    selection_files = list((output_dir / "selections").glob("**/*.json"))
    model_files = list((output_dir / "models").glob("**/*.joblib"))
    leakage: list[str] = []
    heldout_label_uses = 0
    for path in selection_files:
        selection = json.loads(path.read_text())
        leakage.extend(selection.get("train_test_dataset_overlap", []))
        heldout_label_uses += int(selection.get("heldout_labels_used_for_training_selection_or_price", -1))
    reload_failures: list[str] = []
    for path in model_files:
        try:
            checkpoint = joblib.load(path)
            if not {"model", "calibrators", "price", "feature_columns", "actions"}.issubset(checkpoint):
                reload_failures.append(str(path))
        except Exception:
            reload_failures.append(str(path))
    coverage = predictions.groupby(["method", "seed"]).size()
    status = (
        len(predictions) == EXPECTED_PREDICTION_ROWS
        and duplicate == 0
        and len(configs) == EXPECTED_PARTS
        and set(configs["heldout_dataset"]) == set(DATASETS)
        and set(configs["method"]) == set(METHODS)
        and set(configs["seed"]) == set(SEEDS)
        and coverage.eq(6_850).all()
        and len(part_files) == EXPECTED_PARTS
        and len(selection_files) == EXPECTED_PARTS
        and len(model_files) == EXPECTED_PARTS
        and not leakage
        and heldout_label_uses == 0
        and not reload_failures
        and not missing
    )
    audit = {
        "status": "PASS" if status else "FAIL",
        "created_at_utc": base.utc_now(),
        "assets": assets,
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": EXPECTED_PREDICTION_ROWS,
        "configurations": int(len(configs)),
        "expected_parts": EXPECTED_PARTS,
        "duplicate_predictions": duplicate,
        "coverage_rows_per_method_seed": coverage.to_dict(),
        "part_files": len(part_files),
        "selection_files": len(selection_files),
        "model_files": len(model_files),
        "checkpoint_reload_failures": reload_failures,
        "train_test_dataset_overlap": sorted(set(leakage)),
        "heldout_labels_used_for_training_selection_or_price": heldout_label_uses,
        "missing_outputs": missing,
        "storysumm_rows_read": 0,
        "official_or_sealed_test_rows_read": 0,
        "verifier_inference_calls": 0,
        "output_hashes": {
            name: base.sha256_file(output_dir / name)
            for name in required
            if (output_dir / name).is_file()
        },
    }
    base.write_json(output_dir / "AUDIT.json", audit)
    if not status:
        raise AssertionError(f"strict LODO audit failed: {audit}")
    return audit


def run_formal(
    project_root: Path,
    output_dir: Path,
    code_paths: Sequence[Path],
    *,
    quick: bool = False,
) -> dict[str, Any]:
    frame, features, input_audit = load_inputs(project_root)
    base.write_json(output_dir / "INPUT_AUDIT.json", input_audit)
    expected_prereg = preregistration_payload(project_root, code_paths)
    prereg_path = output_dir / "PREREG.json"
    if not prereg_path.is_file():
        raise FileNotFoundError("PREREG.json must be frozen before formal LODO")
    existing = json.loads(prereg_path.read_text())
    for key in (
        "assets",
        "datasets",
        "actions",
        "methods",
        "betas",
        "seeds",
        "target_share",
        "price_search",
        "code_hashes",
    ):
        if existing.get(key) != expected_prereg.get(key):
            raise ValueError(f"LODO preregistration drift in {key}")
    result = run_lodo(frame, features, output_dir, quick=quick)
    reports = build_reports(
        frame,
        features,
        result["predictions"],
        output_dir,
        bootstrap_draws=0 if quick else BOOTSTRAP_DRAWS,
    )
    audit = audit_completed_run(project_root, output_dir)
    return {"input_audit": input_audit, "reports": reports, "audit": audit}
