#!/usr/bin/env python3
"""Train and spatially evaluate RF, XGBoost, and MLP classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import ANNUAL_FEATURE_NAMES, FEATURE_NAMES, log, resolve, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF, XGBoost, and MLP cocoa classifiers.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--season", choices=("djf", "wet", "annual"), default="annual")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--split-attempts",
        type=int,
        default=500,
        help="Candidate spatial-group splits searched for two-class train/test sets.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def build_models(seed: int, n_jobs: int) -> dict:
    try:
        from xgboost import XGBClassifier
    except Exception as error:
        raise RuntimeError(
            "XGBoost could not be loaded. Install the Python requirements; on macOS "
            "also install its OpenMP runtime with: brew install libomp"
        ) from error
    impute = [("imputer", SimpleImputer(strategy="median"))]
    return {
        "random_forest": Pipeline(impute + [("model", RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=n_jobs,
        ))]),
        "xgboost": Pipeline(impute + [("model", XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=seed, n_jobs=n_jobs,
        ))]),
        "mlp": Pipeline(impute + [("scaler", StandardScaler()), ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001,
            batch_size=64, learning_rate_init=0.001, max_iter=1000,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            random_state=seed,
        ))]),
    }


def select_spatial_split(
    data: pd.DataFrame, test_fraction: float, seed: int, attempts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a valid group-disjoint split with both classes on both sides."""
    splitter = GroupShuffleSplit(
        n_splits=attempts, test_size=test_fraction, random_state=seed
    )
    overall_cocoa_rate = float(data["label"].mean())
    best = None
    best_score = float("inf")
    for train_index, test_index in splitter.split(
        data, data["label"], groups=data["spatial_group"]
    ):
        train_labels = data.iloc[train_index]["label"]
        test_labels = data.iloc[test_index]["label"]
        if train_labels.nunique() != 2 or test_labels.nunique() != 2:
            continue
        observed_fraction = len(test_index) / len(data)
        score = abs(observed_fraction - test_fraction) + abs(
            float(test_labels.mean()) - overall_cocoa_rate
        )
        if score < best_score:
            best = (train_index, test_index)
            best_score = score
    if best is None:
        raise RuntimeError(
            f"None of {attempts} spatial-group splits retained both classes in train "
            "and holdout sets. Reduce --spatial-block-km when preparing samples or "
            "collect cocoa samples across more spatial blocks."
        )
    return best


def main() -> int:
    args = parse_args()
    if not 0.1 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction must be between 0.1 and 0.5")
    if args.split_attempts < 1:
        raise ValueError("--split-attempts must be positive")
    samples_path = resolve(args.samples or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/training_samples.csv"))
    output_dir = resolve(args.output_dir or Path(f"models/cocoa_classification/{args.year}/{args.season}"))
    data = pd.read_csv(samples_path)
    feature_names = ANNUAL_FEATURE_NAMES if args.season == "annual" else FEATURE_NAMES
    required = set(feature_names) | {"label", "spatial_group"}
    if missing := required - set(data.columns):
        raise ValueError(f"Training data lacks fields: {sorted(missing)}")
    if data["label"].nunique() != 2 or data["spatial_group"].nunique() < 4:
        raise ValueError("Training requires both classes and at least four spatial groups")
    train_index, test_index = select_spatial_split(
        data, args.test_fraction, args.seed, args.split_attempts
    )
    train, test = data.iloc[train_index], data.iloc[test_index]
    log(
        f"Spatial split selected: train={len(train)} "
        f"({int(train['label'].sum())} cocoa), holdout={len(test)} "
        f"({int(test['label'].sum())} cocoa); no shared blocks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for name, model in build_models(args.seed, args.n_jobs).items():
        log(f"Training {name} on {len(train)} samples; spatial holdout={len(test)}")
        model.fit(train[list(feature_names)], train["label"])
        predicted = model.predict(test[list(feature_names)])
        probability = model.predict_proba(test[list(feature_names)])[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(test["label"], predicted),
            "balanced_accuracy": balanced_accuracy_score(test["label"], predicted),
            "precision_cocoa": precision_score(test["label"], predicted, zero_division=0),
            "recall_cocoa": recall_score(test["label"], predicted, zero_division=0),
            "f1_cocoa": f1_score(test["label"], predicted, zero_division=0),
            "roc_auc": roc_auc_score(test["label"], probability),
        }
        metric_rows.append(metrics)
        artifact = {
            "pipeline": model, "feature_names": list(feature_names),
            "classes": {0: "natural_tree", 1: "cocoa"}, "year": args.year,
            "season": args.season, "spatial_holdout_metrics": metrics,
        }
        joblib.dump(artifact, output_dir / f"{name}.joblib")
        importance = permutation_importance(
            model,
            test[list(feature_names)],
            test["label"],
            scoring="f1",
            n_repeats=10,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        pd.DataFrame(
            {
                "feature": feature_names,
                "permutation_importance_mean": importance.importances_mean,
                "permutation_importance_std": importance.importances_std,
            }
        ).sort_values("permutation_importance_mean", ascending=False).to_csv(
            output_dir / f"{name}_feature_importance.csv", index=False
        )
        report = {
            **metrics,
            "confusion_matrix_rows_actual_cols_predicted": confusion_matrix(test["label"], predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(test["label"], predicted, labels=[0, 1], target_names=["pseudo_natural", "cocoa"], output_dict=True, zero_division=0),
            "train_samples": len(train), "test_samples": len(test),
            "train_groups": int(train["spatial_group"].nunique()), "test_groups": int(test["spatial_group"].nunique()),
        }
        write_json_atomic(report, output_dir / f"{name}_evaluation.json")
    metrics_frame = pd.DataFrame(metric_rows).sort_values("f1_cocoa", ascending=False)
    metrics_frame.to_csv(output_dir / "model_comparison.csv", index=False)
    split = data[["source_id", "label", "spatial_group"]].copy()
    split["split"] = np.where(split.index.isin(test_index), "spatial_holdout", "train")
    split.to_csv(output_dir / "sample_split.csv", index=False)
    log(f"Saved three models and comparison metrics: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
