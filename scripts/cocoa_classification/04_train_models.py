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

from common import FEATURE_NAMES, log, resolve, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF, XGBoost, and MLP cocoa classifiers.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--season", choices=("djf", "amj"), default="djf")
    parser.add_argument("--samples", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def build_models(seed: int, n_jobs: int) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError("XGBoost is required; run: python -m pip install -r requirements.txt") from error
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


def main() -> int:
    args = parse_args()
    if not 0.1 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction must be between 0.1 and 0.5")
    samples_path = resolve(args.samples or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/training_samples.csv"))
    output_dir = resolve(args.output_dir or Path(f"models/cocoa_classification/{args.year}/{args.season}"))
    data = pd.read_csv(samples_path)
    required = set(FEATURE_NAMES) | {"label", "spatial_group"}
    if missing := required - set(data.columns):
        raise ValueError(f"Training data lacks fields: {sorted(missing)}")
    if data["label"].nunique() != 2 or data["spatial_group"].nunique() < 4:
        raise ValueError("Training requires both classes and at least four spatial groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_fraction, random_state=args.seed)
    train_index, test_index = next(splitter.split(data, data["label"], groups=data["spatial_group"]))
    train, test = data.iloc[train_index], data.iloc[test_index]
    if train["label"].nunique() != 2 or test["label"].nunique() != 2:
        raise RuntimeError("Spatial split did not retain both classes; change --seed or spatial block size")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for name, model in build_models(args.seed, args.n_jobs).items():
        log(f"Training {name} on {len(train)} samples; spatial holdout={len(test)}")
        model.fit(train[list(FEATURE_NAMES)], train["label"])
        predicted = model.predict(test[list(FEATURE_NAMES)])
        probability = model.predict_proba(test[list(FEATURE_NAMES)])[:, 1]
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
            "pipeline": model, "feature_names": list(FEATURE_NAMES),
            "classes": {0: "natural_tree", 1: "cocoa"}, "year": args.year,
            "season": args.season, "spatial_holdout_metrics": metrics,
        }
        joblib.dump(artifact, output_dir / f"{name}.joblib")
        importance = permutation_importance(
            model,
            test[list(FEATURE_NAMES)],
            test["label"],
            scoring="f1",
            n_repeats=10,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
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
