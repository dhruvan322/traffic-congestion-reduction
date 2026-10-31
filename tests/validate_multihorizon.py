"""
Validation report for multi-horizon congestion forecasting.

Loads the already-trained +1h, +2h, +3h and +6h models and reports:
- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- Per-class Precision / Recall / F1
- Confusion Matrix
- Majority-class baseline accuracy

This script does NOT retrain or modify any forecasting model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from xgboost import XGBClassifier

from engine.multihorizon_forecaster import (
    CONGESTION_LABELS,
    HORIZONS,
    build_horizon_target,
    chronological_split,
    load_final_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = (
    PROJECT_ROOT
    / "trained_models"
    / "multihorizon"
)


def load_model(horizon_hours: int) -> XGBClassifier:
    """Load a saved XGBoost model."""

    model_path = (
        MODEL_ROOT
        / f"{horizon_hours}h"
        / "xgb_congestion.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = XGBClassifier()
    model.load_model(model_path)

    return model


def prepare_test_data(
    df: pd.DataFrame,
    horizon_hours: int,
):
    """Recreate the exact chronological test split used in training."""

    data = df.copy()

    data["forecast_target"] = build_horizon_target(
        data,
        horizon_hours,
    )

    data = data.dropna(
        subset=["forecast_target"]
    ).copy()

    drop_columns = {
        "future_congestion_level",
        "forecast_target",
        "timestamp",
        "traffic_id",
    }

    feature_columns = [
        column
        for column in data.columns
        if column not in drop_columns
    ]

    X = data[feature_columns].copy()

    label_encoder = joblib.load(
        MODEL_ROOT
        / f"{horizon_hours}h"
        / "label_encoder.joblib"
    )

    y = label_encoder.transform(
        data["forecast_target"].astype(str)
    )

    split = chronological_split(
        X,
        pd.Series(y, index=X.index),
        data["timestamp"],
    )

    preprocessor = joblib.load(
        MODEL_ROOT
        / f"{horizon_hours}h"
        / "traffic_preprocessor.joblib"
    )

    X_test = preprocessor.transform(
        split["X_test"]
    )

    y_test = split["y_test"].to_numpy()

    return X_test, y_test, split


def validate_horizon(
    df: pd.DataFrame,
    horizon_hours: int,
) -> dict:
    """Validate one forecasting horizon."""

    model = load_model(horizon_hours)

    X_test, y_test, split = prepare_test_data(
        df,
        horizon_hours,
    )

    predictions = model.predict(
        X_test
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(CONGESTION_LABELS))),
        target_names=CONGESTION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(CONGESTION_LABELS))),
    )

    # Majority-class baseline on this test set.
    class_counts = np.bincount(
        y_test,
        minlength=len(CONGESTION_LABELS),
    )

    majority_class = int(
        np.argmax(class_counts)
    )

    majority_baseline = (
        class_counts[majority_class]
        / len(y_test)
    )

    result = {
        "horizon_hours": horizon_hours,
        "test_records": int(len(y_test)),
        "accuracy": round(
            float(accuracy),
            4,
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy),
            4,
        ),
        "macro_f1": round(
            float(macro_f1),
            4,
        ),
        "weighted_f1": round(
            float(weighted_f1),
            4,
        ),
        "majority_class_baseline_accuracy": round(
            float(majority_baseline),
            4,
        ),
        "majority_class": CONGESTION_LABELS[
            majority_class
        ],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "test_period": {
            "start": split["test_start"],
            "end": split["test_end"],
        },
    }

    return result


def print_result(result: dict) -> None:
    """Print a human-readable validation report."""

    horizon = result["horizon_hours"]

    print("\n" + "=" * 78)
    print(f"+{horizon} HOUR FORECAST VALIDATION")
    print("=" * 78)

    print(
        f"Test records: {result['test_records']:,}"
    )

    print(
        f"Test period: "
        f"{result['test_period']['start']} "
        f"-> "
        f"{result['test_period']['end']}"
    )

    print()

    print(
        f"Accuracy:           "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{result['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{result['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{result['weighted_f1']:.4f}"
    )

    print(
        f"Majority Baseline:  "
        f"{result['majority_class_baseline_accuracy']:.4f} "
        f"({result['majority_class']})"
    )

    print("\nPER-CLASS METRICS")

    report = result["classification_report"]

    header = (
        f"{'Class':<15}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print(header)
    print("-" * len(header))

    for label in CONGESTION_LABELS:
        values = report[label]

        print(
            f"{label:<15}"
            f"{values['precision']:>12.4f}"
            f"{values['recall']:>12.4f}"
            f"{values['f1-score']:>12.4f}"
            f"{int(values['support']):>12,}"
        )

    print("\nCONFUSION MATRIX")

    matrix = np.asarray(
        result["confusion_matrix"]
    )

    print(
        f"{'Actual / Pred':<15}"
        + "".join(
            f"{label[:10]:>12}"
            for label in CONGESTION_LABELS
        )
    )

    for index, label in enumerate(
        CONGESTION_LABELS
    ):
        print(
            f"{label:<15}"
            + "".join(
                f"{int(value):>12,}"
                for value in matrix[index]
            )
        )


def main() -> None:

    df = load_final_dataset()

    print(
        f"Loaded dataset: {len(df):,} records"
    )

    results = {}

    for horizon_hours in HORIZONS:
        result = validate_horizon(
            df,
            horizon_hours,
        )

        results[str(horizon_hours)] = result

        print_result(result)

    output_path = (
        MODEL_ROOT
        / "validation_report.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 78)
    print(
        f"Validation report saved to:\n{output_path}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
