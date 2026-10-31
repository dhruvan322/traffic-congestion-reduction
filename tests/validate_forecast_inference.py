"""Regression checks for multi-horizon traffic forecasting."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.multihorizon_predictor import (
    get_latest_row_for_segment,
    forecast_all_horizons,
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "final_dataset"
    / "ml_traffic_dataset.csv"
)

EXPECTED_HORIZONS = {"1h", "2h", "3h", "6h"}

EXPECTED_CLASSES = {
    "Free Flow",
    "Low",
    "Moderate",
    "High",
    "Severe",
}


def main():
    """Validate every road segment using its latest traffic record."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    segments = sorted(
        df["road_segment_id"].dropna().unique().tolist()
    )

    errors = []
    checked = 0

    for segment in segments:
        try:
            row = get_latest_row_for_segment(df, segment)

            if row is None:
                raise ValueError("No latest traffic record returned.")

            payload = forecast_all_horizons(row)
            forecasts = payload["forecasts"]

            if set(forecasts.keys()) != EXPECTED_HORIZONS:
                raise ValueError(
                    f"Unexpected horizons: {list(forecasts.keys())}"
                )

            for horizon in sorted(EXPECTED_HORIZONS):
                result = forecasts[horizon]
                probabilities = result["probabilities"]

                if set(probabilities.keys()) != EXPECTED_CLASSES:
                    raise ValueError(
                        f"{horizon}: unexpected probability classes."
                    )

                values = list(probabilities.values())

                if any(
                    not isinstance(value, (int, float))
                    or not 0.0 <= value <= 1.0
                    for value in values
                ):
                    raise ValueError(
                        f"{horizon}: invalid probability values."
                    )

                probability_sum = sum(values)

                if abs(probability_sum - 1.0) > 0.005:
                    raise ValueError(
                        f"{horizon}: probabilities sum to "
                        f"{probability_sum:.4f}, not approximately 1."
                    )

                if result["predicted_class"] not in EXPECTED_CLASSES:
                    raise ValueError(
                        f"{horizon}: invalid predicted class."
                    )

            checked += 1

        except Exception as exc:
            errors.append((segment, str(exc)))

    print(f"ROAD SEGMENTS: {len(segments)}")
    print(f"SUCCESSFULLY VALIDATED: {checked}")
    print(f"FAILED: {len(errors)}")
    print(f"EXPECTED HORIZONS: {sorted(EXPECTED_HORIZONS)}")

    if errors:
        for segment, error in errors[:10]:
            print(f"FAILURE {segment}: {error}")

        raise SystemExit(1)

    print("RESULT: ALL FORECAST VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
