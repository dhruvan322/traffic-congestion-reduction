"""
Traffic Congestion Prediction Module

Uses the final trained XGBoost model and the saved preprocessing
pipeline from the project's final ML dataset.
"""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_models"

MODEL_PATH = MODEL_DIR / "xgb_congestion_final.json"
PREPROCESSOR_PATH = MODEL_DIR / "traffic_preprocessor.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "congestion_label_encoder.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"


# -------------------------------------------------------------------
# Load final trained artifacts
# -------------------------------------------------------------------

_model = None
_preprocessor = None
_label_encoder = None
_feature_columns = None


def _load_artifacts():
    """Load the final XGBoost model and preprocessing artifacts."""

    global _model
    global _preprocessor
    global _label_encoder
    global _feature_columns

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Final XGBoost model not found: {MODEL_PATH}")

        _model = xgb.XGBClassifier()
        _model.load_model(str(MODEL_PATH))

    if _preprocessor is None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Traffic preprocessor not found: {PREPROCESSOR_PATH}"
            )

        _preprocessor = joblib.load(PREPROCESSOR_PATH)

    if _label_encoder is None:
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(
                f"Congestion label encoder not found: {LABEL_ENCODER_PATH}"
            )

        _label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if _feature_columns is None:
        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                f"Feature column definition not found: {FEATURE_COLUMNS_PATH}"
            )

        _feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


# -------------------------------------------------------------------
# Public helpers
# -------------------------------------------------------------------


def get_feature_columns():
    """Return the 28 raw feature columns expected by the model."""

    _load_artifacts()
    return list(_feature_columns)


def get_class_names():
    """Return the five congestion class names."""

    _load_artifacts()
    return list(_label_encoder.classes_)


def predict_congestion(input_data):
    """
    Predict future congestion from the 28 raw model features.

    Parameters
    ----------
    input_data : pandas.DataFrame or dict
        Raw feature values expected by the saved preprocessing pipeline.

    Returns
    -------
    dict
        Prediction result containing:
        - predicted_class
        - predicted_class_id
        - probabilities
        - confidence
    """

    _load_artifacts()

    # Convert a dictionary into a one-row DataFrame.
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    elif not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame or a dictionary.")

    data = input_data.copy()

    # Check that all required raw features are present.
    missing_columns = [
        column for column in _feature_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required model features: " + ", ".join(missing_columns)
        )

    # Keep exactly the feature order used during training.
    data = data[_feature_columns].copy()

    # No accident -> explicit categorical value.
    if "accident_severity" in data.columns:
        data["accident_severity"] = data["accident_severity"].fillna("No Accident")

    # Apply the exact preprocessing pipeline used during training.
    transformed_data = _preprocessor.transform(data)

    # Generate prediction and probabilities.
    prediction = _model.predict(transformed_data)
    probabilities = _model.predict_proba(transformed_data)

    # Convert encoded class back to the original congestion label.
    predicted_class_id = int(prediction[0])
    predicted_class = _label_encoder.inverse_transform([predicted_class_id])[0]

    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(_label_encoder.classes_, probabilities[0])
    }

    confidence = float(probabilities[0].max())

    return {
        "predicted_class": predicted_class,
        "predicted_class_id": predicted_class_id,
        "probabilities": class_probabilities,
        "confidence": confidence,
    }


def predict_from_row(row):
    """
    Convenience function for predicting directly from a pandas Series.
    """

    if isinstance(row, pd.Series):
        row = row.to_frame().T

    return predict_congestion(row)


# -------------------------------------------------------------------
# Compatibility helpers
# -------------------------------------------------------------------


def get_feature_importance():
    """
    Return native XGBoost feature importance.

    These are model-level importance values for the transformed
    features used by XGBoost.
    """

    _load_artifacts()

    feature_names = list(_preprocessor.get_feature_names_out())

    importance_values = _model.feature_importances_

    return sorted(
        zip(feature_names, importance_values),
        key=lambda item: item[1],
        reverse=True,
    )


def train_model(*args, **kwargs):
    """
    Training is intentionally disabled in the application.

    The final model is trained and validated offline in the ML
    training notebook and loaded here for inference.
    """

    raise RuntimeError(
        "Model training is disabled in the application. "
        "Use the ML training notebook to retrain the final model."
    )
