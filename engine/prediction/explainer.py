"""
SHAP Explanation Module.

Explains predictions made by the final trained XGBoost model.
"""

from pathlib import Path

import joblib
import pandas as pd
import shap
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
# Cached artifacts
# -------------------------------------------------------------------

_model = None
_preprocessor = None
_label_encoder = None
_feature_columns = None
_shap_explainer = None


def _load_artifacts():
    """Load the final model and preprocessing artifacts."""

    global _model
    global _preprocessor
    global _label_encoder
    global _feature_columns
    global _shap_explainer

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"XGBoost model not found: {MODEL_PATH}")

        _model = xgb.XGBClassifier()
        _model.load_model(str(MODEL_PATH))

    if _preprocessor is None:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

        _preprocessor = joblib.load(PREPROCESSOR_PATH)

    if _label_encoder is None:
        if not LABEL_ENCODER_PATH.exists():
            raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")

        _label_encoder = joblib.load(LABEL_ENCODER_PATH)

    if _feature_columns is None:
        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                f"Feature columns not found: {FEATURE_COLUMNS_PATH}"
            )

        _feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    if _shap_explainer is None:
        _shap_explainer = shap.TreeExplainer(_model)


# -------------------------------------------------------------------
# Input preparation
# -------------------------------------------------------------------


def _prepare_input(input_data):
    """
    Prepare raw input using the exact preprocessing pipeline
    used during model training.
    """

    _load_artifacts()

    if isinstance(input_data, dict):
        data = pd.DataFrame([input_data])

    elif isinstance(input_data, pd.Series):
        data = input_data.to_frame().T

    elif isinstance(input_data, pd.DataFrame):
        data = input_data.copy()

    else:
        raise TypeError(
            "input_data must be a dictionary, pandas Series, " "or pandas DataFrame."
        )

    missing_columns = [
        column for column in _feature_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required model features: " + ", ".join(missing_columns)
        )

    data = data[_feature_columns].copy()

    if "accident_severity" in data.columns:
        data["accident_severity"] = data["accident_severity"].fillna("No Accident")

    transformed = _preprocessor.transform(data)

    return data, transformed


# -------------------------------------------------------------------
# SHAP value handling
# -------------------------------------------------------------------


def _get_shap_values(transformed_data):
    """Calculate SHAP values and normalize their output format."""

    shap_values = _shap_explainer.shap_values(transformed_data)

    return shap_values


def _select_class_shap_values(shap_values, class_index):
    """
    Extract SHAP values for one class.

    Supports the common SHAP multiclass output formats:
    - list of arrays
    - 3-dimensional numpy array
    """

    # Older SHAP versions:
    # [class_0_array, class_1_array, ...]
    if isinstance(shap_values, list):
        return shap_values[class_index][0]

    # Newer SHAP versions:
    # (samples, features, classes)
    if getattr(shap_values, "ndim", 0) == 3:
        return shap_values[0, :, class_index]

    # Binary/single-output fallback.
    if getattr(shap_values, "ndim", 0) == 2:
        return shap_values[0]

    raise ValueError(
        f"Unexpected SHAP output shape: " f"{getattr(shap_values, 'shape', None)}"
    )


# -------------------------------------------------------------------
# Public explanation function
# -------------------------------------------------------------------


def explain_prediction(input_data, top_n=15):
    """
    Explain one traffic congestion prediction.

    Parameters
    ----------
    input_data : pandas.DataFrame, pandas.Series, or dict
        Raw model features.

    top_n : int
        Number of strongest feature contributions to return.

    Returns
    -------
    dict
        Explanation containing predicted class and SHAP
        contributions.
    """

    try:
        raw_data, transformed_data = _prepare_input(input_data)

        prediction = _model.predict(transformed_data)

        probabilities = _model.predict_proba(transformed_data)

        predicted_class_id = int(prediction[0])

        predicted_class = _label_encoder.inverse_transform([predicted_class_id])[0]

        shap_values = _get_shap_values(transformed_data)

        selected_values = _select_class_shap_values(shap_values, predicted_class_id)

        feature_names = list(_preprocessor.get_feature_names_out())

        contributions = list(zip(feature_names, selected_values))

        # Sort by absolute SHAP magnitude.
        contributions.sort(key=lambda item: abs(float(item[1])), reverse=True)

        top_contributions = contributions[:top_n]

        formatted_contributions = {
            feature: float(value) for feature, value in top_contributions
        }

        return {
            "predicted_class": predicted_class,
            "predicted_class_id": predicted_class_id,
            "confidence": float(probabilities[0].max()),
            "contributions": formatted_contributions,
            "all_contributions": {
                feature: float(value) for feature, value in contributions
            },
            "feature_count": len(feature_names),
        }

    except Exception as exc:
        return {"error": str(exc)}
