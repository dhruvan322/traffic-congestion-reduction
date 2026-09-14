"""
Explainability Module.
Uses SHAP to explain ML model predictions.
"""

import logging
import os
import numpy as np
import joblib

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from config import MODEL_DIR, PREDICTION_FEATURES

logger = logging.getLogger(__name__)

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_congestion.joblib")


def explain_prediction(feature_values: list, feature_names: list = None) -> dict:
    """
    Explain a single prediction using SHAP TreeExplainer.
    Returns SHAP values and base value for visualization.
    """

    if not HAS_SHAP:
        logger.warning("SHAP not installed. Returning empty explanation.")
        return {"error": "SHAP not installed"}

    if not os.path.exists(RF_MODEL_PATH):
        return {"error": "Model not trained yet"}

    try:
        model = joblib.load(RF_MODEL_PATH)

        feature_names = feature_names or PREDICTION_FEATURES

        explainer = shap.TreeExplainer(model)

        X = np.array([feature_values])

        shap_values = explainer.shap_values(X)

        pred_class = int(model.predict(X)[0])

        # ---------------------------------------------------------
        # Handle SHAP output formats across versions
        # ---------------------------------------------------------

        if isinstance(shap_values, list):

            sv = np.array(shap_values[pred_class])

            while sv.ndim > 1:
                sv = sv[0]

            try:
                base = explainer.expected_value[pred_class]
            except Exception:
                base = explainer.expected_value

        else:

            sv = np.array(shap_values)

            if sv.ndim == 3:
                sv = sv[0, :, pred_class]
            elif sv.ndim == 2:
                sv = sv[0]
            elif sv.ndim > 3:
                sv = sv.flatten()

            base = explainer.expected_value

        # ---------------------------------------------------------
        # Normalize SHAP values
        # ---------------------------------------------------------

        sv = np.asarray(sv, dtype=float).flatten()

        if len(sv) > len(feature_names):
            sv = sv[: len(feature_names)]

        if len(sv) < len(feature_names):
            sv = np.pad(sv, (0, len(feature_names) - len(sv)), mode="constant")

        # ---------------------------------------------------------
        # Normalize base value
        # ---------------------------------------------------------

        base_arr = np.asarray(base)

        if base_arr.size == 0:
            base_value = 0.0
        else:
            base_value = float(base_arr.flatten()[0])

        explanation = {
            "feature_names": feature_names,
            "shap_values": [round(float(v), 4) for v in sv],
            "base_value": round(base_value, 4),
            "predicted_class": pred_class,
            "contributions": {
                name: round(float(val), 4)
                for name, val in sorted(
                    zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True
                )
            },
        }

        return explanation

    except Exception as e:
        logger.exception("SHAP explanation failed")
        return {"error": str(e)}


def get_global_explanations(
    X_sample: np.ndarray = None, max_samples: int = 200
) -> dict:
    """
    Get global SHAP feature importance across a sample of data.
    """

    if not HAS_SHAP:
        return {"error": "SHAP not installed"}

    if not os.path.exists(RF_MODEL_PATH):
        return {"error": "Model not trained yet"}

    try:
        model = joblib.load(RF_MODEL_PATH)

        if X_sample is None:
            return {"error": "No data provided for global explanation"}

        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_sample = X_sample[indices]

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):

            mean_shap = np.mean(
                [np.abs(np.asarray(sv)).mean(axis=0) for sv in shap_values], axis=0
            )

        else:

            mean_shap = np.abs(np.asarray(shap_values)).mean(axis=0)

        mean_shap = np.asarray(mean_shap).flatten()

        if len(mean_shap) > len(PREDICTION_FEATURES):
            mean_shap = mean_shap[: len(PREDICTION_FEATURES)]

        importance = {
            name: round(float(val), 4)
            for name, val in sorted(
                zip(PREDICTION_FEATURES, mean_shap), key=lambda x: x[1], reverse=True
            )
        }

        return {"global_importance": importance}

    except Exception as e:
        logger.exception("Global SHAP explanation failed")
        return {"error": str(e)}
