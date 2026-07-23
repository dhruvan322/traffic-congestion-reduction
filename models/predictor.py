"""
Congestion Prediction Module.
Trains RandomForest + XGBoost ensemble, predicts congestion level with confidence.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from config import (
    MODEL_DIR, PREDICTION_FEATURES, CONGESTION_LABELS,
    TEST_SIZE, RANDOM_STATE, DATASET_PATH,
)

logger = logging.getLogger(__name__)

# ──────────────────────────── Paths ────────────────────────────
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_congestion.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_congestion.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")


# ──────────────────────────── Feature Engineering ────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical features and return encoders."""
    df = df.copy()
    encoders = {}

    # Encode weather
    le_weather = LabelEncoder()
    df["weather_encoded"] = le_weather.fit_transform(df["weather"])
    encoders["weather"] = le_weather

    # Encode junction_id
    le_junction = LabelEncoder()
    df["junction_encoded"] = le_junction.fit_transform(df["junction_id"])
    encoders["junction"] = le_junction

    return df, encoders


def prepare_data(df: pd.DataFrame):
    """Prepare features and labels for training."""
    df, encoders = engineer_features(df)

    # Target encoding
    le_target = LabelEncoder()
    le_target.classes_ = np.array(CONGESTION_LABELS)
    y = le_target.transform(df["congestion_level"])
    encoders["target"] = le_target

    X = df[PREDICTION_FEATURES].values
    return X, y, encoders


# ──────────────────────────── Training ────────────────────────────

def train_model(df: pd.DataFrame = None) -> dict:
    """Train RandomForest + XGBoost ensemble on traffic data."""
    if df is None:
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)

    logger.info(f"Training on {len(df)} records...")
    X, y, encoders = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    joblib.dump(rf, RF_MODEL_PATH)
    results["rf_accuracy"] = round(rf_acc, 4)
    logger.info(f"RandomForest accuracy: {rf_acc:.4f}")

    # XGBoost
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
        joblib.dump(xgb, XGB_MODEL_PATH)
        results["xgb_accuracy"] = round(xgb_acc, 4)
        logger.info(f"XGBoost accuracy: {xgb_acc:.4f}")

    # Save encoders
    joblib.dump(encoders, ENCODERS_PATH)
    results["features"] = PREDICTION_FEATURES
    results["report"] = classification_report(
        y_test, rf.predict(X_test),
        target_names=CONGESTION_LABELS, output_dict=True
    )
    logger.info("Models saved successfully.")
    return results


# ──────────────────────────── Prediction ────────────────────────────

def load_models():
    """Load trained models and encoders."""
    models = {}
    if os.path.exists(RF_MODEL_PATH):
        models["rf"] = joblib.load(RF_MODEL_PATH)
    if os.path.exists(XGB_MODEL_PATH):
        models["xgb"] = joblib.load(XGB_MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}
    return models, encoders


def predict_congestion(features: dict) -> dict:
    """
    Predict congestion level for given features.
    features: dict with keys matching PREDICTION_FEATURES
    Returns: {predicted_level, confidence, probabilities, model_used}
    """
    models, encoders = load_models()
    if not models:
        # Auto-train if no models exist
        logger.info("No trained models found. Training now...")
        train_model()
        models, encoders = load_models()

    # Encode categorical inputs
    if "weather" in features and "weather" in encoders:
        weather_val = features["weather"]
        le = encoders["weather"]
        if weather_val in le.classes_:
            features["weather_encoded"] = le.transform([weather_val])[0]
        else:
            features["weather_encoded"] = 0

    if "junction_id" in features and "junction" in encoders:
        jid = features["junction_id"]
        le = encoders["junction"]
        if jid in le.classes_:
            features["junction_encoded"] = le.transform([jid])[0]
        else:
            features["junction_encoded"] = 0

    # Build feature vector
    X = np.array([[features.get(f, 0) for f in PREDICTION_FEATURES]])

    # Ensemble prediction
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        predictions[name] = pred
        probabilities[name] = proba

    # Use best model (XGBoost if available, else RandomForest)
    best_model = "xgb" if "xgb" in predictions else "rf"
    pred_label = encoders["target"].inverse_transform([predictions[best_model]])[0]
    proba = probabilities[best_model]
    confidence = float(np.max(proba))

    return {
        "predicted_level": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": {
            label: round(float(p), 4)
            for label, p in zip(CONGESTION_LABELS, proba)
        },
        "model_used": best_model.upper(),
    }


def get_feature_importance() -> dict:
    """Return feature importance from the RandomForest model."""
    if not os.path.exists(RF_MODEL_PATH):
        return {}
    rf = joblib.load(RF_MODEL_PATH)
    importance = rf.feature_importances_
    return {
        feat: round(float(imp), 4)
        for feat, imp in sorted(
            zip(PREDICTION_FEATURES, importance),
            key=lambda x: x[1], reverse=True
        )
    }
