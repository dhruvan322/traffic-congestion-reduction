"""
Prediction Panel.
Interactive congestion prediction with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from config import DATASET_PATH, PEAK_HOURS, WEATHER_CONDITIONS, CONGESTION_LABELS
from models.predictor import predict_congestion, train_model, get_feature_importance
from models.explainer import explain_prediction
from database import insert_prediction
from datetime import datetime
from engine.reduction_engine import generate_mitigation_actions


def render_prediction_panel():
    """Render the congestion prediction interface."""
    st.markdown("# 🔮 Congestion Prediction Engine")
    st.markdown("Predict traffic congestion using AI (RandomForest + XGBoost ensemble)")

    df = pd.read_csv(DATASET_PATH)
    junctions = sorted(df["junction_id"].unique())

    # ─────────── Model Training ───────────
    with st.expander("🧠 Model Training", expanded=False):
        if st.button("🚀 Train / Retrain Models", type="primary"):
            with st.spinner("Training models... This may take a minute."):
                results = train_model(df)
            st.success("✅ Models trained successfully!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RandomForest Accuracy", f"{results['rf_accuracy']:.2%}")
            with col2:
                if "xgb_accuracy" in results:
                    st.metric("XGBoost Accuracy", f"{results['xgb_accuracy']:.2%}")

    st.divider()

    # ─────────── Prediction Input ───────────
    st.subheader("📝 Enter Traffic Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        junction = st.selectbox("🏢 Junction", junctions)
        hour = st.slider("🕐 Hour of Day", 0, 23, 9)
    with col2:
        vehicle_count = st.number_input("🚗 Vehicle Count", 10, 600, 200)
        day_of_week = st.selectbox("📅 Day", list(range(7)),
                                     format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    with col3:
        avg_speed = st.number_input("⚡ Avg Speed (km/h)", 5.0, 80.0, 35.0, step=1.0)
        weather = st.selectbox("🌤️ Weather", WEATHER_CONDITIONS)

    is_peak = int(hour in PEAK_HOURS)
    is_event = st.checkbox("🎉 Special Event", value=False)
    is_construction = st.checkbox("🚧 Construction Zone", value=False)

    # ─────────── Predict ───────────
    if st.button("🔮 Predict Congestion", type="primary"):
        features = {
            "junction_id": junction,
            "hour": hour,
            "day_of_week": day_of_week,
            "vehicle_count": vehicle_count,
            "avg_speed": avg_speed,
            "is_peak": is_peak,
            "is_event": int(is_event),
            "is_construction": int(is_construction),
            "weather": weather,
        }

        with st.spinner("Predicting..."):
            result = predict_congestion(features)

        # Display result
        level = result["predicted_level"]
        confidence = result["confidence"]
        color_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

        st.markdown("---")
        st.markdown(f"### {color_map.get(level, '⚪')} Predicted: **{level}** Congestion")
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        # Probability bars
        st.subheader("📊 Class Probabilities")
        probs = result["probabilities"]
        fig = go.Figure()
        colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#f44336"}
        for label in CONGESTION_LABELS:
            fig.add_trace(go.Bar(
                x=[label], y=[probs[label]],
                marker_color=colors[label],
                text=[f"{probs[label]:.1%}"],
                textposition="outside",
                name=label,
            ))
        fig.update_layout(
            yaxis_title="Probability", height=300,
            showlegend=False, template="plotly_dark",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Store prediction
        insert_prediction({
            "junction_id": junction,
            "predicted_level": level,
            "confidence": confidence,
            "probabilities": probs,
            "model_used": result["model_used"],
            "timestamp": datetime.now().isoformat(),
        })

        # ─────────── AI Mitigation Actions ───────────
        mitigation = generate_mitigation_actions(
            junction_id=junction,
            predicted_level=level,
            confidence=confidence,
            vehicle_count=vehicle_count,
            avg_speed=avg_speed,
            time_window="next 1-2 hours"
        )

        st.markdown("---")
        st.subheader("🎯 Suggested Mitigation Actions")

        st.info(mitigation["summary"])

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                "Estimated Congestion Reduction",
                f"{mitigation['estimated_total_reduction']}%"
            )

        with col_b:
            st.metric(
                "Recommended Actions",
                len(mitigation["actions"])
            )

        for action in mitigation["actions"]:

            priority_color = {
                "Critical": "🔴",
                "High": "🟠",
                "Medium": "🟡",
                "Low": "🟢"
            }

            icon = priority_color.get(action["priority"], "⚪")

            with st.expander(
                f"{icon} {action['action']} "
                f"({action['expected_reduction']}% reduction)"
            ):

                st.markdown(
                    f"**Priority:** {action['priority']}"
                )

                st.markdown(
                    f"**Expected Reduction:** "
                    f"{action['expected_reduction']}%"
                )

                st.markdown(
                    f"**Confidence:** "
                    f"{action['confidence']:.0%}"
                )

                st.markdown(
                    f"**Category:** "
                    f"{action['category']}"
                )

                st.markdown(
                    f"**Why?** {action['explanation']}"
                )


        # SHAP Explanation
        with st.expander("🧪 AI Explanation (SHAP)", expanded=True):
            feature_values = [
                features.get(f, 0) for f in [
                    "hour", "day_of_week", "vehicle_count", "avg_speed",
                    "is_peak", "is_event", "is_construction",
                ]
            ]
            # Add encoded values
            feature_values.extend([0, 0])  # weather_encoded, junction_encoded placeholder

            explanation = explain_prediction(feature_values)
            if "error" in explanation:
                st.info(f"Explanation unavailable: {explanation['error']}")
            else:
                st.markdown("**Feature Contributions:**")
                contribs = explanation.get("contributions", {})
                for feat, val in contribs.items():
                    direction = "↑" if val > 0 else "↓"
                    color = "red" if val > 0 else "green"
                    st.markdown(
                        f"- `{feat}`: **{direction} {abs(val):.3f}**"
                    )

    # ─────────── Feature Importance ───────────
    st.divider()
    with st.expander("📊 Feature Importance", expanded=False):
        importance = get_feature_importance()
        if importance:
            fig = go.Figure(go.Bar(
                x=list(importance.values()),
                y=list(importance.keys()),
                orientation="h",
                marker_color="#667eea",
            ))
            fig.update_layout(
                xaxis_title="Importance",
                height=350, template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train a model first to see feature importance.")
