import streamlit as st


def render_command_center(data_context):

    st.title("🚨 Executive Traffic Command Center")

    st.markdown("---")

    df = data_context["traffic_data"]

    total_vehicles = int(df["vehicle_count"].sum())
    avg_speed = round(df["avg_speed"].mean(), 1)

    hotspots = data_context.get("hotspots", [])
    anomalies = data_context.get("anomalies", [])

    col1, col2, col3 = st.columns(3)

    col1.metric("🚗 Vehicles", f"{total_vehicles:,}")

    col2.metric("⚡ Avg Speed", f"{avg_speed} km/h")

    col3.metric("⚠️ Anomalies", len(anomalies))

    st.markdown("---")

    st.subheader("🔥 Top Hotspots")

    for hotspot in hotspots[:5]:

        st.warning(f"Cluster {hotspot['cluster_id']} " f"({hotspot['severity']})")

    st.markdown("---")

    st.subheader("🚔 Operational Priorities")

    shown = set()

    for anomaly in anomalies:

        if anomaly["risk_level"] not in ["High", "Critical"]:
            continue

        location = anomaly["junction_id"]

        if location in shown:
            continue

        shown.add(location)

        st.error(f"{location} | " f"Risk: {anomaly['risk_level']}")

        if len(shown) >= 5:
            break
