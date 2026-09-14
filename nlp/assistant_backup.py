"""
NLP Assistant Module.
Intent classification, entity extraction, and response generation.
"""

import re
import logging
import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHAT_CONTEXT_WINDOW
from engine.reduction_engine import (
    generate_mitigation_actions,
    generate_anomaly_response,
)
from engine.route_optimizer import optimize_route
from config import DATASET_PATH
from models.hotspot_detector import get_hotspot_summary
from engine.prediction.predictor import predict_congestion
from config import PEAK_HOURS
from data.generate_dataset import JUNCTION_NAMES
from models.anomaly_detector import get_anomaly_report
from database import insert_chat_turn, get_chat_history, get_next_turn_number
from engine.traffic_reasoner import generate_operational_summary
from engine.management_planner import generate_management_plan
from engine.executive_report import generate_executive_report

logger = logging.getLogger(__name__)

# ──────────────────────────── Intent Patterns ────────────────────────────

INTENT_PATTERNS = {
    "reduction_query": [
        r"improve.*traffic",
        r"improve.*congestion",
        r"better.*traffic",
        r"optimize.*traffic",
        r"optimize.*flow",
        r"how.*improve",
        r"how.*reduce",
        r"(?:reduce|improve|solve|mitigate|fix)\s+(?:traffic|congestion)",
        r"how\s+can.*(?:traffic|congestion).*reduced",
        r"suggest\s+(?:actions|recommendations|solutions)",
    ],
    "management_plan_query": [
        r"generate.*plan",
        r"traffic management plan",
        r"action plan",
        r"mitigation strategy",
        r"what should traffic police do",
        r"what should authorities do",
    ],
    "explanation_query": [
        r"why.*route",
        r"why.*recommended",
        r"why.*prediction",
        r"why.*congestion",
        r"why.*traffic.*bad",
        r"why.*busy",
        r"why.*jam",
        r"why.*congestion",
        r"reason.*traffic",
        r"reason.*congestion",
        r"explain.*route",
        r"explain.*prediction",
        r"explain.*congestion",
    ],
    "prediction_query": [
        r"predict.*traffic",
        r"predict.*congestion",
        r"forecast.*traffic",
        r"forecast.*congestion",
        r"future.*traffic",
        r"future.*congestion",
        r"expected.*traffic",
        r"expected.*congestion",
        r"what.*traffic.*expected",
        r"will.*traffic",
        r"will.*congestion",
        r"(?:predict|forecast|will|expect|future)\s*(?:traffic|congestion)?",
    ],
    "anomaly_query": [
        r"anomaly",
        r"anomalies",
        r"unusual.*traffic",
        r"abnormal.*traffic",
        r"strange.*traffic",
        r"weird.*traffic",
        r"odd.*traffic",
        r"traffic.*anomaly",
        r"traffic.*anomalies",
        r"unusual.*pattern",
        r"abnormal.*pattern",
        r"traffic.*pattern",
        r"unusual.*condition",
        r"abnormal.*condition",
        r"traffic.*condition",
        r"show.*anomal",
        r"detect.*anomal",
        r"any.*anomaly",
        r"any.*anomalies",
        r"are.*there.*anomal",
        r"anything.*unusual",
        r"(?:anomal|unusual|abnormal|strange|weird|odd)\s*(?:traffic)?",
    ],
    "hotspot_query": [
        r"hotspot",
        r"hot\s*spot",
        r"hotspots",
        r"most\s+congested",
        r"worst\s+congestion",
        r"worst\s+traffic",
        r"traffic\s+hotspots",
        r"congestion\s+hotspots",
        r"show\s+hotspots",
        r"show\s+traffic\s+hotspots",
        r"show\s+congestion\s+clusters",
        r"congestion\s+clusters",
        r"which\s+areas\s+are\s+most\s+congested",
        r"which\s+locations\s+are\s+most\s+congested",
        r"where\s+is\s+traffic\s+worst",
        r"where\s+is\s+congestion\s+worst",
        r"highest\s+congestion",
        r"heaviest\s+traffic",
        r"which\s+areas\s+need\s+attention",
        r"traffic\s+blackspots",
        r"(?:which|where|what)\s+(?:junctions?|areas?|places?)\s+(?:are|is)\s+(?:most|worst)",
    ],
    "executive_report_query": [
        r"executive\s+traffic\s+report",
        r"executive\s+report",
        r"traffic\s+report",
        r"city\s+report",
        r"network\s+report",
        r"daily\s+traffic\s+report",
        r"generate\s+report",
        r"traffic\s+briefing",
        r"citywide\s+report",
    ],
    "city_summary_query": [
        r"traffic\s+summary",
        r"citywide\s+traffic",
        r"traffic\s+overview",
        r"traffic\s+report",
        r"current\s+traffic\s+situation",
        r"overall\s+traffic",
        r"summarize\s+traffic",
        r"summary\s+of\s+traffic",
        r"what\s+is\s+the\s+traffic\s+situation",
        r"what\s+is\s+happening\s+across\s+the\s+city",
    ],
    "operational_query": [
        r"traffic\s+police",
        r"focus\s+on\s+today",
        r"top\s+traffic\s+risks",
        r"traffic\s+risks",
        r"priority\s+locations",
        r"where\s+should\s+officers",
        r"where\s+should\s+traffic\s+police",
        r"where\s+should\s+police",
        r"operational\s+recommendations",
        r"what\s+should\s+traffic\s+police",
        r"what\s+should\s+authorities",
        r"areas\s+requiring\s+attention",
        r"critical\s+traffic\s+areas",
        r"deployment\s+recommendations",
    ],
    "congestion_check": [
        r"(?:traffic|congestion|jam|crowded|busy)\s*(?:at|in|on|near|for)?\s*(.+)?",
        r"(?:how|what).*(?:traffic|congestion|busy)",
        r"is\s+(.+)\s+(?:busy|crowded|congested)",
        # More natural traffic phrasing to catch casual user queries
        r"current\s+situation",
        r"traffic\s+situation",
        r"current\s+status",
        r"status\s+at",
        r"what\s+is\s+happening",
        r"how\s+is\s+traffic",
        r"what.*traffic",
        r"how.*traffic",
        r"traffic.*at",
        r"traffic.*near",
        r"situation.*at",
        r"situation.*near",
        r"conditions.*at",
        r"conditions.*near",
        r"status.*at",
        r"status.*near",
    ],
    "intervention_query": [
        r"needs.*intervention",
        r"highest.*risk",
        r"most.*critical",
        r"which.*area.*attention",
        r"which.*junction.*attention",
        r"where.*should.*focus",
    ],
    "route_query": [
        r"best\s+route",
        r"fastest\s+route",
        r"shortest\s+route",
        r"optimal\s+route",
        r"route\s+from",
        r"path\s+from",
        r"from\s+.+\s+to\s+.+",
        r"how\s+to\s+get\s+from",
        r"how\s+to\s+reach",
        r"travel\s+from",
        r"navigate\s+from",
        r"(?:route|path|way|direction|navigate|go)\s*(?:from|between)\s+(.+)\s+(?:to)\s+(.+)",
        r"(?:shortest|fastest|best|optimal)\s*(?:route|path|way)",
        r"how\s+(?:to\s+)?(?:get|go|reach|travel)\s+(?:from|to)\s+(.+)",
    ],
    "general_info": [
        r"(?:hello|hi|hey|help|what\s+can\s+you|tell\s+me\s+about)",
        r"(?:status|overview|summary|report|stats|statistics)",
    ],
}

# ──────────────────────────── Junction Extraction ────────────────────────────

JUNCTION_ALIASES = {
    "mg road": "J001",
    "ameerpet": "J002",
    "begumpet": "J003",
    "secunderabad": "J004",
    "hitec city": "J005",
    "hitech city": "J005",
    "madhapur": "J006",
    "gachibowli": "J007",
    "jubilee hills": "J008",
    "banjara hills": "J009",
    "tank bund": "J010",
    "mehdipatnam": "J011",
    "lb nagar": "J012",
    "uppal": "J013",
    "kukatpally": "J014",
    "miyapur": "J015",
    "charminar": "J016",
    "abids": "J017",
    "nampally": "J018",
    "paradise": "J019",
    "dilsukhnagar": "J020",
}

JUNCTION_NAMES = {jid: alias.title() for alias, jid in JUNCTION_ALIASES.items()}

# Peak traffic hours (9-11 AM, 6-9 PM)
PEAK_HOURS = [9, 10, 18, 19, 20]


def classify_intent(query: str) -> str:
    """Classify user query into an intent category."""
    query_lower = query.lower().strip()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return intent

    return "general_info"


def extract_entities(query: str) -> dict:
    """Extract traffic locations, route endpoints, time, and weather."""

    query_lower = query.lower().strip()
    entities = {}

    # ---------------------------------------------------------
    # Extract explicit road-segment IDs
    # Example: RS0001, RS0031
    # ---------------------------------------------------------
    segment_matches = re.findall(r"\brs\d{4}\b", query_lower)

    if segment_matches:
        entities["road_segment_ids"] = [segment.upper() for segment in segment_matches]

    # ---------------------------------------------------------
    # Final-dataset area names
    # ---------------------------------------------------------
    area_aliases = {
        "hitec city": "Hitech City",
        "hitech city": "Hitech City",
        "hitec": "Hitech City",
        "hitech": "Hitech City",
        "gachibowli": "Gachibowli",
        "ameerpet": "Ameerpet",
        "begumpet": "Begumpet",
        "secunderabad": "Secunderabad",
        "madhapur": "Madhapur",
        "kondapur": "Kondapur",
        "kukatpally": "Kukatpally",
        "miyapur": "Miyapur",
        "punjagutta": "Punjagutta",
        "paradise": "Paradise",
        "lb nagar": "LB Nagar",
        "lb. nagar": "LB Nagar",
        "dilsukhnagar": "Dilsukhnagar",
        "dilsukh nagar": "Dilsukhnagar",
        "mehdipatnam": "Mehdipatnam",
        "jubilee hills": "Jubilee Hills",
        "banjara hills": "Banjara Hills",
        "uppal": "Uppal",
        "nagole": "Nagole",
        "abids": "Abids",
        "charminar": "Charminar",
    }

    detected_locations = []

    # Longest aliases first so multi-word locations are handled cleanly.
    for alias in sorted(area_aliases, key=len, reverse=True):
        if alias in query_lower:
            canonical_name = area_aliases[alias]

            if canonical_name not in detected_locations:
                detected_locations.append(canonical_name)

    if detected_locations:
        entities["locations"] = detected_locations
        entities["location_names"] = detected_locations.copy()

    # ---------------------------------------------------------
    # Explicit legacy J001-J020 IDs
    # Keep support for backward compatibility.
    # ---------------------------------------------------------
    jid_matches = re.findall(r"\bj\d{3}\b", query_lower)

    if jid_matches:
        entities["junction_ids"] = [j.upper() for j in jid_matches]

    # ---------------------------------------------------------
    # Route origin / destination
    # ---------------------------------------------------------
    route_match = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)", query_lower)

    if route_match:
        origin_text = route_match.group(1).strip()
        destination_text = route_match.group(2).strip()

        # Remove common trailing punctuation.
        origin_text = origin_text.rstrip(".,?!")
        destination_text = destination_text.rstrip(".,?!")

        origin = None
        destination = None

        # Match aliases against the complete endpoint text.
        for alias, canonical_name in area_aliases.items():
            if origin_text == alias:
                origin = canonical_name
                break

        for alias, canonical_name in area_aliases.items():
            if destination_text == alias:
                destination = canonical_name
                break

        if origin:
            entities["origin_name"] = origin
            entities["origin"] = origin

        if destination:
            entities["destination_name"] = destination
            entities["destination"] = destination

    # ---------------------------------------------------------
    # Time references
    # ---------------------------------------------------------
    time_matches = re.findall(r"(\d{1,2})\s*(?:am|pm|:00|:30)", query_lower)

    if time_matches:
        entities["hours"] = [int(h) for h in time_matches]

    # ---------------------------------------------------------
    # Weather
    # ---------------------------------------------------------
    for weather in ["rain", "fog", "clear", "cloudy"]:
        if weather in query_lower:
            entities["weather"] = weather.capitalize()
            break

    return entities


def resolve_area_segments(location, data_context):
    """
    Resolve a natural-language location to road segments
    in the final traffic dataset.
    """

    if "traffic_data" not in data_context:
        return []

    df = data_context["traffic_data"]

    if "area" not in df.columns or "road_segment_id" not in df.columns:
        return []

    if not location:
        return []

    normalized = " ".join(str(location).lower().strip().split())

    aliases = {
        "hitec city": "hitech city",
        "hitech city": "hitech city",
        "hitec": "hitech city",
        "hitech": "hitech city",
        "lb. nagar": "lb nagar",
        "dilsukh nagar": "dilsukhnagar",
    }

    normalized = aliases.get(normalized, normalized)

    areas = df["area"].dropna().astype(str).str.strip()

    matches = areas.str.lower() == normalized

    return (
        df.loc[matches, "road_segment_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def build_traffic_context(location, data_context):
    """
    Build traffic context from the final road-segment dataset.
    """

    if "traffic_data" not in data_context:
        return None

    df = data_context["traffic_data"]

    segment_ids = resolve_area_segments(location, data_context)

    if not segment_ids:
        return None

    location_data = df[df["road_segment_id"].astype(str).isin(segment_ids)].copy()

    if location_data.empty:
        return None

    latest_timestamp = location_data["timestamp"].max()

    latest_rows = location_data[location_data["timestamp"] == latest_timestamp]

    if latest_rows.empty:
        return None

    vehicle_count = latest_rows["vehicle_count"].mean()
    average_speed = latest_rows["average_speed_kmh"].mean()

    congestion_values = latest_rows["future_congestion_level"].dropna().astype(str)

    if congestion_values.empty:
        congestion_level = "Unknown"
    else:
        severity_order = {
            "Free Flow": 0,
            "Low": 1,
            "Moderate": 2,
            "High": 3,
            "Severe": 4,
        }

        congestion_level = max(
            congestion_values.tolist(), key=lambda x: severity_order.get(x, -1)
        )

    return {
        "location": location,
        "road_segment_ids": segment_ids,
        "vehicle_count": int(round(vehicle_count)),
        "avg_speed": float(average_speed),
        "average_speed_kmh": float(average_speed),
        "congestion_level": congestion_level,
        "future_congestion_level": congestion_level,
        "timestamp": latest_timestamp,
    }


def build_prediction_context(junction_id, data_context):

    context = build_traffic_context(junction_id, data_context)

    if not context:
        return None

    from datetime import datetime

    now = datetime.now()

    return {
        "junction_id": junction_id,
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "vehicle_count": context["vehicle_count"],
        "avg_speed": context["avg_speed"],
        "is_peak": int(now.hour in PEAK_HOURS),
        "is_event": 0,
        "is_construction": 0,
        "weather": "Clear",
    }


def get_intervention_priorities(data_context, top_n=3):
    """
    Identify junctions requiring immediate intervention.
    """

    if "traffic_data" not in data_context:
        return []

    df = data_context["traffic_data"].copy()

    latest_rows = df.sort_values("timestamp").groupby("junction_id").tail(1)

    def calculate_risk(row):

        risk = 0

        if row["congestion_level"] == "High":
            risk += 100

        elif row["congestion_level"] == "Medium":
            risk += 50

        risk += row["vehicle_count"] * 0.15

        risk += max(0, 60 - row["avg_speed"])

        return risk

    latest_rows["risk_score"] = latest_rows.apply(calculate_risk, axis=1)

    ranked = latest_rows.sort_values("risk_score", ascending=False)

    return ranked.head(top_n)


def build_anomaly_context(data_context, top_n=3):
    """
    Retrieve highest-risk anomalies.
    """

    if "traffic_data" not in data_context:
        return []

    anomalies = get_anomaly_report(data_context["traffic_data"])

    return anomalies[:top_n]


def build_hotspot_context(data_context, top_n=3):
    """
    Retrieve hotspot cluster summaries.
    """

    if "traffic_data" not in data_context:
        return []

    hotspots = get_hotspot_summary(data_context["traffic_data"])

    return hotspots[:top_n]


def generate_response(intent: str, entities: dict, data_context: dict = None) -> str:
    """Generate a response based on intent, entities, and data context."""
    data_context = data_context or {}
    locations = entities.get("locations", [])

    if intent == "congestion_check":
        if locations and "traffic_data" in data_context:
            responses = []

            for location in locations:
                context = build_traffic_context(location, data_context)

                if context:
                    responses.append(
                        f"**{context['location']}**: "
                        f"{context['congestion_level']} congestion "
                        f"({context['vehicle_count']} vehicles, "
                        f"{context['average_speed_kmh']:.1f} km/h)"
                    )
                else:
                    responses.append(f"**{location}**: No traffic data available")

            if responses:
                return "🚦 **Traffic Status:**\n" + "\n".join(responses)

        return (
            "🚦 I can check traffic for specific areas. "
            "Try asking: *'What is the traffic at HITEC City?'* "
            "or *'Is Ameerpet congested?'*"
        )

    elif intent == "reduction_query":
        if not locations:
            return (
                "Please specify a location such as "
                "'How can congestion at HITEC City be reduced?'"
            )

        location = locations[0]
        context = build_traffic_context(location, data_context)

        if not context or "traffic_data" not in data_context:
            return "Traffic data unavailable."

        df = data_context["traffic_data"]
        area_data = df[
            df["road_segment_id"].astype(str).isin(context["road_segment_ids"])
        ].sort_values("timestamp")

        if area_data.empty:
            return "Traffic data unavailable."

        prediction_row = area_data.iloc[-1]

        from engine.prediction.predictor import predict_from_row

        prediction_result = predict_from_row(prediction_row)
        prediction = {
            "congestion_level": prediction_result["predicted_class"],
            "confidence": prediction_result["confidence"],
            "road_segment_id": prediction_row["road_segment_id"],
        }

        hotspots = get_hotspot_summary(df)
        hotspot_context = {
            "detected": False,
            "severity": "None",
            "cluster_id": "Unknown",
            "road_segments": [],
        }

        for hotspot in hotspots:
            hotspot_segments = hotspot.get("road_segments", [])
            if any(
                segment in context["road_segment_ids"]
                for segment in hotspot_segments
            ):
                hotspot_context = {
                    "detected": True,
                    "severity": hotspot.get("severity", "Medium"),
                    "cluster_id": hotspot.get("cluster_id", "Unknown"),
                    "road_segments": hotspot_segments,
                }
                break

        anomalies = get_anomaly_report(df)
        anomaly_context = {
            "detected": False,
            "risk_level": "Low",
            "road_segment_id": "Unknown",
        }

        for anomaly in anomalies:
            anomaly_segment = str(anomaly.get("road_segment_id", ""))
            if anomaly_segment in context["road_segment_ids"]:
                anomaly_context = {
                    "detected": True,
                    "risk_level": anomaly.get("risk_level", "Low"),
                    "road_segment_id": anomaly_segment,
                }
                break

        route_context = {
            "alternative_available": False,
            "shortest_distance_km": 0.0,
            "shortest_time_minutes": 0.0,
            "traffic_aware_distance_km": 0.0,
            "traffic_aware_time_minutes": 0.0,
        }

        from engine.reduction_engine import generate_integrated_recommendations

        decision = generate_integrated_recommendations(
            prediction=prediction,
            hotspot=hotspot_context,
            anomaly=anomaly_context,
            route_conditions=route_context,
        )

        confidence_percent = prediction_result["confidence"] * 100
        response = (
            f"🚦 **Congestion Reduction Plan for {location}**\n\n"
            f"**Current Traffic Context**\n"
            f"• Predicted congestion: {prediction_result['predicted_class']}\n"
            f"• Prediction confidence: {confidence_percent:.1f}%\n"
            f"• Vehicles: {context['vehicle_count']}\n"
            f"• Average speed: {context['average_speed_kmh']:.1f} km/h\n"
            f"• Road segment used for prediction: {prediction_row['road_segment_id']}\n\n"
            f"**Operational Priority:** {decision['overall_priority']}\n\n"
            f"**Recommended Actions**\n"
        )

        for index, recommendation in enumerate(decision["recommendations"][:4], start=1):
            response += (
                f"{index}. **{recommendation['category']}** — "
                f"{recommendation['action']}\n"
            )

        response += "\n**WHY these actions?**\n"
        for recommendation in decision["recommendations"][:4]:
            response += f"• {recommendation['reason']}\n"

        return response

    elif intent == "management_plan_query":
        junctions = entities.get("junction_ids", [])

        if not junctions:
            return (
                "Please specify a location such as "
                "'Generate a traffic management plan for HITEC City'."
            )

        context = build_traffic_context(junctions[0], data_context)
        if not context:
            return "Traffic data unavailable."

        actions = generate_mitigation_actions(
            context["junction_id"],
            context["congestion_level"],
            confidence=0.80,
            vehicle_count=context["vehicle_count"],
            avg_speed=context["avg_speed"],
        )

        junction_name = entities.get("junction_names", ["Target Area"])[0]
        response = (
            f"📋 Traffic Management Plan for {junction_name}\n\n"
            f"Current Situation:\n"
            f"• Congestion Level: {context['congestion_level']}\n"
            f"• Vehicle Count: {context['vehicle_count']}\n"
            f"• Average Speed: {context['avg_speed']:.1f} km/h\n\n"
            f"Recommended Actions:\n"
        )

        for idx, action in enumerate(actions["actions"][:4], start=1):
            response += f"{idx}. {action['action']}\n"

        response += (
            "\nExpected Outcome:\n"
            "Reduced congestion, improved traffic flow, "
            "better travel times, and proactive traffic management."
        )

        return response

    elif intent == "explanation_query":
        junctions = entities.get("junction_ids", [])

        if junctions:
            context = build_traffic_context(junctions[0], data_context)

            if context:
                congestion = context["congestion_level"]
                vehicles = context["vehicle_count"]
                speed = context["avg_speed"]

                if vehicles > 400:
                    vehicle_reason = "Vehicle count is significantly above normal levels."
                elif vehicles > 150:
                    vehicle_reason = "Vehicle volume is moderately elevated."
                else:
                    vehicle_reason = "Vehicle volume remains relatively manageable."

                if speed < 15:
                    speed_reason = "Average speed is extremely low, indicating severe delays."
                elif speed < 30:
                    speed_reason = "Traffic speeds are reduced, indicating congestion."
                else:
                    speed_reason = "Traffic speeds remain within acceptable limits."

                if congestion == "High":
                    analysis = (
                        "Traffic conditions indicate severe congestion. "
                        "High vehicle density and low average speeds suggest "
                        "significant delays and queue buildup."
                    )
                    recommendation = "Immediate intervention is recommended."
                elif congestion == "Medium":
                    analysis = (
                        "Traffic flow is beginning to slow down. "
                        "Vehicle density is increasing and congestion may worsen during peak periods."
                    )
                    recommendation = "Preventive traffic management actions are advised."
                else:
                    analysis = (
                        "Traffic is currently flowing reasonably well. "
                        "Vehicle density remains manageable and no major congestion indicators are present."
                    )
                    recommendation = "Continue monitoring traffic conditions."

                return (
                    f"🧠 Traffic Analysis for {entities.get('junction_names', ['Selected Area'])[0]}\n\n"
                    f"Current Conditions:\n"
                    f"• Congestion Level: {congestion}\n"
                    f"• Vehicle Count: {vehicles}\n"
                    f"• Average Speed: {speed:.1f} km/h\n\n"
                    f"Analysis:\n"
                    f"{analysis}\n\n"
                    f"Contributing Factors:\n"
                    f"• {vehicle_reason}\n"
                    f"• {speed_reason}\n\n"
                    f"Recommendation:\n"
                    f"{recommendation}"
                )

        return "Please specify a location such as 'Why is traffic bad at HITEC City?'"

    elif intent == "city_summary_query":
        response = "📊 Citywide Traffic Summary\n\n"

        if "traffic_data" in data_context:
            df = data_context["traffic_data"]
            total_junctions = df["junction_id"].nunique()
            avg_speed = round(df["avg_speed"].mean(), 1)

            if avg_speed > 35:
                status = "Traffic conditions are generally stable."
            elif avg_speed > 25:
                status = "Moderate congestion observed across the network."
            else:
                status = "Significant congestion detected across multiple corridors."

            total_vehicles = int(df["vehicle_count"].sum())
            response += (
                f"🚦 Current Network Status\n\n"
                f"• Monitored Junctions: {total_junctions}\n"
                f"• Total Vehicles: {total_vehicles}\n"
                f"• Average Speed: {avg_speed} km/h\n\n"
                f"📈 Overall Assessment\n\n{status}\n\n"
            )

        if "hotspots" in data_context:
            hotspots = data_context["hotspots"][:3]
            response += "🔥 Top Congestion Hotspots\n\n"
            for spot in hotspots:
                response += f"• Cluster {spot['cluster_id']} ({spot['severity']} Severity)\n"
            response += "\n"

        if "anomalies" in data_context:
            anomalies = data_context["anomalies"]
            high_risk = [a for a in anomalies if a["risk_level"] in ["High", "Critical"]]
            unique_anomaly_junctions = len(set(a["junction_id"] for a in high_risk))
            response += f"⚠️ Junctions With Active Anomalies: {unique_anomaly_junctions}\n\n"

            if high_risk:
                response += "Top Risk Locations:\n\n"
                for a in high_risk[:3]:
                    junction_name = JUNCTION_NAMES.get(a["junction_id"], a["junction_id"])
                    response += f"• {junction_name} ({a['risk_level']} Risk)\n"
                response += "\n"

        response += (
            "🎯 Recommended Priorities\n\n"
            "• Monitor hotspot regions\n"
            "• Investigate anomaly locations\n"
            "• Optimize signals at congested junctions\n"
            "• Prepare diversion plans if conditions worsen"
        )
        return response

    elif intent == "operational_query":
        summary = generate_operational_summary(data_context)
        if not summary:
            return "No significant operational issues detected at this time."

        response = "🚔 Traffic Operations Brief\n\nPriority Locations:\n\n"
        for idx, item in enumerate(summary[:5], start=1):
            response += (
                f"{idx}. {item['location']}\n"
                f"• Type: {item['type'].title()}\n"
                f"• Severity: {item['severity']}\n"
                f"• Priority Score: {item['score']}\n\n"
            )

        response += (
            "Recommended Actions:\n"
            "• Deploy field personnel\n"
            "• Optimize signal timings\n"
            "• Monitor congestion escalation\n"
            "• Prepare diversion plans\n"
        )
        return response

    elif intent == "intervention_query":
        top_locations = get_intervention_priorities(data_context, top_n=3)
        if len(top_locations) == 0:
            return "No traffic data available for intervention analysis."

        response = "🚨 Immediate Intervention Priority Areas\n\n"
        for idx, (_, row) in enumerate(top_locations.iterrows(), start=1):
            location_name = JUNCTION_NAMES.get(row["junction_id"], row["junction_id"])
            response += (
                f"{idx}. {location_name}\n"
                f"• Congestion: {row['congestion_level']}\n"
                f"• Vehicles: {int(row['vehicle_count'])}\n"
                f"• Avg Speed: {row['avg_speed']:.1f} km/h\n\n"
            )

        response += (
            "These locations currently show the highest traffic risk and should be prioritized for "
            "signal optimization, route diversion, and traffic personnel deployment."
        )
        return response

    elif intent == "route_query":
        origin = entities.get("origin")
        destination = entities.get("destination")

        if not origin or not destination:
            return (
                "Please specify both origin and destination.\n\n"
                "Example:\n"
                "• Best route from Ameerpet to Gachibowli\n"
                "• Fastest route from Charminar to LB Nagar"
            )

        try:
            df = pd.read_csv(DATASET_PATH)
            result = optimize_route(df, origin, destination)

            shortest = result["shortest"]
            least_congested = result["least_congested"]

            recommended_route_names = [
                JUNCTION_NAMES.get(jid, jid) for jid in least_congested["path"]
            ]
            alternative_route_names = [
                JUNCTION_NAMES.get(jid, jid) for jid in shortest["path"]
            ]

            response = (
                f"🗺️ Route Analysis\n\n"
                f"Origin:\n• {entities.get('origin_name', origin)}\n\n"
                f"Destination:\n• {entities.get('destination_name', destination)}\n\n"
                f"🚀 Recommended Route:\n• {' → '.join(recommended_route_names)}\n\n"
                f"Distance:\n• {least_congested['total_distance_km']} km\n\n"
                f"Estimated Time:\n• {least_congested['estimated_minutes']} min\n\n"
                f"Alternative Route:\n• {' → '.join(alternative_route_names)}\n\n"
                f"Recommendation:\n• {result['recommendation']}"
            )
            return response
        except Exception as e:
            return f"⚠️ Unable to calculate route.\n\nError: {str(e)}"

    elif intent == "prediction_query":
        junctions = entities.get("junction_ids", [])

        if not junctions:
            return "Please specify a location such as 'Predict traffic at HITEC City'."

        prediction_features = build_prediction_context(junctions[0], data_context)
        if not prediction_features:
            return "Traffic data unavailable."

        prediction = predict_congestion(prediction_features)
        mitigation = generate_mitigation_actions(
            junction_id=junctions[0],
            predicted_level=prediction["predicted_level"],
            confidence=prediction["confidence"],
            vehicle_count=prediction_features["vehicle_count"],
            avg_speed=prediction_features["avg_speed"],
        )

        junction_name = entities.get("junction_names", ["Selected Area"])[0]
        response = (
            f"🔮 Traffic Forecast for {junction_name}\n\n"
            f"Current Conditions:\n"
            f"• Vehicles: {prediction_features['vehicle_count']}\n"
            f"• Avg Speed: {prediction_features['avg_speed']:.1f} km/h\n\n"
            f"Predicted Congestion:\n• {prediction['predicted_level']}\n\n"
            f"Confidence:\n• {prediction['confidence']:.1%}\n\n"
            f"Recommended Actions:\n"
        )

        for action in mitigation["actions"][:3]:
            response += f"• {action['action']}\n"

        return response

    elif intent == "executive_report_query":
        return generate_executive_report(data_context)

    elif intent == "anomaly_query":
        anomalies = build_anomaly_context(data_context, top_n=3)
        if not anomalies:
            return "✅ No major traffic anomalies detected at this time."

        response = "⚠️ Current Traffic Anomalies\n\n"
        for idx, anomaly in enumerate(anomalies, start=1):
            location = JUNCTION_NAMES.get(anomaly["junction_id"], anomaly["junction_id"])
            response += (
                f"{idx}. {location}\n"
                f"• Risk Level: {anomaly['risk_level']}\n"
                f"• Vehicles: {anomaly['vehicle_count']}\n"
                f"• Avg Speed: {anomaly['avg_speed']:.1f} km/h\n"
                f"• Score: {anomaly['anomaly_score']:.3f}\n\n"
            )

        response += (
            "Recommended Response:\n"
            "• Route diversion\n"
            "• Signal optimization\n"
            "• Traffic personnel deployment"
        )
        return response

    elif intent == "hotspot_query":
        hotspots = build_hotspot_context(data_context, top_n=3)
        if not hotspots:
            return "✅ No major congestion hotspots detected currently."

        response = "🔥 Current Congestion Hotspots\n\n"
        for idx, hotspot in enumerate(hotspots, start=1):
            response += (
                f"{idx}. Cluster {hotspot['cluster_id']}\n"
                f"• Severity: {hotspot['severity']}\n"
                f"• Avg Vehicles: {hotspot['avg_vehicles']:.0f}\n"
                f"• Avg Speed: {hotspot['avg_speed']:.1f} km/h\n"
                f"• Peak Hours: {', '.join(map(str, hotspot['peak_hours']))}\n\n"
            )

        response += (
            "Recommended Actions:\n"
            "• Route diversion\n"
            "• Signal timing optimization\n"
            "• Traffic personnel deployment\n"
            "• Continuous monitoring"
        )
        return response

    else:
        return (
            "👋 Hello! I'm your **Traffic Intelligence Assistant**. I can help with:\n\n"
            "- 🚦 **Traffic status**: *'What's the traffic at HITEC City?'*\n"
            "- 🗺️ **Route planning**: *'Best route from Ameerpet to Gachibowli'*\n"
            "- 🔮 **Predictions**: *'Predict traffic for tomorrow'*\n"
            "- ⚠️ **Anomalies**: *'Any unusual traffic patterns?'*\n"
            "- 🔥 **Hotspots**: *'Which areas are most congested?'*\n"
        )


def process_query(query: str, session_id: str, data_context: dict = None) -> dict:
    """
    Full NLP pipeline:
    1. Detect language → translate to English
    2. Classify intent
    3. Extract entities
    4. Generate response
    5. Translate back
    6. Store in chat history
    """
    from nlp.multilingual import detect_language, translate_text

    # Language detection
    detected_lang = detect_language(query)
    original_query = query

    # Translate to English if needed
    if detected_lang != "en":
        query = translate_text(query, source=detected_lang, target="en")

    # Classify and extract
    intent = classify_intent(query)
    entities = extract_entities(query)

    # Smart fallback:
    # If user mentions a known junction but intent is unclear,
    # assume they want current traffic status.
    if intent == "general_info" and entities.get("junction_ids"):
        intent = "congestion_check"

    # Generate response
    response = generate_response(intent, entities, data_context)

    # Translate response back if needed
    if detected_lang != "en":
        response = translate_text(response, source="en", target=detected_lang)

    # Store in chat history
    turn_number = get_next_turn_number(session_id)
    insert_chat_turn(
        {
            "session_id": session_id,
            "turn_number": turn_number,
            "user_query": original_query,
            "detected_language": detected_lang,
            "intent": intent,
            "entities": entities,
            "module_results": {},
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }
    )

    return {
        "response": response,
        "intent": intent,
        "entities": entities,
        "detected_language": detected_lang,
        "turn_number": turn_number,
    }
