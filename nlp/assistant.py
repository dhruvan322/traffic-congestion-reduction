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
from engine.prediction.predictor import predict_congestion, predict_from_row
from config import PEAK_HOURS
from data.generate_dataset import JUNCTION_NAMES
from models.anomaly_detector import get_anomaly_report
from database import insert_chat_turn, get_chat_history, get_next_turn_number
from engine.traffic_reasoner import generate_operational_summary
from engine.management_planner import generate_management_plan
from engine.executive_report import generate_executive_report

logger = logging.getLogger(__name__)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Intent Patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Junction Extraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                return "ðŸš¦ **Traffic Status:**\n" + "\n".join(responses)

        return (
            "ðŸš¦ I can check traffic for specific areas. "
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

        # Use the most recent anomaly for the selected area,
        # rather than the highest anomaly score from the full year.
        area_anomalies = [
            anomaly
            for anomaly in anomalies
            if str(anomaly.get("road_segment_id", ""))
            in context["road_segment_ids"]
        ]

        anomaly_context = {
            "detected": False,
            "risk_level": "Low",
            "road_segment_id": "Unknown",
        }

        if area_anomalies:
            area_anomalies.sort(
                key=lambda anomaly: str(anomaly.get("timestamp", "")),
                reverse=True,
            )

            latest_anomaly = area_anomalies[0]

            anomaly_context = {
                "detected": True,
                "risk_level": latest_anomaly.get(
                    "risk_level",
                    "Low",
                ),
                "road_segment_id": str(
                    latest_anomaly.get(
                        "road_segment_id",
                        "Unknown",
                    )
                ),
            }

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
            f"ðŸš¦ **Congestion Reduction Plan for {location}**\n\n"
            f"**Traffic Intelligence**\n"
            f"â€¢ Predicted congestion: {prediction_result['predicted_class']}\n"
            f"â€¢ Prediction confidence: {confidence_percent:.1f}%\n"
            f"â€¢ Vehicles: {context['vehicle_count']}\n"
            f"â€¢ Average speed: {context['average_speed_kmh']:.1f} km/h\n"
            f"â€¢ Road segment used for prediction: {prediction_row['road_segment_id']}\n\n"
            f"**Operational Priority:** {decision['overall_priority']}\n\n"
            f"**Recommended Actions**\n"
        )

        for index, recommendation in enumerate(decision["recommendations"][:4], start=1):
            response += (
                f"{index}. **{recommendation['category']}** â€” "
                f"{recommendation['action']}\n"
            )

        response += "\n**WHY these actions?**\n"
        for recommendation in decision["recommendations"][:4]:
            reason = recommendation["reason"]

            if recommendation["category"] == "Anomaly Response":
                reason = "Historical anomaly evidence: " + reason

            response += f"â€¢ {reason}\n"

        return response

    elif intent == "management_plan_query":
        if not locations:
            return (
                "Please specify a location such as "
                "'Generate a traffic management plan for HITEC City'."
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

        
        from engine.reduction_engine import generate_integrated_recommendations

        prediction_result = predict_from_row(prediction_row)

        prediction = {
            "congestion_level": prediction_result["predicted_class"],
            "confidence": prediction_result["confidence"],
            "road_segment_id": str(prediction_row["road_segment_id"]),
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

        area_anomalies = [
            anomaly
            for anomaly in anomalies
            if str(anomaly.get("road_segment_id", ""))
            in context["road_segment_ids"]
        ]

        anomaly_context = {
            "detected": False,
            "risk_level": "Low",
            "road_segment_id": "Unknown",
        }

        if area_anomalies:
            area_anomalies.sort(
                key=lambda anomaly: str(anomaly.get("timestamp", "")),
                reverse=True,
            )

            latest_anomaly = area_anomalies[0]

            anomaly_context = {
                "detected": True,
                "risk_level": latest_anomaly.get(
                    "risk_level",
                    "Low",
                ),
                "road_segment_id": str(
                    latest_anomaly.get(
                        "road_segment_id",
                        "Unknown",
                    )
                ),
            }

        route_context = {
            "alternative_available": False,
            "shortest_distance_km": 0.0,
            "shortest_time_minutes": 0.0,
            "traffic_aware_distance_km": 0.0,
            "traffic_aware_time_minutes": 0.0,
        }

        decision = generate_integrated_recommendations(
            prediction=prediction,
            hotspot=hotspot_context,
            anomaly=anomaly_context,
            route_conditions=route_context,
        )

        response = (
            f"ðŸ“‹ **Traffic Management Plan for {location}**\n\n"
            f"**Traffic Intelligence**\n"
            f"â€¢ Predicted congestion: "
            f"{prediction_result['predicted_class']}\n"
            f"â€¢ Prediction confidence: "
            f"{prediction_result['confidence'] * 100:.1f}%\n"
            f"â€¢ Vehicles: {context['vehicle_count']}\n"
            f"â€¢ Average speed: "
            f"{context['average_speed_kmh']:.1f} km/h\n"
            f"â€¢ Road segment used for prediction: "
            f"{prediction_row['road_segment_id']}\n\n"
            f"**Operational Priority:** "
            f"{decision['overall_priority']}\n\n"
            f"**Management Actions**\n"
        )

        for index, recommendation in enumerate(
            decision["recommendations"][:4],
            start=1,
        ):
            response += (
                f"{index}. **{recommendation['category']}** â€” "
                f"{recommendation['action']}\n"
            )

        response += "\n**Management Rationale**\n"

        for recommendation in decision["recommendations"][:4]:
            reason = recommendation["reason"]

            if recommendation["category"] == "Anomaly Response":
                reason = "Historical anomaly evidence: " + reason

            response += f"â€¢ {reason}\n"

        return response
    elif intent == "explanation_query":
        if not locations:
            return (
                "Please specify a location such as "
                "'Why is traffic bad at HITEC City?'"
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

        explanation_row = area_data.iloc[-1]

        from engine.prediction.explainer import explain_prediction

        shap_result = explain_prediction(
            explanation_row,
            top_n=10,
        )

        if "error" in shap_result:
            return (
                f"Unable to generate the SHAP explanation for {location}: "
                f"{shap_result['error']}"
            )

        predicted_class = shap_result["predicted_class"]
        confidence_percent = shap_result["confidence"] * 100

        def format_shap_feature(name):
            if name.startswith("num__"):
                feature = name.replace("num__", "", 1)
                return feature.replace("_", " ").title()

            if name.startswith("cat__"):
                feature = name.replace("cat__", "", 1)
                return feature.replace("_", " ").title()

            return name.replace("_", " ").title()

        contributions = shap_result.get("contributions", {})

        positive_factors = []
        negative_factors = []

        for feature, value in contributions.items():
            readable_feature = format_shap_feature(feature)
            value = float(value)

            if value >= 0:
                positive_factors.append(
                    (readable_feature, value)
                )
            else:
                negative_factors.append(
                    (readable_feature, value)
                )

        response = (
            f"ðŸ§  **Traffic Analysis for {location}**\n\n"
            f"**Traffic Intelligence**\n"
            f"â€¢ Predicted congestion: {predicted_class}\n"
            f"â€¢ Prediction confidence: "
            f"{confidence_percent:.1f}%\n"
            f"â€¢ Vehicles: {context['vehicle_count']}\n"
            f"â€¢ Average speed: "
            f"{context['average_speed_kmh']:.1f} km/h\n"
            f"â€¢ Road segment used for prediction: "
            f"{explanation_row['road_segment_id']}\n\n"
            f"**Model Explanation (SHAP)**\n"
        )

        for feature, value in positive_factors[:5]:
            response += (
                f"â€¢ **{feature}**: {value:+.4f} — "
                f"pushed the prediction toward {predicted_class}\n"
            )

        for feature, value in negative_factors[:5]:
            response += (
                f"â€¢ **{feature}**: {value:+.4f} — "
                f"pushed the prediction away from {predicted_class}\n"
            )

        if contributions:
            strongest_feature = max(
                contributions.items(),
                key=lambda item: abs(float(item[1])),
            )

            strongest_name = format_shap_feature(strongest_feature[0])
            strongest_value = float(strongest_feature[1])

            if strongest_value >= 0:
                strongest_direction = (
                    f"toward the predicted {predicted_class} class"
                )
            else:
                strongest_direction = (
                    f"away from the predicted {predicted_class} class"
                )

            response += (
                "\n**Interpretation**\n"
                f"The strongest SHAP factor is **{strongest_name}** "
                f"({strongest_value:+.4f}), which pushed the model "
                f"{strongest_direction}.\n"
            )

        response += (
            "\nSHAP values explain how individual model inputs influenced "
            "this prediction; they do not by themselves establish causal "
            "relationships."
        )

        return response
    elif intent == "city_summary_query":
        df = data_context["traffic_data"]

        if df.empty:
            return "No traffic data is available for the citywide summary."

        total_records = len(df)
        total_segments = df["road_segment_id"].nunique()
        total_vehicles = int(df["vehicle_count"].sum())
        avg_speed = float(df["average_speed_kmh"].mean())

        congestion_counts = (
            df["future_congestion_level"]
            .value_counts()
            .reindex(
                ["Free Flow", "Low", "Moderate", "High", "Severe"],
                fill_value=0
            )
        )

        congestion_percentages = (
            congestion_counts / total_records * 100
        )

        congested_records = int(
            congestion_counts["Moderate"]
            + congestion_counts["High"]
            + congestion_counts["Severe"]
        )

        congested_percentage = (
            congested_records / total_records * 100
        )

        if congested_percentage >= 25:
            status = (
                "Significant congestion is present across the monitored "
                "road network."
            )
        elif congested_percentage >= 15:
            status = (
                "Moderate congestion is present across parts of the "
                "monitored road network."
            )
        else:
            status = (
                "Most monitored traffic observations indicate relatively "
                "stable network conditions."
            )

        hotspots = get_hotspot_summary(df)
        anomalies = get_anomaly_report(df)

        high_risk_anomalies = [
            item for item in anomalies
            if item.get("risk_level") in ["High", "Critical"]
        ]

        unique_high_risk_segments = len(
            set(
                str(item["road_segment_id"])
                for item in high_risk_anomalies
            )
        )

        response = (
            "📊 Citywide Traffic Summary\n\n"
            "🚦 Network Status\n\n"
            f"• Monitored Records: {total_records:,}\n"
            f"• Monitored Road Segments: {total_segments}\n"
            f"• Total Vehicles: {total_vehicles:,}\n"
            f"• Average Speed: {avg_speed:.1f} km/h\n"
            f"• Moderate/High/Severe Observations: "
            f"{congested_records:,} ({congested_percentage:.1f}%)\n\n"
            f"📈 Overall Assessment\n\n"
            f"{status}\n\n"
            "Congestion Distribution\n\n"
            f"• Free Flow: {int(congestion_counts['Free Flow']):,} "
            f"({congestion_percentages['Free Flow']:.1f}%)\n"
            f"• Low: {int(congestion_counts['Low']):,} "
            f"({congestion_percentages['Low']:.1f}%)\n"
            f"• Moderate: {int(congestion_counts['Moderate']):,} "
            f"({congestion_percentages['Moderate']:.1f}%)\n"
            f"• High: {int(congestion_counts['High']):,} "
            f"({congestion_percentages['High']:.1f}%)\n"
            f"• Severe: {int(congestion_counts['Severe']):,} "
            f"({congestion_percentages['Severe']:.1f}%)\n\n"
        )

        if hotspots:
            response += "🔥 Top Congestion Hotspots\n\n"

            for spot in hotspots[:3]:
                response += (
                    f"• Cluster {spot['cluster_id']} — "
                    f"{spot['severity']} severity\n"
                    f"  Congestion rate: "
                    f"{spot['congestion_rate'] * 100:.1f}%\n"
                    f"  High/Severe rate: "
                    f"{spot['high_or_severe_rate'] * 100:.1f}%\n"
                    f"  Road segments: {spot['size']}\n"
                )

            response += "\n"

        response += (
            "⚠️ Anomaly Status\n\n"
            f"• High/Critical anomaly observations: "
            f"{len(high_risk_anomalies):,}\n"
            f"• Unique road segments with High/Critical anomalies: "
            f"{unique_high_risk_segments}\n"
        )

        if high_risk_anomalies:
            top_anomaly = high_risk_anomalies[0]

            response += (
                f"• Highest-risk observation: "
                f"{top_anomaly['road_segment_id']} "
                f"({top_anomaly['risk_level']})\n"
                f"• Congestion at detection: "
                f"{top_anomaly['future_congestion_level']}\n"
                f"• Detection time: {top_anomaly['timestamp']}\n"
            )

        response += (
            "\n🎯 Recommended Priorities\n\n"
            "• Monitor identified hotspot clusters\n"
            "• Investigate High/Critical anomaly observations\n"
            "• Review signal and traffic-management strategies "
            "for congested corridors\n"
            "• Prepare diversion options where congestion escalates\n\n"
            "Note: This summary describes the available historical "
            "traffic dataset and does not represent live traffic conditions."
        )

        return response
    elif intent == "operational_query":
        df = data_context["traffic_data"]

        if df.empty:
            return "No traffic data is available for operational analysis."

        hotspots = get_hotspot_summary(df)
        anomalies = get_anomaly_report(df)

        high_risk_anomalies = [
            item for item in anomalies
            if item.get("risk_level") in ["High", "Critical"]
        ]

        response = "🚨 Traffic Operations Brief\n\n"

        if hotspots:
            response += "Priority Hotspot Locations\n\n"

            for idx, spot in enumerate(hotspots[:3], start=1):
                priority = "High"
                if (
                    spot.get("severity") == "Medium"
                    and spot.get("high_or_severe_rate", 0) < 0.05
                ):
                    priority = "Medium"

                response += (
                    f"{idx}. Cluster {spot['cluster_id']} "
                    f"({priority} operational priority)\n"
                    f"• Severity: {spot['severity']}\n"
                    f"• Congestion rate: "
                    f"{spot['congestion_rate'] * 100:.1f}%\n"
                    f"• High/Severe rate: "
                    f"{spot['high_or_severe_rate'] * 100:.1f}%\n"
                    f"• Severe rate: "
                    f"{spot['severe_rate'] * 100:.1f}%\n"
                    f"• Road segments: {spot['size']}\n"
                    f"• Peak hours: {spot['peak_hours']}\n"
                    f"• Segments: "
                    f"{', '.join(spot['road_segments'])}\n\n"
                )
        else:
            response += (
                "Priority Hotspot Locations\n\n"
                "• No significant hotspot clusters detected.\n\n"
            )

        response += "⚠️ Anomaly Response Priorities\n\n"

        if high_risk_anomalies:
            for idx, anomaly in enumerate(
                high_risk_anomalies[:5],
                start=1
            ):
                response += (
                    f"{idx}. {anomaly['road_segment_id']} "
                    f"({anomaly['risk_level']} risk)\n"
                    f"• Congestion: "
                    f"{anomaly['future_congestion_level']}\n"
                    f"• Vehicles: {anomaly['vehicle_count']}\n"
                    f"• Average speed: "
                    f"{anomaly['average_speed_kmh']:.1f} km/h\n"
                    f"• Occupancy: "
                    f"{anomaly['occupancy_percent']:.1f}%\n"
                    f"• Detection type: "
                    f"{anomaly['anomaly_type']}\n"
                    f"• Detection time: {anomaly['timestamp']}\n\n"
                )
        else:
            response += (
                "• No High/Critical anomaly observations require "
                "immediate attention.\n\n"
            )

        critical_count = sum(
            1
            for item in high_risk_anomalies
            if item.get("risk_level") == "Critical"
        )

        response += (
            "Recommended Operational Actions\n\n"
            "• Investigate High/Critical anomaly locations first\n"
            "• Monitor identified hotspot clusters during peak hours\n"
            "• Review signal timing and traffic-management measures "
            "on congested corridors\n"
            "• Prepare route-diversion options if congestion escalates\n"
            "• Continue monitoring anomalous traffic conditions\n\n"
            f"Operational Assessment: "
            f"{len(high_risk_anomalies)} High/Critical anomaly "
            f"observations identified, including "
            f"{critical_count} Critical observations.\n\n"
            "Note: This operational brief is based on the available "
            "historical traffic dataset and does not represent live "
            "field conditions."
        )

        return response
    elif intent == "intervention_query":
        df = data_context["traffic_data"]

        if df.empty:
            return "No traffic data is available for intervention analysis."

        hotspots = get_hotspot_summary(df)
        anomalies = get_anomaly_report(df)

        high_risk_anomalies = [
            item for item in anomalies
            if item.get("risk_level") in ["High", "Critical"]
        ]

        high_risk_anomalies = sorted(
            high_risk_anomalies,
            key=lambda item: (
                0 if item.get("risk_level") == "Critical" else 1,
                -float(item.get("anomaly_score", 0))
            )
        )

        response = "🚨 Immediate Intervention Priority Areas\n\n"

        if high_risk_anomalies:
            response += "High-Risk Road Segments\n\n"

            displayed_segments = set()

            for idx, item in enumerate(
                high_risk_anomalies,
                start=1
            ):
                segment_id = str(item["road_segment_id"])

                if segment_id in displayed_segments:
                    continue

                displayed_segments.add(segment_id)

                response += (
                    f"{len(displayed_segments)}. {segment_id} "
                    f"({item['risk_level']} priority)\n"
                    f"• Congestion: "
                    f"{item['future_congestion_level']}\n"
                    f"• Vehicles: {item['vehicle_count']}\n"
                    f"• Average speed: "
                    f"{item['average_speed_kmh']:.1f} km/h\n"
                    f"• Occupancy: "
                    f"{item['occupancy_percent']:.1f}%\n"
                    f"• Anomaly score: "
                    f"{float(item['anomaly_score']):.3f}\n"
                    f"• Detection type: "
                    f"{item['anomaly_type']}\n"
                    f"• Detection time: "
                    f"{item['timestamp']}\n\n"
                )

                if len(displayed_segments) >= 5:
                    break
        else:
            response += (
                "High-Risk Road Segments\n\n"
                "• No High/Critical anomaly locations detected.\n\n"
            )

        if hotspots:
            response += "🔥 Hotspot Intervention Priorities\n\n"

            for idx, spot in enumerate(hotspots[:3], start=1):
                if spot["high_or_severe_rate"] >= 0.10:
                    intervention_level = "High"
                elif spot["high_or_severe_rate"] >= 0.05:
                    intervention_level = "Medium"
                else:
                    intervention_level = "Monitor"

                response += (
                    f"{idx}. Cluster {spot['cluster_id']} "
                    f"({intervention_level})\n"
                    f"• Congestion rate: "
                    f"{spot['congestion_rate'] * 100:.1f}%\n"
                    f"• High/Severe rate: "
                    f"{spot['high_or_severe_rate'] * 100:.1f}%\n"
                    f"• Severe rate: "
                    f"{spot['severe_rate'] * 100:.1f}%\n"
                    f"• Peak hours: {spot['peak_hours']}\n"
                    f"• Road segments: "
                    f"{', '.join(spot['road_segments'])}\n\n"
                )

        response += (
            "Recommended Interventions\n\n"
            "• Investigate Critical and High-risk road segments first\n"
            "• Verify abnormal traffic conditions in the field\n"
            "• Review signal timing and traffic-management measures "
            "at hotspot corridors\n"
            "• Prepare route-diversion options for severe congestion\n"
            "• Monitor peak-hour escalation in identified hotspots\n\n"
            "Note: These priorities are derived from historical anomaly "
            "and hotspot observations. They are decision-support "
            "recommendations and do not represent live field conditions."
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
            from engine.osm_traffic_router import (
                compare_osm_routes,
                load_traffic_dataset,
            )

            df = load_traffic_dataset().copy()

            df["timestamp"] = pd.to_datetime(
                df["timestamp"]
            )

            def resolve_area_coordinates(area_name):
                normalized = str(area_name).strip().lower()

                area_rows = df[
                    df["area"].astype(str).str.strip().str.lower()
                    == normalized
                ]

                if area_rows.empty:
                    return None

                return (
                    float(area_rows["latitude"].mean()),
                    float(area_rows["longitude"].mean()),
                )

            origin_coords = resolve_area_coordinates(origin)
            destination_coords = resolve_area_coordinates(destination)

            if origin_coords is None:
                return (
                    f"Unable to resolve origin area '{origin}' "
                    "in the project traffic dataset."
                )

            if destination_coords is None:
                return (
                    f"Unable to resolve destination area '{destination}' "
                    "in the project traffic dataset."
                )

            latest_timestamp = df["timestamp"].max()

            result = compare_osm_routes(
                origin=origin_coords,
                destination=destination_coords,
                timestamp=latest_timestamp,
            )

            shortest = result["shortest_metrics"]
            traffic_aware = result["traffic_aware_metrics"]

            shortest_route = result["shortest_route"]
            traffic_route = result["traffic_aware_route"]

            route_changed = shortest_route != traffic_route

            if route_changed:
                recommendation = (
                    "The traffic-aware route is recommended because "
                    "the traffic-weighted network selected a different "
                    "route from the shortest-distance route."
                )
            else:
                recommendation = (
                    "The shortest and traffic-aware routes are the same "
                    "for the selected historical traffic conditions."
                )

            response = (
                "🗺️ OSM Traffic-Aware Route Analysis\n\n"
                f"Origin:\n"
                f"• {entities.get('origin_name', origin)}\n"
                f"• Coordinates: "
                f"{origin_coords[0]:.5f}, {origin_coords[1]:.5f}\n\n"
                f"Destination:\n"
                f"• {entities.get('destination_name', destination)}\n"
                f"• Coordinates: "
                f"{destination_coords[0]:.5f}, "
                f"{destination_coords[1]:.5f}\n\n"
                f"Traffic Data Timestamp:\n"
                f"• {result['timestamp_requested']}\n\n"
                "🚗 Shortest Route\n"
                f"• Distance: "
                f"{shortest['distance_km']:.2f} km\n"
                f"• Estimated time: "
                f"{shortest['estimated_time_minutes']:.2f} min\n"
                f"• Average speed: "
                f"{shortest['average_speed_kmh']:.1f} km/h\n"
                f"• Traffic-covered edges: "
                f"{shortest['traffic_covered_edges']}/"
                f"{shortest['total_edges']}\n\n"
                "🚦 Traffic-Aware Route\n"
                f"• Distance: "
                f"{traffic_aware['distance_km']:.2f} km\n"
                f"• Estimated time: "
                f"{traffic_aware['estimated_time_minutes']:.2f} min\n"
                f"• Average speed: "
                f"{traffic_aware['average_speed_kmh']:.1f} km/h\n"
                f"• Traffic-covered edges: "
                f"{traffic_aware['traffic_covered_edges']}/"
                f"{traffic_aware['total_edges']}\n\n"
                f"Route changed: {'Yes' if route_changed else 'No'}\n\n"
                f"Recommendation:\n"
                f"• {recommendation}\n\n"
                "Note: This route analysis uses the project's "
                "historical traffic observations with the Hyderabad "
                "OSM driving network. It is decision-support and "
                "not live navigation."
            )

            return response

        except Exception as e:
            return (
                "⚠️ Unable to calculate the OSM traffic-aware route.\n\n"
                f"Error: {str(e)}"
            )
    elif intent == "prediction_query":
        location_names = entities.get("location_names", [])
        locations = entities.get("locations", [])
        road_segment_ids = entities.get("road_segment_ids", [])

        location = (
            locations[0]
            if locations
            else (
                location_names[0]
                if location_names
                else None
            )
        )

        if not location and not road_segment_ids:
            return (
                "Please specify a location such as "
                "'Predict traffic at HITEC City' "
                "or 'Predict traffic at RS0031'."
            )

        df = data_context["traffic_data"]

        if location:
            traffic_context = build_traffic_context(
                location,
                data_context
            )
        else:
            matching_rows = df[
                df["road_segment_id"].astype(str).isin(
                    [str(segment) for segment in road_segment_ids]
                )
            ].copy()

            if matching_rows.empty:
                return "Traffic data unavailable for the specified road segment."

            matching_rows["timestamp"] = pd.to_datetime(
                matching_rows["timestamp"]
            )

            latest_timestamp = matching_rows["timestamp"].max()

            latest_rows = matching_rows[
                matching_rows["timestamp"] == latest_timestamp
            ]

            if latest_rows.empty:
                return "Traffic data unavailable for the specified road segment."

            latest_row = latest_rows.iloc[0]

            traffic_context = {
                "location": str(
                    latest_row.get("area", road_segment_ids[0])
                ),
                "road_segment_ids": [
                    str(latest_row["road_segment_id"])
                ],
                "vehicle_count": int(
                    latest_row["vehicle_count"]
                ),
                "avg_speed": float(
                    latest_row["average_speed_kmh"]
                ),
                "average_speed_kmh": float(
                    latest_row["average_speed_kmh"]
                ),
                "congestion_level": str(
                    latest_row["future_congestion_level"]
                ),
                "future_congestion_level": str(
                    latest_row["future_congestion_level"]
                ),
                "timestamp": str(
                    latest_timestamp
                ),
            }

        segment_ids = traffic_context.get(
            "road_segment_ids",
            []
        )

        if not segment_ids:
            return "Traffic data unavailable for the specified location."

        latest_rows = df[
            df["road_segment_id"].astype(str).isin(
                [str(segment) for segment in segment_ids]
            )
        ].copy()

        if latest_rows.empty:
            return "Traffic data unavailable for the specified location."

        latest_rows["timestamp"] = pd.to_datetime(
            latest_rows["timestamp"]
        )

        latest_timestamp = latest_rows["timestamp"].max()

        prediction_rows = latest_rows[
            latest_rows["timestamp"] == latest_timestamp
        ]

        if prediction_rows.empty:
            return "Traffic data unavailable for the specified location."

        prediction_row = prediction_rows.iloc[0]

        prediction = predict_from_row(prediction_row)

        predicted_class = prediction["predicted_class"]
        confidence = float(prediction["confidence"])

        location_display = traffic_context.get(
            "location",
            location or road_segment_ids[0]
        )

        response = (
            f"🔮 Traffic Prediction for {location_display}\n\n"
            f"Traffic Intelligence\n"
            f"• Predicted congestion: {predicted_class}\n"
            f"• Prediction confidence: {confidence:.1%}\n"
            f"• Vehicles: {int(prediction_row['vehicle_count'])}\n"
            f"• Average speed: "
            f"{float(prediction_row['average_speed_kmh']):.1f} km/h\n"
            f"• Road segment used for prediction: "
            f"{prediction_row['road_segment_id']}\n"
            f"• Prediction timestamp: {latest_timestamp}\n\n"
            f"Model Output\n"
            f"• The prediction is generated by the trained "
            f"five-class congestion model.\n"
            f"• Predicted class: {predicted_class}\n"
        )

        return response
    elif intent == "executive_report_query":
        return generate_executive_report(data_context)

    elif intent == "anomaly_query":
        df = data_context["traffic_data"]

        anomalies = get_anomaly_report(df)

        if not anomalies:
            return "No significant traffic anomalies were detected."

        high_risk = [
            anomaly
            for anomaly in anomalies
            if anomaly.get("risk_level") in ["High", "Critical"]
        ]

        response = "⚠️ Current Traffic Anomalies\n\n"

        if high_risk:
            response += "High-Risk Anomalies\n\n"

            for idx, anomaly in enumerate(high_risk[:5], start=1):
                segment = anomaly.get(
                    "road_segment_id",
                    "Unknown"
                )
                risk = anomaly.get(
                    "risk_level",
                    "Unknown"
                )
                timestamp = anomaly.get(
                    "timestamp",
                    "Unknown"
                )
                vehicles = int(
                    float(anomaly.get("vehicle_count", 0))
                )
                speed = float(
                    anomaly.get("average_speed_kmh", 0.0)
                )
                occupancy = float(
                    anomaly.get("occupancy_percent", 0.0)
                )
                congestion = anomaly.get(
                    "future_congestion_level",
                    "Unknown"
                )
                score = float(
                    anomaly.get("anomaly_score", 0.0)
                )
                anomaly_type = anomaly.get(
                    "anomaly_type",
                    "Unknown"
                )

                response += (
                    f"{idx}. {segment}\n"
                    f"• Risk Level: {risk}\n"
                    f"• Future congestion: {congestion}\n"
                    f"• Vehicles: {vehicles}\n"
                    f"• Average speed: {speed:.1f} km/h\n"
                    f"• Occupancy: {occupancy:.1f}%\n"
                    f"• Anomaly score: {score:.3f}\n"
                    f"• Detection type: {anomaly_type}\n"
                    f"• Timestamp: {timestamp}\n\n"
                )

        else:
            response += (
                "No High or Critical risk anomalies were detected.\n\n"
            )

            response += (
                f"Total anomaly records analyzed: "
                f"{len(anomalies):,}\n"
            )

        response += (
            "Recommended Response\n"
            "• Investigate High/Critical anomaly locations\n"
            "• Verify abnormal traffic conditions on affected road segments\n"
            "• Review signal and corridor conditions where appropriate\n"
            "• Consider diversion measures if congestion escalates\n"
            "• Continue monitoring anomaly indicators"
        )

        return response
    elif intent == "hotspot_query":
        df = data_context["traffic_data"]

        hotspots = get_hotspot_summary(df)

        if not hotspots:
            return (
                "No significant congestion hotspots were detected "
                "in the analyzed traffic data."
            )

        response = "🔥 Current Congestion Hotspots\n\n"

        for idx, hotspot in enumerate(hotspots[:3], start=1):
            cluster_id = hotspot.get("cluster_id", "Unknown")
            severity = hotspot.get("severity", "Unknown")
            size = int(hotspot.get("size", 0))
            congestion_rate = float(
                hotspot.get("congestion_rate", 0.0)
            )
            high_or_severe_rate = float(
                hotspot.get("high_or_severe_rate", 0.0)
            )
            severe_rate = float(
                hotspot.get("severe_rate", 0.0)
            )
            avg_speed = float(
                hotspot.get("avg_speed", 0.0)
            )
            avg_vehicles = float(
                hotspot.get("avg_vehicles", 0.0)
            )
            peak_hours = hotspot.get("peak_hours", [])
            road_segments = hotspot.get("road_segments", [])

            response += (
                f"{idx}. Cluster {cluster_id}\n"
                f"• Severity: {severity}\n"
                f"• Road segments: {size}\n"
                f"• Congestion rate: {congestion_rate:.1%}\n"
                f"• High/Severe rate: {high_or_severe_rate:.1%}\n"
                f"• Severe rate: {severe_rate:.1%}\n"
                f"• Average vehicles: {avg_vehicles:.1f}\n"
                f"• Average speed: {avg_speed:.1f} km/h\n"
            )

            if peak_hours:
                response += (
                    f"• Peak congestion hours: "
                    f"{', '.join(map(str, peak_hours))}\n"
                )

            if road_segments:
                response += (
                    f"• Road segments: "
                    f"{', '.join(map(str, road_segments))}\n"
                )

            response += "\n"

        response += (
            "Recommended Actions\n"
            "• Prioritize monitoring of identified hotspot clusters\n"
            "• Review signal timing and corridor management\n"
            "• Consider diversion strategies if congestion escalates\n"
            "• Continue monitoring hotspot severity and congestion rates"
        )

        return response
    else:
        return (
            "ðŸ‘‹ Hello! I'm your **Traffic Intelligence Assistant**. I can help with:\n\n"
            "- ðŸš¦ **Traffic status**: *'What's the traffic at HITEC City?'*\n"
            "- ðŸ—ºï¸ **Route planning**: *'Best route from Ameerpet to Gachibowli'*\n"
            "- ðŸ”® **Predictions**: *'Predict traffic for tomorrow'*\n"
            "- âš ï¸ **Anomalies**: *'Any unusual traffic patterns?'*\n"
            "- ðŸ”¥ **Hotspots**: *'Which areas are most congested?'*\n"
        )


def process_query(query: str, session_id: str, data_context: dict = None) -> dict:
    """
    Full NLP pipeline:
    1. Detect language â†’ translate to English
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
















