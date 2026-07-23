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
from models.predictor import predict_congestion
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
    """Extract entities like junction IDs and time references from query."""
    query_lower = query.lower()
    entities = {}

    # Extract junction IDs (J001-J020)
    jid_matches = re.findall(r"j\d{3}", query_lower)
    if jid_matches:
        entities["junction_ids"] = [j.upper() for j in jid_matches]

    # Extract junction names
    for alias, jid in JUNCTION_ALIASES.items():
        if alias in query_lower:
            if "junction_ids" not in entities:
                entities["junction_ids"] = []
            if jid not in entities["junction_ids"]:
                entities["junction_ids"].append(jid)
                entities.setdefault("junction_names", []).append(alias.title())

    # Extract route origin/destination

    route_match = re.search(r"from\s+(.+?)\s+to\s+(.+)", query_lower)

    if route_match:
        origin_text = route_match.group(1).strip()
        destination_text = route_match.group(2).strip()

        for alias, jid in JUNCTION_ALIASES.items():
            if alias == origin_text:
                entities["origin"] = jid
                entities["origin_name"] = alias.title()

            if alias == destination_text:
                entities["destination"] = jid
                entities["destination_name"] = alias.title()

    # Extract time references
    time_matches = re.findall(r"(\d{1,2})\s*(?:am|pm|:00|:30)", query_lower)
    if time_matches:
        entities["hours"] = [int(h) for h in time_matches]

    # Extract weather
    for w in ["rain", "fog", "clear", "cloudy"]:
        if w in query_lower:
            entities["weather"] = w.capitalize()

    return entities


def build_traffic_context(junction_id, data_context):
    """
    Build real traffic context from dataset.
    """

    if "traffic_data" not in data_context:
        return None

    df = data_context["traffic_data"]

    row = df[df["junction_id"] == junction_id]

    if row.empty:
        return None

    latest = row.iloc[-1]

    return {
        "junction_id": junction_id,
        "vehicle_count": int(latest["vehicle_count"]),
        "avg_speed": float(latest["avg_speed"]),
        "congestion_level": latest["congestion_level"],
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

    if intent == "congestion_check":
        junctions = entities.get("junction_ids", [])
        if junctions and "traffic_data" in data_context:
            df = data_context["traffic_data"]
            responses = []
            for jid in junctions:
                jdata = df[df["junction_id"] == jid]
                if not jdata.empty:
                    latest = jdata.iloc[-1]
                    responses.append(
                        f"**{jid}**: {latest['congestion_level']} congestion "
                        f"({latest['vehicle_count']} vehicles, {latest['avg_speed']} km/h)"
                    )
                else:
                    responses.append(f"**{jid}**: No data available")
            return "🚦 **Traffic Status:**\n" + "\n".join(responses)

        return (
            "🚦 I can check traffic for specific junctions. "
            "Try asking: *'What is the traffic at HITEC City?'* or *'Is Ameerpet busy?'*"
        )

    elif intent == "reduction_query":

        junctions = entities.get("junction_ids", [])

        if junctions:

            junction_name = entities.get("junction_names", ["selected area"])[0]

            context = build_traffic_context(junctions[0], data_context)

            if not context:
                return "Traffic data unavailable."

            actions = generate_mitigation_actions(
                context["junction_id"],
                context["congestion_level"],
                confidence=0.75,
                vehicle_count=context["vehicle_count"],
                avg_speed=context["avg_speed"],
            )

            response = (
                f"🚦 Congestion Reduction Plan for "
                f"**{junction_name}**\n\n"
                f"Current Status:\n"
                f"• Congestion: {context['congestion_level']}\n"
                f"• Vehicles: {context['vehicle_count']}\n"
                f"• Avg Speed: {context['avg_speed']:.1f} km/h\n\n"
                f"Recommended Actions:\n"
            )

            for action in actions["actions"][:4]:
                response += f"• {action['action']}\n"

            response += (
                "\nThese measures can significantly "
                "reduce queue formation and improve "
                "traffic flow."
            )

            return response

        return (
            "Please specify a location such as "
            "'How can congestion at HITEC City be reduced?'"
        )
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
            f"📋 Traffic Management Plan for "
            f"{junction_name}\n\n"
            f"Current Situation:\n"
            f"• Congestion Level: "
            f"{context['congestion_level']}\n"
            f"• Vehicle Count: "
            f"{context['vehicle_count']}\n"
            f"• Average Speed: "
            f"{context['avg_speed']:.1f} km/h\n\n"
            f"Recommended Actions:\n"
        )

        for idx, action in enumerate(actions["actions"][:4], start=1):

            response += f"{idx}. {action['action']}\n"

        response += (
            "\nExpected Outcome:\n"
            "Reduced congestion, improved traffic flow, "
            "better travel times, and proactive traffic "
            "management."
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

                vehicle_reason = ""

                if vehicles > 400:
                    vehicle_reason = (
                        "Vehicle count is significantly above normal levels."
                    )
                elif vehicles > 150:
                    vehicle_reason = "Vehicle volume is moderately elevated."
                else:
                    vehicle_reason = "Vehicle volume remains relatively manageable."

                speed_reason = ""
                if speed < 15:
                    speed_reason = (
                        "Average speed is extremely low, indicating severe delays."
                    )
                elif speed < 30:
                    speed_reason = "Traffic speeds are reduced, indicating congestion."
                else:
                    speed_reason = "Traffic speeds remain within acceptable limits."

                analysis = ""

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
                        "Vehicle density is increasing and congestion "
                        "may worsen during peak periods."
                    )

                    recommendation = (
                        "Preventive traffic management actions are advised."
                    )

                else:

                    analysis = (
                        "Traffic is currently flowing reasonably well. "
                        "Vehicle density remains manageable and no major "
                        "congestion indicators are present."
                    )

                    recommendation = "Continue monitoring traffic conditions."

                return (
                    f"🧠 Traffic Analysis for "
                    f"{entities.get('junction_names', ['Selected Area'])[0]}\n\n"
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

        return (
            "Please specify a location such as " "'Why is traffic bad at HITEC City?'"
        )

    elif intent == "city_summary_query":

        response = "📊 Citywide Traffic Summary\n\n"

        # Traffic overview
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
                f"📈 Overall Assessment\n\n"
                f"{status}\n\n"
            )

        # Hotspots
        if "hotspots" in data_context:

            hotspots = data_context["hotspots"][:3]

            response += "🔥 Top Congestion Hotspots\n\n"

            for spot in hotspots:

                response += (
                    f"• Cluster {spot['cluster_id']} "
                    f"({spot['severity']} Severity)\n"
                )

            response += "\n"

        # Anomalies
        if "anomalies" in data_context:

            anomalies = data_context["anomalies"]

            high_risk = [
                a for a in anomalies if a["risk_level"] in ["High", "Critical"]
            ]

            unique_anomaly_junctions = len(set(a["junction_id"] for a in high_risk))

            response += (
                f"⚠️ Junctions With Active Anomalies: "
                f"{unique_anomaly_junctions}\n\n"
            )

            if high_risk:

                response += "Top Risk Locations:\n\n"

                for a in high_risk[:3]:

                    junction_name = JUNCTION_NAMES.get(
                        a["junction_id"], a["junction_id"]
                    )

                    response += f"• {junction_name} " f"({a['risk_level']} Risk)\n"

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

            return "No significant operational issues " "detected at this time."

        response = "🚔 Traffic Operations Brief\n\n" "Priority Locations:\n\n"

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

            return "No traffic data available " "for intervention analysis."

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
            "These locations currently show the highest "
            "traffic risk and should be prioritized for "
            "signal optimization, route diversion, and "
            "traffic personnel deployment."
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
                f"Origin:\n"
                f"• {entities.get('origin_name', origin)}\n\n"
                f"Destination:\n"
                f"• {entities.get('destination_name', destination)}\n\n"
                f"🚀 Recommended Route:\n"
                f"• {' → '.join(recommended_route_names)}\n\n"
                f"Distance:\n"
                f"• {least_congested['total_distance_km']} km\n\n"
                f"Estimated Time:\n"
                f"• {least_congested['estimated_minutes']} min\n\n"
                f"Alternative Route:\n"
                f"• {' → '.join(alternative_route_names)}\n\n"
                f"Recommendation:\n"
                f"• {result['recommendation']}"
            )

            return response

        except Exception as e:

            return f"⚠️ Unable to calculate route.\n\n" f"Error: {str(e)}"

    elif intent == "prediction_query":

        junctions = entities.get("junction_ids", [])

        if not junctions:

            return (
                "Please specify a location such as " "'Predict traffic at HITEC City'."
            )

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
            f"🔮 Traffic Forecast for "
            f"{junction_name}\n\n"
            f"Current Conditions:\n"
            f"• Vehicles: "
            f"{prediction_features['vehicle_count']}\n"
            f"• Avg Speed: "
            f"{prediction_features['avg_speed']:.1f} km/h\n\n"
            f"Predicted Congestion:\n"
            f"• {prediction['predicted_level']}\n\n"
            f"Confidence:\n"
            f"• {prediction['confidence']:.1%}\n\n"
            f"Recommended Actions:\n"
        )

        for action in mitigation["actions"][:3]:

            response += f"• {action['action']}\n"

        return response

    elif intent == "executive_report_query":
        return generate_executive_report(data_context)

    elif intent == "management_plan_query":
        from engine.management_planner import generate_management_plan

        plan = generate_management_plan(data_context)
        response = "📋 Traffic Management Plan\n\n"

        for item in plan:
            response += f"• {item}\n"

        return response

    elif intent == "anomaly_query":

        anomalies = build_anomaly_context(data_context, top_n=3)

        if not anomalies:

            return "✅ No major traffic anomalies " "detected at this time."

        response = "⚠️ Current Traffic Anomalies\n\n"

        for idx, anomaly in enumerate(anomalies, start=1):

            location = JUNCTION_NAMES.get(
                anomaly["junction_id"], anomaly["junction_id"]
            )

            response += (
                f"{idx}. {location}\n"
                f"• Risk Level: "
                f"{anomaly['risk_level']}\n"
                f"• Vehicles: "
                f"{anomaly['vehicle_count']}\n"
                f"• Avg Speed: "
                f"{anomaly['avg_speed']:.1f} km/h\n"
                f"• Score: "
                f"{anomaly['anomaly_score']:.3f}\n\n"
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

            return "✅ No major congestion hotspots " "detected currently."

        response = "🔥 Current Congestion Hotspots\n\n"

        for idx, hotspot in enumerate(hotspots, start=1):

            response += (
                f"{idx}. Cluster {hotspot['cluster_id']}\n"
                f"• Severity: {hotspot['severity']}\n"
                f"• Avg Vehicles: "
                f"{hotspot['avg_vehicles']:.0f}\n"
                f"• Avg Speed: "
                f"{hotspot['avg_speed']:.1f} km/h\n"
                f"• Peak Hours: "
                f"{', '.join(map(str, hotspot['peak_hours']))}\n\n"
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
