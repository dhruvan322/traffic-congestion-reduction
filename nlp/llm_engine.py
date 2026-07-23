"""
LLM Engine Module.
Integrates with Ollama for enhanced responses, with rule-based fallback.
"""

import logging
import requests
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT_SEC

logger = logging.getLogger(__name__)


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def query_ollama(prompt: str, context: str = "") -> dict:
    """
    Query Ollama LLM for enhanced traffic analysis response.
    Falls back to returning None if Ollama is unavailable.
    """
    if not check_ollama_available():
        logger.info("Ollama not available, using rule-based response.")
        return {"success": False, "error": "Ollama not available"}

    system_prompt = (
        "You are an AI traffic intelligence assistant for Hyderabad, India. "
        "Provide helpful, concise answers about traffic conditions, routes, "
        "and congestion patterns. Use the provided data context to give "
        "accurate answers."
    )

    full_prompt = f"{context}\n\nUser: {prompt}" if context else prompt

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "system": system_prompt,
                "options": {
                    "num_predict": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE,
                },
                "stream": False,
            },
            timeout=LLM_TIMEOUT_SEC,
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "response": result.get("response", ""),
                "model": OLLAMA_MODEL,
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}

    except requests.Timeout:
        logger.warning("Ollama request timed out.")
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return {"success": False, "error": str(e)}


def build_traffic_prompt(intent: str, data_summary: dict) -> str:
    """Build a context-rich prompt for the LLM based on intent and data."""
    context_parts = [
        "Traffic Data Context:",
        f"- Total records: {data_summary.get('total_records', 'N/A')}",
        f"- Average vehicles: {data_summary.get('avg_vehicles', 'N/A')}",
        f"- Average speed: {data_summary.get('avg_speed', 'N/A')} km/h",
    ]

    if "high_congestion_pct" in data_summary:
        context_parts.append(
            f"- High congestion: {data_summary['high_congestion_pct']}%"
        )

    if "top_junctions" in data_summary:
        context_parts.append(
            f"- Busiest junctions: {', '.join(data_summary['top_junctions'])}"
        )

    return "\n".join(context_parts)
