"""
Global configuration for the AI Traffic Intelligence Platform.
All constants, paths, thresholds, and settings in one place.
"""

import os

# ──────────────────────────── Paths ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
DB_PATH = os.path.join(BASE_DIR, "traffic_intelligence.db")
DATASET_PATH = os.path.join(
    DATA_DIR,
    "final_dataset",
    "ml_traffic_dataset.csv",
)

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────── Dataset ────────────────────────────
DATASET_ROWS = 10000
NUM_JUNCTIONS = 20
JUNCTION_PREFIX = "J"
DATE_RANGE_DAYS = 30

# ──────────────────────────── Map / Geospatial ────────────────────────────
# Center: Hyderabad, India
MAP_CENTER_LAT = 17.385
MAP_CENTER_LON = 78.4867
LAT_RANGE = (17.35, 17.45)
LON_RANGE = (78.40, 78.55)
MAP_ZOOM = 12

# ──────────────────────────── Traffic Thresholds ────────────────────────────
CONGESTION_THRESHOLDS = {
    "low_max_vehicles": 180,
    "medium_max_vehicles": 350,
    # Above medium_max_vehicles → High
}
PEAK_HOURS = [8, 9, 10, 17, 18, 19]
SPEED_LIMITS = {
    "free_flow": 60,  # km/h
    "moderate": 30,
    "congested": 15,
}

# ──────────────────────────── ML Models ────────────────────────────
PREDICTION_FEATURES = [
    "hour",
    "day_of_week",
    "vehicle_count",
    "avg_speed",
    "is_peak",
    "is_event",
    "is_construction",
    "weather_encoded",
    "junction_encoded",
]
CONGESTION_LABELS = ["Low", "Medium", "High"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ──────────────────────────── Anomaly Detection ────────────────────────────
ANOMALY_CONTAMINATION = 0.05  # Expected proportion of anomalies
ZSCORE_THRESHOLD = 3.0

# ──────────────────────────── Clustering ────────────────────────────
DBSCAN_EPS = 0.01  # ~1 km in lat/lon degrees
DBSCAN_MIN_SAMPLES = 5
KMEANS_N_CLUSTERS = 5

# ──────────────────────────── Route Optimization ────────────────────────────
ROUTE_SPEED_FACTOR = 1.5  # Weight for traffic speed in edge cost
ROUTE_DISTANCE_FACTOR = 1.0

# ──────────────────────────── Simulation ────────────────────────────
SIMULATION_INTERVAL_SEC = 2.0
SIMULATION_INCIDENT_PROBABILITY = 0.05

# FastAPI Backend
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)
API_TIMEOUT_SECONDS = 30

# ──────────────────────────── LLM / Ollama ────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TIMEOUT_SEC = 30

# ──────────────────────────── NLP ────────────────────────────
SUPPORTED_LANGUAGES = ["en", "hi", "te"]
DEFAULT_LANGUAGE = "en"
CHAT_CONTEXT_WINDOW = 5  # Number of past turns for memory

# ──────────────────────────── Alerts ────────────────────────────
ALERT_CONGESTION_THRESHOLD = "High"
ALERT_ANOMALY_RISK_THRESHOLD = "High"
MAX_ALERTS_DISPLAY = 20

# ──────────────────────────── Weather ────────────────────────────
WEATHER_CONDITIONS = ["Clear", "Rain", "Fog", "Cloudy"]
WEATHER_WEIGHTS = [0.55, 0.20, 0.10, 0.15]

# ──────────────────────────── Dashboard ────────────────────────────
PAGE_TITLE = "🚦 AI Traffic Intelligence Platform"
PAGE_ICON = "🚦"
LAYOUT = "wide"

# ──────────────────────────── Congestion Reduction Engine ────────────────────────────

# Minimum confidence required before showing recommendations
REDUCTION_CONFIDENCE_THRESHOLD = 0.60

# Signal timing adjustment used by the AI recommendations
SIGNAL_TIMING_INCREMENT = 15

# Priority thresholds
REDUCTION_PRIORITY_LEVELS = {
    "critical": 25,
    "high": 15,
    "medium": 8,
    "low": 0,
}

# Optional route names for recommendation explanations
ALTERNATE_ROUTE_NAMES = {
    "J005": "Madhapur Corridor",
    "J006": "Cyber Towers Road",
    "J007": "Financial District Road",
    "J014": "NH-65 Bypass",
    "J012": "Inner Ring Road",
}
