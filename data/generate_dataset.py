"""
Synthetic Traffic Dataset Generator.
Generates realistic traffic data for 20 junctions over 30 days.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_PATH,
    DATASET_ROWS,
    NUM_JUNCTIONS,
    DATE_RANGE_DAYS,
    LAT_RANGE,
    LON_RANGE,
    PEAK_HOURS,
    WEATHER_CONDITIONS,
    WEATHER_WEIGHTS,
    CONGESTION_THRESHOLDS,
    RANDOM_STATE,
    JUNCTION_PREFIX,
)

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# ──────────────────────────── Junction Layout ────────────────────────────


def generate_junction_locations(n_junctions: int) -> dict:
    """Generate fixed junction locations on a grid."""
    junctions = {}
    grid_size = int(np.ceil(np.sqrt(n_junctions)))
    lats = np.linspace(LAT_RANGE[0], LAT_RANGE[1], grid_size)
    lons = np.linspace(LON_RANGE[0], LON_RANGE[1], grid_size)

    idx = 0
    for lat in lats:
        for lon in lons:
            if idx >= n_junctions:
                break
            jid = f"{JUNCTION_PREFIX}{idx + 1:03d}"
            # Add small random offset for realism
            junctions[jid] = {
                "latitude": round(lat + np.random.uniform(-0.005, 0.005), 6),
                "longitude": round(lon + np.random.uniform(-0.005, 0.005), 6),
            }
            idx += 1
    return junctions


# ──────────────────────────── Junction Names ────────────────────────────

JUNCTION_NAMES = {
    "J001": "MG Road Junction",
    "J002": "Ameerpet Circle",
    "J003": "Begumpet Flyover",
    "J004": "Secunderabad Station",
    "J005": "HITEC City Signal",
    "J006": "Madhapur Junction",
    "J007": "Gachibowli Crossing",
    "J008": "Jubilee Hills Checkpost",
    "J009": "Banjara Hills Road No.1",
    "J010": "Tank Bund Junction",
    "J011": "Mehdipatnam Bus Stand",
    "J012": "LB Nagar Crossroads",
    "J013": "Uppal Ring Road",
    "J014": "Kukatpally Y Junction",
    "J015": "Miyapur Metro",
    "J016": "Charminar Circle",
    "J017": "Abids GPO Junction",
    "J018": "Nampally Station",
    "J019": "Paradise Circle",
    "J020": "Dilsukhnagar Junction",
}


# ──────────────────────────── Data Generation ────────────────────────────


def compute_congestion_level(
    vehicle_count: int, avg_speed: float, is_peak: bool, weather: str
) -> str:
    """Determine congestion level based on multiple factors."""
    score = 0

    # Vehicle count contribution
    if vehicle_count > CONGESTION_THRESHOLDS["medium_max_vehicles"]:
        score += 3
    elif vehicle_count > CONGESTION_THRESHOLDS["low_max_vehicles"]:
        score += 2
    else:
        score += 1

    # Speed contribution (inverse)
    if avg_speed < 15:
        score += 3
    elif avg_speed < 30:
        score += 2
    else:
        score += 1

    # Peak hour bonus
    if is_peak:
        score += 1

    # Weather bonus
    if weather in ("Rain", "Fog"):
        score += 1

    # Classify
    if score >= 7:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"


def generate_dataset(n_rows: int = DATASET_ROWS) -> pd.DataFrame:
    """Generate the complete synthetic traffic dataset."""
    junctions = generate_junction_locations(NUM_JUNCTIONS)
    junction_ids = list(junctions.keys())

    # Some junctions are inherently busier (multiplier)
    junction_busyness = {jid: np.random.uniform(0.6, 1.4) for jid in junction_ids}
    # Make HITEC City and Ameerpet busiest
    junction_busyness["J005"] = 1.5
    junction_busyness["J002"] = 1.4

    # Construction zones (sticky per junction for the month)
    construction_junctions = set(random.sample(junction_ids, 3))

    start_date = datetime(2026, 5, 1)
    records = []

    for _ in range(n_rows):
        # Random timestamp within date range
        offset_minutes = random.randint(0, DATE_RANGE_DAYS * 24 * 60 - 1)
        ts = start_date + timedelta(minutes=offset_minutes)
        hour = ts.hour
        day_of_week = ts.weekday()  # 0=Monday

        # Select junction
        jid = random.choice(junction_ids)
        loc = junctions[jid]
        busyness = junction_busyness[jid]

        # Peak hour
        is_peak = hour in PEAK_HOURS

        # Weather (time-correlated: morning fog, afternoon rain)
        if 5 <= hour <= 8:
            weather = np.random.choice(WEATHER_CONDITIONS, p=[0.40, 0.15, 0.30, 0.15])
        elif 14 <= hour <= 18:
            weather = np.random.choice(WEATHER_CONDITIONS, p=[0.45, 0.30, 0.05, 0.20])
        else:
            weather = np.random.choice(WEATHER_CONDITIONS, p=WEATHER_WEIGHTS)

        # Events (weekends + random)
        is_event = (
            day_of_week >= 5 and random.random() < 0.15
        ) or random.random() < 0.03

        # Construction
        is_construction = jid in construction_junctions

        # Vehicle count (base + peak + weather + events + busyness)
        base_vehicles = 120
        if is_peak:
            base_vehicles += random.randint(80, 180)
        if day_of_week < 5:  # Weekdays busier
            base_vehicles += random.randint(20, 60)
        if weather == "Rain":
            base_vehicles += random.randint(10, 40)
        if is_event:
            base_vehicles += random.randint(50, 150)
        if is_construction:
            base_vehicles += random.randint(20, 60)

        # Night hours (12am-5am) — very low traffic
        if 0 <= hour <= 5:
            base_vehicles = int(base_vehicles * 0.2)

        vehicle_count = max(10, int(base_vehicles * busyness + np.random.normal(0, 25)))

        # Average speed (inversely correlated with vehicle count)
        base_speed = 55 - (vehicle_count / 10)
        if weather in ("Rain", "Fog"):
            base_speed -= random.randint(5, 15)
        if is_construction:
            base_speed -= random.randint(5, 10)
        avg_speed = max(5, round(base_speed + np.random.normal(0, 5), 1))

        # Congestion level
        congestion = compute_congestion_level(
            vehicle_count, avg_speed, is_peak, weather
        )

        records.append(
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "junction_id": jid,
                "junction_name": JUNCTION_NAMES.get(jid, jid),
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "vehicle_count": vehicle_count,
                "avg_speed": avg_speed,
                "weather": weather,
                "day_of_week": day_of_week,
                "hour": hour,
                "is_peak": int(is_peak),
                "is_event": int(is_event),
                "is_construction": int(is_construction),
                "congestion_level": congestion,
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ──────────────────────────── Main ────────────────────────────

if __name__ == "__main__":
    print("🚦 Generating synthetic traffic dataset...")
    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"✅ Dataset saved to {DATASET_PATH}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n📊 Congestion distribution:")
    print(df["congestion_level"].value_counts().to_string())
    print(f"\n🌤️ Weather distribution:")
    print(df["weather"].value_counts().to_string())
    print(f"\n📍 Junctions: {df['junction_id'].nunique()}")
