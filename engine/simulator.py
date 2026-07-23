"""
Traffic Simulation Module.
Generates live traffic updates with random incidents for dashboard simulation.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    NUM_JUNCTIONS, JUNCTION_PREFIX, PEAK_HOURS,
    WEATHER_CONDITIONS, WEATHER_WEIGHTS,
    SIMULATION_INCIDENT_PROBABILITY, LAT_RANGE, LON_RANGE,
)
from data.generate_dataset import (
    generate_junction_locations, compute_congestion_level,
)


def simulate_traffic_step(
    junctions: dict = None,
    current_hour: int = None,
) -> list[dict]:
    """
    Generate one step of simulated traffic data for all junctions.
    Returns a list of traffic records.
    """
    if junctions is None:
        junctions = generate_junction_locations(NUM_JUNCTIONS)

    if current_hour is None:
        current_hour = datetime.now().hour

    now = datetime.now()
    is_peak = current_hour in PEAK_HOURS
    day_of_week = now.weekday()

    records = []
    for jid, loc in junctions.items():
        weather = np.random.choice(WEATHER_CONDITIONS, p=WEATHER_WEIGHTS)
        is_event = random.random() < 0.05
        is_construction = random.random() < 0.15

        # Base vehicles
        base = 120
        if is_peak:
            base += random.randint(80, 180)
        if day_of_week < 5:
            base += random.randint(20, 60)
        if weather in ("Rain", "Fog"):
            base += random.randint(10, 40)
        if is_event:
            base += random.randint(50, 150)

        # Random incident
        incident = random.random() < SIMULATION_INCIDENT_PROBABILITY
        if incident:
            base += random.randint(100, 250)

        if 0 <= current_hour <= 5:
            base = int(base * 0.2)

        vehicle_count = max(10, int(base + np.random.normal(0, 25)))

        # Speed
        speed = max(5, round(55 - vehicle_count / 10 + np.random.normal(0, 5), 1))
        if weather in ("Rain", "Fog"):
            speed = max(5, speed - random.randint(5, 15))

        congestion = compute_congestion_level(vehicle_count, speed, is_peak, weather)

        records.append({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "junction_id": jid,
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "vehicle_count": vehicle_count,
            "avg_speed": speed,
            "weather": weather,
            "day_of_week": day_of_week,
            "hour": current_hour,
            "is_peak": int(is_peak),
            "is_event": int(is_event),
            "is_construction": int(is_construction),
            "congestion_level": congestion,
            "has_incident": incident,
        })

    return records


def run_simulation(steps: int = 10) -> list[list[dict]]:
    """Run a multi-step traffic simulation."""
    junctions = generate_junction_locations(NUM_JUNCTIONS)
    all_steps = []
    hour = datetime.now().hour

    for i in range(steps):
        step_data = simulate_traffic_step(junctions, current_hour=(hour + i) % 24)
        all_steps.append(step_data)

    return all_steps
