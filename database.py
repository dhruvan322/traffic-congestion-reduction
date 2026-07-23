"""
SQLite database layer for the AI Traffic Intelligence Platform.
8 tables: traffic_records, predictions, hotspots, anomalies,
          chat_history, reports, route_history, alerts.
"""

import sqlite3
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from config import DB_PATH

logger = logging.getLogger(__name__)


# ──────────────────────────── Schema ────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS traffic_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    junction_id TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    vehicle_count INTEGER NOT NULL,
    avg_speed REAL NOT NULL,
    weather TEXT NOT NULL,
    day_of_week INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    is_peak INTEGER NOT NULL DEFAULT 0,
    is_event INTEGER NOT NULL DEFAULT 0,
    is_construction INTEGER NOT NULL DEFAULT 0,
    congestion_level TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    junction_id TEXT NOT NULL,
    predicted_level TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities_json TEXT,
    explanation TEXT,
    model_used TEXT
);

CREATE TABLE IF NOT EXISTS hotspots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    cluster_id INTEGER NOT NULL,
    severity TEXT NOT NULL,
    peak_hours_json TEXT
);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT NOT NULL,
    junction_id TEXT NOT NULL,
    anomaly_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    user_query TEXT NOT NULL,
    detected_language TEXT DEFAULT 'en',
    intent TEXT,
    entities_json TEXT,
    module_results_json TEXT,
    response TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type TEXT NOT NULL,
    period_start TEXT,
    period_end TEXT,
    content TEXT NOT NULL,
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS route_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin TEXT NOT NULL,
    destination TEXT NOT NULL,
    route_path_json TEXT,
    distance_km REAL,
    estimated_minutes REAL,
    time_saved_minutes REAL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    junction_id TEXT,
    is_read INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traffic_junction ON traffic_records(junction_id);
CREATE INDEX IF NOT EXISTS idx_traffic_timestamp ON traffic_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_alerts_unread ON alerts(is_read);
"""


# ──────────────────────────── Connection ────────────────────────────

@contextmanager
def get_connection():
    """Context manager for safe SQLite connections."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_database():
    """Initialize the database schema. Safe to call multiple times."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info("Database initialized successfully.")


# ──────────────────────────── Traffic Records ────────────────────────────

def insert_traffic_records(records: list[dict]):
    """Bulk insert traffic records."""
    sql = """
        INSERT INTO traffic_records 
        (timestamp, junction_id, latitude, longitude, vehicle_count,
         avg_speed, weather, day_of_week, hour, is_peak, is_event,
         is_construction, congestion_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.executemany(sql, [
            (r["timestamp"], r["junction_id"], r["latitude"], r["longitude"],
             r["vehicle_count"], r["avg_speed"], r["weather"], r["day_of_week"],
             r["hour"], int(r["is_peak"]), int(r["is_event"]),
             int(r["is_construction"]), r["congestion_level"])
            for r in records
        ])


def get_traffic_records(junction_id=None, limit=1000):
    """Retrieve traffic records, optionally filtered by junction."""
    with get_connection() as conn:
        if junction_id:
            rows = conn.execute(
                "SELECT * FROM traffic_records WHERE junction_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (junction_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM traffic_records ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


def get_traffic_stats():
    """Get aggregate traffic statistics."""
    with get_connection() as conn:
        row = conn.execute("""
            SELECT 
                COUNT(*) as total_records,
                AVG(vehicle_count) as avg_vehicles,
                AVG(avg_speed) as avg_speed,
                SUM(CASE WHEN congestion_level = 'High' THEN 1 ELSE 0 END) as high_count,
                SUM(CASE WHEN congestion_level = 'Medium' THEN 1 ELSE 0 END) as medium_count,
                SUM(CASE WHEN congestion_level = 'Low' THEN 1 ELSE 0 END) as low_count
            FROM traffic_records
        """).fetchone()
    return dict(row) if row else {}


# ──────────────────────────── Predictions ────────────────────────────

def insert_prediction(prediction: dict):
    """Insert a prediction record."""
    sql = """
        INSERT INTO predictions 
        (timestamp, junction_id, predicted_level, confidence,
         probabilities_json, explanation, model_used)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            prediction.get("timestamp", datetime.now().isoformat()),
            prediction["junction_id"],
            prediction["predicted_level"],
            prediction["confidence"],
            json.dumps(prediction.get("probabilities", {})),
            prediction.get("explanation", ""),
            prediction.get("model_used", ""),
        ))


def get_recent_predictions(limit=50):
    """Get recent predictions."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────── Hotspots ────────────────────────────

def insert_hotspots(hotspots: list[dict]):
    """Bulk insert detected hotspots."""
    sql = """
        INSERT INTO hotspots
        (detected_at, latitude, longitude, cluster_id, severity, peak_hours_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.executemany(sql, [
            (h.get("detected_at", datetime.now().isoformat()),
             h["latitude"], h["longitude"], h["cluster_id"],
             h["severity"], json.dumps(h.get("peak_hours", [])))
            for h in hotspots
        ])


def get_hotspots(limit=100):
    """Get detected hotspots."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM hotspots ORDER BY detected_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────── Anomalies ────────────────────────────

def insert_anomaly(anomaly: dict):
    """Insert an anomaly record."""
    sql = """
        INSERT INTO anomalies
        (detected_at, junction_id, anomaly_type, risk_level, description)
        VALUES (?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            anomaly.get("detected_at", datetime.now().isoformat()),
            anomaly["junction_id"],
            anomaly["anomaly_type"],
            anomaly["risk_level"],
            anomaly.get("description", ""),
        ))


def get_anomalies(limit=50):
    """Get recent anomalies."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM anomalies ORDER BY detected_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────── Chat History ────────────────────────────

def insert_chat_turn(turn: dict):
    """Insert a chat conversation turn."""
    sql = """
        INSERT INTO chat_history
        (session_id, turn_number, user_query, detected_language,
         intent, entities_json, module_results_json, response, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            turn["session_id"],
            turn["turn_number"],
            turn["user_query"],
            turn.get("detected_language", "en"),
            turn.get("intent", ""),
            json.dumps(turn.get("entities", {})),
            json.dumps(turn.get("module_results", {})),
            turn.get("response", ""),
            turn.get("timestamp", datetime.now().isoformat()),
        ))


def get_chat_history(session_id: str, limit=5):
    """Get recent chat turns for a session."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE session_id = ? "
            "ORDER BY turn_number DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_next_turn_number(session_id: str) -> int:
    """Get the next turn number for a session."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(turn_number) as max_turn FROM chat_history WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    return (row["max_turn"] or 0) + 1


# ──────────────────────────── Reports ────────────────────────────

def insert_report(report: dict):
    """Insert a generated report."""
    sql = """
        INSERT INTO reports
        (report_type, period_start, period_end, content, generated_at)
        VALUES (?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            report["report_type"],
            report.get("period_start", ""),
            report.get("period_end", ""),
            report["content"],
            report.get("generated_at", datetime.now().isoformat()),
        ))


def get_reports(report_type=None, limit=20):
    """Get generated reports, optionally filtered by type."""
    with get_connection() as conn:
        if report_type:
            rows = conn.execute(
                "SELECT * FROM reports WHERE report_type = ? "
                "ORDER BY generated_at DESC LIMIT ?",
                (report_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM reports ORDER BY generated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────── Route History ────────────────────────────

def insert_route(route: dict):
    """Insert a route query result."""
    sql = """
        INSERT INTO route_history
        (origin, destination, route_path_json, distance_km,
         estimated_minutes, time_saved_minutes, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            route["origin"],
            route["destination"],
            json.dumps(route.get("path", [])),
            route.get("distance_km", 0),
            route.get("estimated_minutes", 0),
            route.get("time_saved_minutes", 0),
            route.get("timestamp", datetime.now().isoformat()),
        ))


# ──────────────────────────── Alerts ────────────────────────────

def insert_alert(alert: dict):
    """Insert an alert."""
    sql = """
        INSERT INTO alerts
        (alert_type, severity, message, junction_id, is_read, created_at)
        VALUES (?, ?, ?, ?, 0, ?)
    """
    with get_connection() as conn:
        conn.execute(sql, (
            alert["alert_type"],
            alert["severity"],
            alert["message"],
            alert.get("junction_id", ""),
            alert.get("created_at", datetime.now().isoformat()),
        ))


def get_active_alerts(limit=20):
    """Get unread alerts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE is_read = 0 "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def mark_alert_read(alert_id: int):
    """Mark an alert as read."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE alerts SET is_read = 1 WHERE id = ?", (alert_id,)
        )


def get_alert_count():
    """Get count of unread alerts."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM alerts WHERE is_read = 0"
        ).fetchone()
    return row["cnt"] if row else 0


# ──────────────────────────── Error Logging ────────────────────────────

def log_error_to_db(module: str, error_msg: str):
    """Log a module error as a system alert."""
    insert_alert({
        "alert_type": "system_error",
        "severity": "Warning",
        "message": f"[{module}] {error_msg}",
        "junction_id": "",
    })
