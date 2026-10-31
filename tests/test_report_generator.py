import pandas as pd
import pytest

from config import DATASET_PATH
import reports.report_generator as report_generator


@pytest.fixture
def traffic_df(monkeypatch):
    # Avoid writing test-generated reports to the application database.
    monkeypatch.setattr(
        report_generator,
        "insert_report",
        lambda report: None,
    )
    return pd.read_csv(DATASET_PATH)


def test_daily_report_contains_all_five_classes(traffic_df):
    timestamps = pd.to_datetime(
        traffic_df["timestamp"],
        errors="coerce",
    )
    report_date = timestamps.max().strftime("%Y-%m-%d")

    report = report_generator.generate_daily_report(
        traffic_df,
        report_date,
    )

    assert "error" not in report
    assert report["period_start"] == report_date
    assert report["period_end"] == report_date

    content = report["content"]

    for label in ["Free Flow", "Low", "Moderate", "High", "Severe"]:
        assert f"| {label} |" in content

    expected_records = int(
        (timestamps.dt.strftime("%Y-%m-%d") == report_date).sum()
    )
    assert f"| Total Records | {expected_records} |" in content


def test_weekly_report_covers_latest_seven_dates(traffic_df):
    timestamps = pd.to_datetime(
        traffic_df["timestamp"],
        errors="coerce",
    )
    available_dates = sorted(
        timestamps.dropna().dt.strftime("%Y-%m-%d").unique().tolist()
    )
    expected_dates = available_dates[-7:]

    report = report_generator.generate_weekly_report(traffic_df)

    assert "error" not in report
    assert report["period_start"] == expected_dates[0]
    assert report["period_end"] == expected_dates[-1]
    assert len(report["content"]) > 0
    assert "High/Severe (%)" in report["content"]


def test_daily_report_returns_error_for_missing_date(traffic_df):
    report = report_generator.generate_daily_report(
        traffic_df,
        "1900-01-01",
    )

    assert "error" in report
    assert "No data found" in report["error"]
