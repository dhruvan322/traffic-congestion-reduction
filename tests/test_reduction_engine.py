from engine.reduction_engine import generate_integrated_recommendations


def print_scenario(name, result):
    print()
    print("-" * 70)
    print(f"SCENARIO: {name}")
    print("-" * 70)

    print(f"ROAD SEGMENT: {result['road_segment_id']}")
    print(f"OVERALL PRIORITY: {result['overall_priority']}")

    prediction = result["prediction"]
    print(
        f"PREDICTION: {prediction['congestion_level']} "
        f"({prediction['confidence']:.1f}%)"
    )

    hotspot = result["hotspot"]
    print(f"HOTSPOT: {hotspot['detected']} " f"({hotspot['severity']})")

    anomaly = result["anomaly"]
    print(f"ANOMALY: {anomaly['detected']} " f"({anomaly['risk_level']})")

    route = result["route"]
    print(f"ALTERNATIVE ROUTE: " f"{route['alternative_available']}")

    print()
    print("RECOMMENDATIONS:")

    for i, recommendation in enumerate(result["recommendations"], start=1):
        print(
            f"{i}. "
            f"[{recommendation['priority']}] "
            f"{recommendation['category']}: "
            f"{recommendation['action']}"
        )

    print()
    print("DECISION SUMMARY:")
    print(result["decision_summary"])


def main():

    # ---------------------------------------------------------
    # Scenario 1: Free Flow
    # ---------------------------------------------------------
    result_1 = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0001",
            "congestion_level": "Free Flow",
            "confidence": 90.0,
        },
        hotspot={
            "detected": False,
            "severity": None,
        },
        anomaly={
            "detected": False,
            "risk_level": "Low",
        },
        route_conditions={
            "alternative_available": False,
            "shortest_time_minutes": 10.0,
            "traffic_aware_time_minutes": 10.0,
        },
    )

    print_scenario("FREE FLOW", result_1)

    assert result_1["road_segment_id"] == "RS0001"
    assert result_1["overall_priority"] == "Low"
    assert len(result_1["recommendations"]) >= 1

    # ---------------------------------------------------------
    # Scenario 2: Moderate Congestion + Hotspot
    # ---------------------------------------------------------
    result_2 = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0031",
            "congestion_level": "Moderate",
            "confidence": 82.0,
        },
        hotspot={
            "detected": True,
            "severity": "High",
        },
        anomaly={
            "detected": False,
            "risk_level": "Low",
        },
        route_conditions={
            "alternative_available": True,
            "shortest_time_minutes": 20.0,
            "traffic_aware_time_minutes": 16.0,
        },
    )

    print_scenario("MODERATE CONGESTION + HOTSPOT", result_2)

    assert result_2["road_segment_id"] == "RS0031"

    assert result_2["overall_priority"] in {
        "Medium",
        "High",
        "Critical",
    }

    assert any(
        r["category"] == "Hotspot Management" for r in result_2["recommendations"]
    )

    # ---------------------------------------------------------
    # Scenario 3: High Congestion + Alternative Route
    # ---------------------------------------------------------
    result_3 = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0019",
            "congestion_level": "High",
            "confidence": 88.0,
        },
        hotspot={
            "detected": True,
            "severity": "High",
        },
        anomaly={
            "detected": False,
            "risk_level": "Low",
        },
        route_conditions={
            "alternative_available": True,
            "shortest_time_minutes": 25.0,
            "traffic_aware_time_minutes": 18.0,
        },
    )

    print_scenario("HIGH CONGESTION + ALTERNATIVE ROUTE", result_3)

    assert result_3["road_segment_id"] == "RS0019"

    assert result_3["overall_priority"] in {
        "High",
        "Critical",
    }

    assert any(r["category"] == "Route Diversion" for r in result_3["recommendations"])

    # ---------------------------------------------------------
    # Scenario 4: Severe Congestion + Critical Anomaly
    # ---------------------------------------------------------
    result_4 = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0028",
            "congestion_level": "Severe",
            "confidence": 94.0,
        },
        hotspot={
            "detected": True,
            "severity": "Critical",
        },
        anomaly={
            "detected": True,
            "risk_level": "Critical",
        },
        route_conditions={
            "alternative_available": True,
            "shortest_time_minutes": 30.0,
            "traffic_aware_time_minutes": 19.0,
        },
    )

    print_scenario("SEVERE CONGESTION + CRITICAL ANOMALY", result_4)

    assert result_4["road_segment_id"] == "RS0028"
    assert result_4["overall_priority"] == "Critical"

    assert any(r["category"] == "Route Diversion" for r in result_4["recommendations"])

    assert any(r["category"] == "Anomaly Response" for r in result_4["recommendations"])

    # ---------------------------------------------------------
    # Scenario 5: High Congestion + No Alternative
    # ---------------------------------------------------------
    result_5 = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0059",
            "congestion_level": "High",
            "confidence": 86.0,
        },
        hotspot={
            "detected": False,
            "severity": None,
        },
        anomaly={
            "detected": False,
            "risk_level": "Low",
        },
        route_conditions={
            "alternative_available": False,
            "shortest_time_minutes": 22.0,
            "traffic_aware_time_minutes": 22.0,
        },
    )

    print_scenario("HIGH CONGESTION + NO ALTERNATIVE", result_5)

    assert result_5["road_segment_id"] == "RS0059"

    assert result_5["overall_priority"] in {
        "Medium",
        "High",
        "Critical",
    }

    assert any(
        r["category"] == "Traffic Management" for r in result_5["recommendations"]
    )

    # ---------------------------------------------------------
    # Determinism Test
    # ---------------------------------------------------------
    repeat_a = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0019",
            "congestion_level": "High",
            "confidence": 88.0,
        },
        hotspot={
            "detected": True,
            "severity": "High",
        },
        anomaly={
            "detected": False,
            "risk_level": "Low",
        },
        route_conditions={
            "alternative_available": True,
            "shortest_time_minutes": 25.0,
            "traffic_aware_time_minutes": 18.0,
        },
    )

    repeat_b = generate_integrated_recommendations(
        prediction={
            "road_segment_id": "RS0019",
            "congestion_level": "High",
            "confidence": 88.0,
        },
        hotspot={
            "detected": True,
            "severity": "High",
        },
        anomaly={
            "detected": False,
            "risk_level": "Low",
        },
        route_conditions={
            "alternative_available": True,
            "shortest_time_minutes": 25.0,
            "traffic_aware_time_minutes": 18.0,
        },
    )

    assert repeat_a == repeat_b

    print()
    print("-" * 70)
    print("REDUCTION ENGINE TEST: PASS")
    print("-" * 70)


if __name__ == "__main__":
    main()
