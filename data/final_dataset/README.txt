HYDERABAD SMART AI TRAFFIC DATASET
====================================

Purpose:
Synthetic academic dataset for the Smart AI Traffic Congestion Reduction System.

Language:
All values and descriptions are in English.

Source:
Programmatically generated using Python, NumPy and Pandas.
This is NOT real traffic-sensor data.

Source datasets:
1. road_segments.csv
2. traffic_data.csv
3. weather_data.csv
4. accidents.csv
5. events.csv

Integrated ML dataset:
ml_traffic_dataset.csv

Relationships:
- road_segments.road_segment_id -> traffic_data.road_segment_id
- road_segments.road_segment_id -> accidents.road_segment_id
- road_segments.road_segment_id -> events.road_segment_id
- traffic_data timestamp/area -> weather_data timestamp/area
- ml_traffic_dataset integrates road, traffic, weather and incident information.

Prediction design:
Traffic observations are generated on a fixed hourly grid for each road segment.
The ML target is future_congestion_level, representing the next hourly congestion
condition on the same road segment.

For ML training, current/future congestion scores are NOT used as input features.
The target is future_congestion_level.

Congestion score bands:
0-20: Free Flow
21-40: Low
41-60: Moderate
61-80: High
81-100: Severe

The dataset is synthetic and must be described as synthetic in the project report.
