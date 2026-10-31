# Project Viva Preparation Guide

## Smart AI Traffic Congestion Prediction and Reduction System Using Explainable Machine Learning and OpenStreetMap-Based Routing

This guide is based on the current project implementation and the training/evaluation outputs shared during development. Use the wording below as a preparation aid; answer in your own words and show the corresponding source code or output when asked.

---

## 1. Thirty-second project introduction

“Our project is a Hyderabad-focused traffic-intelligence prototype. It uses a synthetic, hourly traffic dataset for 60 road segments to forecast congestion at 1-hour, 2-hour, 3-hour and 6-hour horizons. It also provides SHAP-based prediction explanations, geographic hotspot grouping, anomaly screening, OpenStreetMap-based route calculation, REST API endpoints through FastAPI, and an interactive Streamlit dashboard. The current version is an offline decision-support prototype: it does not ingest live traffic feeds, control traffic signals, or measure real-world congestion reduction.”

## 2. Project objective

The objective is to bring several traffic-analysis capabilities into one application:

- Forecast future congestion classes for selected road segments.
- Explain model predictions using SHAP feature contributions.
- Group geographically close road segments into hotspots.
- Flag unusual traffic observations for review.
- Calculate candidate routes over an OpenStreetMap road graph, applying traffic-related costs when traffic information is available.
- Expose core operations through FastAPI and present them through Streamlit.

Do not claim that the prototype directly reduces traffic, changes signal timings, or provides live navigation.

## 3. Dataset and its origin

The project uses a custom synthetic Hyderabad-focused dataset rather than a collected live sensor feed. The integrated ML dataset contains 525,480 hourly records covering 60 road segments over 2025. The source tables are:

- `road_segments.csv`
- `traffic_data.csv`
- `weather_data.csv`
- `accidents.csv`
- `events.csv`

The integrated file is `data/final_dataset/ml_traffic_dataset.csv` and the prediction target is `future_congestion_level`, with five classes: Free Flow, Low, Moderate, High and Severe.

**Suggested answer — Why synthetic data?**

“We did not have access to a sufficiently detailed, consistently timestamped traffic dataset for the selected road segments. We therefore generated an interconnected synthetic dataset for academic experimentation. This allowed us to control the schema and create traffic, road, weather, accident and event features. The limitation is that model results on synthetic data do not establish real-world performance; validation on independent real traffic data would be needed.”

## 4. Machine-learning approach

### 4.1 Current training implementation

The current multi-horizon training code is:

`engine/multihorizon_forecaster.py`

Useful sections in the current file (line numbers may change if the file is edited):

- Dataset loading and validation: `load_final_dataset()`.
- Forecast target creation: `build_horizon_target()`.
- Feature/target preparation: `prepare_horizon_dataset()`.
- Chronological split: `chronological_split()`.
- Preprocessing: `build_preprocessor()`.
- XGBoost configuration: `build_model()`.
- Metric calculation: `evaluate_predictions()`.
- One-horizon training: `train_horizon()`.
- All four horizons: `train_all_horizons()`.
- Command-line entry point: `if __name__ == "__main__":`.

The current model configuration in `build_model()` is:

```python
XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=180,
    max_depth=8,
    learning_rate=0.10,
    subsample=0.90,
    colsample_bytree=0.90,
    min_child_weight=1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
```

### 4.2 Four prediction horizons

The target mapping in the current source is:

```python
HORIZONS = {
    1: 0,
    2: 1,
    3: 2,
    6: 5,
}
```

The existing `future_congestion_level` is used as the +1-hour target. For longer horizons, that target is shifted forward within each `road_segment_id` group. Since the dataset is hourly, the shifts correspond to the number of hourly records needed to reach the requested horizon.

**Suggested answer — Why separate models?**

“We train an independent XGBoost classifier for each forecast horizon. This lets each model learn the target associated with its own prediction distance and keeps the preprocessing pipeline and model artifacts separate for each horizon.”

### 4.3 Preprocessing

The current training pipeline uses:

- Median imputation for numerical columns.
- Most-frequent imputation for categorical columns.
- One-hot encoding for categorical values, with `handle_unknown="ignore"`.
- A preprocessing transformer fitted on the training partition, then applied to validation and test partitions.

The current training output reports 117 transformed features.

### 4.4 Chronological split

The data is split chronologically rather than randomly. For the +1-hour model, the recorded split is:

| Partition | Rows |
|---|---:|
| Training | 367,800 |
| Validation | 78,840 |
| Test | 78,840 |

The longer-horizon datasets contain slightly fewer usable records because the last observations in each road segment do not have the future target required by that horizon.

**Suggested answer — Why chronological splitting?**

“Traffic observations are time ordered. A chronological split trains on earlier records and evaluates on later records, which better approximates forecasting future observations and reduces direct leakage from randomly mixing future records into training.”

## 5. Current model results

The latest successful local training run produced these test metrics:

| Horizon | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| +1 hour | 63.60% | 48.52% | 48.33% | 61.82% |
| +2 hours | 63.66% | 48.89% | 49.32% | 62.10% |
| +3 hours | 63.53% | 49.06% | 48.71% | 61.86% |
| +6 hours | 63.58% | 48.21% | 48.29% | 61.52% |

The +2-hour model has the highest accuracy and weighted F1 among the four. The +3-hour model has the highest balanced accuracy. These metrics describe classification on the held-out synthetic test records; they are not percentages of traffic reduced.

### 5.1 Interpreting the class imbalance

For the +1-hour model, the recorded recall values were:

| Class | Recall |
|---|---:|
| Free Flow | 85.74% |
| Low | 13.02% |
| Moderate | 38.54% |
| High | 38.10% |
| Severe | 67.20% |

The results are uneven across classes. The majority-class baseline is approximately 54.23% for +1 hour. This is why accuracy should be discussed together with balanced accuracy and macro F1. Low-class recall is a notable weakness and should be acknowledged honestly.

**Suggested answer — Why is accuracy higher than macro F1?**

“The class distribution is imbalanced. The model performs better on more frequent classes than on some less frequent classes. Accuracy is affected by the number of records in each class, while macro F1 gives each class equal weight. The lower macro F1 and balanced accuracy reveal uneven class performance.”

## 6. Evaluation and reproducibility

The separate evaluation script is:

`tests/validate_multihorizon.py`

It loads the integrated dataset and saved models, evaluates each horizon on its held-out test partition, and prints:

- Accuracy, balanced accuracy, macro F1 and weighted F1.
- Per-class precision, recall, F1 and support.
- Confusion matrices.
- Test-period information and majority-class baseline.

The report is saved to:

`trained_models/multihorizon/validation_report.json`

The training summary is saved to:

`trained_models/multihorizon/multihorizon_summary.json`

### Commands

Run from the project root with the virtual environment activated.

Train all four models (this overwrites the existing multi-horizon artifacts):

```powershell
python .\engine\multihorizon_forecaster.py
```

Evaluate the saved models separately:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python .\tests\validate_multihorizon.py
Remove-Item Env:PYTHONPATH
```

Do not retrain immediately before a viva unless necessary. Training overwrites the saved multi-horizon models and their summary files. If demonstrating evaluation only, use the validation command.

## 7. Saved model artifacts

The current multi-horizon artifacts are organized as:

```text
trained_models/
└── multihorizon/
    ├── multihorizon_summary.json
    ├── validation_report.json
    ├── 1h/
    │   ├── xgb_congestion.json
    │   ├── traffic_preprocessor.joblib
    │   ├── label_encoder.joblib
    │   └── metadata.json
    ├── 2h/
    ├── 3h/
    └── 6h/
```

Each horizon stores an XGBoost model, preprocessing transformer, label encoder and metadata. The summary and validation JSON files contain aggregate and class-wise evaluation evidence.

## 8. Original Colab experiment versus current project training

The project also has an original Colab notebook named `Hyderabad_Traffic_Final_ML_Training.ipynb`. It documents an earlier next-hour experiment comparing Random Forest and XGBoost, including preprocessing, model comparison, classification report, confusion matrix and SHAP analysis.

Keep the experiments distinct:

- Original Colab experiment: Random Forest versus XGBoost for the initial next-hour model. The previously recorded XGBoost test accuracy was 55.45%.
- Current project implementation: four independently trained XGBoost models for +1h, +2h, +3h and +6h. The latest current-project metrics are listed in Section 5.

Do not present the original 55.45% result as the result of the current four-horizon models.

## 9. Explainable AI (SHAP)

SHAP is used to explain how input features contribute to a model prediction. In the original Colab experiment, global feature importance was also examined. Feature contribution is an explanation of model behaviour, not proof that a feature causes congestion or that changing it will reduce traffic.

**Suggested answer — Why SHAP?**

“A predicted congestion class alone may be difficult to interpret. SHAP helps show which input features contributed to a particular model output. This makes the prediction easier to inspect, although it does not establish causal relationships.”

## 10. Other system modules

### Hotspot detection

DBSCAN groups geographically close project road segments using their coordinates. The application summarizes traffic characteristics for the resulting groups. A hotspot is an analytical grouping, not an officially verified congestion zone.

### Anomaly detection

The project combines Isolation Forest and a z-score rule to flag unusual observations. These are indicators for review, not confirmed accidents or real-world incidents.

### OpenStreetMap routing

The project uses a local Hyderabad drive graph and NetworkX graph search to calculate candidate routes. Traffic-related edge costs are applied when mapped traffic data is available. Route coverage must be checked; a route with no traffic coverage should not be described as validated traffic-aware navigation.

### FastAPI and Streamlit

FastAPI exposes health, traffic, hotspot, anomaly, prediction, route and forecast operations. Streamlit provides the user interface for the analytical modules and visualizations.

## 11. Likely viva questions and suggested answers

### Q1. Where did you train the current models?

“The current four-horizon models are trained using the Python training pipeline in `engine/multihorizon_forecaster.py` in our project. We also conducted an earlier Random Forest versus XGBoost experiment in Google Colab. The current four-horizon training pipeline is part of the local project source.”

### Q2. Show me the exact training statement.

Open `engine/multihorizon_forecaster.py` and show the `train_horizon()` function, particularly the `model = build_model()` and `model.fit(...)` section. Then show the `train_all_horizons()` function and the CLI entry point.

### Q3. How did you evaluate the models?

“We used a chronological train-validation-test split. After fitting the preprocessing pipeline on training data and training the model, we predicted on the held-out test data. We calculated accuracy, balanced accuracy, macro F1 and weighted F1, and generated class-wise reports and confusion matrices using `tests/validate_multihorizon.py`.”

### Q4. Why XGBoost?

“Our data is structured tabular data with numerical and categorical traffic, road and environmental features. XGBoost is a gradient-boosted tree model suited to structured classification tasks. In our earlier experiment we compared Random Forest and XGBoost, and XGBoost performed better on the metrics we evaluated. For the current forecasting extension, we trained separate XGBoost classifiers for the four horizons.”

### Q5. Is XGBoost the best algorithm?

“We do not claim it is universally the best. It performed better than Random Forest in our initial comparison. We selected it for the current multi-horizon implementation, but other algorithms could be evaluated in future work.”

### Q6. Why do you have four models?

“Each model predicts a different future horizon. The target distribution and useful patterns can differ with the prediction distance, so we train and store an independent model and preprocessor for each horizon.”

### Q7. What is the difference between accuracy and balanced accuracy?

“Accuracy is the fraction of all test examples classified correctly. Balanced accuracy averages recall across classes, so it is more informative when class counts are unequal.”

### Q8. What does macro F1 mean?

“It calculates F1 for each class and gives each class equal weight when averaging. It helps show performance on less frequent classes instead of letting the majority class dominate the summary.”

### Q9. Does 63.66% mean traffic was reduced by 63.66%?

“No. It is the +2-hour classifier's accuracy on its held-out synthetic test records. It is not a measurement of traffic reduction.”

### Q10. Is the system using live traffic?

“No. The current prototype uses historical synthetic records ending in 2025. It is an offline decision-support prototype, not a live traffic-monitoring deployment.”

### Q11. Does route optimization guarantee a faster route?

“No. It calculates candidate routes using the road graph and traffic-related costs when traffic coverage is available. If coverage is absent, the result should be interpreted as a graph-based candidate route rather than a verified live traffic-aware route.”

### Q12. Are anomaly flags confirmed accidents?

“No. They identify unusual observations based on the anomaly-screening methods. They require external verification before being treated as actual incidents.”

### Q13. What are the main limitations?

“The dataset is synthetic, the current prototype does not ingest live traffic feeds, class performance is uneven—particularly for the Low class—and route traffic coverage can be incomplete. Real traffic data and independent validation would be required before operational deployment.”

### Q14. What is the role of FastAPI?

“FastAPI exposes the system's analytical functions as REST endpoints so that prediction, forecasting, route and other operations can be accessed programmatically. Streamlit provides the interactive dashboard.”

### Q15. What is the role of Streamlit?

“Streamlit provides the interactive interface for exploring forecasts, prediction explanations, hotspots, anomalies, routes and reports.”

## 12. Recommended live demonstration sequence

1. Start FastAPI and verify `GET /health`.
2. Start Streamlit and open the dashboard.
3. Show the Overview page and explain that its data is historical.
4. Open the Multi-Horizon Forecast page and select a road segment.
5. Show the +1h, +2h, +3h and +6h outputs.
6. In VS Code, open `engine/multihorizon_forecaster.py` and show target creation, preprocessing, XGBoost configuration, `model.fit()`, evaluation and artifact saving.
7. Open `trained_models/multihorizon/multihorizon_summary.json` and `validation_report.json` to show results.
8. Open `tests/validate_multihorizon.py` and explain the separate evaluation process.
9. Demonstrate hotspot, anomaly and route modules, stating their limitations accurately.

## 13. Final points to remember

- Clearly distinguish the original Colab experiment from the current four-horizon training implementation.
- Use the current four-horizon metrics when discussing the current forecasting module.
- The data are synthetic and historical; do not call the output live traffic.
- Accuracy is not a traffic-reduction percentage.
- SHAP contributions are not causal effects.
- Anomaly flags are not confirmed incidents.
- Route outputs are only traffic-aware to the extent that mapped traffic observations are available.
- Be honest about limitations. A clear explanation of limitations is better than an unsupported claim.
