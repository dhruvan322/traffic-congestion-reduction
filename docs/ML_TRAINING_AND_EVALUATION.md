# ML Training and Evaluation Documentation

**Project:** Smart AI Traffic Congestion Prediction and Reduction System Using Explainable Machine Learning and OpenStreetMap-Based Routing<br>
**Module:** Multi-horizon traffic congestion forecasting<br>
**Training implementation:** `engine/multihorizon_forecaster.py`<br>
**Evaluation implementation:** `tests/validate_multihorizon.py`<br>
**Documentation date:** 31 October 2026

---

## 1. Purpose and scope

This document records the current project's four-horizon XGBoost training and evaluation workflow. It describes the code that is present in the project, the model configuration, the chronological evaluation approach, the results reproduced from the latest local training and evaluation runs, and the artifacts used by the application.

The current multi-horizon workflow trains **four separate XGBoost multiclass classifiers** to forecast one of five congestion classes at +1, +2, +3, and +6 hours. It is distinct from the earlier Google Colab experiment, which compared Random Forest and XGBoost for a single next-hour target. The original Colab experiment's reported 55.45% XGBoost test accuracy must not be mixed up with the results in this document.

The current results are classification metrics on a held-out portion of the project's synthetic dataset. They are **not** percentages of congestion reduction, vehicles rerouted, or real-world traffic accuracy.

## 2. Dataset

### 2.1 Source datasets

The project uses a Hyderabad-focused synthetic academic traffic dataset. The source tables and integrated model dataset were recorded with these dimensions:

| Dataset | Rows | Columns |
|---|---:|---:|
| `road_segments.csv` | 60 | 9 |
| `traffic_data.csv` | 525,480 | 27 |
| `weather_data.csv` | 175,200 | 8 |
| `accidents.csv` | 600 | 9 |
| `events.csv` | 350 | 9 |
| `ml_traffic_dataset.csv` | 525,480 | 31 |

The integrated dataset contains hourly records from 1 January 2025 through 31 December 2025 and covers 60 road segments. The current training code reads:

```text
data/final_dataset/ml_traffic_dataset.csv
```

### 2.2 Target classes

The model predicts one of five congestion classes:

1. Free Flow
2. Low
3. Moderate
4. High
5. Severe

The class distribution is imbalanced. For the +1-hour test set, for example, the supports were 42,753 Free Flow, 2,880 Low, 22,133 Moderate, 9,827 High, and 1,247 Severe records. This imbalance is important when interpreting accuracy and class-wise recall.

## 3. Training implementation

### 3.1 Source file and entry point

The current training source is:

```text
engine/multihorizon_forecaster.py
```

The file includes dataset loading and validation, target construction, feature preparation, chronological splitting, preprocessing, model construction, training, test-set evaluation, artifact saving, and the command-line entry point.

The main functions include:

| Function | Responsibility |
|---|---|
| `load_final_dataset()` | Loads the integrated dataset, checks required columns and timestamps, and sorts by road segment and timestamp. |
| `build_horizon_target()` | Creates the target label for a requested future horizon. |
| `prepare_horizon_dataset()` | Builds the feature matrix and target vector while excluding target/leakage and identifier columns. |
| `build_preprocessor()` | Constructs the numeric and categorical preprocessing pipelines. |
| `build_model()` | Creates the XGBoost classifier using the project hyperparameters. |
| `evaluate_predictions()` | Calculates the four reported classification metrics. |
| `train_horizon()` | Trains, evaluates, and saves one horizon's model and supporting artifacts. |
| `train_all_horizons()` | Runs training for +1h, +2h, +3h, and +6h, then writes the summary JSON. |

To inspect or present the implementation in VS Code, open `engine/multihorizon_forecaster.py`. Useful code sections in the inspected version include dataset loading (approximately lines 86–122), target construction (129–170), preprocessing (291–359), model configuration (366–382), metrics (389–424), horizon training (431 onward), model fitting and test prediction (542–563), artifact saving (565 onward), and the all-horizons entry point (673–738). Line numbers may change if the source file is edited.

### 3.2 Future target construction

The base target column is `future_congestion_level`. The code treats it as the +1-hour target and constructs longer-horizon targets by shifting the target forward **within each `road_segment_id`**:

| Forecast | Shift applied to the base target |
|---|---:|
| +1 hour | 0 rows (existing target) |
| +2 hours | 1 hourly row |
| +3 hours | 2 hourly rows |
| +6 hours | 5 hourly rows |

The corresponding configuration is:

```python
HORIZONS = {
    1: 0,
    2: 1,
    3: 2,
    6: 5,
}
```

Rows without a future target at the end of a segment's time series are dropped for that horizon. Consequently, the usable row counts differ slightly across horizons.

### 3.3 Feature preparation and leakage safeguards

The inspected training code excludes these columns from the model feature matrix:

- `future_congestion_level`
- `forecast_target`
- `timestamp`
- `traffic_id`

The newly constructed `forecast_target` is used as the label, not as an input feature. The raw timestamp and row identifier are also excluded. The remaining feature set contains 28 columns before preprocessing/encoding; the training run reported 117 transformed features.

### 3.4 Preprocessing

The preprocessor is built from the training partition and then applied to the validation and test partitions:

| Feature type | Processing |
|---|---|
| Numerical | Median imputation |
| Categorical | Most-frequent imputation, followed by one-hot encoding |

The one-hot encoder uses `handle_unknown="ignore"`. The preprocessing pipeline is fitted on the training data only; validation and test data are transformed using that fitted pipeline.

### 3.5 Chronological partitions

The training workflow uses chronological partitions close to 70% training, 15% validation, and 15% testing. It does not randomly shuffle records before splitting. This is intended to evaluate later observations after fitting on earlier observations.

The latest training execution printed these partition sizes:

| Horizon | Training rows | Validation rows | Test rows |
|---|---:|---:|---:|
| +1 hour | 367,800 | 78,840 | 78,840 |
| +2 hours | 367,740 | 78,840 | 78,840 |
| +3 hours | 367,740 | 78,780 | 78,840 |
| +6 hours | 367,620 | 78,780 | 78,780 |

The latest independent evaluation printed these held-out test periods:

| Horizon | Test period | Test rows |
|---|---|---:|
| +1 hour | 7 Nov 2025 04:00 to 31 Dec 2025 21:00 | 78,840 |
| +2 hours | 7 Nov 2025 03:00 to 31 Dec 2025 20:00 | 78,840 |
| +3 hours | 7 Nov 2025 02:00 to 31 Dec 2025 19:00 | 78,840 |
| +6 hours | 7 Nov 2025 00:00 to 31 Dec 2025 16:00 | 78,780 |

### 3.6 XGBoost configuration

The `build_model()` function defines the following configuration for each of the four independent models:

| Parameter | Value |
|---|---|
| `objective` | `multi:softprob` |
| `eval_metric` | `mlogloss` |
| `n_estimators` | 180 |
| `max_depth` | 8 |
| `learning_rate` | 0.10 |
| `subsample` | 0.90 |
| `colsample_bytree` | 0.90 |
| `min_child_weight` | 1 |
| `reg_lambda` | 1.0 |
| `random_state` | 42 |
| `n_jobs` | -1 |
| `tree_method` | `hist` |

The classifier is fitted on the transformed training data. The transformed validation partition is passed through `eval_set`; the held-out test partition is used for the reported test predictions and metrics.

## 4. Evaluation methodology

The evaluation source is:

```text
tests/validate_multihorizon.py
```

The validation script loads the final dataset and each saved horizon-specific model and preprocessing artifact, constructs the appropriate horizon target, transforms the held-out data, predicts the test labels, and reports metrics. It prints the per-class classification report and confusion matrix for each horizon and writes the detailed results to `trained_models/multihorizon/validation_report.json`.

The four headline metrics are:

- **Accuracy:** fraction of test records classified correctly.
- **Balanced accuracy:** average recall across classes, giving each class equal weight.
- **Macro F1:** unweighted average of the class-level F1 scores.
- **Weighted F1:** average of class-level F1 scores weighted by each class's support.

The validation output also reports a majority-class baseline, which predicts the most frequent class (Free Flow) for every test record. The recorded baseline was 54.23% for +1h, +2h, and +3h, and 54.21% for +6h.

## 5. Latest reproduced test results

The following values are from the training execution and the subsequent independent evaluation run performed on 31 October 2026.

### 5.1 Headline metrics

| Horizon | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Majority baseline |
|---|---:|---:|---:|---:|---:|
| +1 hour | 63.60% | 48.52% | 48.33% | 61.82% | 54.23% |
| +2 hours | 63.66% | 48.89% | 49.32% | 62.10% | 54.23% |
| +3 hours | 63.53% | 49.06% | 48.71% | 61.86% | 54.23% |
| +6 hours | 63.58% | 48.21% | 48.29% | 61.52% | 54.21% |

The +2-hour model has the highest accuracy (63.66%) and weighted F1 (62.10%). The +3-hour model has the highest balanced accuracy (49.06%). The +2-hour model has the highest macro F1 (49.32%).

### 5.2 Class-wise results

Rows below show precision / recall / F1-score. Support is the number of actual test examples for the class.

**+1-hour model**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Free Flow | 0.7506 | 0.8574 | 0.8005 | 42,753 |
| Low | 0.3609 | 0.1302 | 0.1914 | 2,880 |
| Moderate | 0.4599 | 0.3854 | 0.4194 | 22,133 |
| High | 0.4209 | 0.3810 | 0.3999 | 9,827 |
| Severe | 0.5510 | 0.6720 | 0.6055 | 1,247 |

**+2-hour model**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Free Flow | 0.7546 | 0.8516 | 0.8001 | 42,753 |
| Low | 0.3654 | 0.1701 | 0.2322 | 2,880 |
| Moderate | 0.4567 | 0.4111 | 0.4327 | 22,133 |
| High | 0.4263 | 0.3421 | 0.3796 | 9,827 |
| Severe | 0.5795 | 0.6696 | 0.6213 | 1,247 |

**+3-hour model**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Free Flow | 0.7515 | 0.8564 | 0.8005 | 42,753 |
| Low | 0.3059 | 0.1653 | 0.2146 | 2,880 |
| Moderate | 0.4620 | 0.3888 | 0.4223 | 22,133 |
| High | 0.4217 | 0.3603 | 0.3886 | 9,827 |
| Severe | 0.5508 | 0.6824 | 0.6096 | 1,247 |

**+6-hour model**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Free Flow | 0.7429 | 0.8688 | 0.8009 | 42,706 |
| Low | 0.3473 | 0.1441 | 0.2037 | 2,880 |
| Moderate | 0.4631 | 0.3678 | 0.4100 | 22,121 |
| High | 0.4194 | 0.3672 | 0.3916 | 9,826 |
| Severe | 0.5623 | 0.6624 | 0.6082 | 1,247 |

### 5.3 Confusion matrices

The complete confusion matrices are stored in `trained_models/multihorizon/validation_report.json`. In each matrix, rows are actual classes and columns are predicted classes. The class order is:

```text
Free Flow, Low, Moderate, High, Severe
```

Refer to the JSON report for the full matrices rather than manually transcribing them into another source of record.

## 6. Interpretation and limitations

1. The models outperform the simple majority-class baseline on overall accuracy in the recorded test results.
2. Performance is not uniform across classes. The +1-hour model's Low-class recall is 13.02%; the corresponding recall is also relatively weak for the other horizons. This is consistent with the imbalance in the target distribution.
3. Balanced accuracy and macro F1 are lower than overall accuracy because they give more weight to minority classes.
4. The dataset is synthetic academic data. These metrics measure performance on a chronological held-out portion of that dataset; they do not establish performance on live Hyderabad traffic.
5. The application currently uses historical dataset records. It does not have a connected live traffic sensor/GPS feed.
6. Route optimization is a graph-based decision-support feature. The project has not measured a real-world before-and-after reduction in congestion.
7. SHAP explanations describe model contributions and do not prove causal effects.

For presentations, describe the results as **test-set congestion-classification metrics**. Do not call them traffic-reduction percentages or real-world accuracy.

## 7. Saved artifacts

The training script writes one directory per horizon:

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
    │   └── ...
    ├── 3h/
    │   └── ...
    └── 6h/
        └── ...
```

Each horizon folder contains the XGBoost model, fitted preprocessing pipeline, label encoder, and metadata. The summary JSON stores aggregate training/split/metric information. The validation report JSON stores the detailed classification reports and confusion matrices.

## 8. Reproducing the workflow

Run commands from the project root with the project's virtual environment activated.

### 8.1 Train all four models

```powershell
python .\engine\multihorizon_forecaster.py
```

This command **re-trains all four models and overwrites the corresponding files** under `trained_models\multihorizon`. Do not run it casually during a demonstration if you only intend to show existing results.

### 8.2 Independently evaluate the saved models

The evaluation script imports `engine` from the project root. When launched as a file under `tests`, the project root may need to be added to Python's module search path. In PowerShell, run:

```powershell
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = (Get-Location).Path
try {
    python .\tests\validate_multihorizon.py
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-horizon evaluation failed."
    }
}
finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }
}
```

Successful evaluation prints all four horizons' headline metrics, per-class metrics, and confusion matrices. It writes/updates:

```text
trained_models\multihorizon\validation_report.json
```

### 8.3 Open the saved results

```powershell
Get-Content .\trained_models\multihorizon\multihorizon_summary.json
Get-Content .\trained_models\multihorizon\validation_report.json
```

## 9. Suggested project-review demonstration

1. Open `engine/multihorizon_forecaster.py` and show `load_final_dataset()`, `build_horizon_target()`, `build_preprocessor()`, `build_model()`, `train_horizon()`, and `train_all_horizons()`.
2. Show the actual training terminal output from `python .\engine\multihorizon_forecaster.py`.
3. Open `tests/validate_multihorizon.py` and explain how the saved models are reloaded and evaluated.
4. Show the independent evaluation terminal output and the four confusion matrices.
5. Open `trained_models/multihorizon/multihorizon_summary.json` and `validation_report.json`.
6. Show the horizon-specific saved model directories.
7. Demonstrate the Multi-Horizon Forecast page and explain that the application loads the saved artifacts to generate forecasts.

### Suggested explanation for the project guide

> We implemented the current multi-horizon training pipeline in Python in our project repository. We used our integrated Hyderabad synthetic traffic dataset and prepared separate targets for +1, +2, +3, and +6 hours by shifting the future congestion label within each road segment. For each horizon, we used a chronological training, validation, and test split, fitted a preprocessing pipeline on the training partition, and trained an independent XGBoost classifier. We evaluated the held-out test predictions using accuracy, balanced accuracy, macro F1, weighted F1, class-wise classification reports, and confusion matrices. We saved each model with its preprocessing pipeline and label encoder, and the Streamlit/FastAPI application uses the saved artifacts for forecasting.

---

**Source files:** `engine/multihorizon_forecaster.py`, `tests/validate_multihorizon.py`, `trained_models/multihorizon/multihorizon_summary.json`, and `trained_models/multihorizon/validation_report.json`.
