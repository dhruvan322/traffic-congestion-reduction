# Smart AI Traffic Congestion Prediction and Reduction System
## Project Setup and User Guide

**Project type:** B.Tech AIML academic prototype<br>
**Geographic context:** Hyderabad, India<br>
**Application stack:** Python, XGBoost, scikit-learn, FastAPI, Streamlit, NetworkX and OpenStreetMap data

---

## 1. About this guide

This guide explains how to prepare the Windows development environment, install dependencies, verify the project files, start the API and dashboard, run the tests, and use the main application modules.

Commands are written for **Windows PowerShell** and assume that the terminal is opened at the project root (the folder containing `app.py`, `api`, `engine`, and `requirements.txt`). Do not run project commands from inside the `tests`, `engine`, or `dashboard` folders.

The guide describes the current project implementation. Paths and behavior should be checked against the delivered project folder if files are moved or renamed.

## 2. System requirements

- Windows computer with PowerShell.
- Python installed and available through the `py` or `python` command.
- Sufficient disk space for the CSV datasets, trained model files, and cached OpenStreetMap road graph.
- A modern web browser.
- Internet access may be needed to display external map tiles in the route map.

The project was developed and tested in a Python virtual environment on Windows. The exact Python version used for the final environment can be checked with:

```powershell
python --version
```

## 3. Project directory overview

The main folders and files are organized by responsibility:

```text
minor_project/
├── api/                         # FastAPI application and route handlers
├── dashboard/                   # Streamlit dashboard pages and panels
├── data/
│   ├── final_dataset/            # Integrated and source traffic datasets
│   └── osm/                      # Hyderabad OpenStreetMap road graph files
├── engine/                       # Forecasting, routing and decision-support logic
├── models/                       # Prediction, hotspot and anomaly model logic
├── nlp/                          # AI chat assistant and language components
├── reports/                      # Report generation logic
├── tests/                        # Automated and validation tests
├── trained_models/
│   └── multihorizon/             # Saved 1h, 2h, 3h and 6h forecasting artifacts
├── docs/                         # Project documentation
├── app.py                        # Streamlit application entry point
├── requirements.txt              # Main application dependencies
└── requirements-dev.txt          # Development and test dependencies
```

Some folders may contain additional files not shown in this abbreviated overview.

## 4. Dataset and model files required at runtime

### 4.1 Integrated traffic dataset

The current multi-horizon training pipeline reads:

```text
data/final_dataset/ml_traffic_dataset.csv
```

The integrated dataset contains approximately 525,480 hourly records for 60 Hyderabad-context road segments, covering January 1, 2025 through December 31, 2025. It was generated as synthetic academic data; it is not a live traffic feed.

The project also contains five source datasets:

- `road_segments.csv`
- `traffic_data.csv`
- `weather_data.csv`
- `accidents.csv`
- `events.csv`

Keep the expected files and folder structure intact. The training code checks for required fields such as `road_segment_id`, `timestamp`, and `future_congestion_level`.

### 4.2 OpenStreetMap graph

The route engine uses the local Hyderabad drive graph files:

```text
data/osm/hyderabad_drive.graphml
data/osm/hyderabad_drive.pkl
```

Keep these files in the expected location. The graph represents the road network used by the prototype; it does not itself provide live traffic conditions.

### 4.3 Multi-horizon trained model artifacts

The forecasting models and preprocessing artifacts are stored separately for each horizon:

```text
trained_models/multihorizon/
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

Each horizon folder contains its own model, preprocessor, label encoder and metadata. Preserve these artifacts together; the model must be used with the matching preprocessor and label encoder.

## 5. Set up the Python environment

### 5.1 Open the project root

Open PowerShell and change to the project folder. For the current local layout:

```powershell
cd "$HOME\OneDrive\Desktop\minor_project"
```

If the project is stored somewhere else, replace the path with that folder.

Confirm that the terminal is at the project root:

```powershell
Get-ChildItem -Force | Select-Object Name, Mode
```

You should see project items such as `app.py`, `api`, `engine`, `data`, and `requirements.txt`.

### 5.2 Create and activate a virtual environment

For a fresh setup:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this terminal session, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The prompt should show `(.venv)` after activation.

If `.venv` already exists and is the environment used for the project, activate it rather than creating another environment.

### 5.3 Install dependencies

With the virtual environment activated, run:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

The main requirements file contains the application libraries. The development requirements file includes testing dependencies.

Check for dependency conflicts:

```powershell
python -m pip check
```

Expected healthy result:

```text
No broken requirements found.
```

## 6. Start the application

The application has two processes: the FastAPI backend and the Streamlit dashboard. Start them in **separate PowerShell terminals**, both opened at the project root and using the same virtual environment.

The Traffic Predictions dashboard sends prediction requests to the FastAPI backend through `POST /api/v1/predict`. Therefore, keep FastAPI running while using the Predictions page. Other dashboards continue to use their respective engine modules directly.

The prediction endpoint supports an optional historical timestamp. If no timestamp is supplied, it uses the latest available observation. SHAP explanations and feature-importance calculations continue to run locally within Streamlit.

### 6.1 Start FastAPI

In Terminal 1:

```powershell
cd "$HOME\OneDrive\Desktop\minor_project"
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open while using the application.

Verify the health endpoint from another terminal or browser:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 5
```

The verified response has the following structure:

```json
{
  "status": "healthy",
  "service": "ai-traffic-intelligence-api",
  "version": "1.0.0"
}
```

A traffic endpoint can be checked with:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/traffic/Hitech%20City" |
    ConvertTo-Json -Depth 5
```

The project has returned historical records for Hitech City through this endpoint.

### 6.2 Start Streamlit

In Terminal 2:

```powershell
cd "$HOME\OneDrive\Desktop\minor_project"
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

Open the local dashboard address printed by Streamlit. The standard local address is:

```text
http://127.0.0.1:8501
```

Keep both the FastAPI and Streamlit terminals running while demonstrating the complete application.

### 6.3 Stop the services

Use `Ctrl+C` in each terminal to stop its process. Stop the Streamlit and API processes separately.

## 7. API overview

The current FastAPI application exposes these endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| GET | `/api/v1/traffic/{area}` | Retrieve traffic information for an area |
| GET | `/api/v1/hotspots` | Retrieve hotspot results |
| GET | `/api/v1/anomalies` | Retrieve anomaly results |
| POST | `/api/v1/predict` | Generate a congestion prediction |
| POST | `/api/v1/route` | Calculate a route using the routing engine |
| POST | `/api/v1/forecast` | Generate multi-horizon forecasts |

Where the FastAPI interactive documentation is enabled, it can be opened at:

```text
http://127.0.0.1:8000/docs
```

Use that interactive API page to inspect request schemas and try endpoints. The exact JSON request body should be taken from the live API schema rather than guessed from this overview.

## 8. Using the Streamlit dashboard

The main dashboard provides the following areas. Exact controls and data displayed can depend on the selected page and the currently loaded records.

### Overview

Review summary traffic information and visual analytics derived from the available dataset.

### Predictions

Select the available area or road segment and inspect the congestion prediction, class probabilities and explanation information provided by the prediction interface.

### Multi-Horizon Forecast

Inspect the independent forecast outputs for:

- +1 hour
- +2 hours
- +3 hours
- +6 hours

The saved horizon-specific models and their corresponding preprocessing artifacts are used for inference.

### Hotspots

Explore geographic groupings of project road segments and their traffic summaries. Hotspot grouping is an analytical indicator, not an official traffic-control boundary.

### Anomaly Detection

Inspect unusual traffic records and their anomaly indicators. An anomaly flag is a prompt for review; it is not confirmation of a real-world incident.

### Route Optimizer

Select route endpoints and inspect candidate paths on the OpenStreetMap-based road graph. Review the route distance, estimated travel metrics and traffic-data coverage. A route with insufficient mapped traffic coverage should not be described as validated traffic-aware navigation.

### AI Chat Assistant

Use the available assistant interfaces to ask questions about project traffic information and available functions. Their responses are limited by the data and integrations configured in the current application.

### Command Center

Review combined traffic intelligence and decision-support information made available by the dashboard.

### Reports

Generate and download the available daily, weekly and anomaly reports. These reports summarize the project's dataset; they are not official operational traffic reports.

## 9. Reproduce model training

The current multi-horizon training implementation is:

```text
engine/multihorizon_forecaster.py
```

From the project root, with the virtual environment activated, run:

```powershell
python .\engine\multihorizon_forecaster.py
```

The script loads `data/final_dataset/ml_traffic_dataset.csv`, creates the horizon targets, prepares features, applies chronological splits, fits preprocessing and trains four independent XGBoost models. It reports accuracy, balanced accuracy, macro F1 and weighted F1, and saves the model and supporting artifacts under `trained_models/multihorizon/`.

**Important:** Running this command retrains all four models and overwrites the existing horizon model artifacts, metadata and summary files. Do not run it during a live demonstration unless retraining is intended. Keep a backup of the trained-model directory before deliberate retraining.

## 10. Reproduce saved-model evaluation

The separate evaluation script is:

```text
 tests/validate_multihorizon.py
```

Run it from the project root. It imports the `engine` package, so set the project root in `PYTHONPATH` for this direct-script invocation:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python .\tests\validate_multihorizon.py
Remove-Item Env:PYTHONPATH
```

The script loads the dataset and saved horizon models, evaluates the test predictions, prints class-wise metrics and confusion matrices, and writes:

```text
trained_models/multihorizon/validation_report.json
```

This command evaluates the saved models; it does not retrain them.

## 11. Run the automated tests and validation checks

Run the full automated test suite from the project root:

```powershell
python -m pytest -v
```

A previous complete regression run recorded **24 passed**. Re-run the suite after making code or environment changes; do not assume an earlier test result applies to a modified version.

Run the multi-horizon inference validation:

```powershell
python .\tests\validate_forecast_inference.py
```

The previously verified result was 60 road segments successfully validated, zero failures, and all four expected horizons (`1h`, `2h`, `3h`, `6h`).

Check dependencies:

```powershell
python -m pip check
```

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'engine'`

Confirm the terminal is at the project root. For `tests/validate_multihorizon.py`, set the project root temporarily:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python .\tests\validate_multihorizon.py
Remove-Item Env:PYTHONPATH
```

### Dataset file not found

Confirm that the integrated dataset exists at:

```text
data/final_dataset/ml_traffic_dataset.csv
```

Also confirm that the expected source datasets have not been moved or renamed.

### Forecast model or preprocessing artifact not found

Check that each horizon directory contains all its expected files: `xgb_congestion.json`, `traffic_preprocessor.joblib`, `label_encoder.joblib`, and `metadata.json`.

### API or dashboard port is already in use

Stop the earlier process using that port before restarting the service. Do not start multiple copies of the same service unintentionally.

### Map tiles do not appear

External map tiles may require internet access. The local OSM road graph and external map-tile display are separate dependencies.

### Pandas deprecation warning during training

A pandas deprecation warning may be printed for `is_categorical_dtype`. The previously observed warning did not prevent the four training runs from completing. It is a compatibility warning, not a model evaluation result.

## 13. How to present the system

A suggested demonstration sequence is:

1. Introduce the problem and the synthetic Hyderabad traffic dataset.
2. Show the data and project folder structure.
3. Open `engine/multihorizon_forecaster.py` and explain target creation, preprocessing, chronological splitting, XGBoost configuration, fitting and artifact saving.
4. Show the training command and its four-horizon results.
5. Open `tests/validate_multihorizon.py` and show the separate evaluation command, class-wise metrics and confusion matrices.
6. Show `trained_models/multihorizon/` and explain that each horizon has its own model and preprocessing artifacts.
7. Demonstrate the Streamlit forecasting page and relevant analytical modules.
8. Show the FastAPI health endpoint and, where useful, the API documentation.
9. Explain the limitations listed below.

## 14. Dataset, model and operational limitations

The following limitations should be stated clearly in reports and demonstrations:

- The traffic dataset is synthetic academic data, not a live feed from Hyderabad roads.
- The dataset's observations cover 2025; a displayed timestamp such as `2025-12-31 21:00:00` is historical dataset time, not the current time.
- The four classification models have uneven class-wise performance. Overall accuracy should be considered together with balanced accuracy, macro F1, weighted F1 and the class-wise confusion matrices.
- Model accuracy is not a percentage of traffic reduced.
- SHAP values explain model outputs; they do not prove that changing a feature will cause a traffic improvement.
- Anomaly indicators do not confirm real-world incidents.
- OSM-based routes are candidate paths over the stored road graph. Traffic-aware usefulness depends on the availability of mapped traffic observations for the requested route and timestamp.
- The current prototype does not ingest live GPS/sensor traffic feeds, actuate traffic signals, or provide a measured before-and-after result for real-world congestion reduction.
- Independent validation using real external traffic data would be required before operational deployment.

## 15. Final handover checklist

Before submitting or demonstrating the project, verify:

- [ ] The full project folder is present and the expected directory structure is intact.
- [ ] `requirements.txt` and `requirements-dev.txt` are included.
- [ ] The five source datasets and integrated ML dataset are present.
- [ ] The Hyderabad OSM graph files are present.
- [ ] All four horizon model directories contain their model and matching preprocessing artifacts.
- [ ] FastAPI starts and `/health` returns a healthy response.
- [ ] Streamlit starts and the main pages open.
- [ ] The forecast page returns predictions for the expected horizons.
- [ ] The multi-horizon validation script runs successfully.
- [ ] The full pytest suite passes on the submitted copy.
- [ ] The documentation folder contains the ML training/evaluation and system architecture/API guides, along with this setup guide.
- [ ] A backup of the final working project and trained model directory is stored separately.
- [ ] Synthetic-data and no-live-feed limitations are disclosed during the demonstration.

---

**End of guide.**
