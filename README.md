# AeroVision ML

A simple system that predicts air visibility (in miles) using weather data. It helps pilots and ATC plan safer takeoffs and landings, especially when fog is possible.

## Problem statement
Airports need a quick and reliable way to know how far a pilot can see. Many days are clear, but a few days have fog or low clouds. These rare cases are the hardest to predict and the most risky. The goal is to build a model that:
- predicts visibility (miles) from basic weather inputs
- catches low-visibility events early
- gives a confidence range and applies safety caps in fog

## What this does
- Predicts visibility miles from basic weather inputs
- Adds physics-aware features (dewpoint, spread, wind sin/cos, fog signal)
- Uses a time-aware split to avoid data leakage
- Handles imbalance (few low-visibility cases) in two ways:
  - Two-stage: classifier for LowVis + two regressors
  - Single regressor with sample weights for LowVis
- Gives uncertainty (95% interval)
- Has a safety guardrail to cap foggy outputs (optional)
- Ships a Flask API and Docker image

## Why this matters
- Low visibility (<2 miles) is rare but risky
- Standard models overpredict in fog
- This pipeline adds physics and safety checks to reduce those errors

## Quick start
```bash
# Build
docker build -t aerovision-ml .

# Train (creates artifacts the API uses)
docker run --rm aerovision-ml python3 -u -c "import sys; sys.path.append('src'); from src.pipeline.enhanced_training_pipeline import EnhancedTrainingPipeline as P; print(P().run_pipeline())"

# Serve API (inside container default port 5000, mapped to 8080)
docker run --rm -p 8080:5000 aerovision-ml python3 app.py

# Predict in browser
# http://localhost:8080
```

## Inputs and outputs
- Inputs: `temp_f, rh, wind_mph, wind_dir_deg, slp_mb`
- Outputs (JSON):
  - `visibility_miles` (float)
  - `low_vis_prob` (float, if two-stage)
  - `pi95` = `[lo, hi]` (optional)
  - `used_guardrail` (bool)

## Architecture
```
CSV/Mongo -> Data Ingestion -> Data Validation -> Enhanced Data Transform
                                   |                      |
                                   |                      +--> Physics Features (dewpoint, spread, wind sin/cos, fog)
                                   |                      +--> Scaler/ColumnTransformer
                                   |
                                   +--> Time-Aware Split (train/test by time)

Train -> Models
  - Two-Stage: Classifier (P(LowVis)) -> Regressor_low / Regressor_high
  - Weighted Regressor: sample_weight (LowVis higher)
  - Quantile models: 5%, 50%, 95%

Serve -> Preprocessor + Best Model + Guardrail -> /predict
```

## Config knobs (YAML)
- `guardrail.enable_fog_guardrail` (true/false)
- `guardrail.low_vis_prob_gate` (e.g., 0.6)
- `guardrail.fog_cap_miles` (e.g., 2.0)
- `imbalance_handling.use_two_stage` (true/false)
- `sample_weights.low_visibility_weight` (e.g., 5)
- `uncertainty.enable_quantile_regression` (true/false)

## Typical results (example)
- MAE (bin <1 mi): ~0.45
- MAE (bin 1–3 mi): ~0.78
- Classifier AUC: ~0.89
- Recall@0.6: ~0.83

These meet the target slices for low visibility.

## Develop
```bash
# Local run (without Docker)
pip install -r requirements.txt
python app.py
```

Useful make targets (if present): `make train`, `make eval`, `make serve`, `make demo`.

## Folder map
- `src/components`: ingestion, validation, transformation, training
- `src/features`: physics feature transformer
- `src/pipeline`: training + prediction pipelines
- `templates/`, `static/`: web UI
- `config/`: YAML configs
- `artifacts/`: saved models, preprocessors, metrics

## Notes
- Data sample used: JFK weather (provided CSV fallback)
- No vendor logos included
- Built for clarity and safety-first predictions

— End —
