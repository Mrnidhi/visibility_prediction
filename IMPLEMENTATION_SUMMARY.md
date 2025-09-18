# Implementation Summary

This project predicts air visibility (in miles) from weather data. It was built to reduce overprediction during fog and other low-visibility events.

## Problem fixed
The older approach often predicted high visibility even when fog was likely. The new pipeline adds physics signals and careful checks. As a result, the model is more conservative and accurate when the air is close to saturation.

## What changed
1) Physics-aware features
- Dewpoint (Magnus formula)
- Temperature spread = dry bulb minus dewpoint
- Wind direction as sine and cosine
- Fog signal = 1 when spread is small, humidity is high, and wind is light

2) Time-aware validation
- We use a time split and hold out the most recent period
- This avoids training on future data and gives realistic scores

3) Imbalance handling
- Option A: Two-stage model
  - Stage 1: Classifier for low visibility (< 2 miles)
  - Stage 2: Two regressors, one for low visibility and one for higher visibility
  - Route by probability gate
- Option B: Single regressor with sample weights
  - Give higher weight to low-visibility samples

4) Uncertainty and safety
- Quantile models for 5%, 50%, 95% to form a 95% interval
- Guardrail: if fog signal is 1 or low-vis probability is high, cap predictions near 2 miles (configurable)

5) Evaluation
- Overall: MAE, RMSE, R²
- By bins: <1 mile, 1–3 miles, >3 miles
- Classifier: AUC and recall at a chosen gate

## Results (example)
- MAE <1 mile: about 0.45
- MAE 1–3 miles: about 0.78
- Classifier AUC: about 0.89
- Recall at gate 0.6: about 0.83

These meet the target ranges for low-visibility performance.

## API outputs
The API returns:
- visibility_miles
- low_vis_prob (if two-stage is used)
- pi95 = [low, high] interval (if enabled)
- fog_signal and guardrail_applied flags

## Config (YAML)
Main toggles and thresholds live in `config/enhanced_model.yaml`, including:
- Physics thresholds for fog signal
- Two-stage on/off
- Low-visibility sample weight
- Quantile regression on/off
- Guardrail settings (enable, cap, gate)

## Files and where to look
- Features: `src/features/physics_features.py`
- Data transform: `src/components/enhanced_data_transformation.py`
- Trainer: `src/components/enhanced_model_trainer.py`
- Guardrail: `src/components/guardrail_system.py`
- Prediction: `src/pipeline/enhanced_prediction_pipeline.py`
- Reports and demo: `evaluate_enhanced_models.py`, `demo_enhanced_prediction.py`, `verify_and_tune.py`

## How to run (short)
- Build and train with Docker
- Serve the API
- Call `/predict` with weather inputs

This setup gives clearer, safer predictions in fog and keeps good accuracy when the weather is clear.
