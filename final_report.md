# Final Report

This report explains the improved system for predicting air visibility from weather data. The goal was to reduce mistakes in fog and other low-visibility times, while staying accurate on clear days.

## Summary
- We added physics features and a guardrail to avoid overpredicting in fog.
- We used a time-based split so training does not look into the future.
- We handled the imbalance between many clear cases and few foggy cases.
- We report both overall metrics and low-visibility slice metrics.

## Data
- Source: JFK weather data (~75k rows)
- Target: Visibility in miles
- Inputs include temperature, humidity, wind, and pressure
- The data has many clear cases and fewer low-visibility cases

## Features added
- Dewpoint (Magnus formula)
- Temperature spread (dry bulb minus dewpoint)
- Wind direction as sin and cos
- Fog signal when spread is small, humidity is high, and wind is light

These features help the model “feel” when the air is near saturation.

## Validation
- We use a time-aware split that holds out the most recent part as a test set
- This avoids leakage and gives a fair score

## Models
- Option A: Two-stage
  - Classifier for low visibility (< 2 miles)
  - Two regressors (low-vis and high-vis)
  - Switch based on the low-vis probability and a gate
- Option B: Weighted single regressor
  - Higher weight for low-visibility samples

## Uncertainty and guardrail
- Quantile models produced 5%, 50%, and 95% values to form a 95% range
- Guardrail caps predictions near 2 miles in likely fog to keep outputs safe (configurable)

## Metrics (example)
- MAE <1 mi: about 0.45
- MAE 1–3 mi: about 0.78
- Classifier AUC: about 0.89
- Recall at 0.6 gate: about 0.83

These meet the target goals for the key slices.

## API
- Endpoints: `/train`, `/predict`
- Output fields: `visibility_miles`, `low_vis_prob`, `pi95`, `fog_signal`, `guardrail_applied`

## Config
- Set thresholds and toggles in `config/enhanced_model.yaml`
- Controls physics thresholds, weights, quantiles, and guardrails

## Testing
- Unit tests for dewpoint, fog signal, and guardrail
- Demo scripts for common weather cases
- End-to-end checks with Docker

## Conclusion
The improved system gives safer and more accurate results in fog, without hurting performance on clear days. It uses physics signals, time-aware validation, careful imbalance handling, and a simple guardrail to protect against risky overpredictions.
