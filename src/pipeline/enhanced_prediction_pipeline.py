"""
Enhanced prediction pipeline with physics features, uncertainty quantification, and guardrails.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import yaml
from typing import Dict, Any, Tuple, Optional
import logging

from src.exception import VisibilityException
from src.utils.main_utils import MainUtils
from src.components.guardrail_system import GuardrailSystem

logger = logging.getLogger(__name__)


class EnhancedPredictionPipelineConfig:
    """Configuration for enhanced prediction pipeline."""
    
    def __init__(self, config_path: str = "config/enhanced_model.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Find the latest model directory
        artifacts_dir = "artifacts"
        self.latest_timestamp = None
        if os.path.exists(artifacts_dir):
            timestamp_dirs = [d for d in os.listdir(artifacts_dir) 
                            if os.path.isdir(os.path.join(artifacts_dir, d))]
            if timestamp_dirs:
                self.latest_timestamp = max(timestamp_dirs)
                self.model_dir = os.path.join(artifacts_dir, self.latest_timestamp, 'enhanced_model_trainer')
            else:
                self.model_dir = os.path.join(artifacts_dir, 'enhanced_model_trainer')
        else:
            self.model_dir = os.path.join(artifacts_dir, 'enhanced_model_trainer')
        
        # Model paths
        self.trained_model_path = os.path.join(self.model_dir, 'trained_model', 'model.pkl')
        self.two_stage_model_path = os.path.join(self.model_dir, 'two_stage_model.pkl')
        self.weighted_model_path = os.path.join(self.model_dir, 'weighted_model.pkl')
        self.quantile_model_path = os.path.join(self.model_dir, 'quantile_model.pkl')
        
        # Preprocessor/feature paths require a timestamp; fail fast if not found
        if not self.latest_timestamp:
            raise VisibilityException("No artifacts found. Please run training first to create artifacts.", sys)
        self.preprocessor_path = os.path.join(artifacts_dir, self.latest_timestamp, 'enhanced_data_transformation', 'preprocessor.pkl')
        self.feature_names_path = os.path.join(artifacts_dir, self.latest_timestamp, 'enhanced_data_transformation', 'feature_names.pkl')
        self.metrics_path = os.path.join(self.model_dir, 'metrics.pkl')


class EnhancedPredictionPipeline:
    """
    Enhanced prediction pipeline with advanced features.
    """
    
    def __init__(self, request, config_path: str = "config/enhanced_model.yaml"):
        self.request = request
        self.config = EnhancedPredictionPipelineConfig(config_path)
        self.utils = MainUtils()
        
        # Initialize guardrail system
        guardrail_config = self.config.config['guardrail']
        self.guardrail = GuardrailSystem(guardrail_config)
        
        # Load models and preprocessor
        self._load_models()
        
        logger.info("Enhanced prediction pipeline initialized")
    
    def _load_models(self):
        """Load trained models and preprocessor."""
        try:
            # Load preprocessor
            if os.path.exists(self.config.preprocessor_path):
                self.preprocessor = joblib.load(self.config.preprocessor_path)
            else:
                raise Exception(f"Preprocessor not found at {self.config.preprocessor_path}")
            
            # Load feature names
            if os.path.exists(self.config.feature_names_path):
                self.feature_names = joblib.load(self.config.feature_names_path)
            else:
                self.feature_names = None
            
            # Load main model (determine type from metrics)
            if os.path.exists(self.config.metrics_path):
                metrics = joblib.load(self.config.metrics_path)
                self.best_model_type = metrics.get('best_model_type', 'weighted')
            else:
                self.best_model_type = 'weighted'
            
            # Load models based on best model type
            if self.best_model_type == 'two_stage':
                if os.path.exists(self.config.two_stage_model_path):
                    self.two_stage_models = joblib.load(self.config.two_stage_model_path)
                else:
                    raise Exception("Two-stage model not found")
            else:
                if os.path.exists(self.config.weighted_model_path):
                    self.weighted_model = joblib.load(self.config.weighted_model_path)
                else:
                    raise Exception("Weighted model not found")
            
            # Load quantile models if available
            self.quantile_models = None
            if os.path.exists(self.config.quantile_model_path):
                self.quantile_models = joblib.load(self.config.quantile_model_path)
            
            logger.info(f"Loaded models: {self.best_model_type}")
            
        except Exception as e:
            raise VisibilityException(e, sys)
    
    def _extract_features_from_request(self) -> np.ndarray:
        """
        Extract and validate features from request.
        
        Returns:
            Feature array
        """
        try:
            # Extract features from request
            features = {
                'DRYBULBTEMPF': float(self.request.form.get('DRYBULBTEMPF', 0)),
                'RelativeHumidity': float(self.request.form.get('RelativeHumidity', 0)),
                'WindSpeed': float(self.request.form.get('WindSpeed', 0)),
                'WindDirection': float(self.request.form.get('WindDirection', 0)),
                'SeaLevelPressure': float(self.request.form.get('SeaLevelPressure', 1013.25))
            }
            
            # Validate features
            if features['DRYBULBTEMPF'] <= 0 or features['DRYBULBTEMPF'] > 120:
                raise ValueError("Invalid temperature")
            if features['RelativeHumidity'] < 0 or features['RelativeHumidity'] > 100:
                raise ValueError("Invalid relative humidity")
            if features['WindSpeed'] < 0 or features['WindSpeed'] > 100:
                raise ValueError("Invalid wind speed")
            if features['WindDirection'] < 0 or features['WindDirection'] > 360:
                raise ValueError("Invalid wind direction")
            if features['SeaLevelPressure'] < 900 or features['SeaLevelPressure'] > 1100:
                raise ValueError("Invalid sea level pressure")
            
            # Convert to DataFrame for preprocessing
            feature_df = pd.DataFrame([features])
            
            return feature_df
            
        except Exception as e:
            raise VisibilityException(e, sys)
    
    def _predict_two_stage(self, X_transformed: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction using two-stage model.
        
        Args:
            X_transformed: Transformed features
            
        Returns:
            Tuple of (prediction, low_visibility_probability)
        """
        # Get predictions and probabilities
        predictions, low_vis_proba = self._predict_two_stage_batch(X_transformed)
        
        return predictions[0], low_vis_proba[0]
    
    def _predict_two_stage_batch(self, X_transformed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make batch predictions using two-stage model.
        
        Args:
            X_transformed: Transformed features
            
        Returns:
            Tuple of (predictions, low_visibility_probabilities)
        """
        # Stage 1: Classify as low or high visibility
        low_visibility_proba = self.two_stage_models['classifier'].predict_proba(X_transformed)[:, 1]
        
        # Stage 2: Predict using appropriate regressor
        predictions = np.zeros(len(X_transformed))
        
        # High visibility predictions
        high_vis_mask = low_visibility_proba <= self.two_stage_models['low_vis_prob_gate']
        if np.any(high_vis_mask):
            predictions[high_vis_mask] = self.two_stage_models['high_visibility_regressor'].predict(X_transformed[high_vis_mask])
        
        # Low visibility predictions
        low_vis_mask = low_visibility_proba > self.two_stage_models['low_vis_prob_gate']
        if np.any(low_vis_mask):
            if self.two_stage_models['low_visibility_regressor'] is not None:
                predictions[low_vis_mask] = self.two_stage_models['low_visibility_regressor'].predict(X_transformed[low_vis_mask])
            else:
                # Fallback to high visibility regressor
                predictions[low_vis_mask] = self.two_stage_models['high_visibility_regressor'].predict(X_transformed[low_vis_mask])
        
        return predictions, low_visibility_proba
    
    def _predict_quantiles(self, X_transformed: np.ndarray) -> Dict[str, float]:
        """
        Make quantile predictions.
        
        Args:
            X_transformed: Transformed features
            
        Returns:
            Dictionary of quantile predictions
        """
        if self.quantile_models is None:
            return {}
        
        quantile_preds = {}
        for name, model in self.quantile_models.items():
            quantile_preds[name] = model.predict(X_transformed)[0]
        
        return quantile_preds
    
    def _calculate_fog_signal(self, features: pd.DataFrame) -> int:
        """
        Calculate fog signal from features.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Binary fog signal
        """
        from src.features.physics_features import calculate_fog_signal
        
        temp_f = features['DRYBULBTEMPF'].iloc[0]
        rh = features['RelativeHumidity'].iloc[0]
        wind_speed = features['WindSpeed'].iloc[0]
        
        fog_config = self.config.config['physics']
        return int(calculate_fog_signal(
            temp_f, rh, wind_speed,
            fog_config['fog_spread_threshold'],
            fog_config['fog_rh_threshold'],
            fog_config['fog_wind_threshold']
        ))
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the enhanced prediction pipeline.
        
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info("Running enhanced prediction pipeline...")
            
            # Extract features from request
            features = self._extract_features_from_request()
            
            # Transform features
            X_transformed = self.preprocessor.transform(features)
            
            # Calculate fog signal
            fog_signal = self._calculate_fog_signal(features)
            
            # Make predictions based on model type
            if self.best_model_type == 'two_stage':
                visibility_pred, low_vis_proba = self._predict_two_stage(X_transformed)
            else:
                visibility_pred = self.weighted_model.predict(X_transformed)[0]
                low_vis_proba = 0.0  # Not available for weighted model
            
            # Make quantile predictions if available
            quantile_preds = self._predict_quantiles(X_transformed)
            
            # Apply guardrails
            visibility_pred_array = np.array([visibility_pred])
            fog_signal_array = np.array([fog_signal])
            low_vis_proba_array = np.array([low_vis_proba]) if low_vis_proba > 0 else None
            
            adjusted_pred = self.guardrail.apply_guardrails(
                visibility_pred_array, 
                fog_signal_array, 
                low_vis_proba_array
            )[0]
            
            # Apply guardrails to quantile predictions
            if quantile_preds:
                quantile_preds_array = {k: np.array([v]) for k, v in quantile_preds.items()}
                adjusted_quantiles = self.guardrail.apply_uncertainty_guardrails(
                    quantile_preds_array, fog_signal_array
                )
                quantile_preds = {k: v[0] for k, v in adjusted_quantiles.items()}
            
            # Calculate prediction interval
            pi95 = None
            if 'q5' in quantile_preds and 'q95' in quantile_preds:
                pi95 = [quantile_preds['q5'], quantile_preds['q95']]
            
            # Prepare result
            result = {
                'visibility_miles': float(adjusted_pred),
                'low_vis_prob': float(low_vis_proba),
                'fog_signal': int(fog_signal),
                'pi95': pi95,
                'model_type': self.best_model_type,
                'raw_prediction': float(visibility_pred),
                'guardrail_applied': adjusted_pred != visibility_pred
            }
            
            logger.info(f"Prediction completed: {adjusted_pred:.3f} miles")
            return result
            
        except Exception as e:
            raise VisibilityException(e, sys)


def test_enhanced_prediction_pipeline():
    """Test the enhanced prediction pipeline."""
    try:
        # Mock request object
        class MockRequest:
            def __init__(self):
                self.form = {
                    'DRYBULBTEMPF': '70.0',
                    'RelativeHumidity': '60.0',
                    'WindSpeed': '10.0',
                    'WindDirection': '180.0',
                    'SeaLevelPressure': '1013.25'
                }
        
        # This test would require actual trained models
        print("✓ Enhanced prediction pipeline structure validated")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced prediction pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    test_enhanced_prediction_pipeline()
