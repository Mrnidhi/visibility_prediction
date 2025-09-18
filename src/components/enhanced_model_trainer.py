"""
Enhanced model trainer with two-stage modeling, weighted regression, and uncertainty quantification.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import yaml
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, classification_report
from sklearn.linear_model import QuantileRegressor
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from src.exception import VisibilityException
from src.utils.main_utils import MainUtils

logger = logging.getLogger(__name__)


class EnhancedModelTrainerConfig:
    """Configuration for enhanced model training."""
    
    def __init__(self, config_path: str = "config/enhanced_model.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create timestamped directory
        timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        self.artifact_folder = os.path.join("artifacts", timestamp)
        
        self.model_trainer_dir = os.path.join(self.artifact_folder, 'enhanced_model_trainer')
        self.trained_model_path = os.path.join(self.model_trainer_dir, 'trained_model', 'model.pkl')
        self.two_stage_model_path = os.path.join(self.model_trainer_dir, 'two_stage_model.pkl')
        self.weighted_model_path = os.path.join(self.model_trainer_dir, 'weighted_model.pkl')
        self.quantile_model_path = os.path.join(self.model_trainer_dir, 'quantile_model.pkl')
        self.metrics_path = os.path.join(self.model_trainer_dir, 'metrics.pkl')
        
        # Create directories
        os.makedirs(self.model_trainer_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_trainer_dir, 'trained_model'), exist_ok=True)


class EnhancedModelTrainer:
    """
    Enhanced model trainer with advanced techniques for imbalanced visibility prediction.
    """
    
    def __init__(self, config_path: str = "config/enhanced_model.yaml"):
        self.config = EnhancedModelTrainerConfig(config_path)
        self.utils = MainUtils()
        self.schema_config = self.utils.read_schema_config_file()
        
        # Get configuration parameters
        self.low_vis_threshold = self.config.config['physics']['low_vis_threshold']
        self.low_vis_prob_gate = self.config.config['physics']['low_vis_prob_gate']
        self.low_vis_weight = self.config.config['sample_weights']['low_visibility_weight']
        self.normal_weight = self.config.config['sample_weights']['normal_weight']
        
        # Evaluation bins
        self.visibility_bins = self.config.config['evaluation']['visibility_bins']
        self.bin_labels = self.config.config['evaluation']['bin_labels']
        
        logger.info("Enhanced model trainer initialized")
    
    def create_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Create sample weights for imbalanced data.
        
        Args:
            y: Target values (visibility)
            
        Returns:
            Sample weights array
        """
        weights = np.where(y < self.low_vis_threshold, self.low_vis_weight, self.normal_weight)
        weights = weights / weights.mean()  # Normalize
        return weights
    
    def train_two_stage_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train two-stage model: classifier for low visibility + separate regressors.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary containing trained models
        """
        logger.info("Training two-stage model...")
        
        # Stage 1: Binary classifier for low visibility
        y_binary = (y_train < self.low_vis_threshold).astype(int)
        
        # Get classifier configuration
        classifier_config = self.config.config['models']['two_stage']['classifier']
        classifier = RandomForestClassifier(**classifier_config['params'])
        
        # Train classifier
        classifier.fit(X_train, y_binary)
        
        # Stage 2: Separate regressors for each class
        low_vis_mask = y_binary == 1
        high_vis_mask = y_binary == 0
        
        # Low visibility regressor
        low_vis_regressor = None
        if np.sum(low_vis_mask) > 10:  # Need sufficient samples
            low_vis_config = self.config.config['models']['two_stage']['regressor_low_vis']
            low_vis_regressor = GradientBoostingRegressor(**low_vis_config['params'])
            low_vis_regressor.fit(X_train[low_vis_mask], y_train[low_vis_mask])
            logger.info(f"Trained low visibility regressor on {np.sum(low_vis_mask)} samples")
        
        # High visibility regressor
        high_vis_config = self.config.config['models']['two_stage']['regressor_high_vis']
        high_vis_regressor = GradientBoostingRegressor(**high_vis_config['params'])
        high_vis_regressor.fit(X_train[high_vis_mask], y_train[high_vis_mask])
        logger.info(f"Trained high visibility regressor on {np.sum(high_vis_mask)} samples")
        
        two_stage_models = {
            'classifier': classifier,
            'low_visibility_regressor': low_vis_regressor,
            'high_visibility_regressor': high_vis_regressor,
            'low_visibility_threshold': self.low_vis_threshold,
            'low_vis_prob_gate': self.low_vis_prob_gate
        }
        
        return two_stage_models
    
    def predict_two_stage(self, models: Dict[str, Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using two-stage model.
        
        Args:
            models: Trained two-stage models
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, low_visibility_probabilities)
        """
        # Stage 1: Classify as low or high visibility
        low_visibility_proba = models['classifier'].predict_proba(X)[:, 1]
        
        # Stage 2: Predict using appropriate regressor
        predictions = np.zeros(len(X))
        
        # High visibility predictions
        high_vis_mask = low_visibility_proba <= self.low_vis_prob_gate
        if np.sum(high_vis_mask) > 0:
            predictions[high_vis_mask] = models['high_visibility_regressor'].predict(X[high_vis_mask])
        
        # Low visibility predictions
        low_vis_mask = low_visibility_proba > self.low_vis_prob_gate
        if np.sum(low_vis_mask) > 0:
            if models['low_visibility_regressor'] is not None:
                predictions[low_vis_mask] = models['low_visibility_regressor'].predict(X[low_vis_mask])
            else:
                # Fallback to high visibility regressor
                predictions[low_vis_mask] = models['high_visibility_regressor'].predict(X[low_vis_mask])
        
        return predictions, low_visibility_proba
    
    def train_weighted_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train single regressor with sample weights.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained weighted model
        """
        logger.info("Training weighted single model...")
        
        # Create sample weights
        sample_weights = self.create_sample_weights(y_train)
        
        # Get model configuration
        model_config = self.config.config['models']['weighted_single']
        model = GradientBoostingRegressor(**model_config['params'])
        
        # Train with sample weights
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        logger.info(f"Trained weighted model with {len(sample_weights)} samples")
        return model
    
    def train_quantile_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train quantile regression models for uncertainty quantification.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of quantile models
        """
        logger.info("Training quantile regression models...")
        
        quantiles = self.config.config['uncertainty']['quantiles']
        quantile_models = {}
        
        for quantile in quantiles:
            model = QuantileRegressor(quantile=quantile, alpha=0.0)
            model.fit(X_train, y_train)
            quantile_models[f'q{int(quantile*100)}'] = model
            logger.info(f"Trained quantile model for {quantile*100}th percentile")
        
        return quantile_models
    
    def predict_quantiles(self, quantile_models: Dict[str, Any], X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Make quantile predictions.
        
        Args:
            quantile_models: Trained quantile models
            X: Feature matrix
            
        Returns:
            Dictionary of quantile predictions
        """
        predictions = {}
        for name, model in quantile_models.items():
            predictions[name] = model.predict(X)
        
        return predictions
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray, 
                       two_stage_models: Dict[str, Any] = None,
                       weighted_model: Any = None,
                       quantile_models: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate all trained models with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_test: Test targets
            two_stage_models: Two-stage model dictionary
            weighted_model: Weighted single model
            quantile_models: Quantile regression models
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating models...")
        
        metrics = {}
        
        # Two-stage model evaluation
        if two_stage_models is not None:
            y_pred_2stage, low_vis_proba = self.predict_two_stage(two_stage_models, X_test)
            
            # Overall metrics
            metrics['two_stage'] = {
                'mae': mean_absolute_error(y_test, y_pred_2stage),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_2stage)),
                'r2': r2_score(y_test, y_pred_2stage)
            }
            
            # Classifier metrics
            y_binary_true = (y_test < self.low_vis_threshold).astype(int)
            metrics['two_stage']['classifier_auc'] = roc_auc_score(y_binary_true, low_vis_proba)
            
            # Binned metrics
            metrics['two_stage']['binned_metrics'] = self._calculate_binned_metrics(y_test, y_pred_2stage)
        
        # Weighted model evaluation
        if weighted_model is not None:
            y_pred_weighted = weighted_model.predict(X_test)
            
            metrics['weighted'] = {
                'mae': mean_absolute_error(y_test, y_pred_weighted),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_weighted)),
                'r2': r2_score(y_test, y_pred_weighted),
                'binned_metrics': self._calculate_binned_metrics(y_test, y_pred_weighted)
            }
        
        # Quantile model evaluation
        if quantile_models is not None:
            quantile_preds = self.predict_quantiles(quantile_models, X_test)
            
            # Calculate prediction intervals
            if 'q5' in quantile_preds and 'q95' in quantile_preds:
                pi_width = quantile_preds['q95'] - quantile_preds['q5']
                coverage = np.mean((y_test >= quantile_preds['q5']) & (y_test <= quantile_preds['q95']))
                
                metrics['quantile'] = {
                    'mean_pi_width': np.mean(pi_width),
                    'coverage_95': coverage,
                    'predictions': quantile_preds
                }
        
        return metrics
    
    def _calculate_binned_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for different visibility bins.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of binned metrics
        """
        y_binned = pd.cut(y_true, bins=self.visibility_bins, labels=self.bin_labels, include_lowest=True)
        
        binned_metrics = {}
        for bin_label in self.bin_labels:
            mask = y_binned == bin_label
            if np.sum(mask) > 0:
                y_true_bin = y_true[mask]
                y_pred_bin = y_pred[mask]
                
                binned_metrics[bin_label] = {
                    'mae': mean_absolute_error(y_true_bin, y_pred_bin),
                    'rmse': np.sqrt(mean_squared_error(y_true_bin, y_pred_bin)),
                    'r2': r2_score(y_true_bin, y_pred_bin),
                    'samples': np.sum(mask)
                }
        
        return binned_metrics
    
    def initiate_enhanced_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray, 
                                      preprocessor_path: str) -> Dict[str, Any]:
        """
        Initiate enhanced model training with all approaches.
        
        Args:
            train_array: Training data array
            test_array: Test data array
            preprocessor_path: Path to preprocessor
            
        Returns:
            Dictionary of trained models and metrics
        """
        try:
            logger.info("Starting enhanced model training...")
            
            # Split features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            
            logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
            
            # Train two-stage model
            two_stage_models = self.train_two_stage_model(X_train, y_train)
            
            # Train weighted model
            weighted_model = self.train_weighted_model(X_train, y_train)
            
            # Train quantile models
            quantile_models = None
            if self.config.config['uncertainty']['enable_quantile_regression']:
                quantile_models = self.train_quantile_models(X_train, y_train)
            
            # Evaluate all models
            metrics = self.evaluate_models(X_test, y_test, two_stage_models, weighted_model, quantile_models)
            
            # Save models
            joblib.dump(two_stage_models, self.config.two_stage_model_path)
            joblib.dump(weighted_model, self.config.weighted_model_path)
            if quantile_models:
                joblib.dump(quantile_models, self.config.quantile_model_path)
            
            # Save metrics
            joblib.dump(metrics, self.config.metrics_path)
            
            # Print summary
            self._print_evaluation_summary(metrics)
            
            # Select best model based on low visibility performance
            best_model_type = self._select_best_model(metrics)
            logger.info(f"Best model type: {best_model_type}")
            
            # Save best model as main model
            if best_model_type == 'two_stage':
                joblib.dump(two_stage_models, self.config.trained_model_path)
            else:
                joblib.dump(weighted_model, self.config.trained_model_path)
            
            return {
                'two_stage_models': two_stage_models,
                'weighted_model': weighted_model,
                'quantile_models': quantile_models,
                'metrics': metrics,
                'best_model_type': best_model_type
            }
            
        except Exception as e:
            raise VisibilityException(e, sys)
    
    def _print_evaluation_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("ENHANCED MODEL EVALUATION SUMMARY")
        print("="*60)
        
        for model_type, model_metrics in metrics.items():
            if model_type in ['two_stage', 'weighted']:
                print(f"\n{model_type.upper()} MODEL:")
                print(f"  Overall MAE: {model_metrics['mae']:.3f} miles")
                print(f"  Overall RMSE: {model_metrics['rmse']:.3f} miles")
                print(f"  Overall R²: {model_metrics['r2']:.3f}")
                
                if 'binned_metrics' in model_metrics:
                    print("  Binned Metrics:")
                    for bin_name, bin_metrics in model_metrics['binned_metrics'].items():
                        print(f"    {bin_name}: MAE={bin_metrics['mae']:.3f}, "
                              f"RMSE={bin_metrics['rmse']:.3f}, "
                              f"R²={bin_metrics['r2']:.3f}, "
                              f"Samples={bin_metrics['samples']}")
                
                if 'classifier_auc' in model_metrics:
                    print(f"  Classifier AUC: {model_metrics['classifier_auc']:.3f}")
        
        if 'quantile' in metrics:
            print(f"\nQUANTILE REGRESSION:")
            print(f"  Mean PI Width: {metrics['quantile']['mean_pi_width']:.3f} miles")
            print(f"  95% Coverage: {metrics['quantile']['coverage_95']:.3f}")
        
        print("="*60)
    
    def _select_best_model(self, metrics: Dict[str, Any]) -> str:
        """
        Select best model based on low visibility performance.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Best model type
        """
        target_mae_1mi = self.config.config['evaluation']['target_mae_1mi']
        target_mae_3mi = self.config.config['evaluation']['target_mae_3mi']
        
        best_score = float('inf')
        best_model = 'weighted'
        
        for model_type in ['two_stage', 'weighted']:
            if model_type in metrics and 'binned_metrics' in metrics[model_type]:
                binned_metrics = metrics[model_type]['binned_metrics']
                
                # Score based on low visibility performance
                score = 0
                if '<1mi' in binned_metrics:
                    score += binned_metrics['<1mi']['mae'] / target_mae_1mi
                if '1-3mi' in binned_metrics:
                    score += binned_metrics['1-3mi']['mae'] / target_mae_3mi
                
                if score < best_score:
                    best_score = score
                    best_model = model_type
        
        return best_model


def test_enhanced_model_trainer():
    """Test the enhanced model trainer."""
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        X_train = np.random.randn(n_samples, 9)  # 9 features from physics transformer
        y_train = np.random.exponential(5, n_samples)
        
        X_test = np.random.randn(200, 9)
        y_test = np.random.exponential(5, 200)
        
        train_array = np.column_stack([X_train, y_train])
        test_array = np.column_stack([X_test, y_test])
        
        # Test trainer
        trainer = EnhancedModelTrainer()
        results = trainer.initiate_enhanced_model_trainer(train_array, test_array, "dummy_path")
        
        print("✓ Enhanced model trainer test passed")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced model trainer test failed: {e}")
        return False


if __name__ == "__main__":
    test_enhanced_model_trainer()
