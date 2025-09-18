"""
Imbalanced Regression Handler for Aviation Visibility Prediction

This module provides strategies to handle imbalanced regression datasets,
specifically for aviation visibility prediction where low visibility events
are rare but critical for safety.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOGN, fall back to simple oversampling if not available
try:
    from smogn import SMOGN
    SMOGN_AVAILABLE = True
except ImportError:
    SMOGN_AVAILABLE = False
    logging.warning("SMOGN not available. Using simple oversampling instead.")


class ImbalancedRegressionHandler:
    """
    Handles imbalanced regression datasets using various strategies:
    1. Weighted loss functions
    2. SMOGN oversampling for regression
    3. Two-stage modeling (classification + regression)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.visibility_bins = [0, 1, 3, 5, 10]  # Visibility bins in miles
        self.bin_labels = ['<1mi', '1-3mi', '3-5mi', '>5mi']
        
    def analyze_imbalance(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the imbalance in the visibility dataset.
        
        Args:
            y: Target values (visibility distances)
            
        Returns:
            Dictionary with imbalance statistics
        """
        self.logger.info("Analyzing data imbalance...")
        
        # Create visibility bins
        y_binned = pd.cut(y, bins=self.visibility_bins, labels=self.bin_labels, include_lowest=True)
        bin_counts = y_binned.value_counts().sort_index()
        bin_percentages = (bin_counts / len(y) * 100).round(2)
        
        # Calculate imbalance ratio
        max_count = bin_counts.max()
        min_count = bin_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Low visibility events (critical for aviation safety)
        low_visibility_count = bin_counts.get('<1mi', 0) + bin_counts.get('1-3mi', 0)
        low_visibility_percentage = (low_visibility_count / len(y) * 100)
        
        imbalance_stats = {
            'total_samples': len(y),
            'bin_counts': bin_counts.to_dict(),
            'bin_percentages': bin_percentages.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'low_visibility_count': low_visibility_count,
            'low_visibility_percentage': low_visibility_percentage,
            'mean_visibility': np.mean(y),
            'std_visibility': np.std(y)
        }
        
        self.logger.info(f"Imbalance Analysis:")
        self.logger.info(f"  Total samples: {len(y)}")
        self.logger.info(f"  Low visibility events (<3mi): {low_visibility_count} ({low_visibility_percentage:.2f}%)")
        self.logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        self.logger.info(f"  Visibility distribution: {bin_percentages.to_dict()}")
        
        return imbalance_stats
    
    def create_sample_weights(self, y: np.ndarray, method: str = 'inverse_frequency') -> np.ndarray:
        """
        Create sample weights to handle class imbalance.
        
        Args:
            y: Target values
            method: Weighting method ('inverse_frequency', 'exponential', 'custom')
            
        Returns:
            Sample weights array
        """
        if method == 'inverse_frequency':
            # Weight inversely proportional to frequency
            y_binned = pd.cut(y, bins=self.visibility_bins, labels=self.bin_labels, include_lowest=True)
            bin_counts = y_binned.value_counts()
            weights = 1.0 / bin_counts[y_binned].values
            weights = weights / weights.mean()  # Normalize
            
        elif method == 'exponential':
            # Exponential weighting favoring low visibility
            weights = np.exp(-y / np.mean(y))
            weights = weights / weights.mean()
            
        elif method == 'custom':
            # Custom weighting: higher weight for low visibility events
            weights = np.ones(len(y))
            low_visibility_mask = y < 3.0  # Less than 3 miles
            weights[low_visibility_mask] = 5.0  # 5x weight for low visibility
            weights = weights / weights.mean()
            
        else:
            weights = np.ones(len(y))
            
        self.logger.info(f"Created sample weights using {method} method")
        self.logger.info(f"  Weight range: {weights.min():.3f} - {weights.max():.3f}")
        self.logger.info(f"  Mean weight: {weights.mean():.3f}")
        
        return weights
    
    def smogn_oversampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOGN (Synthetic Minority Oversampling for Regression) or simple oversampling.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Oversampled X and y
        """
        if SMOGN_AVAILABLE:
            try:
                # Combine X and y for SMOGN
                data = pd.DataFrame(X)
                data['target'] = y
                
                # Apply SMOGN
                smogn = SMOGN(
                    random_state=42,
                    phi=0.5,  # Relevance threshold
                    rel_method='auto',
                    rel_xtrm_type='both',
                    rel_coef=1.5,
                    rel_ctrl_pts_rg=None
                )
                
                data_resampled = smogn.fit_resample(data, data['target'])
                X_resampled = data_resampled.drop('target', axis=1).values
                y_resampled = data_resampled['target'].values
                
                self.logger.info(f"SMOGN oversampling: {len(X)} -> {len(X_resampled)} samples")
                return X_resampled, y_resampled
                
            except Exception as e:
                self.logger.warning(f"SMOGN failed: {e}. Using simple oversampling.")
        
        # Fallback: Simple oversampling for low visibility events
        return self._simple_oversampling(X, y)
    
    def _simple_oversampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple oversampling by duplicating low visibility samples.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Oversampled X and y
        """
        # Identify low visibility samples (< 3 miles)
        low_visibility_mask = y < 3.0
        low_visibility_X = X[low_visibility_mask]
        low_visibility_y = y[low_visibility_mask]
        
        if len(low_visibility_X) == 0:
            self.logger.warning("No low visibility samples found for oversampling")
            return X, y
        
        # Duplicate low visibility samples 3 times
        X_oversampled = np.vstack([X, low_visibility_X, low_visibility_X, low_visibility_X])
        y_oversampled = np.hstack([y, low_visibility_y, low_visibility_y, low_visibility_y])
        
        self.logger.info(f"Simple oversampling: {len(X)} -> {len(X_oversampled)} samples")
        self.logger.info(f"  Low visibility samples: {len(low_visibility_X)} -> {len(low_visibility_X) * 4}")
        
        return X_oversampled, y_oversampled
    
    def two_stage_modeling(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Two-stage modeling: classifier for low visibility + regressors for each class.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary containing trained models
        """
        self.logger.info("Training two-stage model...")
        
        # Stage 1: Binary classifier for low visibility (< 3 miles)
        y_binary = (y < 3.0).astype(int)
        
        # Train classifier
        classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        classifier.fit(X, y_binary)
        
        # Stage 2: Separate regressors for each class
        low_visibility_mask = y_binary == 1
        high_visibility_mask = y_binary == 0
        
        # Low visibility regressor
        if np.sum(low_visibility_mask) > 10:  # Need sufficient samples
            low_vis_regressor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            low_vis_regressor.fit(X[low_visibility_mask], y[low_visibility_mask])
        else:
            low_vis_regressor = None
            self.logger.warning("Insufficient low visibility samples for separate regressor")
        
        # High visibility regressor
        high_vis_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        high_vis_regressor.fit(X[high_visibility_mask], y[high_visibility_mask])
        
        two_stage_models = {
            'classifier': classifier,
            'low_visibility_regressor': low_vis_regressor,
            'high_visibility_regressor': high_vis_regressor,
            'low_visibility_threshold': 3.0
        }
        
        self.logger.info("Two-stage model training completed")
        return two_stage_models
    
    def predict_two_stage(self, models: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """
        Make predictions using two-stage model.
        
        Args:
            models: Trained two-stage models
            X: Feature matrix
            
        Returns:
            Predictions
        """
        # Stage 1: Classify as low or high visibility
        low_visibility_proba = models['classifier'].predict_proba(X)[:, 1]
        low_visibility_pred = low_visibility_proba > 0.5
        
        # Stage 2: Predict using appropriate regressor
        predictions = np.zeros(len(X))
        
        # High visibility predictions
        high_vis_mask = ~low_visibility_pred
        if np.sum(high_vis_mask) > 0:
            predictions[high_vis_mask] = models['high_visibility_regressor'].predict(X[high_vis_mask])
        
        # Low visibility predictions
        low_vis_mask = low_visibility_pred
        if np.sum(low_vis_mask) > 0:
            if models['low_visibility_regressor'] is not None:
                predictions[low_vis_mask] = models['low_visibility_regressor'].predict(X[low_vis_mask])
            else:
                # Fallback to high visibility regressor
                predictions[low_vis_mask] = models['high_visibility_regressor'].predict(X[low_vis_mask])
        
        return predictions
    
    def stratified_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance across different visibility bins.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with metrics for each visibility bin
        """
        # Create visibility bins
        y_binned = pd.cut(y_true, bins=self.visibility_bins, labels=self.bin_labels, include_lowest=True)
        
        results = {}
        for bin_label in self.bin_labels:
            mask = y_binned == bin_label
            if np.sum(mask) > 0:
                y_true_bin = y_true[mask]
                y_pred_bin = y_pred[mask]
                
                mae = mean_absolute_error(y_true_bin, y_pred_bin)
                rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
                r2 = r2_score(y_true_bin, y_pred_bin)
                
                results[bin_label] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'samples': np.sum(mask)
                }
        
        return results
    
    def train_weighted_models(self, X: np.ndarray, y: np.ndarray, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train models with sample weights.
        
        Args:
            X: Feature matrix
            y: Target values
            models: Dictionary of models to train
            sample_weights: Sample weights
            
        Returns:
            Dictionary of trained models
        """
        weights = self.create_sample_weights(y, method='custom')
        trained_models = {}
        
        for name, model in models.items():
            try:
                # Train with sample weights
                model.fit(X, y, sample_weight=weights)
                trained_models[f"{name}_weighted"] = model
                self.logger.info(f"Trained weighted model: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to train weighted model {name}: {e}")
                # Train without weights as fallback
                model.fit(X, y)
                trained_models[f"{name}_unweighted"] = model
        
        return trained_models
