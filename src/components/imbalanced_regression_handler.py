import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImbalancedRegressionHandler:
    """
    Handles imbalanced regression data for aviation visibility prediction.
    Implements multiple strategies to address data imbalance in regression tasks.
    """
    
    def __init__(self, visibility_bins=None):
        """
        Initialize the imbalanced regression handler.
        
        Args:
            visibility_bins (list): Bins for visibility stratification
        """
        self.visibility_bins = visibility_bins or [0, 1, 3, 5, 10, float('inf')]
        self.bin_labels = ['<1mi', '1-3mi', '3-5mi', '5-10mi', '>10mi']
        self.stratified_metrics = {}
        
    def analyze_imbalance(self, y):
        """
        Analyze the distribution of visibility values.
        
        Args:
            y (array): Target visibility values
            
        Returns:
            dict: Distribution statistics
        """
        bins = pd.cut(y, bins=self.visibility_bins, labels=self.bin_labels, include_lowest=True)
        distribution = bins.value_counts().sort_index()
        
        print("=== Visibility Distribution Analysis ===")
        print(f"Total samples: {len(y)}")
        print(f"Mean visibility: {y.mean():.2f} miles")
        print(f"Std visibility: {y.std():.2f} miles")
        print("\nDistribution by visibility bins:")
        for bin_name, count in distribution.items():
            percentage = (count / len(y)) * 100
            print(f"{bin_name}: {count} samples ({percentage:.1f}%)")
            
        return {
            'distribution': distribution,
            'mean': y.mean(),
            'std': y.std(),
            'min': y.min(),
            'max': y.max()
        }
    
    def create_sample_weights(self, y, method='inverse_frequency'):
        """
        Create sample weights to handle imbalanced regression data.
        
        Args:
            y (array): Target values
            method (str): Weighting method ('inverse_frequency', 'exponential', 'custom')
            
        Returns:
            array: Sample weights
        """
        if method == 'inverse_frequency':
            # Inverse frequency weighting
            bins = pd.cut(y, bins=self.visibility_bins, labels=self.bin_labels, include_lowest=True)
            bin_counts = bins.value_counts()
            weights = np.ones(len(y))
            
            for i, bin_name in enumerate(bins):
                if bin_name in bin_counts:
                    weights[i] = len(y) / bin_counts[bin_name]
                    
        elif method == 'exponential':
            # Exponential weighting - give more weight to low visibility samples
            weights = np.exp(-y / 2)  # Exponential decay based on visibility
            
        elif method == 'custom':
            # Custom weighting for aviation safety
            weights = np.ones(len(y))
            weights[y < 1] = 10.0    # Very low visibility: highest weight
            weights[(y >= 1) & (y < 3)] = 5.0   # Low visibility: high weight
            weights[(y >= 3) & (y < 5)] = 2.0   # Moderate visibility: medium weight
            weights[y >= 5] = 1.0    # High visibility: normal weight
            
        return weights
    
    def smogn_oversampling(self, X, y, k_neighbors=5, threshold=0.5):
        """
        Implement SMOGN (Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise)
        for handling imbalanced regression data.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            k_neighbors (int): Number of neighbors for SMOTE
            threshold (float): Threshold for minority class definition
            
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Identify minority samples (low visibility)
        minority_threshold = np.percentile(y, threshold * 100)
        minority_indices = np.where(y < minority_threshold)[0]
        majority_indices = np.where(y >= minority_threshold)[0]
        
        if len(minority_indices) == 0:
            return X, y
            
        X_minority = X[minority_indices]
        y_minority = y[minority_indices]
        
        # Find k-nearest neighbors for each minority sample
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_minority)
        distances, indices = nbrs.kneighbors(X_minority)
        
        # Generate synthetic samples
        synthetic_samples = []
        synthetic_targets = []
        
        for i in range(len(X_minority)):
            # Randomly select a neighbor
            neighbor_idx = np.random.choice(indices[i][1:])  # Exclude self
            
            # Generate synthetic sample with Gaussian noise
            noise_factor = np.random.normal(0, 0.1)
            synthetic_sample = X_minority[i] + noise_factor * (X_minority[neighbor_idx] - X_minority[i])
            synthetic_target = y_minority[i] + noise_factor * (y_minority[neighbor_idx] - y_minority[i])
            
            synthetic_samples.append(synthetic_sample)
            synthetic_targets.append(synthetic_target)
        
        # Combine original and synthetic samples
        X_balanced = np.vstack([X, np.array(synthetic_samples)])
        y_balanced = np.concatenate([y, np.array(synthetic_targets)])
        
        return X_balanced, y_balanced
    
    def two_stage_modeling(self, X, y, classification_threshold=3.0):
        """
        Implement two-stage modeling: classifier for low visibility + regression.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            classification_threshold (float): Threshold for low visibility classification
            
        Returns:
            dict: Two-stage model components
        """
        # Stage 1: Binary classification (low vs high visibility)
        y_binary = (y < classification_threshold).astype(int)
        
        # Train classifier
        classifier = RandomForestRegressor(n_estimators=100, random_state=42)
        classifier.fit(X, y_binary)
        
        # Stage 2: Regression for each class
        low_vis_mask = y_binary == 1
        high_vis_mask = y_binary == 0
        
        regressor_low = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor_high = RandomForestRegressor(n_estimators=100, random_state=42)
        
        if np.sum(low_vis_mask) > 0:
            regressor_low.fit(X[low_vis_mask], y[low_vis_mask])
        
        if np.sum(high_vis_mask) > 0:
            regressor_high.fit(X[high_vis_mask], y[high_vis_mask])
        
        return {
            'classifier': classifier,
            'regressor_low': regressor_low,
            'regressor_high': regressor_high,
            'threshold': classification_threshold
        }
    
    def predict_two_stage(self, model_dict, X):
        """
        Make predictions using two-stage model.
        
        Args:
            model_dict (dict): Two-stage model components
            X (array): Feature matrix
            
        Returns:
            array: Predictions
        """
        # Stage 1: Classify visibility level
        visibility_class = model_dict['classifier'].predict(X)
        
        # Stage 2: Predict visibility value
        predictions = np.zeros(len(X))
        
        low_vis_mask = visibility_class < 0.5
        high_vis_mask = visibility_class >= 0.5
        
        if np.sum(low_vis_mask) > 0:
            predictions[low_vis_mask] = model_dict['regressor_low'].predict(X[low_vis_mask])
        
        if np.sum(high_vis_mask) > 0:
            predictions[high_vis_mask] = model_dict['regressor_high'].predict(X[high_vis_mask])
        
        return predictions
    
    def stratified_evaluation(self, y_true, y_pred, model_name="Model"):
        """
        Evaluate model performance across different visibility bins.
        
        Args:
            y_true (array): True visibility values
            y_pred (array): Predicted visibility values
            model_name (str): Name of the model for reporting
            
        Returns:
            dict: Stratified evaluation metrics
        """
        bins = pd.cut(y_true, bins=self.visibility_bins, labels=self.bin_labels, include_lowest=True)
        
        stratified_metrics = {}
        
        print(f"\n=== Stratified Evaluation for {model_name} ===")
        
        for bin_name in self.bin_labels:
            mask = bins == bin_name
            if np.sum(mask) > 0:
                y_true_bin = y_true[mask]
                y_pred_bin = y_pred[mask]
                
                mae = mean_absolute_error(y_true_bin, y_pred_bin)
                rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
                r2 = r2_score(y_true_bin, y_pred_bin)
                
                stratified_metrics[bin_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'sample_count': len(y_true_bin)
                }
                
                print(f"{bin_name} ({len(y_true_bin)} samples):")
                print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        # Overall metrics
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_r2 = r2_score(y_true, y_pred)
        
        print(f"\nOverall Performance:")
        print(f"MAE: {overall_mae:.3f}, RMSE: {overall_rmse:.3f}, R²: {overall_r2:.3f}")
        
        return {
            'stratified': stratified_metrics,
            'overall': {
                'mae': overall_mae,
                'rmse': overall_rmse,
                'r2': overall_r2
            }
        }
    
    def train_weighted_models(self, X_train, y_train, X_test, y_test):
        """
        Train models with different weighting strategies.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            dict: Trained models and their performance
        """
        models = {}
        results = {}
        
        # 1. Standard Random Forest (baseline)
        rf_standard = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_standard.fit(X_train, y_train)
        y_pred_standard = rf_standard.predict(X_test)
        
        models['standard_rf'] = rf_standard
        results['standard_rf'] = self.stratified_evaluation(y_test, y_pred_standard, "Standard RF")
        
        # 2. Weighted Random Forest
        weights = self.create_sample_weights(y_train, method='custom')
        rf_weighted = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_weighted.fit(X_train, y_train, sample_weight=weights)
        y_pred_weighted = rf_weighted.predict(X_test)
        
        models['weighted_rf'] = rf_weighted
        results['weighted_rf'] = self.stratified_evaluation(y_test, y_pred_weighted, "Weighted RF")
        
        # 3. SMOGN Oversampled Random Forest
        X_train_smogn, y_train_smogn = self.smogn_oversampling(X_train, y_train)
        rf_smogn = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_smogn.fit(X_train_smogn, y_train_smogn)
        y_pred_smogn = rf_smogn.predict(X_test)
        
        models['smogn_rf'] = rf_smogn
        results['smogn_rf'] = self.stratified_evaluation(y_test, y_pred_smogn, "SMOGN RF")
        
        # 4. Two-Stage Model
        two_stage_model = self.two_stage_modeling(X_train, y_train)
        y_pred_two_stage = self.predict_two_stage(two_stage_model, X_test)
        
        models['two_stage'] = two_stage_model
        results['two_stage'] = self.stratified_evaluation(y_test, y_pred_two_stage, "Two-Stage")
        
        return models, results
    
    def get_best_model(self, results):
        """
        Select the best model based on stratified performance.
        
        Args:
            results (dict): Model evaluation results
            
        Returns:
            str: Best model name
        """
        # Focus on low visibility performance (safety-critical)
        low_vis_scores = {}
        
        for model_name, result in results.items():
            if '<1mi' in result['stratified']:
                # Weighted score favoring low visibility performance
                low_vis_score = (
                    result['stratified']['<1mi']['mae'] * 0.4 +
                    result['stratified']['1-3mi']['mae'] * 0.3 +
                    result['overall']['mae'] * 0.3
                )
                low_vis_scores[model_name] = low_vis_score
        
        best_model = min(low_vis_scores, key=low_vis_scores.get)
        print(f"\nBest model for low visibility prediction: {best_model}")
        
        return best_model
