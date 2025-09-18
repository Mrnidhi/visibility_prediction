"""
Enhanced data transformation with physics-aware features and time-series validation.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import joblib
import yaml
from datetime import datetime, timedelta
import logging

from src.exception import VisibilityException
from src.utils.main_utils import MainUtils
from src.features.physics_features import PhysicsFeatureTransformer

logger = logging.getLogger(__name__)


class EnhancedDataTransformationConfig:
    """Configuration for enhanced data transformation."""
    
    def __init__(self, config_path: str = "config/enhanced_model.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create timestamped directory
        timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
        self.artifact_folder = os.path.join("artifacts", timestamp)
        
        self.data_transformation_dir = os.path.join(self.artifact_folder, 'enhanced_data_transformation')
        self.transformed_train_file_path = os.path.join(self.data_transformation_dir, 'train.npy')
        self.transformed_test_file_path = os.path.join(self.data_transformation_dir, 'test.npy')
        self.preprocessor_path = os.path.join(self.data_transformation_dir, 'preprocessor.pkl')
        self.feature_names_path = os.path.join(self.data_transformation_dir, 'feature_names.pkl')
        self.time_splits_path = os.path.join(self.data_transformation_dir, 'time_splits.pkl')


class EnhancedDataTransformation:
    """
    Enhanced data transformation with physics features and time-series validation.
    """
    
    def __init__(self, valid_data_dir: str, config_path: str = "config/enhanced_model.yaml"):
        self.valid_data_dir = valid_data_dir
        self.config = EnhancedDataTransformationConfig(config_path)
        self.utils = MainUtils()
        self.schema_config = self.utils.read_schema_config_file()
        
        # Create directories
        os.makedirs(self.config.data_transformation_dir, exist_ok=True)
        
        # Initialize physics feature transformer
        physics_config = self.config.config['physics']
        self.physics_transformer = PhysicsFeatureTransformer(
            fog_spread_threshold=physics_config['fog_spread_threshold'],
            fog_rh_threshold=physics_config['fog_rh_threshold'],
            fog_wind_threshold=physics_config['fog_wind_threshold']
        )
        
    def create_time_series_splits(self, data: pd.DataFrame) -> tuple:
        """
        Create time-series splits for validation.
        
        Args:
            data: DataFrame with DATE column
            
        Returns:
            Tuple of (train_indices, test_indices)
        """
        # Sort by date
        data_sorted = data.sort_values('DATE').reset_index(drop=True)
        
        # Calculate split point (hold out last month)
        test_size_months = self.config.config['time_validation']['test_size_months']
        n_splits = self.config.config['time_validation']['n_splits']
        
        # For simplicity, use last 20% as test set
        split_idx = int(len(data_sorted) * 0.8)
        
        train_indices = data_sorted.index[:split_idx].values
        test_indices = data_sorted.index[split_idx:].values
        
        logger.info(f"Time series split: {len(train_indices)} train, {len(test_indices)} test samples")
        
        return train_indices, test_indices
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Create preprocessing pipeline with physics features.
        
        Returns:
            Fitted preprocessing pipeline
        """
        # Define column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('physics', self.physics_transformer, 
                 ['DRYBULBTEMPF', 'RelativeHumidity', 'WindSpeed', 'WindDirection']),
                ('scaler', StandardScaler(), 
                 ['SeaLevelPressure'])  # Scale pressure separately
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def initiate_data_transformation(self) -> tuple:
        """
        Initiate enhanced data transformation.
        
        Returns:
            Tuple of (train_array, test_array, preprocessor_path)
        """
        try:
            logger.info("Starting enhanced data transformation...")
            
            # Load and validate data
            data_files = [f for f in os.listdir(self.valid_data_dir) if f.endswith('.csv')]
            if not data_files:
                raise Exception("No validated data files found")
            
            # Load the first validated file
            data_path = os.path.join(self.valid_data_dir, data_files[0])
            data = pd.read_csv(data_path)
            
            logger.info(f"Loaded data: {data.shape}")
            logger.info(f"Columns: {data.columns.tolist()}")
            
            # Create time series splits
            train_indices, test_indices = self.create_time_series_splits(data)
            
            # Split data
            train_data = data.iloc[train_indices].copy()
            test_data = data.iloc[test_indices].copy()
            
            # Separate features and target
            feature_columns = ['DRYBULBTEMPF', 'RelativeHumidity', 'WindSpeed', 
                             'WindDirection', 'SeaLevelPressure']
            target_column = 'VISIBILITY'
            
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Create and fit preprocessing pipeline
            preprocessor = self.create_preprocessing_pipeline()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # Get feature names
            feature_names = self.physics_transformer.get_feature_names() + ['SeaLevelPressure']
            
            # Create final arrays
            train_array = np.column_stack([X_train_transformed, y_train.values])
            test_array = np.column_stack([X_test_transformed, y_test.values])
            
            # Save artifacts
            np.save(self.config.transformed_train_file_path, train_array)
            np.save(self.config.transformed_test_file_path, test_array)
            joblib.dump(preprocessor, self.config.preprocessor_path)
            joblib.dump(feature_names, self.config.feature_names_path)
            
            # Save time split information
            time_splits = {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_dates': train_data['DATE'].tolist(),
                'test_dates': test_data['DATE'].tolist()
            }
            joblib.dump(time_splits, self.config.time_splits_path)
            
            logger.info(f"Enhanced transformation completed:")
            logger.info(f"  Train shape: {train_array.shape}")
            logger.info(f"  Test shape: {test_array.shape}")
            logger.info(f"  Features: {feature_names}")
            
            return train_array, test_array, self.config.preprocessor_path
            
        except Exception as e:
            raise VisibilityException(e, sys)


def test_enhanced_transformation():
    """Test the enhanced data transformation."""
    try:
        # Create test data
        test_data = pd.DataFrame({
            'DATE': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'DRYBULBTEMPF': np.random.normal(70, 10, 1000),
            'RelativeHumidity': np.random.normal(60, 20, 1000),
            'WindSpeed': np.random.exponential(5, 1000),
            'WindDirection': np.random.uniform(0, 360, 1000),
            'SeaLevelPressure': np.random.normal(1013, 10, 1000),
            'VISIBILITY': np.random.exponential(5, 1000)
        })
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data
            test_file = os.path.join(temp_dir, 'test_data.csv')
            test_data.to_csv(test_file, index=False)
            
            # Test transformation
            transformer = EnhancedDataTransformation(temp_dir)
            train_array, test_array, preprocessor_path = transformer.initiate_data_transformation()
            
            print(f"✓ Enhanced transformation test passed:")
            print(f"  Train shape: {train_array.shape}")
            print(f"  Test shape: {test_array.shape}")
            
            return True
            
    except Exception as e:
        print(f"✗ Enhanced transformation test failed: {e}")
        return False


if __name__ == "__main__":
    test_enhanced_transformation()
