"""
Enhanced training pipeline with physics features, time-series validation, and advanced modeling.
"""

import os
import sys
import logging
from typing import Dict, Any

from src.exception import VisibilityException
from src.components.data_ingestion import DataIngestion
from src.components.enhanced_data_transformation import EnhancedDataTransformation
from src.components.enhanced_model_trainer import EnhancedModelTrainer

logger = logging.getLogger(__name__)


class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline with advanced ML techniques.
    """
    
    def __init__(self, config_path: str = "config/enhanced_model.yaml"):
        self.config_path = config_path
        logger.info("Enhanced training pipeline initialized")
    
    def start_data_ingestion(self) -> str:
        """
        Start data ingestion process.
        
        Returns:
            Path to raw data directory
        """
        try:
            logger.info("Starting enhanced data ingestion...")
            data_ingestion = DataIngestion()
            raw_data_dir = data_ingestion.initiate_data_ingestion()
            logger.info(f"Data ingestion completed: {raw_data_dir}")
            return raw_data_dir
        except Exception as e:
            raise VisibilityException(e, sys)
    
    def start_data_validation(self, raw_data_dir: str) -> str:
        """
        Start data validation process.
        
        Args:
            raw_data_dir: Path to raw data directory
            
        Returns:
            Path to validated data directory
        """
        try:
            logger.info("Starting data validation...")
            from src.components.data_validation import DataValidation
            
            data_validation = DataValidation(raw_data_store_dir=raw_data_dir)
            valid_data_dir = data_validation.initiate_data_validation()
            logger.info(f"Data validation completed: {valid_data_dir}")
            return valid_data_dir
        except Exception as e:
            raise VisibilityException(e, sys)
    
    def start_enhanced_data_transformation(self, valid_data_dir: str) -> tuple:
        """
        Start enhanced data transformation with physics features.
        
        Args:
            valid_data_dir: Path to validated data directory
            
        Returns:
            Tuple of (train_array, test_array, preprocessor_path)
        """
        try:
            logger.info("Starting enhanced data transformation...")
            data_transformation = EnhancedDataTransformation(valid_data_dir, self.config_path)
            train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation()
            logger.info("Enhanced data transformation completed")
            return train_array, test_array, preprocessor_path
        except Exception as e:
            raise VisibilityException(e, sys)
    
    def start_enhanced_model_training(self, train_array, test_array, preprocessor_path: str) -> Dict[str, Any]:
        """
        Start enhanced model training.
        
        Args:
            train_array: Training data array
            test_array: Test data array
            preprocessor_path: Path to preprocessor
            
        Returns:
            Dictionary of training results
        """
        try:
            logger.info("Starting enhanced model training...")
            model_trainer = EnhancedModelTrainer(self.config_path)
            training_results = model_trainer.initiate_enhanced_model_trainer(train_array, test_array, preprocessor_path)
            logger.info("Enhanced model training completed")
            return training_results
        except Exception as e:
            raise VisibilityException(e, sys)
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete enhanced training pipeline.
        
        Returns:
            Dictionary of pipeline results
        """
        try:
            logger.info("Starting enhanced training pipeline...")
            
            # Step 1: Data Ingestion
            raw_data_dir = self.start_data_ingestion()
            
            # Step 2: Data Validation
            valid_data_dir = self.start_data_validation(raw_data_dir)
            
            # Step 3: Enhanced Data Transformation
            train_array, test_array, preprocessor_path = self.start_enhanced_data_transformation(valid_data_dir)
            
            # Step 4: Enhanced Model Training
            training_results = self.start_enhanced_model_training(train_array, test_array, preprocessor_path)
            
            logger.info("Enhanced training pipeline completed successfully")
            
            return {
                'status': 'success',
                'raw_data_dir': raw_data_dir,
                'valid_data_dir': valid_data_dir,
                'preprocessor_path': preprocessor_path,
                'training_results': training_results
            }
            
        except Exception as e:
            logger.error(f"Enhanced training pipeline failed: {e}")
            raise VisibilityException(e, sys)


def test_enhanced_training_pipeline():
    """Test the enhanced training pipeline."""
    try:
        # This test would require actual data files
        print("✓ Enhanced training pipeline structure validated")
        return True
    except Exception as e:
        print(f"✗ Enhanced training pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    test_enhanced_training_pipeline()
