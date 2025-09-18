"""
Physics-aware feature engineering for aviation visibility prediction.

This module implements meteorological features based on atmospheric physics
to improve visibility prediction accuracy, especially in foggy conditions.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, List
import logging

logger = logging.getLogger(__name__)


class PhysicsFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that adds physics-based features for visibility prediction.
    
    Features added:
    - Dewpoint temperature (Magnus formula)
    - Temperature spread (dry bulb - dewpoint)
    - Wind direction encoding (sin/cos)
    - Fog signal (binary indicator)
    """
    
    def __init__(self, 
                 fog_spread_threshold: float = 3.0,
                 fog_rh_threshold: float = 95.0,
                 fog_wind_threshold: float = 4.0):
        """
        Initialize the physics feature transformer.
        
        Args:
            fog_spread_threshold: Temperature spread threshold for fog detection (°F)
            fog_rh_threshold: Relative humidity threshold for fog detection (%)
            fog_wind_threshold: Wind speed threshold for fog detection (mph)
        """
        self.fog_spread_threshold = fog_spread_threshold
        self.fog_rh_threshold = fog_rh_threshold
        self.fog_wind_threshold = fog_wind_threshold
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform input data by adding physics-based features.
        
        Args:
            X: Input data with columns [DRYBULBTEMPF, RelativeHumidity, WindSpeed, WindDirection]
            
        Returns:
            Array with original features plus physics features
        """
        if isinstance(X, np.ndarray):
            # Convert to DataFrame for easier column access
            feature_names = ['DRYBULBTEMPF', 'RelativeHumidity', 'WindSpeed', 'WindDirection']
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
        
        # Ensure we have the required columns
        required_cols = ['DRYBULBTEMPF', 'RelativeHumidity', 'WindSpeed', 'WindDirection']
        missing_cols = [col for col in required_cols if col not in X_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate dewpoint using Magnus formula
        dewpoint_f = self._calculate_dewpoint(
            X_df['DRYBULBTEMPF'], 
            X_df['RelativeHumidity']
        )
        
        # Calculate temperature spread
        spread_f = X_df['DRYBULBTEMPF'] - dewpoint_f
        
        # Wind direction encoding
        wind_rad = np.radians(X_df['WindDirection'])
        wind_sin = np.sin(wind_rad)
        wind_cos = np.cos(wind_rad)
        
        # Fog signal
        fog_signal = self._calculate_fog_signal(
            spread_f, 
            X_df['RelativeHumidity'], 
            X_df['WindSpeed']
        )
        
        # Combine original features with new physics features
        physics_features = np.column_stack([
            X_df['DRYBULBTEMPF'].values,
            X_df['RelativeHumidity'].values,
            X_df['WindSpeed'].values,
            X_df['WindDirection'].values,
            dewpoint_f,
            spread_f,
            wind_sin,
            wind_cos,
            fog_signal
        ])
        
        logger.info(f"Added physics features. Shape: {physics_features.shape}")
        return physics_features
    
    def _calculate_dewpoint(self, temp_f: pd.Series, rh: pd.Series) -> np.ndarray:
        """
        Calculate dewpoint temperature using Magnus formula.
        
        Args:
            temp_f: Dry bulb temperature in Fahrenheit
            rh: Relative humidity in percent
            
        Returns:
            Dewpoint temperature in Fahrenheit
        """
        # Convert to Celsius for calculation
        temp_c = (temp_f - 32) * 5/9
        rh_decimal = rh / 100.0
        
        # Magnus formula parameters
        a = 17.27
        b = 237.7
        
        # Calculate dewpoint in Celsius
        gamma = (a * temp_c) / (b + temp_c) + np.log(rh_decimal)
        dewpoint_c = (b * gamma) / (a - gamma)
        
        # Convert back to Fahrenheit
        dewpoint_f = (dewpoint_c * 9/5) + 32
        
        return dewpoint_f
    
    def _calculate_fog_signal(self, spread_f: np.ndarray, rh: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
        """
        Calculate binary fog signal based on meteorological conditions.
        
        Args:
            spread_f: Temperature spread in Fahrenheit
            rh: Relative humidity in percent
            wind_speed: Wind speed in mph
            
        Returns:
            Binary fog signal (1 if fog conditions, 0 otherwise)
        """
        fog_conditions = (
            (spread_f <= self.fog_spread_threshold) &
            (rh >= self.fog_rh_threshold) &
            (wind_speed <= self.fog_wind_threshold)
        )
        
        return fog_conditions.astype(int)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features (original + physics)."""
        return [
            'DRYBULBTEMPF',
            'RelativeHumidity', 
            'WindSpeed',
            'WindDirection',
            'DEWPOINT_F',
            'SPREAD_F',
            'WIND_SIN',
            'WIND_COS',
            'FOG_SIGNAL'
        ]


def calculate_dewpoint_magnus(temp_f: float, rh: float) -> float:
    """
    Calculate dewpoint temperature using Magnus formula.
    
    Args:
        temp_f: Dry bulb temperature in Fahrenheit
        rh: Relative humidity in percent
        
    Returns:
        Dewpoint temperature in Fahrenheit
    """
    # Convert to Celsius
    temp_c = (temp_f - 32) * 5/9
    rh_decimal = rh / 100.0
    
    # Magnus formula
    a, b = 17.27, 237.7
    gamma = (a * temp_c) / (b + temp_c) + np.log(rh_decimal)
    dewpoint_c = (b * gamma) / (a - gamma)
    
    # Convert back to Fahrenheit
    return (dewpoint_c * 9/5) + 32


def calculate_fog_signal(temp_f: float, rh: float, wind_speed: float, 
                        spread_threshold: float = 3.0,
                        rh_threshold: float = 95.0,
                        wind_threshold: float = 4.0) -> bool:
    """
    Calculate fog signal for given meteorological conditions.
    
    Args:
        temp_f: Dry bulb temperature in Fahrenheit
        rh: Relative humidity in percent
        wind_speed: Wind speed in mph
        spread_threshold: Temperature spread threshold for fog detection
        rh_threshold: Relative humidity threshold for fog detection
        wind_threshold: Wind speed threshold for fog detection
        
    Returns:
        True if fog conditions are detected, False otherwise
    """
    dewpoint_f = calculate_dewpoint_magnus(temp_f, rh)
    spread_f = temp_f - dewpoint_f
    
    return (
        spread_f <= spread_threshold and
        rh >= rh_threshold and
        wind_speed <= wind_threshold
    )


# Unit test functions
def test_dewpoint_calculation():
    """Test dewpoint calculation with known values."""
    # Test case: 70°F, 60% RH should give dewpoint around 55°F
    dewpoint = calculate_dewpoint_magnus(70.0, 60.0)
    expected_range = (54.0, 56.0)
    
    assert expected_range[0] <= dewpoint <= expected_range[1], \
        f"Dewpoint {dewpoint:.1f}°F not in expected range {expected_range}"
    
    print(f"✓ Dewpoint test passed: {dewpoint:.1f}°F")
    return True


def test_fog_signal():
    """Test fog signal calculation."""
    # Foggy conditions
    assert calculate_fog_signal(58.0, 98.0, 1.0) == True, "Should detect fog"
    
    # Clear conditions
    assert calculate_fog_signal(72.0, 35.0, 8.0) == False, "Should not detect fog"
    
    print("✓ Fog signal test passed")
    return True


if __name__ == "__main__":
    # Run tests
    test_dewpoint_calculation()
    test_fog_signal()
    print("All physics feature tests passed!")
