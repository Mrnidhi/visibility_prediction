"""
Guardrail system for aviation visibility prediction with fog-aware safety constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GuardrailSystem:
    """
    Guardrail system that applies physics-based safety constraints to predictions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize guardrail system.
        
        Args:
            config: Configuration dictionary with guardrail parameters
        """
        self.enable_fog_guardrail = config.get('enable_fog_guardrail', True)
        self.fog_cap_enabled = config.get('fog_cap_enabled', True)
        self.max_visibility_cap = config.get('max_visibility_cap', 15.0)
        self.fog_cap = config.get('fog_cap', 2.0)
        
        logger.info(f"Guardrail system initialized: fog_guardrail={self.enable_fog_guardrail}")
    
    def apply_guardrails(self, 
                        predictions: np.ndarray,
                        fog_signals: np.ndarray,
                        low_vis_proba: Optional[np.ndarray] = None,
                        low_vis_threshold: float = 2.0) -> np.ndarray:
        """
        Apply guardrails to predictions.
        
        Args:
            predictions: Raw model predictions
            fog_signals: Binary fog signals (1 if fog conditions, 0 otherwise)
            low_vis_proba: Low visibility probabilities (optional)
            low_vis_threshold: Low visibility threshold
            
        Returns:
            Guardrail-adjusted predictions
        """
        adjusted_predictions = predictions.copy()
        
        # Apply fog guardrail
        if self.enable_fog_guardrail and self.fog_cap_enabled:
            fog_mask = fog_signals == 1
            if np.any(fog_mask):
                adjusted_predictions[fog_mask] = np.minimum(
                    adjusted_predictions[fog_mask], 
                    self.fog_cap
                )
                logger.info(f"Applied fog guardrail to {np.sum(fog_mask)} predictions")
        
        # Apply low visibility probability guardrail
        if low_vis_proba is not None and self.enable_fog_guardrail:
            low_vis_proba = np.array(low_vis_proba)  # Ensure it's a numpy array
            high_low_vis_mask = low_vis_proba >= 0.8  # Very high probability of low visibility
            if np.any(high_low_vis_mask):
                adjusted_predictions[high_low_vis_mask] = np.minimum(
                    adjusted_predictions[high_low_vis_mask],
                    low_vis_threshold
                )
                logger.info(f"Applied low visibility guardrail to {np.sum(high_low_vis_mask)} predictions")
        
        # Apply maximum visibility cap
        adjusted_predictions = np.minimum(adjusted_predictions, self.max_visibility_cap)
        
        # Ensure non-negative predictions
        adjusted_predictions = np.maximum(adjusted_predictions, 0.0)
        
        return adjusted_predictions
    
    def apply_uncertainty_guardrails(self,
                                   predictions: Dict[str, np.ndarray],
                                   fog_signals: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply guardrails to uncertainty quantification predictions.
        
        Args:
            predictions: Dictionary of quantile predictions
            fog_signals: Binary fog signals
            
        Returns:
            Guardrail-adjusted quantile predictions
        """
        adjusted_predictions = {}
        
        for quantile_name, quantile_preds in predictions.items():
            adjusted_predictions[quantile_name] = self.apply_guardrails(
                quantile_preds, fog_signals
            )
        
        return adjusted_predictions


def test_guardrail_system():
    """Test the guardrail system."""
    try:
        # Test configuration
        config = {
            'enable_fog_guardrail': True,
            'fog_cap_enabled': True,
            'fog_cap': 2.0,
            'max_visibility_cap': 15.0
        }
        
        # Create test data
        predictions = np.array([10.0, 5.0, 1.5, 0.5, 8.0])
        fog_signals = np.array([1, 0, 1, 0, 0])  # Fog in positions 0 and 2
        low_vis_proba = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
        
        # Test guardrail system
        guardrail = GuardrailSystem(config)
        adjusted_preds = guardrail.apply_guardrails(predictions, fog_signals, low_vis_proba)
        
        # Check results
        expected = np.array([2.0, 5.0, 1.5, 0.5, 8.0])  # First prediction capped at 2.0
        assert np.allclose(adjusted_preds, expected), f"Expected {expected}, got {adjusted_preds}"
        
        print("✓ Guardrail system test passed")
        return True
        
    except Exception as e:
        print(f"✗ Guardrail system test failed: {e}")
        return False


if __name__ == "__main__":
    test_guardrail_system()
