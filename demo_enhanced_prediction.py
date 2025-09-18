#!/usr/bin/env python3
"""
Demo script for enhanced aviation visibility prediction system.

This script demonstrates the enhanced prediction capabilities with physics-aware features,
uncertainty quantification, and guardrail systems.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging

# Add src to path
sys.path.append('src')

from src.features.physics_features import (
    calculate_dewpoint_magnus, 
    calculate_fog_signal,
    test_dewpoint_calculation,
    test_fog_signal
)
from src.components.guardrail_system import GuardrailSystem, test_guardrail_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPredictionDemo:
    """
    Demo class for enhanced visibility prediction.
    """
    
    def __init__(self):
        self.test_cases = self._load_test_cases()
        self.guardrail_config = {
            'enable_fog_guardrail': True,
            'fog_cap_enabled': True,
            'fog_cap': 2.0,
            'max_visibility_cap': 15.0
        }
        self.guardrail = GuardrailSystem(self.guardrail_config)
    
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases for demonstration."""
        return [
            {
                "name": "Foggy Conditions",
                "description": "High humidity, low wind, small temperature spread",
                "input": {
                    "temp_f": 58,
                    "rh": 98,
                    "wind_mph": 1,
                    "wind_dir_deg": 10,
                    "slp_mb": 1018
                },
                "expected_range": (0.5, 2.0),
                "expected_fog_signal": True
            },
            {
                "name": "Dense Fog",
                "description": "Very high humidity, no wind, minimal temperature spread",
                "input": {
                    "temp_f": 50,
                    "rh": 99,
                    "wind_mph": 0,
                    "wind_dir_deg": 0,
                    "slp_mb": 1016
                },
                "expected_range": (0.1, 1.0),
                "expected_fog_signal": True
            },
            {
                "name": "Clear/Dry Conditions",
                "description": "Low humidity, moderate wind, large temperature spread",
                "input": {
                    "temp_f": 72,
                    "rh": 35,
                    "wind_mph": 8,
                    "wind_dir_deg": 270,
                    "slp_mb": 1022
                },
                "expected_range": (8.0, 12.0),
                "expected_fog_signal": False
            },
            {
                "name": "Muggy Light Wind",
                "description": "High humidity, light wind, moderate temperature spread",
                "input": {
                    "temp_f": 75,
                    "rh": 95,
                    "wind_mph": 3,
                    "wind_dir_deg": 160,
                    "slp_mb": 1012
                },
                "expected_range": (2.0, 5.0),
                "expected_fog_signal": True
            }
        ]
    
    def calculate_physics_features(self, case: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate physics-based features for a test case.
        
        Args:
            case: Test case dictionary
            
        Returns:
            Dictionary of calculated features
        """
        input_data = case["input"]
        
        # Calculate dewpoint
        dewpoint_f = calculate_dewpoint_magnus(input_data["temp_f"], input_data["rh"])
        
        # Calculate temperature spread
        spread_f = input_data["temp_f"] - dewpoint_f
        
        # Calculate wind direction encoding
        wind_rad = np.radians(input_data["wind_dir_deg"])
        wind_sin = np.sin(wind_rad)
        wind_cos = np.cos(wind_rad)
        
        # Calculate fog signal
        fog_signal = calculate_fog_signal(
            input_data["temp_f"], 
            input_data["rh"], 
            input_data["wind_mph"]
        )
        
        return {
            "dewpoint_f": dewpoint_f,
            "spread_f": spread_f,
            "wind_sin": wind_sin,
            "wind_cos": wind_cos,
            "fog_signal": fog_signal
        }
    
    def simulate_prediction(self, case: Dict[str, Any], physics_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate prediction using physics-based heuristics.
        
        Args:
            case: Test case dictionary
            physics_features: Calculated physics features
            
        Returns:
            Dictionary of prediction results
        """
        input_data = case["input"]
        
        # Base visibility calculation (simplified heuristic)
        base_visibility = 10.0  # Start with clear conditions
        
        # Reduce visibility based on humidity
        humidity_factor = 1.0 - (input_data["rh"] / 100.0) * 0.7
        base_visibility *= humidity_factor
        
        # Reduce visibility based on temperature spread
        if physics_features["spread_f"] < 3.0:
            base_visibility *= 0.3  # Significant reduction for small spread
        elif physics_features["spread_f"] < 5.0:
            base_visibility *= 0.6  # Moderate reduction
        
        # Reduce visibility based on wind speed
        if input_data["wind_mph"] < 2.0:
            base_visibility *= 0.5  # Low wind reduces visibility
        elif input_data["wind_mph"] > 15.0:
            base_visibility *= 1.2  # High wind can improve visibility
        
        # Add some randomness to simulate model uncertainty
        noise = np.random.normal(0, 0.5)
        raw_prediction = max(0.1, base_visibility + noise)
        
        # Calculate low visibility probability
        low_vis_prob = 0.0
        if physics_features["fog_signal"]:
            low_vis_prob = 0.8
        elif input_data["rh"] > 90:
            low_vis_prob = 0.6
        elif input_data["rh"] > 80:
            low_vis_prob = 0.3
        
        # Apply guardrails
        fog_signal_array = np.array([int(physics_features["fog_signal"])])
        low_vis_proba_array = np.array([low_vis_prob])
        raw_pred_array = np.array([raw_prediction])
        
        adjusted_prediction = self.guardrail.apply_guardrails(
            raw_pred_array, 
            fog_signal_array, 
            low_vis_proba_array
        )[0]
        
        # Calculate prediction interval (simplified)
        pi95 = [max(0.1, adjusted_prediction - 1.0), adjusted_prediction + 1.0]
        
        return {
            "visibility_miles": adjusted_prediction,
            "low_vis_prob": low_vis_prob,
            "fog_signal": int(physics_features["fog_signal"]),
            "pi95": pi95,
            "raw_prediction": raw_prediction,
            "guardrail_applied": adjusted_prediction != raw_prediction
        }
    
    def run_demo(self):
        """Run the complete demonstration."""
        print("="*80)
        print("ENHANCED AVIATION VISIBILITY PREDICTION DEMO")
        print("="*80)
        
        # Run unit tests first
        print("\n1. RUNNING UNIT TESTS")
        print("-" * 40)
        test_dewpoint_calculation()
        test_fog_signal()
        test_guardrail_system()
        
        # Run test cases
        print("\n2. RUNNING TEST CASES")
        print("-" * 40)
        
        results = []
        for i, case in enumerate(self.test_cases, 1):
            print(f"\nTest Case {i}: {case['name']}")
            print(f"Description: {case['description']}")
            print(f"Input: {json.dumps(case['input'], indent=2)}")
            
            # Calculate physics features
            physics_features = self.calculate_physics_features(case)
            print(f"Physics Features:")
            for key, value in physics_features.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            
            # Simulate prediction
            prediction_result = self.simulate_prediction(case, physics_features)
            
            print(f"Prediction Results:")
            print(f"  Visibility: {prediction_result['visibility_miles']:.3f} miles")
            print(f"  Low Vis Probability: {prediction_result['low_vis_prob']:.3f}")
            print(f"  Fog Signal: {prediction_result['fog_signal']}")
            print(f"  95% PI: [{prediction_result['pi95'][0]:.3f}, {prediction_result['pi95'][1]:.3f}]")
            print(f"  Guardrail Applied: {prediction_result['guardrail_applied']}")
            
            # Check if prediction is in expected range
            pred_vis = prediction_result['visibility_miles']
            expected_min, expected_max = case['expected_range']
            in_range = expected_min <= pred_vis <= expected_max
            
            print(f"Expected Range: {expected_min}-{expected_max} miles")
            print(f"✓ In Range: {in_range}")
            
            # Check fog signal
            fog_correct = prediction_result['fog_signal'] == case['expected_fog_signal']
            print(f"✓ Fog Signal Correct: {fog_correct}")
            
            results.append({
                'case': case['name'],
                'prediction': pred_vis,
                'expected_range': case['expected_range'],
                'in_range': in_range,
                'fog_correct': fog_correct
            })
        
        # Summary
        print("\n3. DEMO SUMMARY")
        print("-" * 40)
        total_cases = len(results)
        in_range_count = sum(1 for r in results if r['in_range'])
        fog_correct_count = sum(1 for r in results if r['fog_correct'])
        
        print(f"Total Test Cases: {total_cases}")
        print(f"Predictions in Expected Range: {in_range_count}/{total_cases} ({in_range_count/total_cases*100:.1f}%)")
        print(f"Fog Signal Correct: {fog_correct_count}/{total_cases} ({fog_correct_count/total_cases*100:.1f}%)")
        
        # Detailed results table
        print(f"\nDetailed Results:")
        print(f"{'Case':<20} {'Prediction':<12} {'Expected':<15} {'In Range':<10} {'Fog OK':<8}")
        print("-" * 70)
        for result in results:
            expected_str = f"{result['expected_range'][0]}-{result['expected_range'][1]}"
            print(f"{result['case']:<20} {result['prediction']:<12.3f} {expected_str:<15} "
                  f"{'✓' if result['in_range'] else '✗':<10} {'✓' if result['fog_correct'] else '✗':<8}")
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return results


def main():
    """Main function to run the demo."""
    try:
        demo = EnhancedPredictionDemo()
        results = demo.run_demo()
        
        # Check if all tests passed
        all_in_range = all(r['in_range'] for r in results)
        all_fog_correct = all(r['fog_correct'] for r in results)
        
        if all_in_range and all_fog_correct:
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("\n⚠️  SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
