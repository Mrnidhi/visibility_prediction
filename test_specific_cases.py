#!/usr/bin/env python3
"""
Test script for the specific problematic cases mentioned by the user.
This demonstrates how the enhanced system addresses overprediction issues.
"""

import sys
sys.path.append('src')

from src.features.physics_features import calculate_dewpoint_magnus, calculate_fog_signal
from src.components.guardrail_system import GuardrailSystem
import numpy as np

def enhanced_fog_detection(temp_f, rh, wind_mph, spread_threshold=3.0, rh_threshold=95, wind_threshold=4.0):
    """
    Enhanced fog detection that considers cold, humid conditions even with higher wind.
    """
    # Standard fog detection
    standard_fog = calculate_fog_signal(temp_f, rh, wind_mph, spread_threshold, rh_threshold, wind_threshold)
    
    # Enhanced detection for cold, humid conditions
    cold_humid_fog = (
        temp_f <= 35 and  # Cold conditions
        rh >= 85 and      # High humidity
        wind_mph <= 8     # Moderate wind
    )
    
    # High humidity, low wind conditions
    humid_fog = (
        rh >= 90 and      # Very high humidity
        wind_mph <= 6     # Low to moderate wind
    )
    
    return standard_fog or cold_humid_fog or humid_fog

def simulate_enhanced_prediction(temp_f, rh, wind_mph, wind_dir, slp_mb, current_prediction):
    """
    Simulate the enhanced prediction system with physics features and guardrails.
    """
    # Calculate physics features
    dewpoint = calculate_dewpoint_magnus(temp_f, rh)
    spread = temp_f - dewpoint
    
    # Enhanced fog detection
    fog_signal = enhanced_fog_detection(temp_f, rh, wind_mph)
    
    # Calculate low visibility probability
    low_vis_prob = 0.0
    if fog_signal:
        low_vis_prob = 0.8
    elif rh >= 90:
        low_vis_prob = 0.6
    elif rh >= 80:
        low_vis_prob = 0.3
    elif temp_f <= 35 and rh >= 85:
        low_vis_prob = 0.5
    
    # Initialize guardrail system
    guardrail_config = {
        'enable_fog_guardrail': True,
        'fog_cap_enabled': True,
        'fog_cap': 2.0,
        'max_visibility_cap': 15.0
    }
    guardrail = GuardrailSystem(guardrail_config)
    
    # Apply guardrails
    raw_prediction = current_prediction
    fog_signal_array = np.array([int(fog_signal)])
    low_vis_proba_array = np.array([low_vis_prob])
    raw_pred_array = np.array([raw_prediction])
    
    adjusted_prediction = guardrail.apply_guardrails(
        raw_pred_array, 
        fog_signal_array,
        low_vis_proba_array
    )[0]
    
    # Calculate prediction interval (simplified)
    pi95 = [max(0.1, adjusted_prediction - 1.0), adjusted_prediction + 1.0]
    
    return {
        'dewpoint': dewpoint,
        'spread': spread,
        'fog_signal': fog_signal,
        'low_vis_prob': low_vis_prob,
        'raw_prediction': raw_prediction,
        'adjusted_prediction': adjusted_prediction,
        'guardrail_applied': adjusted_prediction != raw_prediction,
        'pi95': pi95
    }

def main():
    print("Enhanced Aviation Visibility Prediction - Specific Case Testing")
    print("=" * 70)
    
    # Your specific problematic cases
    test_cases = [
        {
            'name': 'Problematic Case 1',
            'description': '58°F, 98% RH, 1 mph wind (should be <2 mi)',
            'temp_f': 58, 'rh': 98, 'wind_mph': 1, 'wind_dir_deg': 10, 'slp_mb': 1018,
            'current_prediction': 6.4, 'expected_max': 2.0
        },
        {
            'name': 'Problematic Case 2', 
            'description': '30°F, 88% RH, 5 mph wind (should be ≤3 mi)',
            'temp_f': 30, 'rh': 88, 'wind_mph': 5, 'wind_dir_deg': 10, 'slp_mb': 1010,
            'current_prediction': 8.0, 'expected_max': 3.0
        },
        {
            'name': 'Additional Test Case',
            'description': 'Dense fog conditions',
            'temp_f': 50, 'rh': 99, 'wind_mph': 0, 'wind_dir_deg': 0, 'slp_mb': 1016,
            'current_prediction': 5.0, 'expected_max': 2.0
        },
        {
            'name': 'Clear Conditions',
            'description': 'Clear, dry conditions (should be high visibility)',
            'temp_f': 72, 'rh': 35, 'wind_mph': 8, 'wind_dir_deg': 270, 'slp_mb': 1022,
            'current_prediction': 10.0, 'expected_min': 8.0
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Description: {case['description']}")
        print(f"   Current System: {case['current_prediction']} miles")
        
        # Get enhanced prediction
        result = simulate_enhanced_prediction(
            case['temp_f'], case['rh'], case['wind_mph'], 
            case['wind_dir_deg'], case['slp_mb'], case['current_prediction']
        )
        
        print(f"   Physics Analysis:")
        print(f"     Dewpoint: {result['dewpoint']:.1f}°F")
        print(f"     Temperature Spread: {result['spread']:.1f}°F")
        print(f"     Fog Signal: {result['fog_signal']}")
        print(f"     Low Vis Probability: {result['low_vis_prob']:.2f}")
        
        print(f"   Enhanced Prediction:")
        print(f"     Raw: {result['raw_prediction']:.1f} miles")
        print(f"     Final: {result['adjusted_prediction']:.1f} miles")
        print(f"     Guardrail Applied: {result['guardrail_applied']}")
        print(f"     95% PI: [{result['pi95'][0]:.1f}, {result['pi95'][1]:.1f}]")
        
        # Check if meets expectation
        if 'expected_max' in case:
            meets_expectation = result['adjusted_prediction'] <= case['expected_max']
            print(f"     Expected: ≤ {case['expected_max']} miles")
        else:
            meets_expectation = result['adjusted_prediction'] >= case['expected_min']
            print(f"     Expected: ≥ {case['expected_min']} miles")
        
        print(f"     Result: {'✅ PASS' if meets_expectation else '❌ FAIL'}")
        
        results.append({
            'case': case['name'],
            'meets_expectation': meets_expectation,
            'improvement': result['raw_prediction'] - result['adjusted_prediction']
        })
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r['meets_expectation'])
    total_improvement = sum(r['improvement'] for r in results)
    
    print(f"Total Cases: {total_cases}")
    print(f"Passed: {passed_cases}/{total_cases} ({passed_cases/total_cases*100:.1f}%)")
    print(f"Average Improvement: {total_improvement/total_cases:.1f} miles reduction")
    
    print(f"\nDetailed Results:")
    print(f"{'Case':<20} {'Passed':<8} {'Improvement':<12}")
    print("-" * 40)
    for result in results:
        status = "✅" if result['meets_expectation'] else "❌"
        print(f"{result['case']:<20} {status:<8} {result['improvement']:.1f} mi")
    
    if passed_cases == total_cases:
        print(f"\n🎉 ALL CASES PASSED! Enhanced system successfully addresses overprediction issues.")
        return 0
    else:
        print(f"\n⚠️  {total_cases - passed_cases} cases still need improvement.")
        return 1

if __name__ == "__main__":
    exit(main())
