#!/usr/bin/env python3
import sys
import math
import numpy as np

sys.path.append('src')
from src.features.physics_features import calculate_dewpoint_magnus
from src.components.guardrail_system import GuardrailSystem

# Reuse the enhanced fog detection and simulation from test_specific_cases

def enhanced_fog_detection(temp_f, rh, wind_mph, spread_threshold=3.0, rh_threshold=95, wind_threshold=4.0):
    dewpoint = calculate_dewpoint_magnus(temp_f, rh)
    spread = temp_f - dewpoint
    standard_fog = (spread <= spread_threshold) and (rh >= rh_threshold) and (wind_mph <= wind_threshold)
    cold_humid_fog = (temp_f <= 35) and (rh >= 85) and (wind_mph <= 8)
    humid_fog = (rh >= 90) and (wind_mph <= 6)
    return standard_fog or cold_humid_fog or humid_fog


def simulate_enhanced_prediction(temp_f, rh, wind_mph, wind_dir, slp_mb, current_guess=None):
    # Physics
    dewpoint = calculate_dewpoint_magnus(temp_f, rh)
    spread = temp_f - dewpoint

    # Base heuristic guess if none provided (very lightweight, not a trained model)
    if current_guess is None:
        base = 10.0
        # Humidity impact
        base *= (1.0 - min(max(rh, 0), 100) / 100.0 * 0.7)
        # Spread impact
        if spread < 3.0:
            base *= 0.3
        elif spread < 5.0:
            base *= 0.6
        # Wind impact
        if wind_mph < 2.0:
            base *= 0.5
        elif wind_mph > 20.0:
            base *= 1.2
        raw_prediction = max(0.1, base)
    else:
        raw_prediction = float(current_guess)

    # Low vis probability proxy
    fog_signal = enhanced_fog_detection(temp_f, rh, wind_mph)
    if fog_signal:
        low_vis_prob = 0.8
    elif rh >= 90:
        low_vis_prob = 0.6
    elif rh >= 80:
        low_vis_prob = 0.3
    else:
        low_vis_prob = 0.05

    # Guardrails
    guardrail = GuardrailSystem({
        'enable_fog_guardrail': True,
        'fog_cap_enabled': True,
        'fog_cap': 2.0,
        'max_visibility_cap': 15.0
    })

    adjusted = guardrail.apply_guardrails(
        np.array([raw_prediction]),
        np.array([1 if fog_signal else 0]),
        np.array([low_vis_prob])
    )[0]

    # Simple PI
    pi95 = [max(0.1, adjusted - 1.0), adjusted + 1.0]

    return {
        'visibility_miles': float(adjusted),
        'low_vis_prob': float(low_vis_prob),
        'fog_signal': int(fog_signal),
        'pi95': pi95,
        'raw': float(raw_prediction),
        'guardrail': bool(abs(adjusted - raw_prediction) > 1e-6)
    }


def main():
    rows = [
        (72,35,8,270,1022,"8-10 (clear/dry high)"),
        (55,25,12,320,1025,"9-10 (very dry post-frontal)"),
        (40,45,6,330,1030,"8-10 (cold clear)"),
        (92,55,7,190,1009,"5-8 (hazy hot)"),
        (65,60,25,300,1015,"7-10 (gusty clearing)"),
        (75,95,3,160,1012,"2-5 (muggy evening)"),
        (58,98,1,10,1018,"<2 (radiation fog)"),
        (50,99,0,0,1016,"<1-2 (dense fog)"),
        (64,97,6,140,1016,"1-3 (sea fog onshore)"),
        (57,98,2,220,1011,"1-3 (drizzle calm)"),
        (60,95,7,110,1019,"2-6 (fog breakup)"),
        (62,94,4,200,1008,"2-5 (light rain)"),
        (60,92,15,220,1004,"3-6 (moderate rain + breeze)"),
        (68,96,8,180,999,"1-4 (humid low pressure)"),
        (70,90,28,240,1002,"1-4 (TS outflow)"),
        (30,88,5,10,1010,"0.5-2 (snow + light wind)"),
        (28,82,22,350,1007,"0.5-3 (blowing snow)"),
        (34,92,4,20,1006,"1-3 (wet snow/drizzle)"),
        (85,30,5,250,1013,"2-6 (smoke/haze)"),
        (58,100,0,0,1018,"<1 (saturated + calm)"),
        (58,95,4,180,1018,"1-3 (near-fog threshold)"),
        (58,94,5,180,1018,"2-5 (just above threshold)"),
        (45,97,1,50,1021,"<2 (cold fog)"),
        (80,85,14,180,1014,"6-9 (humid but breezy)"),
        (52,90,0,0,1024,"1-3 (calm humid night)"),
        (60,90,8,0,1012,"2-5"),
        (60,90,8,90,1012,"2-5"),
        (60,90,8,180,1012,"2-5"),
        (60,90,8,270,1012,"2-5"),
        (66,92,10,210,950,"1-4 (deep low, rain likely)"),
        (45,40,12,330,1040,"8-10 (very strong high)"),
        (61,96,3,150,1007,"1-3 (humid, light wind)"),
        (59,93,2,20,1015,"1-3 (near saturation)"),
        (63,88,2,200,1013,"3-6 (humid calm)"),
        (48,85,6,30,1009,"2-5 (cold drizzle)"),
        (77,70,2,160,1010,"4-7 (humid calm, not saturated)"),
        (72,80,20,200,1006,"6-9 (windy mixing)"),
        (41,96,3,10,1018,"1-3 (freezing fog risk)"),
        (55,75,1,220,1016,"4-8 (light wind)"),
        (68,60,5,140,1018,"6-9 (fair)"),
        (64,40,2,140,1018,"8-10 (dry calm)")
    ]

    print(f"{'TempF':>5} {'RH%':>4} {'Wspd':>5} {'Dir':>4} {'SLP':>5}  {'Pred(mi)':>8}  {'PI95':>15}  {'LowVisP':>7}  {'Fog':>3}  {'Guard':>5}  Expected")
    for (t,rh,w,wd,slp,exp) in rows:
        res = simulate_enhanced_prediction(t, rh, w, wd, slp)
        pi = f"[{res['pi95'][0]:.1f},{res['pi95'][1]:.1f}]"
        print(f"{t:5d} {rh:4d} {w:5d} {wd:4d} {slp:5d}  {res['visibility_miles']:8.2f}  {pi:>15}  {res['low_vis_prob']:7.2f}  {res['fog_signal']:>3d}  {str(res['guardrail'])[0]:>5}  {exp}")

if __name__ == '__main__':
    main()
