#!/usr/bin/env python3
import sys
import json
import numpy as np

sys.path.append('src')
from src.pipeline.enhanced_prediction_pipeline import EnhancedPredictionPipeline


def make_case(temp_f, rh, wind_mph, wind_dir_deg, slp_mb, note):
    return {
        "form": {
            "DRYBULBTEMPF": str(temp_f),
            "RelativeHumidity": str(rh),
            "WindSpeed": str(wind_mph),
            "WindDirection": str(wind_dir_deg),
            "SeaLevelPressure": str(slp_mb),
        },
        "note": note,
    }


class MockRequest:
    def __init__(self, form_dict):
        self.form = form_dict


def main():
    cases = [
        make_case(72,35,8,270,1022,"8-10 (clear/dry high)"),
        make_case(55,25,12,320,1025,"9-10 (very dry post-frontal)"),
        make_case(40,45,6,330,1030,"8-10 (cold clear)"),
        make_case(92,55,7,190,1009,"5-8 (hazy hot)"),
        make_case(65,60,25,300,1015,"7-10 (gusty clearing)"),
        make_case(75,95,3,160,1012,"2-5 (muggy evening)"),
        make_case(58,98,1,10,1018,"<2 (radiation fog)"),
        make_case(50,99,0,0,1016,"<1-2 (dense fog)"),
        make_case(64,97,6,140,1016,"1-3 (sea fog onshore)"),
        make_case(57,98,2,220,1011,"1-3 (drizzle calm)"),
        make_case(60,95,7,110,1019,"2-6 (fog breakup)"),
        make_case(62,94,4,200,1008,"2-5 (light rain)"),
        make_case(60,92,15,220,1004,"3-6 (moderate rain + breeze)"),
        make_case(68,96,8,180,999,"1-4 (humid low pressure)"),
        make_case(70,90,28,240,1002,"1-4 (TS outflow)"),
        make_case(30,88,5,10,1010,"0.5-2 (snow + light wind)"),
        make_case(28,82,22,350,1007,"0.5-3 (blowing snow)"),
        make_case(34,92,4,20,1006,"1-3 (wet snow/drizzle)"),
        make_case(85,30,5,250,1013,"2-6 (smoke/haze)"),
        make_case(58,100,0,0,1018,"<1 (saturated + calm)"),
        make_case(58,95,4,180,1018,"1-3 (near-fog threshold)"),
        make_case(58,94,5,180,1018,"2-5 (just above threshold)"),
        make_case(45,97,1,50,1021,"<2 (cold fog)"),
        make_case(80,85,14,180,1014,"6-9 (humid but breezy)"),
        make_case(52,90,0,0,1024,"1-3 (calm humid night)"),
        make_case(60,90,8,0,1012,"2-5"),
        make_case(60,90,8,90,1012,"2-5"),
        make_case(60,90,8,180,1012,"2-5"),
        make_case(60,90,8,270,1012,"2-5"),
        make_case(66,92,10,210,950,"1-4 (deep low, rain likely)"),
        make_case(45,40,12,330,1040,"8-10 (very strong high)"),
        make_case(61,96,3,150,1007,"1-3 (humid, light wind)"),
        make_case(59,93,2,20,1015,"1-3 (near saturation)"),
        make_case(63,88,2,200,1013,"3-6 (humid calm)"),
        make_case(48,85,6,30,1009,"2-5 (cold drizzle)"),
        make_case(77,70,2,160,1010,"4-7 (humid calm, not saturated)"),
        make_case(72,80,20,200,1006,"6-9 (windy mixing)"),
        make_case(41,96,3,10,1018,"1-3 (freezing fog risk)"),
        make_case(55,75,1,220,1016,"4-8 (light wind)"),
        make_case(68,60,5,140,1018,"6-9 (fair)"),
        make_case(64,40,2,140,1018,"8-10 (dry calm)"),
    ]

    print("TempF,RH,Wind,Dir,SLP,visibility_miles,low_vis_prob,pi95_lo,pi95_hi,fog_signal,guardrail,note")
    for c in cases:
        req = MockRequest(c["form"]) 
        pipeline = EnhancedPredictionPipeline(request=req)
        res = pipeline.run_pipeline()
        print(
            f"{c['form']['DRYBULBTEMPF']},{c['form']['RelativeHumidity']},{c['form']['WindSpeed']}"
            f",{c['form']['WindDirection']},{c['form']['SeaLevelPressure']},{res['visibility_miles']:.2f}"
            f",{res['low_vis_prob']:.2f},{(res['pi95'][0] if res['pi95'] else '')}"
            f",{(res['pi95'][1] if res['pi95'] else '')},{res['fog_signal']},{int(res['guardrail_applied'])},"
            f"{c['note']}"
        )


if __name__ == "__main__":
    main()
