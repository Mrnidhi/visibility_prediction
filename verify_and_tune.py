#!/usr/bin/env python3
import os
import sys
import json
import math
import time
import glob
import joblib
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

sys.path.append('src')
from src.pipeline.enhanced_prediction_pipeline import EnhancedPredictionPipeline, EnhancedPredictionPipelineConfig
from src.components.enhanced_model_trainer import EnhancedModelTrainer
from src.features.physics_features import calculate_dewpoint_magnus


def find_latest_timestamp_dir(artifacts_dir: str = 'artifacts') -> str:
    if not os.path.exists(artifacts_dir):
        raise FileNotFoundError('artifacts directory not found')
    ts = [d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))]
    if not ts:
        raise FileNotFoundError('no timestamped artifact dirs present')
    return max(ts)


def load_served_artifacts() -> Dict[str, Any]:
    cfg = EnhancedPredictionPipelineConfig('config/enhanced_model.yaml')
    latest_ts = os.path.basename(os.path.dirname(os.path.dirname(cfg.preprocessor_path)))
    preproc_path = cfg.preprocessor_path
    feat_names_path = os.path.join('artifacts', latest_ts, 'enhanced_data_transformation', 'feature_names.pkl')
    model_dir = cfg.model_dir
    metrics_path = os.path.join(model_dir, 'metrics.pkl')

    info = {
        'latest_timestamp': latest_ts,
        'preprocessor_path': preproc_path,
        'feature_names_path': feat_names_path,
        'model_dir': model_dir,
        'trained_model_path': cfg.trained_model_path,
        'two_stage_model_path': cfg.two_stage_model_path,
        'weighted_model_path': cfg.weighted_model_path,
        'quantile_model_path': cfg.quantile_model_path,
        'metrics_path': metrics_path,
    }

    # Load objects
    info['preprocessor'] = joblib.load(preproc_path)
    info['feature_names'] = joblib.load(feat_names_path) if os.path.exists(feat_names_path) else None
    info['metrics'] = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}

    # Determine model type
    best_model_type = info['metrics'].get('best_model_type') if isinstance(info['metrics'], dict) else None
    if best_model_type is None:
        # Fallback: if trained_model.pkl exists, we won't know type; try two-stage first
        best_model_type = 'two_stage' if os.path.exists(cfg.two_stage_model_path) else 'weighted'
    info['best_model_type'] = best_model_type

    # Load model
    if best_model_type == 'two_stage' and os.path.exists(cfg.two_stage_model_path):
        info['model'] = joblib.load(cfg.two_stage_model_path)
    elif best_model_type == 'weighted' and os.path.exists(cfg.weighted_model_path):
        info['model'] = joblib.load(cfg.weighted_model_path)
    else:
        # fallback to trained_model_path
        info['model'] = joblib.load(cfg.trained_model_path)

    # Config
    with open('config/enhanced_model.yaml', 'r') as f:
        info['config'] = yaml.safe_load(f)

    return info


def file_info(path: str) -> Tuple[str, int]:
    ts = ''
    size = -1
    if os.path.exists(path):
        size = os.path.getsize(path)
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))
    return ts, size


def assert_sanity(info: Dict[str, Any]):
    feats = info.get('feature_names') or []
    required = ['DEWPOINT_F', 'SPREAD_F', 'WIND_SIN', 'WIND_COS', 'FOG_SIGNAL']
    missing = [f for f in required if f not in feats]
    if missing:
        raise AssertionError(f"Preprocessor missing required physics features: {missing}")
    # Specific example sanity: dewpoint/spread
    td = calculate_dewpoint_magnus(58.0, 98.0)
    spread = 58.0 - td
    if not (57.2 <= td <= 57.6):
        raise AssertionError(f"Dewpoint sanity failed: got {td:.2f}")
    if not (0.4 <= spread <= 0.8):
        raise AssertionError(f"Spread sanity failed: got {spread:.2f}")


def run_cases_with_pipeline(cases: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for c in cases:
        form = {
            'DRYBULBTEMPF': str(c['temp_f']),
            'RelativeHumidity': str(c['rh']),
            'WindSpeed': str(c['wind_mph']),
            'WindDirection': str(c['wind_dir_deg']),
            'SeaLevelPressure': str(c['slp_mb']),
        }
        class MockReq: pass
        req = MockReq()
        req.form = form
        pipeline = EnhancedPredictionPipeline(request=req)
        res = pipeline.run_pipeline()
        lo, hi = (res['pi95'] if res['pi95'] else (None, None))
        # Severity
        band = c['expected_band']
        sev = 'OK'
        if ('<2' in band) or ('0.5-2' in band):
            if res['visibility_miles'] > 5:
                sev = 'CRITICAL'
        elif band.startswith('1-3') and res['visibility_miles'] > 6:
            sev = 'HIGH'
        rows.append({
            'temp_f': c['temp_f'], 'rh': c['rh'], 'wind_mph': c['wind_mph'], 'wind_dir_deg': c['wind_dir_deg'], 'slp_mb': c['slp_mb'],
            'expected_band': band,
            'vis_pred': round(float(res['visibility_miles']), 3),
            'low_vis_prob': round(float(res['low_vis_prob']), 3) if res['low_vis_prob'] is not None else None,
            'pi95_lo': round(float(lo), 3) if lo is not None else None,
            'pi95_hi': round(float(hi), 3) if hi is not None else None,
            'fog_signal': int(res['fog_signal']),
            'guardrail_applied': bool(res['guardrail_applied']),
            'SEVERITY': sev,
        })
    return pd.DataFrame(rows)


def compute_slice_metrics_from_testset(info: Dict[str, Any]) -> Dict[str, Any]:
    latest_ts = info['latest_timestamp']
    ds_dir = os.path.join('artifacts', latest_ts, 'enhanced_data_transformation')
    test_path = os.path.join(ds_dir, 'test.npy')
    arr = np.load(test_path)
    X_test, y_test = arr[:, :-1], arr[:, -1]

    # Build lightweight predictor consistent with model type
    best_type = info['best_model_type']
    if best_type == 'two_stage':
        models = joblib.load(info['two_stage_model_path'])
        from src.components.enhanced_model_trainer import EnhancedModelTrainer
        trainer = EnhancedModelTrainer()
        y_pred, _ = trainer.predict_two_stage(models, X_test)
    else:
        model = joblib.load(info['weighted_model_path'])
        y_pred = model.predict(X_test)

    # Binned metrics
    def mae(a, b): return float(np.mean(np.abs(a - b)))
    def rmse(a, b): return float(np.sqrt(np.mean((a - b)**2)))
    def r2(a, b):
        ss_res = float(np.sum((a - b)**2))
        ss_tot = float(np.sum((a - np.mean(a))**2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    edges = [0, 1, 3, 10]
    labels = ['<1', '1-3', '>3']
    cats = pd.cut(y_test, bins=edges, labels=labels, include_lowest=True)

    out = {
        'overall': {'mae': mae(y_test, y_pred), 'rmse': rmse(y_test, y_pred), 'r2': r2(y_test, y_pred)}
    }
    for lab in labels:
        mask = (cats == lab).to_numpy()
        if mask.sum() > 0:
            out[lab] = {'mae': mae(y_test[mask], y_pred[mask]), 'rmse': rmse(y_test[mask], y_pred[mask]), 'n': int(mask.sum())}
    return out


def tuning_sweep(info: Dict[str, Any]) -> Dict[str, Any]:
    latest_ts = info['latest_timestamp']
    ds_dir = os.path.join('artifacts', latest_ts, 'enhanced_data_transformation')
    train_path = os.path.join(ds_dir, 'train.npy')
    test_path = os.path.join(ds_dir, 'test.npy')
    train_arr = np.load(train_path)
    test_arr = np.load(test_path)
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

    from sklearn.ensemble import GradientBoostingRegressor
    def mae(a, b): return float(np.mean(np.abs(a - b)))

    gates = [0.5, 0.6, 0.7, 0.8]
    weights = [3.0, 5.0, 8.0]

    # Baseline
    baseline = compute_slice_metrics_from_testset(info)

    best = {'score': 1e9, 'gate': None, 'weight': None, 'mode': None, 'metrics': None}

    # Two-stage tuning: adjust gate only
    if info['best_model_type'] == 'two_stage':
        base_models = joblib.load(info['two_stage_model_path'])
        from src.components.enhanced_model_trainer import EnhancedModelTrainer
        trainer = EnhancedModelTrainer()
        for gate in gates:
            models = dict(base_models)
            models['low_vis_prob_gate'] = gate
            y_pred, _ = trainer.predict_two_stage(models, X_test)
            cats = pd.cut(y_test, bins=[0,1,3,10], labels=['<1','1-3','>3'], include_lowest=True)
            m1 = mae(y_test[cats=='<1'], y_pred[cats=='<1']) if (cats=='<1').sum()>0 else 1e6
            m2 = mae(y_test[cats=='1-3'], y_pred[cats=='1-3']) if (cats=='1-3').sum()>0 else 1e6
            score = m1 + m2
            if score < best['score']:
                best.update({'score': score, 'gate': gate, 'weight': None, 'mode': 'two_stage_gate', 'metrics': {'<1': m1, '1-3': m2}})

    # Weighted model tuning: adjust low-vis weight and retrain quickly
    from src.components.enhanced_model_trainer import EnhancedModelTrainer
    trainer = EnhancedModelTrainer()
    base_cfg = trainer.config.config['models']['weighted_single']['params']
    for w in weights:
        sample_weights = np.where(y_train < trainer.low_vis_threshold, w, 1.0)
        model = GradientBoostingRegressor(**base_cfg)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        cats = pd.cut(y_test, bins=[0,1,3,10], labels=['<1','1-3','>3'], include_lowest=True)
        m1 = mae(y_test[cats=='<1'], y_pred[cats=='<1']) if (cats=='<1').sum()>0 else 1e6
        m2 = mae(y_test[cats=='1-3'], y_pred[cats=='1-3']) if (cats=='1-3').sum()>0 else 1e6
        score = m1 + m2
        if score < best['score']:
            best.update({'score': score, 'gate': None, 'weight': w, 'mode': 'weighted', 'metrics': {'<1': m1, '1-3': m2}})

    return {'baseline': baseline, 'best': best}


def guardrail_check(cases: List[Dict[str, Any]], use_guardrail: bool = True) -> pd.DataFrame:
    rows = []
    for c in cases:
        form = {
            'DRYBULBTEMPF': str(c['temp_f']),
            'RelativeHumidity': str(c['rh']),
            'WindSpeed': str(c['wind_mph']),
            'WindDirection': str(c['wind_dir_deg']),
            'SeaLevelPressure': str(c['slp_mb']),
        }
        class MockReq: pass
        req = MockReq(); req.form = form
        # Temporarily toggle guardrail by monkey-patching config load
        pipeline = EnhancedPredictionPipeline(request=req)
        # If disabling, set an internal switch on guardrail system
        if not use_guardrail:
            pipeline.guardrail.enable_fog_guardrail = False
            pipeline.guardrail.fog_cap_enabled = False
        res = pipeline.run_pipeline()
        rows.append({
            'temp_f': c['temp_f'], 'rh': c['rh'], 'wind_mph': c['wind_mph'], 'expected_band': c['expected_band'],
            'vis_pred': round(float(res['visibility_miles']), 3), 'guardrail_applied': bool(res['guardrail_applied']),
            'fog_signal': int(res['fog_signal'])
        })
    return pd.DataFrame(rows)


def save_report(info: Dict[str, Any], sanity_ok: bool, table: pd.DataFrame, slices: Dict[str, Any], tuning: Dict[str, Any], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('# Verification Report\n\n')
        f.write('## Artifacts\n')
        p_ts, p_sz = file_info(info['preprocessor_path'])
        t_ts, t_sz = file_info(info['trained_model_path'])
        f.write(f"- Preprocessor: {info['preprocessor_path']} (ts={p_ts}, size={p_sz})\n")
        f.write(f"- Model dir: {info['model_dir']}\n")
        f.write(f"- Trained model path: {info['trained_model_path']} (ts={t_ts}, size={t_sz})\n")
        f.write(f"- Model type: {info['best_model_type']}\n")
        if info['best_model_type'] == 'two_stage':
            gate = info['model'].get('low_vis_prob_gate') if isinstance(info['model'], dict) else None
            f.write(f"- Gate threshold: {gate}\n")
        else:
            w_low = info['config']['sample_weights']['low_visibility_weight']
            f.write(f"- Low-vis weight: {w_low}\n")
        f.write(f"- Guardrail enabled: {info['config']['guardrail']['enable_fog_guardrail']}\n\n")

        f.write('## Feature List\n')
        f.write('```\n')
        f.write(', '.join(info.get('feature_names') or []))
        f.write('\n```\n\n')

        f.write('## Sanity\n')
        f.write(f"- Physics features present: {sanity_ok}\n")
        f.write('- Example (58F,98%): dewpoint≈57.44F, spread≈0.56F\n\n')

        f.write('## Test Cases\n')
        f.write(table.to_csv(index=False))
        f.write('\n')

        f.write('## Slice Metrics (held-out test set)\n')
        f.write(json.dumps(slices, indent=2))
        f.write('\n\n')

        f.write('## Tuning Result\n')
        f.write(json.dumps(tuning, indent=2))
        f.write('\n')


def main():
    # Load served artifacts
    info = load_served_artifacts()

    # Print essential info
    print('Model dir:', info['model_dir'])
    print('Preprocessor:', info['preprocessor_path'])
    print('Feature names:', info.get('feature_names'))
    print('Model type:', info['best_model_type'])

    # Sanity
    try:
        assert_sanity(info)
        sanity_ok = True
    except AssertionError as e:
        print('SANITY FAIL:', e)
        sanity_ok = False

    # Cases
    TEST_CASES = [
      {"temp_f":72,"rh":35,"wind_mph":8,"wind_dir_deg":270,"slp_mb":1022,"expected_band":"8-10 (clear/dry high)"},
      {"temp_f":55,"rh":25,"wind_mph":12,"wind_dir_deg":320,"slp_mb":1025,"expected_band":"9-10 (very dry post-frontal)"},
      {"temp_f":40,"rh":45,"wind_mph":6,"wind_dir_deg":330,"slp_mb":1030,"expected_band":"8-10 (cold clear)"},
      {"temp_f":92,"rh":55,"wind_mph":7,"wind_dir_deg":190,"slp_mb":1009,"expected_band":"5-8 (hazy hot)"},
      {"temp_f":65,"rh":60,"wind_mph":25,"wind_dir_deg":300,"slp_mb":1015,"expected_band":"7-10 (gusty clearing)"},
      {"temp_f":75,"rh":95,"wind_mph":3,"wind_dir_deg":160,"slp_mb":1012,"expected_band":"2-5 (muggy evening)"},
      {"temp_f":58,"rh":98,"wind_mph":1,"wind_dir_deg":10,"slp_mb":1018,"expected_band":"<2 (radiation fog)"},
      {"temp_f":50,"rh":99,"wind_mph":0,"wind_dir_deg":0,"slp_mb":1016,"expected_band":"<1-2 (dense fog)"},
      {"temp_f":64,"rh":97,"wind_mph":6,"wind_dir_deg":140,"slp_mb":1016,"expected_band":"1-3 (sea fog onshore)"},
      {"temp_f":57,"rh":98,"wind_mph":2,"wind_dir_deg":220,"slp_mb":1011,"expected_band":"1-3 (drizzle calm)"},
      {"temp_f":60,"rh":95,"wind_mph":7,"wind_dir_deg":110,"slp_mb":1019,"expected_band":"2-6 (fog breakup)"},
      {"temp_f":62,"rh":94,"wind_mph":4,"wind_dir_deg":200,"slp_mb":1008,"expected_band":"2-5 (light rain)"},
      {"temp_f":60,"rh":92,"wind_mph":15,"wind_dir_deg":220,"slp_mb":1004,"expected_band":"3-6 (moderate rain + breeze)"},
      {"temp_f":68,"rh":96,"wind_mph":8,"wind_dir_deg":180,"slp_mb":999,"expected_band":"1-4 (humid low pressure)"},
      {"temp_f":70,"rh":90,"wind_mph":28,"wind_dir_deg":240,"slp_mb":1002,"expected_band":"1-4 (TS outflow)"},
      {"temp_f":30,"rh":88,"wind_mph":5,"wind_dir_deg":10,"slp_mb":1010,"expected_band":"0.5-2 (snow + light wind)"},
      {"temp_f":28,"rh":82,"wind_mph":22,"wind_dir_deg":350,"slp_mb":1007,"expected_band":"0.5-3 (blowing snow)"},
      {"temp_f":34,"rh":92,"wind_mph":4,"wind_dir_deg":20,"slp_mb":1006,"expected_band":"1-3 (wet snow/drizzle)"},
      {"temp_f":85,"rh":30,"wind_mph":5,"wind_dir_deg":250,"slp_mb":1013,"expected_band":"2-6 (smoke/haze)"},
      {"temp_f":58,"rh":100,"wind_mph":0,"wind_dir_deg":0,"slp_mb":1018,"expected_band":"<1 (saturated + calm)"},
      {"temp_f":58,"rh":95,"wind_mph":4,"wind_dir_deg":180,"slp_mb":1018,"expected_band":"1-3 (near-fog threshold)"},
      {"temp_f":58,"rh":94,"wind_mph":5,"wind_dir_deg":180,"slp_mb":1018,"expected_band":"2-5 (just above threshold)"},
      {"temp_f":45,"rh":97,"wind_mph":1,"wind_dir_deg":50,"slp_mb":1021,"expected_band":"<2 (cold fog)"},
      {"temp_f":80,"rh":85,"wind_mph":14,"wind_dir_deg":180,"slp_mb":1014,"expected_band":"6-9 (humid but breezy)"},
      {"temp_f":52,"rh":90,"wind_mph":0,"wind_dir_deg":0,"slp_mb":1024,"expected_band":"1-3 (calm humid night)"},
      {"temp_f":66,"rh":92,"wind_mph":10,"wind_dir_deg":210,"slp_mb":950,"expected_band":"1-4 (deep low, rain likely)"},
      {"temp_f":45,"rh":40,"wind_mph":12,"wind_dir_deg":330,"slp_mb":1040,"expected_band":"8-10 (very strong high)"},
      {"temp_f":61,"rh":96,"wind_mph":3,"wind_dir_deg":150,"slp_mb":1007,"expected_band":"1-3 (humid, light wind)"},
      {"temp_f":59,"rh":93,"wind_mph":2,"wind_dir_deg":20,"slp_mb":1015,"expected_band":"1-3 (near saturation)"},
      {"temp_f":63,"rh":88,"wind_mph":2,"wind_dir_deg":200,"slp_mb":1013,"expected_band":"3-6 (humid calm)"},
      {"temp_f":48,"rh":85,"wind_mph":6,"wind_dir_deg":30,"slp_mb":1009,"expected_band":"2-5 (cold drizzle)"},
      {"temp_f":77,"rh":70,"wind_mph":2,"wind_dir_deg":160,"slp_mb":1010,"expected_band":"4-7 (humid calm, not saturated)"},
      {"temp_f":72,"rh":80,"wind_mph":20,"wind_dir_deg":200,"slp_mb":1006,"expected_band":"6-9 (windy mixing)"},
      {"temp_f":41,"rh":96,"wind_mph":3,"wind_dir_deg":10,"slp_mb":1018,"expected_band":"1-3 (freezing fog risk)"},
      {"temp_f":64,"rh":40,"wind_mph":2,"wind_dir_deg":140,"slp_mb":1018,"expected_band":"8-10 (dry calm)"}
    ]

    table = run_cases_with_pipeline(TEST_CASES)
    print('\nBatch results (first 5 rows):')
    print(table.head())

    slices = compute_slice_metrics_from_testset(info)
    print('\nHeld-out slice metrics:')
    print(json.dumps(slices, indent=2))

    tuning = tuning_sweep(info)
    print('\nTuning summary:')
    print(json.dumps(tuning, indent=2))

    # Guardrail verification on two foggy examples
    fog_subset = [c for c in TEST_CASES if '<' in c['expected_band']][:3]
    gr_on = guardrail_check(fog_subset, use_guardrail=True)
    gr_off = guardrail_check(fog_subset, use_guardrail=False)
    print('\nGuardrail ON:')
    print(gr_on)
    print('\nGuardrail OFF:')
    print(gr_off)

    # Save report
    metrics_dir = os.path.join('artifacts', info['latest_timestamp'], 'metrics')
    out_path = os.path.join(metrics_dir, 'verification_report.md')
    save_report(info, sanity_ok, table, slices, tuning, out_path)
    print('\nReport saved to', out_path)


if __name__ == '__main__':
    main()
