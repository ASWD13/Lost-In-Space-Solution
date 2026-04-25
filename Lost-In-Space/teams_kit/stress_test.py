import sys, copy, math
from datetime import timedelta
sys.path.insert(0, 'basilisk_harness')
from basilisk_harness.config import load_pass_config
from basilisk_harness.mock_sim import run_mock
from basilisk_harness import sgp4_utils as s4u

import importlib.util
spec = importlib.util.spec_from_file_location('sub', '../solution/my_submission.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['sub'] = mod
spec.loader.exec_module(mod)

def evaluate(cfg_modifier=None):
    st = 0
    weights = {'case1': 0.25, 'case2': 0.35, 'case3': 0.4}
    for case_id in ['case1', 'case2', 'case3']:
        cfg = load_pass_config(case_id)
        if cfg_modifier:
            cfg_modifier(cfg)
        
        sched = mod.plan_imaging(cfg.tle1, cfg.tle2, cfg.aoi_polygon, cfg.pass_start, cfg.pass_end, cfg.sc_params)
        res = run_mock(cfg, sched)
        d = res.as_dict()
        st += weights[case_id] * d['S_orbit']
    return st

print(f"Baseline S_total: {evaluate():.4f}")

def mod_A(cfg):
    cfg.sc_params['off_nadir_max_deg'] = 58.5
    cfg.sc_params['smear_max_degps'] = 0.03
    cfg.sc_params['wheel_max_mnms'] = 25.0
print(f"Test A S_total: {evaluate(mod_A):.4f}")

def mod_B(cfg):
    t_start = s4u.parse_iso_utc(cfg.pass_start) + timedelta(seconds=1.0)
    t_end = s4u.parse_iso_utc(cfg.pass_end) + timedelta(seconds=1.0)
    cfg.pass_start = t_start.isoformat()
    if not cfg.pass_start.endswith('Z') and '+00:00' not in cfg.pass_start: cfg.pass_start += 'Z'
    cfg.pass_end = t_end.isoformat()
    if not cfg.pass_end.endswith('Z') and '+00:00' not in cfg.pass_end: cfg.pass_end += 'Z'
print(f"Test B S_total: {evaluate(mod_B):.4f}")

def mod_C(cfg):
    cfg.aoi_polygon = [(lat + 0.005, lon + 0.005) for lat, lon in cfg.aoi_polygon]
print(f"Test C S_total: {evaluate(mod_C):.4f}")
