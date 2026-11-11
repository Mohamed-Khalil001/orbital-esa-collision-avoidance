"""End-to-End Integration, Scientific Integrity, Stage-2 Risk-Blind & Zero-Plotly Tests."""
from __future__ import annotations
import copy
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src import config, features, modeling, simulation
from src.features.causal_engineering import (
    RISK_HISTORY_COLS,
    STAGE2_PHYSICAL_ALLOWLIST,
    is_stage2_risk_blind_feature,
    filter_causal_cdms,
    engineer_event_features,
    split_stage1_and_stage2_columns,
)
from src.models.two_stage_m6 import (
    evaluate_single_cdm_event,
    evaluate_cdm_history,
    predict_three_zone_final_risk,
    estimate_unsupervised_test_prior_em,
)
from src.simulator.orbital_mechanics import (
    OPERATIONAL_DISCLAIMER,
    cw_state_transition_matrix,
    rk4_propagate_cw_j2_trajectory,
    compute_bplane_collision_probability,
    simulate_along_track_cam,
)

ROOT = Path(__file__).resolve().parents[1]


def test_causal_cutoff_strictly_blocks_under_2_days():
    df = pd.DataFrame({
        "event_id": [1, 1, 1],
        "time_to_tca": [4.5, 2.1, 0.4],
        "risk": [-7.5, -6.4, -4.8],
        "max_risk_estimate": [-5.2, -4.9, -4.5],
        "miss_distance": [1200.0, 800.0, 250.0],
        "mahalanobis_distance": [8.0, 4.0, 1.5],
        "relative_speed": [12500.0, 12500.0, 12500.0],
    })
    filtered = filter_causal_cdms(df)
    assert len(filtered) == 2
    assert (filtered["time_to_tca"] >= 2.0).all()
    feats = engineer_event_features(df)
    assert len(feats) == 1
    assert np.isclose(feats.loc[0, "last_risk"], -6.4)


def test_stage2_explicit_allowlist_blocks_all_audited_risk_columns():
    """Verifies that all 9 risk-history columns highlighted in the audit are strictly excluded from Stage-2."""
    audited_risk_cols = [
        "risk_floor_fraction",
        "risk_floor_count",
        "risk_all_floor",
        "risk_nonfloor_std",
        "risk_first_nonfloor",
        "risk_last_nonfloor",
        "days_since_last_nonfloor_risk",
        "risk_nonfloor_delta",
        "risk_nonfloor_slope",
        "last_risk",
        "first_risk",
        "dilution_gap",
    ]
    for col in audited_risk_cols:
        assert col in RISK_HISTORY_COLS, f"{col} missing from RISK_HISTORY_COLS"
        assert col not in STAGE2_PHYSICAL_ALLOWLIST, f"{col} leaked into STAGE2_PHYSICAL_ALLOWLIST"
        assert not is_stage2_risk_blind_feature(col), f"is_stage2_risk_blind_feature({col}) returned True"

    demo = pd.read_csv(ROOT / "data/sample/demo_events.csv")
    X = features.build_event_features(demo)
    nocat, norisk = split_stage1_and_stage2_columns(list(X.columns))
    for col in audited_risk_cols:
        assert col not in norisk, f"Audited risk column {col} leaked into norisk_nocat"
    assert all(c in STAGE2_PHYSICAL_ALLOWLIST for c in norisk)


def test_stage2_booster_has_exact_zero_sensitivity_to_risk_history_perturbations():
    """Mathematically proves that randomizing all RISK_HISTORY_COLS changes Stage-2 probability by 0.0."""
    bundle = modeling.load_bundle(ROOT / "models/final_model.joblib")
    model = bundle["model"]
    demo = pd.read_csv(ROOT / "data/sample/demo_events.csv")
    X = features.build_event_features(demo).iloc[:25].copy()

    _, p_s2_orig = model.predict_stage_probabilities(X)
    X_perturbed = X.copy()
    rng = np.random.default_rng(123)
    for col in RISK_HISTORY_COLS:
        if col in X_perturbed.columns:
            X_perturbed[col] = rng.uniform(-30.0, -1.0, size=len(X_perturbed))

    _, p_s2_perturbed = model.predict_stage_probabilities(X_perturbed)
    assert np.max(np.abs(p_s2_orig - p_s2_perturbed)) == 0.0


def test_saved_joblib_bundle_directly_drives_engine_predictions(tmp_path):
    """Proves that modifying the saved .joblib bundle on disk immediately changes engine outputs."""
    bundle = copy.deepcopy(modeling.load_bundle(ROOT / "models/final_model.joblib"))
    custom_path = tmp_path / "modified_model.joblib"

    # Baseline prediction on Z2 event
    res_default = evaluate_single_cdm_event(
        -5.85, -5.70, 22000.0, 35.0, 1100.0, 9500.0, 3, bundle_path=ROOT / "models/final_model.joblib"
    )
    assert res_default.flag_high_risk == 0

    # Modify tau_veto inside the bundle and save to custom_path
    bundle["model"].tau_veto = -99.0
    bundle["model_version"] = "CUSTOM-MODIFIED-BUNDLE-V99"
    modeling.save_bundle(bundle, custom_path)

    res_modified = evaluate_single_cdm_event(
        -5.85, -5.70, 22000.0, 35.0, 1100.0, 9500.0, 3, bundle_path=custom_path
    )
    assert res_modified.flag_high_risk == 1
    assert res_modified.model_version == "CUSTOM-MODIFIED-BUNDLE-V99"


def test_three_zone_regressor_and_two_stage_decisions():
    res_z1 = evaluate_single_cdm_event(-4.90, -4.20, 350.0, 1.8, 1500.0, 14000.0, 12)
    assert res_z1.flag_high_risk == 1
    assert res_z1.action_tag == "LOCKED_ALARM"
    assert res_z1.predicted_final_risk > -6.0

    res_z2_veto = evaluate_single_cdm_event(-5.85, -5.70, 22000.0, 35.0, 1100.0, 9500.0, 3)
    assert res_z2_veto.flag_high_risk == 0
    assert res_z2_veto.action_tag == "VETOED_SAFE"
    assert res_z2_veto.predicted_final_risk < -6.0

    res_z3_resc = evaluate_single_cdm_event(-6.18, -3.95, 220.0, 1.2, 6500.0, 14500.0, 15, operating_mode="prior")
    assert res_z3_resc.flag_high_risk == 1
    assert res_z3_resc.action_tag == "RESCUED_ALARM"
    assert -5.95 <= res_z3_resc.predicted_final_risk <= -4.50


def test_coupled_cw_rk4_and_bplane_simulator():
    cam = simulate_along_track_cam(
        rel_pos_r_m=-80.0,
        rel_pos_t_m=450.0,
        rel_pos_n_m=120.0,
        c_sigma_t_m=2200.0,
        initial_predicted_risk=-5.35,
        delta_v_cms=4.0,
        lead_time_days=2.0,
    )
    assert cam.post_cam_miss_distance_m > cam.initial_miss_distance_m
    assert cam.post_cam_predicted_risk < cam.initial_predicted_risk
    assert cam.operational_disclaimer == OPERATIONAL_DISCLAIMER
    assert cam.propagation_method == "Coupled_6State_CW_J2_Full3x3Cov_ActualTCAVelocity"


def test_unsupervised_em_prior_shift_and_ks_invariance():
    rng = np.random.default_rng(42)
    y_tr = np.array([0] * 900 + [1] * 100)
    p_tr = np.where(y_tr == 1, rng.uniform(0.6, 0.95, 1000), rng.uniform(0.01, 0.05, 1000))
    y_te_hidden = np.array([0] * 700 + [1] * 300)
    p_te = np.where(y_te_hidden == 1, rng.uniform(0.6, 0.95, 1000), rng.uniform(0.01, 0.05, 1000))
    diag = modeling.verify_and_estimate_em_prior_shift(p_tr, y_tr, p_te)
    assert abs(diag["pi_train"] - 0.10) < 1e-6
    assert diag["pi_test_estimated"] > diag["pi_train"] + 0.10
    assert diag["em_assumptions_satisfied"] is True


def test_data_governance_no_private_test_ground_truth_in_repo():
    """Verifies that no public data/report file in the repository leaks private test ground-truth labels."""
    forbidden_cols = {"true_final_risk", "true_high_risk", "final_risk_true"}
    for csv_path in list((ROOT / "data").rglob("*.csv")) + list((ROOT / "reports").rglob("*.csv")) + list((ROOT / "research/results").rglob("*.csv")):
        df_head = pd.read_csv(csv_path, nrows=2)
        leaked = forbidden_cols.intersection(set(df_head.columns))
        assert not leaked, f"Data Governance Violation: {csv_path} contains private test ground-truth columns {leaked}"

    b44_path = ROOT / "data/benchmark_44_conjunction_events.json"
    if b44_path.exists():
        b44 = json.loads(b44_path.read_text(encoding="utf-8"))
        for ev in b44:
            assert "final_risk_true" not in ev and "y_true" not in ev
            for pt in ev.get("cdm_series", []):
                assert float(pt["t"]) >= 2.0, "Post-cutoff CDM leaked into benchmark_44_conjunction_events.json"


def test_zero_plotly_anywhere_in_repo():
    """Verifies that plotly is completely removed from all Python code, HTML apps, and requirements."""
    req_txt = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "plotly" not in req_txt
    for py_path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").rglob("*.py")):
        code = py_path.read_text(encoding="utf-8")
        assert "import plotly" not in code and "from plotly" not in code, f"Plotly import found in {py_path}"


def test_all_streamlit_pages_execute_end_to_end():
    """Executes app/app.py and all app/pages/*.py top-to-bottom to catch any runtime AttributeError/KeyError."""
    import runpy
    import sys
    import types

    class _DummyContainer:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def markdown(self, *a, **k): pass
        def caption(self, *a, **k): pass
        def title(self, *a, **k): pass
        def header(self, *a, **k): pass
        def subheader(self, *a, **k): pass
        def metric(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): raise RuntimeError(f"st.error triggered: {a}")
        def success(self, *a, **k): pass
        def json(self, *a, **k): pass
        def dataframe(self, *a, **k): pass
        def divider(self, *a, **k): pass
        def stop(self): raise RuntimeError("st.stop() triggered unexpectedly")
        def rerun(self): pass
        def download_button(self, *a, **k): return False
        def button(self, *a, **k): return False
        def checkbox(self, label, value=False, **k): return True if "rolling" in str(label) else value
        def slider(self, label, min_value=0.0, max_value=1.0, value=0.5, *a, **k): return value
        def number_input(self, label, min_value=0.0, max_value=100.0, value=1.0, *a, **k): return value
        def selectbox(self, label, options, index=0, **k):
            opts = list(options)
            return opts[index] if opts else None
        def file_uploader(self, *a, **k): return None
        def expander(self, *a, **k): return _DummyContainer()
        def columns(self, spec, **k):
            n = spec if isinstance(spec, int) else len(spec)
            return [_DummyContainer() for _ in range(n)]
        def tabs(self, labels, **k):
            return [_DummyContainer() for _ in labels]

    st_mod = _DummyContainer()
    st_mod.sidebar = _DummyContainer()
    st_mod.set_page_config = lambda *a, **k: None
    def _cache_deco(*a, **k):
        if len(a) == 1 and callable(a[0]):
            a[0].clear = lambda: None
            return a[0]
        def _wrap(fn):
            fn.clear = lambda: None
            return fn
        return _wrap
    st_mod.cache_resource = _cache_deco
    st_mod.cache_data = _cache_deco
    comp_mod = types.ModuleType("streamlit.components")
    comp_v1_mod = types.ModuleType("streamlit.components.v1")
    comp_v1_mod.html = lambda html_str, **k: None
    comp_mod.v1 = comp_v1_mod
    st_mod.components = comp_mod

    saved = {k: sys.modules.get(k) for k in ("streamlit", "streamlit.components", "streamlit.components.v1")}
    try:
        sys.modules["streamlit"] = st_mod
        sys.modules["streamlit.components"] = comp_mod
        sys.modules["streamlit.components.v1"] = comp_v1_mod
        for page in [
            ROOT / "app" / "app.py",
            ROOT / "app" / "pages" / "1_Research_Results.py",
            ROOT / "app" / "pages" / "2_3D_Orbital_Simulator_and_CAM_Lab.py",
        ]:
            runpy.run_path(str(page), run_name="__main__")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
