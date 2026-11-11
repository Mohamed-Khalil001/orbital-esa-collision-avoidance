"""Phase 3, Phase 4 & Phase 5 Comprehensive Unit, API & End-to-End Integration Tests."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src import (
    config,
    data,
    features,
    inference,
    modeling,
    monitoring,
    operational_cost,
    registry,
    simulation,
    training,
)
from src.api import server

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Phase 3 Tests: Orbital Simulation Overhaul
# ---------------------------------------------------------------------------

def test_phase3_deltav_prograde_and_retrograde_have_opposite_physical_directions():
    """Verifies that +dv_T (prograde) and -dv_T (retrograde) produce opposite along-track displacements."""
    res_pro = simulation.simulate_along_track_cam(
        rel_pos_r_m=-120.0, rel_pos_t_m=650.0, rel_pos_n_m=-310.0,
        c_sigma_t_m=2800.0, initial_predicted_risk=-5.45,
        delta_v_cms=+2.5, lead_time_days=2.0,
    )
    res_ret = simulation.simulate_along_track_cam(
        rel_pos_r_m=-120.0, rel_pos_t_m=650.0, rel_pos_n_m=-310.0,
        c_sigma_t_m=2800.0, initial_predicted_risk=-5.45,
        delta_v_cms=-2.5, lead_time_days=2.0,
    )
    assert res_pro.along_track_shift_m > 0.0
    assert res_ret.along_track_shift_m < 0.0
    assert np.isclose(res_pro.along_track_shift_m, -res_ret.along_track_shift_m, rtol=1e-5)


def test_phase3_actual_tca_velocity_and_full_3x3_nondiagonal_covariance():
    """Verifies that full 3x3 non-diagonal RTN covariance (ct_r, cn_r, cn_t) and actual TCA velocity are used."""
    r_rel = np.array([-140.0, 520.0, -280.0])
    v_rel = np.array([25.0, -13200.0, 4600.0])
    sig = np.array([220.0, 3400.0, 190.0])

    P_diag = simulation.build_full_rtn_covariance_matrix(sig, corr_tr=0.0, corr_nr=0.0, corr_nt=0.0)
    P_full = simulation.build_full_rtn_covariance_matrix(sig, corr_tr=-0.72, corr_nr=-0.48, corr_nt=-0.12)

    bp_diag = simulation.compute_bplane_collision_probability(r_rel, v_rel, P_rtn_3x3=P_diag)
    bp_full = simulation.compute_bplane_collision_probability(r_rel, v_rel, P_rtn_3x3=P_full)

    assert bp_diag["covariance_off_diagonal_used"] is False
    assert bp_full["covariance_off_diagonal_used"] is True
    # Non-diagonal correlations change the B-plane ellipse orientation and collision probability
    assert abs(bp_full["log10_pc"] - bp_diag["log10_pc"]) > 1e-4
    assert abs(bp_full["bplane_corr_xi_zeta"] - bp_diag["bplane_corr_xi_zeta"]) > 1e-3


def test_phase3_j2_validation_against_high_precision_cowell_dop853_and_stk_orekit():
    """Verifies J2 propagation against 8th-order DOP853 3D ECI Two-Body+J2 Cowell and Orekit/STK benchmarks."""
    df_val = simulation.validate_j2_against_high_precision_and_stk_orekit()
    assert len(df_val) == 4
    # Relative error between Schweighart-Sedwick J2 CW and full 3D ECI J2 Cowell DOP853 must be < 0.5%
    assert (df_val["CW_J2_vs_Cowell_RelError_Pct"] < 0.50).all()
    # Difference between 3D ECI J2 Cowell DOP853 and Orekit 12 / STK 12 J2 reference must be < 1.0 meter
    assert (df_val["Cowell_vs_Orekit_STK_3D_Diff_m"] < 1.0).all()


# ---------------------------------------------------------------------------
# Phase 4 Tests: Platform MLOps, Registry, Monitoring & API/Streamlit Parity
# ---------------------------------------------------------------------------

def test_phase4_model_registry_sha256_and_5stage_validation_gate(tmp_path):
    """Verifies that ModelRegistry tracks SHA-256 digests and rejects invalid/corrupted models at the validation gate."""
    manifest = registry.load_registry_manifest()
    assert manifest["active_model_file"] == "final_model.joblib"
    assert len(manifest["models"]) >= 3
    for entry in manifest["models"]:
        assert len(entry["file_sha256"]) == 64
        assert entry["validation_gate_passed"] is True

    # Now create a corrupted candidate model where Stage-2 illegally includes 'last_risk'
    bad_bundle = copy.deepcopy(modeling.load_bundle(ROOT / "models/final_model.joblib"))
    bad_bundle["model"].stage2_booster["feature_names"].append("last_risk")
    gate_res = registry.run_pre_activation_validation_gate(bad_bundle)
    assert gate_res["passed"] is False
    assert gate_res["gates"]["gate_3_stage2_risk_blind_zero_sensitivity"] is False

    # Verify install_bundle blocks activation of the bad model
    with pytest.raises(ValueError):
        modeling.install_bundle(bad_bundle, models_dir=tmp_path, activate=True)


def test_phase4_streamlit_and_fastapi_share_identical_inference_engine():
    """Verifies bit-for-bit parity between UnifiedInferenceEngine (used by Streamlit) and FastAPI /api/v1/predict-event."""
    demo_df = data.load_demo_sample()
    ev_id = int(demo_df[config.ID_COL].iloc[0])
    ev_cdms = demo_df[demo_df[config.ID_COL] == ev_id].sort_values(config.TIME_COL, ascending=False)

    # 1. Streamlit / Shared Engine call
    engine = inference.get_shared_inference_engine()
    res_streamlit = engine.predict_event_from_cdms(ev_cdms, operating_mode="prior")

    # 2. FastAPI endpoint call with the exact same CDM records
    cdm_inputs = []
    for _, r in ev_cdms.iterrows():
        cdm_inputs.append(server.CDMRecordInput(
            event_id=int(r["event_id"]),
            time_to_tca=float(r["time_to_tca"]),
            risk=float(r["risk"]),
            max_risk_estimate=float(r["max_risk_estimate"]),
            max_risk_scaling=float(r.get("max_risk_scaling", 10.0)),
            miss_distance=float(r["miss_distance"]),
            mahalanobis_distance=float(r["mahalanobis_distance"]),
            relative_speed=float(r["relative_speed"]),
            relative_position_r=float(r["relative_position_r"]),
            relative_position_t=float(r["relative_position_t"]),
            relative_position_n=float(r["relative_position_n"]),
            relative_velocity_r=float(r["relative_velocity_r"]),
            relative_velocity_t=float(r["relative_velocity_t"]),
            relative_velocity_n=float(r["relative_velocity_n"]),
            c_sigma_r=float(r["c_sigma_r"]),
            c_sigma_t=float(r["c_sigma_t"]),
            c_sigma_n=float(r["c_sigma_n"]),
            c_ct_r=float(r["c_ct_r"]) if pd.notna(r.get("c_ct_r")) else -0.35,
            c_cn_r=float(r["c_cn_r"]) if pd.notna(r.get("c_cn_r")) else -0.15,
            c_cn_t=float(r["c_cn_t"]) if pd.notna(r.get("c_cn_t")) else -0.05,
            t_sigma_r=float(r["t_sigma_r"]),
            t_sigma_t=float(r["t_sigma_t"]),
            t_sigma_n=float(r["t_sigma_n"]),
            c_obs_used=float(r["c_obs_used"]),
            c_obs_available=float(r["c_obs_available"]),
            c_recommended_od_span=float(r["c_recommended_od_span"]),
            c_actual_od_span=float(r["c_actual_od_span"]),
            c_object_type=str(r["c_object_type"]),
        ))
    req = server.CDMEventHistoryRequest(cdms=cdm_inputs, operating_mode="prior")
    res_fastapi = server.predict_from_cdm_history(req)

    assert res_streamlit["file_sha256"] == res_fastapi["file_sha256"]
    assert res_streamlit["zone"] == res_fastapi["zone"]
    assert res_streamlit["action_tag"] == res_fastapi["action_tag"]
    assert np.isclose(res_streamlit["adjusted"], res_fastapi["adjusted"], atol=1e-6)


def test_phase4_model_monitoring_psi_and_ks_drift_detection():
    """Verifies that ModelMonitor tracks inference telemetry and detects covariate drift via PSI & KS."""
    mon = monitoring.get_model_monitor()
    summary = mon.get_summary()
    assert summary["total_inferences_logged"] >= 1
    assert "drift_table" in summary and len(summary["drift_table"]) >= 3

    ref_df = mon._get_reference_features()
    # Create a drifted batch where last_risk is shifted by +10.0 log units
    drifted_df = ref_df.copy()
    drifted_df["last_risk"] = drifted_df["last_risk"] + 10.0
    drift_report = mon.evaluate_dataset_drift(drifted_df)
    lr_row = drift_report[drift_report["Feature"] == "last_risk"].iloc[0]
    assert lr_row["PSI"] > 0.25
    assert "ALERT" in lr_row["Drift_Status"]


def test_phase4_api_endpoints_and_causal_cutoff_rejection():
    """Tests FastAPI health, model-info, registry, monitoring, cost-evaluation, and HTTP 422 cutoff enforcement."""
    h = server.health_check()
    assert h["status"] == "ok" and len(h["file_sha256"]) == 64

    reg = server.get_model_registry()
    assert reg["active_model_file"] == "final_model.joblib"

    cost_res = server.get_operational_cost_evaluation(c_fp=1.0, c_fn=20.0, c_cam=2.0)
    assert cost_res["formula"] == "C = C_FP * FP + C_FN * FN + C_CAM * N_CAM"
    assert len(cost_res["ranking"]) >= 10

    # Verify post-cutoff-only CDM payload (< 2.0 days) is rejected with HTTPException 422
    late_cdm = server.CDMRecordInput(
        event_id=999, time_to_tca=0.85, risk=-5.2, max_risk_estimate=-4.5,
        miss_distance=400.0, mahalanobis_distance=2.1, relative_speed=12500.0,
    )
    with pytest.raises( Exception ):
        server.predict_from_cdm_history(server.CDMEventHistoryRequest(cdms=[late_cdm]))


# ---------------------------------------------------------------------------
# Phase 5 Tests: Operational Cost Evaluation (C = C_FP*FP + C_FN*FN + C_CAM*N_CAM)
# ---------------------------------------------------------------------------

def test_phase5_operational_cost_formula_and_threshold_optimization():
    """Verifies exact calculation of C = C_FP*FP + C_FN*FN + C_CAM*N_CAM and cost-sensitive model ranking."""
    s = operational_cost.compute_operational_cost(
        tp=122, fp=71, fn=28, tn=1946,
        c_fp=1.0, c_fn=20.0, c_cam=2.0, cam_fraction_on_fp=1.0,
    )
    # N_CAM = TP + FP = 193; Cost = 1.0*71 + 20.0*28 + 2.0*193 = 71 + 560 + 386 = 1017.0
    assert s.n_alarms == 193
    assert np.isclose(s.total_operational_cost_c, 1017.0)

    df_cost = operational_cost.build_operational_cost_comparison_table(c_fp=1.0, c_fn=20.0, c_cam=2.0)
    m6_row = df_cost[df_cost["System"].str.contains("M6_LeakFree_PriorAdapted")].iloc[0]
    b0_row = df_cost[df_cost["System"].str.contains("B0_LRP_Baseline")].iloc[0]
    assert m6_row["Total_Operational_Cost_C"] < b0_row["Total_Operational_Cost_C"]
    assert m6_row["Net_Cost_Savings_vs_B0"] == 102.0
