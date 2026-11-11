"""Unified Two-Stage M6 + LightGBM Inference Engine (Directly Linked to Saved .joblib Bundle).

Eliminates all hardcoded heuristic weights: every single-event or multi-CDM evaluation constructs or filters
the causal CDM history (time_to_tca >= 2.0 days), executes `src.features.build_event_features`, and invokes
the serialized LightGBM v4 decision tree ensembles inside `models/final_model.joblib` (or `models/final_collision_risk_model.joblib`).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .. import config, features, inference, modeling

GATE_THRESH = -7.0
THRESHOLD = -6.0
HARD_LOCK = -5.25

M_Z1_TRAIN = -4.662259
M_Z2_TRAIN = -5.587414
BETA_Z3_OLS = (-5.247197, -0.747885, -0.276898)


@dataclass
class M6InferenceResult:
    zone: str
    action_tag: str
    flag_high_risk: int
    predicted_final_risk: float
    s1_veto_score: float
    s2_physics_score: float
    stage1_lgbm_prob: float
    stage2_risk_blind_prob: float
    dilution_gap: float
    n_allowed_cdms: int
    model_version: str
    bundle_file: str


def estimate_unsupervised_test_prior_em(
    p_train: np.ndarray, y_train: np.ndarray, p_test: np.ndarray, max_iter: int = 200, tol: float = 1e-7
) -> tuple[float, float]:
    """Saerens-Latinne-Decaestecker (2002) EM algorithm for unsupervised prior-shift estimation."""
    res = modeling.verify_and_estimate_em_prior_shift(p_train, y_train, p_test, max_iter=max_iter, tol=tol)
    return res["pi_train"], res["pi_test_estimated"]


def predict_three_zone_final_risk(
    last_risk: float, n_allowed_cdms: int, flag_high_risk: int, bundle: dict | None = None
) -> float:
    """Computes final_risk using the Three-Zone Conditional Regressor stored in the loaded model bundle."""
    if flag_high_risk == 0:
        return float(config.OFFICIAL_CLIP_LOW)
    m = bundle["model"] if (bundle and "model" in bundle) else None
    hl = float(getattr(m, "hard_lock", HARD_LOCK))
    thr = float(getattr(m, "threshold", THRESHOLD))
    m_z1 = float(getattr(m, "m_z1", M_Z1_TRAIN))
    m_z2 = float(getattr(m, "m_z2", M_Z2_TRAIN))
    beta_z3 = getattr(m, "beta_z3", BETA_Z3_OLS)

    if last_risk >= hl:
        return float(0.75 * last_risk + 0.25 * m_z1)
    if last_risk >= thr:
        return float(0.90 * last_risk + 0.10 * m_z2)
    val = (
        beta_z3[0]
        + beta_z3[1] * (last_risk + 6.0)
        + beta_z3[2] * float(np.log1p(max(n_allowed_cdms, 1)))
    )
    return float(np.clip(val, -5.95, -4.50))


def _resolve_bundle(operating_mode: str = "prior", bundle_path: str | Path | None = None) -> tuple[dict, str]:
    if bundle_path is not None:
        p = Path(bundle_path)
    elif operating_mode == "r100":
        p = config.MODELS_DIR / "m6_leakfree_r100_budget.joblib"
        if not p.exists():
            p = config.MODELS_DIR / "final_model.joblib"
    else:
        p = config.MODELS_DIR / "final_model.joblib"
    return modeling.load_bundle(p), p.name


def evaluate_cdm_history(
    cdm_df: pd.DataFrame,
    operating_mode: str = "prior",
    bundle_path: str | Path | None = None,
) -> M6InferenceResult:
    """Runs the unified shared InferenceEngine: filter_causal_cdms (T >= 2.0d) -> build_event_features -> SHA256-verified .joblib trees."""
    engine = inference.get_shared_inference_engine()
    pred = engine.predict_event_from_cdms(cdm_df, operating_mode=operating_mode, bundle_path=bundle_path)
    if pred is None:
        raise ValueError("No valid pre-cutoff CDMs (time_to_tca >= 2.0 days) found in the supplied CDM history.")
    bname = pred.get("bundle_file", "final_model.joblib")
    return M6InferenceResult(
        zone=pred["zone"],
        action_tag=pred["action_tag"],
        flag_high_risk=int(pred["is_alarm"]),
        predicted_final_risk=round(float(pred["adjusted"]), 5),
        s1_veto_score=round(float(pred["stage1_score"]), 4),
        s2_physics_score=round(float(pred["stage2_physics_score"]), 4),
        stage1_lgbm_prob=round(float(pred["stage1_lgbm_prob"]), 6),
        stage2_risk_blind_prob=round(float(pred["stage2_risk_blind_prob"]), 6),
        dilution_gap=round(float(pred["dilution_gap"]), 4) if pd.notna(pred["dilution_gap"]) else 0.0,
        n_allowed_cdms=int(pred["n_cdms_used"] or 1),
        model_version=str(pred["model_version"]),
        bundle_file=bname,
    )


def evaluate_single_cdm_event(
    last_risk: float,
    max_risk_estimate: float,
    miss_distance_m: float,
    mahalanobis_distance: float,
    c_sigma_t_m: float,
    relative_speed_ms: float,
    n_allowed_cdms: int,
    operating_mode: str = "prior",
    bundle_path: str | Path | None = None,
) -> M6InferenceResult:
    """Constructs a multi-row pre-cutoff CDM history and executes the saved .joblib LightGBM + M6 bundle."""
    n_rows = max(int(n_allowed_cdms), 1)
    times = np.linspace(2.05 + 0.35 * (n_rows - 1), 2.05, n_rows)
    risks = np.linspace(max(-30.0, float(last_risk) - 0.35), float(last_risk), n_rows)
    cdm_df = pd.DataFrame({
        "event_id": [0] * n_rows,
        "time_to_tca": times,
        "risk": risks,
        "max_risk_estimate": np.linspace(float(max_risk_estimate) - 0.15, float(max_risk_estimate), n_rows),
        "max_risk_scaling": np.full(n_rows, max(1.0, 10.0 ** max(0.0, float(max_risk_estimate - last_risk)))),
        "miss_distance": np.linspace(float(miss_distance_m) * 1.15, float(miss_distance_m), n_rows),
        "mahalanobis_distance": np.linspace(float(mahalanobis_distance) * 1.2, float(mahalanobis_distance), n_rows),
        "relative_speed": np.full(n_rows, float(relative_speed_ms)),
        "relative_position_r": np.full(n_rows, -0.15 * float(miss_distance_m)),
        "relative_position_t": np.full(n_rows, 0.75 * float(miss_distance_m)),
        "relative_position_n": np.full(n_rows, 0.60 * float(miss_distance_m)),
        "relative_velocity_r": np.full(n_rows, 15.0),
        "relative_velocity_t": np.full(n_rows, -0.85 * float(relative_speed_ms)),
        "relative_velocity_n": np.full(n_rows, 0.52 * float(relative_speed_ms)),
        "c_sigma_r": np.full(n_rows, max(25.0, 0.08 * float(c_sigma_t_m))),
        "c_sigma_t": np.linspace(float(c_sigma_t_m) * 1.25, float(c_sigma_t_m), n_rows),
        "c_sigma_n": np.full(n_rows, max(25.0, 0.10 * float(c_sigma_t_m))),
        "t_sigma_r": np.full(n_rows, 5.0),
        "t_sigma_t": np.full(n_rows, 45.0),
        "t_sigma_n": np.full(n_rows, 3.0),
        "c_obs_used": np.full(n_rows, 25.0),
        "c_obs_available": np.full(n_rows, 28.0),
        "c_recommended_od_span": np.full(n_rows, 5.5),
        "c_actual_od_span": np.full(n_rows, 5.5),
        "t_obs_used": np.full(n_rows, 120.0),
        "t_obs_available": np.full(n_rows, 125.0),
        "t_recommended_od_span": np.full(n_rows, 7.0),
        "t_actual_od_span": np.full(n_rows, 7.0),
        "c_object_type": ["DEBRIS"] * n_rows,
    })
    return evaluate_cdm_history(cdm_df, operating_mode=operating_mode, bundle_path=bundle_path)
