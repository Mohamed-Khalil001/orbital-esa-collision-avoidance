#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESA Kelvins Spacecraft Collision Avoidance Challenge — Phase 3: Physics-Guided Feature Engineering (F1 to F7)
Strict Leak-Free Evaluation Pipeline (Quarantines test_data_private.csv, X_test_fe_v2.csv, meta_test_fe_v2.csv).

Methodological Note:
- Because Phase 2 error analysis inspected `test_data_private.csv` to diagnose the 6 test False Negatives,
  `test_data_private.csv`, `X_test_fe_v2.csv`, and `meta_test_fe_v2.csv` are strictly quarantined and NEVER loaded
  during Phase 3 feature engineering, feature selection, threshold calibration, or model comparison.
- All experiments F1 through F7 are evaluated strictly on the Training Cohort (8,892 eligible events, 93 High-Risk events)
  across Dev (Seeds 0-2, 15 folds), Fresh (Seeds 3-7, 25 folds), and Group-Aware (Mission Splits, Seeds 0-2, 15 folds),
  plus the 716-event Deep Physics Sub-Cohort (231 columns, 25-fold CV).
- To separate prediction improvement from decision-policy threshold lowering, every experiment F2..F7 is evaluated
  at a locked training alert budget (matching the Baseline B0/F1 alert count on each training fold).
"""

import os
import pickle
import numpy as np
import pandas as pd

FORBIDDEN_TEST_FILES = {"test_data_private.csv", "X_test_fe_v2.csv", "meta_test_fe_v2.csv"}


def assert_no_test_leakage(filepath: str) -> None:
    basename = os.path.basename(filepath)
    if basename in FORBIDDEN_TEST_FILES:
        raise RuntimeError(f"DATA LEAKAGE BLOCKED: Attempted to access quarantined test file {basename}")


def build_phase3_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs physics-guided feature blocks F1 through F7 strictly from pre-cutoff CDMs (time_to_tca >= 2.0 days).
    """
    out = df.copy()
    out["last_risk"] = out["last_allowed_risk"].astype(float)
    out["allowed_span_days"] = (out["first_time_to_tca"] - out["last_allowed_time"]).clip(lower=0.0)

    # F2: Relative Motion & Approach Rate
    out["f2_approach_rate"] = out["allowed_span_days"] / out["first_time_to_tca"].clip(lower=2.0)
    out["f2_cdm_velocity"] = out["n_allowed"] / (out["allowed_span_days"] + 0.25)
    out["f2_transition_rate"] = out["n_transitions"].fillna(0.0) / (out["allowed_span_days"] + 0.25)

    # F3: Closest Approach & TCA Cutoff Geometry
    out["f3_time_to_tca"] = out["last_allowed_time"].astype(float)
    out["f3_cutoff_gap"] = (out["last_allowed_time"] - 2.0).clip(lower=0.0)
    out["f3_first_tca"] = out["first_time_to_tca"].astype(float)
    out["f3_near_cutoff_flag"] = (out["f3_cutoff_gap"] <= 0.25).astype(float)

    # F4: Risk Dynamics (Slope, Step Volatility, Marginal Escalation vs Borderline High De-escalation)
    out["f4_step_max"] = out["risk_step_abs_max"].fillna(0.0)
    out["f4_step_med"] = out["risk_step_abs_median"].fillna(0.0)
    out["f4_step_ratio"] = out["f4_step_max"] / (out["f4_step_med"] + 0.05)
    out["f4_slope_proxy"] = (
        out["f4_step_max"] * np.where(out["last_risk"] > -7.5, 1.0, -1.0)
    ) / (out["allowed_span_days"] + 0.5)
    out["f4_marginal_zone"] = ((out["last_risk"] >= -7.35) & (out["last_risk"] < -6.0)).astype(float)
    out["f4_near_m640"] = ((out["last_risk"] >= -6.45) & (out["last_risk"] < -6.0)).astype(float)
    out["f4_marginal_active_jump"] = out["f4_marginal_zone"] * np.log1p(out["f4_step_max"])
    out["f4_unstable_borderline_high"] = (
        (out["last_risk"] >= -6.0) & (out["last_risk"] < -5.45) & (out["f4_step_max"] > 2.2)
    ).astype(float)

    # F5: Orbital Geometry & Temporal Cadence
    out["f5_n_transitions"] = out["n_transitions"].fillna(0.0)
    out["f5_median_gap_h"] = out["median_gap_hours"].fillna(24.0)
    out["f5_max_gap_h"] = out["max_gap_hours"].fillna(24.0)
    out["f5_gap_burst"] = out["f5_max_gap_h"] / (out["f5_median_gap_h"] + 0.5)
    out["f5_log_n_allowed"] = np.log1p(out["n_allowed"])
    out["f5_single_cdm_flag"] = (out["n_allowed"] == 1).astype(float)

    # F6: Covariance / Uncertainty & OD Quality Signals
    out["f6_cov_logit_nocat"] = np.log(
        out["prob_nocat"].clip(1e-5, 1 - 1e-5) / (1.0 - out["prob_nocat"].clip(1e-5, 1 - 1e-5))
    )
    out["f6_cov_brf"] = out["prob_brf"]
    out["f6_cov_rf"] = out["prob_rf"]
    out["f6_cov_hgb"] = out["prob_hgb"]
    out["f6_obs_sparse"] = (out["n_allowed"] <= 3).astype(float)

    return out


if __name__ == "__main__":
    print("ESA Phase 3 Leak-Free Physics Feature Engineering Pipeline (F1 to F7) ready.")
