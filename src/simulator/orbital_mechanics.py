"""Phase 3 Upgraded Orbital Mechanics, Full 3x3/6x6 Covariance, Actual TCA Velocity & STK/Orekit J2 Validation."""
from __future__ import annotations
from ..simulation import (
    OPERATIONAL_DISCLAIMER,
    CAMSimulationResult,
    mean_motion_from_sma,
    build_full_rtn_covariance_matrix,
    cw_state_transition_matrix,
    rk4_propagate_cw_j2_trajectory,
    compute_bplane_collision_probability,
    simulate_along_track_cam,
    recommend_optimal_cam_burn,
    propagate_nonlinear_eci_j2_cowell,
    validate_j2_against_high_precision_and_stk_orekit,
    extract_last_cdm_kinematics,
    compute_cam_tradeoff_table,
)

__all__ = [
    "OPERATIONAL_DISCLAIMER",
    "CAMSimulationResult",
    "mean_motion_from_sma",
    "build_full_rtn_covariance_matrix",
    "cw_state_transition_matrix",
    "rk4_propagate_cw_j2_trajectory",
    "compute_bplane_collision_probability",
    "simulate_along_track_cam",
    "recommend_optimal_cam_burn",
    "propagate_nonlinear_eci_j2_cowell",
    "validate_j2_against_high_precision_and_stk_orekit",
    "extract_last_cdm_kinematics",
    "compute_cam_tradeoff_table",
]
