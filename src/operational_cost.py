"""Phase 5: Operational Decision Cost Engine & Cost-Sensitive Threshold Optimization.

Implements the explicit ESOC / Satellite Operator Decision Cost Function:
    C = C_FP * FP + C_FN * FN + C_CAM * N_CAM
where:
  - FP: False Positive alarms (unnecessary analyst screening & tracking escalation)
  - FN: False Negative misses (unmitigated high-risk conjunctions at TCA)
  - N_CAM: Executed Collision Avoidance Maneuvers (typically TP + alpha_cam * FP, carrying propellant
    depletion, mission downtime, and orbit perturbation cost C_CAM)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from . import config

MISSION_PROFILES: dict[str, dict[str, float | str]] = {
    "LEO_Operational_R20": {
        "name": "Standard LEO Earth Observation (ESOC Sentinel Profile)",
        "c_fp": 1.0,
        "c_fn": 20.0,
        "c_cam": 2.0,
        "cam_fraction_on_fp": 0.50,
    },
    "Flagship_Safety_Critical_R50": {
        "name": "High-Value Flagship / Crewed Asset (Safety-Critical)",
        "c_fp": 1.0,
        "c_fn": 50.0,
        "c_cam": 1.5,
        "cam_fraction_on_fp": 0.75,
    },
    "Constellation_Fuel_Constrained": {
        "name": "Propellant-Constrained SmallSat / CubeSat Constellation",
        "c_fp": 1.0,
        "c_fn": 15.0,
        "c_cam": 5.0,
        "cam_fraction_on_fp": 0.40,
    },
}


@dataclass
class OperationalCostSummary:
    profile_key: str
    c_fp: float
    c_fn: float
    c_cam: float
    tp: int
    fp: int
    fn: int
    tn: int
    n_alarms: int
    n_cam_executed: float
    cost_fp_total: float
    cost_fn_total: float
    cost_cam_total: float
    total_operational_cost_c: float
    cost_per_event: float
    baseline_b0_cost_c: float
    net_cost_savings_vs_b0: float
    f2_score: float
    esa_loss_l: float


def compute_operational_cost(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    c_fp: float = 1.0,
    c_fn: float = 20.0,
    c_cam: float = 2.0,
    cam_fraction_on_fp: float = 1.0,
    f2_score: float = np.nan,
    esa_loss_l: float = np.nan,
    baseline_tp: int = 50,
    baseline_fp: int = 16,
    baseline_fn: int = 100,
    profile_key: str = "Custom",
) -> OperationalCostSummary:
    """Computes C = C_FP * FP + C_FN * FN + C_CAM * N_CAM and compares against the B0 baseline."""
    tp_i, fp_i, fn_i, tn_i = int(tp), int(fp), int(fn), int(tn)
    n_alarms = tp_i + fp_i
    n_cam = float(tp_i) + float(cam_fraction_on_fp) * float(fp_i)

    cost_fp = float(c_fp) * fp_i
    cost_fn = float(c_fn) * fn_i
    cost_cam = float(c_cam) * n_cam
    total_c = cost_fp + cost_fn + cost_cam

    b0_cam = float(baseline_tp) + float(cam_fraction_on_fp) * float(baseline_fp)
    b0_cost = float(c_fp) * baseline_fp + float(c_fn) * baseline_fn + float(c_cam) * b0_cam
    n_total = max(tp_i + fp_i + fn_i + tn_i, 1)

    return OperationalCostSummary(
        profile_key=profile_key,
        c_fp=round(float(c_fp), 3),
        c_fn=round(float(c_fn), 3),
        c_cam=round(float(c_cam), 3),
        tp=tp_i,
        fp=fp_i,
        fn=fn_i,
        tn=tn_i,
        n_alarms=n_alarms,
        n_cam_executed=round(n_cam, 2),
        cost_fp_total=round(cost_fp, 2),
        cost_fn_total=round(cost_fn, 2),
        cost_cam_total=round(cost_cam, 2),
        total_operational_cost_c=round(total_c, 2),
        cost_per_event=round(total_c / n_total, 4),
        baseline_b0_cost_c=round(b0_cost, 2),
        net_cost_savings_vs_b0=round(b0_cost - total_c, 2),
        f2_score=round(float(f2_score), 5) if pd.notna(f2_score) else np.nan,
        esa_loss_l=round(float(esa_loss_l), 5) if pd.notna(esa_loss_l) else np.nan,
    )


def compute_single_event_decision_cost(
    calibrated_collision_prob: float,
    c_fp: float = 1.0,
    c_fn: float = 20.0,
    c_cam: float = 2.0,
    dv_mag_cms: float = 2.5,
) -> dict:
    """Computes Bayesian expected operational cost for a single conjunction event.

    - Expected cost of NO ALARM / NO CAM: E[C | a=0] = p * C_FN
    - Expected cost of ALARM + CAM:       E[C | a=1] = (1 - p) * C_FP + C_CAM * (1 + 0.05 * |dv|)
    - Bayes-optimal cost threshold p* = (C_FP + C_CAM_eff) / (C_FN + C_FP)
    """
    p = float(np.clip(calibrated_collision_prob, 0.0, 1.0))
    c_cam_eff = float(c_cam) * (1.0 + 0.05 * abs(float(dv_mag_cms)))
    e_cost_no_action = p * float(c_fn)
    e_cost_alarm_cam = (1.0 - p) * float(c_fp) + c_cam_eff
    p_star = min(0.99, (float(c_fp) + c_cam_eff) / max(float(c_fn) + float(c_fp), 1e-6))
    cost_optimal_alarm = bool(e_cost_alarm_cam <= e_cost_no_action)
    return {
        "calibrated_prob": round(p, 6),
        "c_fp": float(c_fp),
        "c_fn": float(c_fn),
        "c_cam_effective": round(c_cam_eff, 4),
        "expected_cost_no_action": round(e_cost_no_action, 4),
        "expected_cost_alarm_and_cam": round(e_cost_alarm_cam, 4),
        "bayes_optimal_prob_threshold": round(p_star, 4),
        "cost_optimal_alarm": cost_optimal_alarm,
        "recommended_operational_decision": "ALARM_AND_PLAN_CAM" if cost_optimal_alarm else "MONITOR_WITHOUT_CAM",
    }


def optimize_thresholds_for_operational_cost(
    last_risk: np.ndarray,
    s1_scores: np.ndarray,
    s2_scores: np.ndarray,
    y_true: np.ndarray,
    c_fp: float = 1.0,
    c_fn: float = 20.0,
    c_cam: float = 2.0,
    hard_lock: float = -5.25,
    gate_thresh: float = -7.0,
) -> dict:
    """Grid-searches (tau_veto, tau_rescue) on training OOF scores to minimize C = C_FP*FP + C_FN*FN + C_CAM*N_CAM."""
    lr = np.asarray(last_risk, dtype=float)
    s1 = np.asarray(s1_scores, dtype=float)
    s2 = np.asarray(s2_scores, dtype=float)
    y = np.asarray(y_true, dtype=int)

    z1 = lr >= hard_lock
    z2 = (lr >= -6.0) & (lr < hard_lock)
    z3 = (lr >= gate_thresh) & (lr < -6.0)

    veto_grid = np.quantile(s1[z2], np.linspace(0.05, 0.60, 12)) if z2.sum() > 0 else np.array([-2.65])
    resc_grid = np.quantile(s2[z3], np.linspace(0.65, 0.98, 12)) if z3.sum() > 0 else np.array([-2.50])

    best_cost = float("inf")
    best_cfg = {}
    for tv in veto_grid:
        for tr in resc_grid:
            pred = z1 | (z2 & (s1 >= tv)) | (z3 & (s2 >= tr))
            tp = int(np.sum((pred == 1) & (y == 1)))
            fp = int(np.sum((pred == 1) & (y == 0)))
            fn = int(np.sum((pred == 0) & (y == 1)))
            n_cam = tp + fp
            cost = float(c_fp) * fp + float(c_fn) * fn + float(c_cam) * n_cam
            if cost < best_cost:
                best_cost = cost
                denom = 5 * tp + 4 * fn + fp
                f2 = (5 * tp / denom) if denom > 0 else 0.0
                best_cfg = {
                    "tau_veto_cost_optimal": round(float(tv), 4),
                    "tau_rescue_cost_optimal": round(float(tr), 4),
                    "min_operational_cost_c": round(float(cost), 2),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "n_cam": n_cam,
                    "f2_at_min_cost": round(float(f2), 4),
                }
    return best_cfg


def build_operational_cost_comparison_table(
    c_fp: float = 1.0,
    c_fn: float = 20.0,
    c_cam: float = 2.0,
    cam_fraction_on_fp: float = 1.0,
) -> pd.DataFrame:
    """Builds the Operational Cost C = C_FP*FP + C_FN*FN + C_CAM*N_CAM table across all evaluated models."""
    comp_path = config.RESEARCH_RESULTS_DIR / "colab_run_outputs" / "06_Final_Test_and_ESA_Leaderboard_Comparison.csv"
    df_board = pd.read_csv(comp_path)
    # Filter to systems that have explicit confusion matrix counts (TP, FP, FN)
    df_valid = df_board.dropna(subset=["TP", "FP", "FN"]).copy()

    b0_row = df_valid[df_valid["System"].astype(str).str.contains("B0_LRP_Baseline")].iloc[0]
    b0_tp, b0_fp, b0_fn = int(b0_row["TP"]), int(b0_row["FP"]), int(b0_row["FN"])

    rows = []
    for _, r in df_valid.iterrows():
        tp, fp, fn = int(r["TP"]), int(r["FP"]), int(r["FN"])
        tn = 2167 - tp - fp - fn
        summary = compute_operational_cost(
            tp=tp, fp=fp, fn=fn, tn=tn,
            c_fp=c_fp, c_fn=c_fn, c_cam=c_cam,
            cam_fraction_on_fp=cam_fraction_on_fp,
            f2_score=float(r["F2"]),
            esa_loss_l=float(r["ESA_Loss_L"]),
            baseline_tp=b0_tp, baseline_fp=b0_fp, baseline_fn=b0_fn,
        )
        rows.append({
            "System": str(r["System"]),
            "Category": str(r["Category"]),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Alarms_N_CAM": summary.n_alarms,
            "F2": summary.f2_score,
            "ESA_Loss_L": summary.esa_loss_l,
            "Cost_FP": summary.cost_fp_total,
            "Cost_FN": summary.cost_fn_total,
            "Cost_CAM": summary.cost_cam_total,
            "Total_Operational_Cost_C": summary.total_operational_cost_c,
            "Net_Cost_Savings_vs_B0": summary.net_cost_savings_vs_b0,
        })
    out_df = pd.DataFrame(rows).sort_values("Total_Operational_Cost_C", ascending=True).reset_index(drop=True)
    out_df["Cost_Rank"] = np.arange(1, len(out_df) + 1)
    return out_df
