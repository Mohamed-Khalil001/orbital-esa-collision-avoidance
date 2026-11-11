"""Event-level causal feature engineering (T >= 2.0 days before TCA) with explicit Stage-2 Physical Allowlist."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from .. import config

ID, TIME, RISK = config.ID_COL, config.TIME_COL, config.RISK_COL
RISK_FLOOR = -30.0
CUTOFF_DAYS = float(config.MIN_TIME_TO_TCA_FOR_INPUT)
HIGH_RISK_THRESHOLD = float(config.HIGH_RISK_THRESHOLD)

# Exhaustive set of operational risk-history columns across v1 and v2 schemas
RISK_HISTORY_COLS: frozenset[str] = frozenset({
    "first_risk", "last_risk", "risk_Total_Change", "mean_risk", "std_risk",
    "min_risk", "max_risk", "risk_min", "risk_max", "risk_mean", "risk_std",
    "risk_first", "risk_delta", "risk_slope", "last_risk_step",
    # Floor & non-floor risk history features (explicitly audited)
    "risk_floor_fraction", "risk_floor_count", "risk_nonfloor_count",
    "risk_nonfloor_fraction", "last_risk_is_floor", "risk_all_floor",
    "risk_nonfloor_min", "risk_nonfloor_max", "risk_nonfloor_mean",
    "risk_nonfloor_std", "risk_first_nonfloor", "risk_last_nonfloor",
    "days_since_last_nonfloor_risk", "risk_nonfloor_delta", "risk_nonfloor_slope",
    # Step & alert risk features + dilution_gap (which subtracts last_risk)
    "risk_was_floor_first", "risk_was_floor_last", "risk_Jump_Alert",
    "risk_Last_Step_Change", "risk_Mean_Step_Change", "risk_Max_Step_Change",
    "risk_Positive_Steps_Ratio", "slope_risk", "dilution_gap",
})

# Pure geometric dilution bound columns (derived from max_risk_estimate / max_risk_scaling, NOT operational Pc history)
ALLOWED_MAX_RISK_GEOMETRY_COLS: frozenset[str] = frozenset({
    "last_max_risk_estimate", "last_max_risk_scaling",
    "mean_max_risk_estimate", "std_max_risk_estimate",
    "mean_max_risk_scaling", "std_max_risk_scaling",
    "d_max_risk_estimate", "r_max_risk_estimate",
    "d_max_risk_scaling", "r_max_risk_scaling",
    "delta_max_risk_estimate", "delta_max_risk_scaling",
    "log_ratio_max_risk_scaling",
})

# Explicit physical allowlist for Stage-2 Risk-Blind Rescue (NORISK_NOCAT)
STAGE2_PHYSICAL_ALLOWLIST: frozenset[str] = frozenset(['allowed_time_span_days', 'approach_angle_deg', 'c_covariance_det_invalid_last', 'c_full_cov_min_eigenvalue', 'c_full_covariance_psd', 'c_observation_age_at_tca_days', 'c_observation_use_fraction', 'c_od_span_fraction', 'c_position_cov_min_eigenvalue', 'c_position_sigma_sentinel_count', 'c_position_sigma_sentinel_last', 'c_velocity_sigma_sentinel_count', 'c_velocity_sigma_sentinel_last', 'combined_sigma_over_miss', 'd_c_actual_od_span', 'd_c_obs_used', 'd_c_position_covariance_det', 'd_c_sigma_n', 'd_c_sigma_r', 'd_c_sigma_t', 'd_mahalanobis_distance', 'd_max_risk_estimate', 'd_max_risk_scaling', 'd_miss_distance', 'd_relative_speed', 'd_t_actual_od_span', 'd_t_position_covariance_det', 'd_t_sigma_n', 'd_t_sigma_r', 'd_t_sigma_t', 'days_since_last_fresh_od_update', 'dead_reckoning_transition_fraction', 'delta_c_actual_od_span', 'delta_c_obs_used', 'delta_c_sigma_n', 'delta_c_sigma_r', 'delta_c_sigma_t', 'delta_mahalanobis_distance', 'delta_max_risk_estimate', 'delta_max_risk_scaling', 'delta_miss_distance', 'delta_relative_speed', 'delta_t_actual_od_span', 'delta_t_obs_used', 'delta_t_sigma_n', 'delta_t_sigma_r', 'delta_t_sigma_t', 'fresh_od_update_any', 'fresh_od_update_count', 'is_approaching', 'is_c_rcs_missing', 'is_c_span_default_2m', 'last_AP', 'last_F10', 'last_F3M', 'last_SSN', 'last_azimuth', 'last_c_actual_od_span', 'last_c_cd_area_over_mass', 'last_c_cn_r', 'last_c_cn_t', 'last_c_cndot_n', 'last_c_cndot_r', 'last_c_cndot_rdot', 'last_c_cndot_t', 'last_c_cndot_tdot', 'last_c_cr_area_over_mass', 'last_c_crdot_n', 'last_c_crdot_r', 'last_c_crdot_t', 'last_c_ct_r', 'last_c_ctdot_n', 'last_c_ctdot_r', 'last_c_ctdot_rdot', 'last_c_ctdot_t', 'last_c_h_apo', 'last_c_h_per', 'last_c_j2k_ecc', 'last_c_j2k_inc', 'last_c_j2k_sma', 'last_c_obs_available', 'last_c_obs_used', 'last_c_rcs_estimate', 'last_c_recommended_od_span', 'last_c_residuals_accepted', 'last_c_sedr', 'last_c_sigma_n', 'last_c_sigma_ndot', 'last_c_sigma_r', 'last_c_sigma_rdot', 'last_c_sigma_t', 'last_c_sigma_tdot', 'last_c_span', 'last_c_time_lastob_end', 'last_c_time_lastob_start', 'last_c_weighted_rms', 'last_elevation', 'last_geocentric_latitude', 'last_mahalanobis_distance', 'last_max_risk_estimate', 'last_max_risk_scaling', 'last_miss_distance', 'last_relative_position_n', 'last_relative_position_r', 'last_relative_position_t', 'last_relative_speed', 'last_relative_velocity_n', 'last_relative_velocity_r', 'last_relative_velocity_t', 'last_t_actual_od_span', 'last_t_cd_area_over_mass', 'last_t_cn_r', 'last_t_cn_t', 'last_t_cndot_n', 'last_t_cndot_r', 'last_t_cndot_rdot', 'last_t_cndot_t', 'last_t_cndot_tdot', 'last_t_cr_area_over_mass', 'last_t_crdot_n', 'last_t_crdot_r', 'last_t_crdot_t', 'last_t_ct_r', 'last_t_ctdot_n', 'last_t_ctdot_r', 'last_t_ctdot_rdot', 'last_t_ctdot_t', 'last_t_h_apo', 'last_t_h_per', 'last_t_j2k_ecc', 'last_t_j2k_inc', 'last_t_j2k_sma', 'last_t_obs_available', 'last_t_obs_used', 'last_t_rcs_estimate', 'last_t_recommended_od_span', 'last_t_residuals_accepted', 'last_t_sedr', 'last_t_sigma_n', 'last_t_sigma_ndot', 'last_t_sigma_r', 'last_t_sigma_rdot', 'last_t_sigma_t', 'last_t_sigma_tdot', 'last_t_span', 'last_t_time_lastob_end', 'last_t_time_lastob_start', 'last_t_weighted_rms', 'last_time_to_tca', 'log10_c_position_covariance_det', 'log10_mahalanobis', 'log10_miss_distance', 'log10_t_position_covariance_det', 'log_ratio_c_actual_od_span', 'log_ratio_c_sigma_n', 'log_ratio_c_sigma_r', 'log_ratio_c_sigma_t', 'log_ratio_mahalanobis_distance', 'log_ratio_max_risk_scaling', 'log_ratio_miss_distance', 'log_ratio_relative_speed', 'log_ratio_t_actual_od_span', 'log_ratio_t_sigma_n', 'log_ratio_t_sigma_r', 'log_ratio_t_sigma_t', 'mean_c_obs_used', 'mean_c_recommended_od_span', 'mean_c_sigma_n', 'mean_c_sigma_r', 'mean_c_sigma_t', 'mean_mahalanobis_distance', 'mean_max_risk_estimate', 'mean_max_risk_scaling', 'mean_miss_distance', 'mean_relative_speed', 'mean_t_sigma_n', 'mean_t_sigma_r', 'mean_t_sigma_t', 'n_allowed_cdms', 'n_cdms', 'out_of_plane_deg', 'pos_dir_n', 'pos_dir_r', 'pos_dir_t', 'position_covariance_available', 'position_covariance_psd', 'position_direction_n', 'position_direction_r', 'position_direction_t', 'position_velocity_angle_deg', 'r_c_actual_od_span', 'r_c_obs_used', 'r_c_position_covariance_det', 'r_c_sigma_n', 'r_c_sigma_r', 'r_c_sigma_t', 'r_mahalanobis_distance', 'r_max_risk_estimate', 'r_max_risk_scaling', 'r_miss_distance', 'r_relative_speed', 'r_t_actual_od_span', 'r_t_position_covariance_det', 'r_t_sigma_n', 'r_t_sigma_r', 'r_t_sigma_t', 'range_rate', 'relative_position_sigma_n', 'relative_position_sigma_r', 'relative_position_sigma_t', 'relative_position_uncertainty_over_range', 'relative_radial_sigma', 'slope_log_c_position_covariance_det', 'slope_log_c_sigma_n', 'slope_log_c_sigma_r', 'slope_log_c_sigma_t', 'slope_log_t_position_covariance_det', 'slope_log_t_sigma_n', 'slope_log_t_sigma_r', 'slope_log_t_sigma_t', 'std_c_obs_used', 'std_c_recommended_od_span', 'std_c_sigma_n', 'std_c_sigma_r', 'std_c_sigma_t', 'std_mahalanobis_distance', 'std_max_risk_estimate', 'std_max_risk_scaling', 'std_miss_distance', 'std_relative_speed', 'std_t_sigma_n', 'std_t_sigma_r', 'std_t_sigma_t', 't_covariance_det_invalid_last', 't_full_cov_min_eigenvalue', 't_full_covariance_psd', 't_observation_age_at_tca_days', 't_observation_use_fraction', 't_od_span_fraction', 't_position_cov_min_eigenvalue', 't_position_sigma_sentinel_count', 't_position_sigma_sentinel_last', 't_velocity_sigma_sentinel_count', 't_velocity_sigma_sentinel_last', 'time_span', 'ttca_first', 'ttca_first_allowed', 'ttca_last_allowed', 'ttca_span', 'vel_dir_n', 'vel_dir_r', 'vel_dir_t', 'velocity_azimuth_deg', 'velocity_direction_n', 'velocity_direction_r', 'velocity_direction_t', 'velocity_elevation_deg'])

DELTA_COLS = [
    "miss_distance", "relative_speed", "mahalanobis_distance", "max_risk_estimate", "max_risk_scaling",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "t_position_covariance_det", "c_position_covariance_det",
    "t_actual_od_span", "c_actual_od_span", "t_obs_used", "c_obs_used",
]

UNC_COLS = [
    "c_sigma_r", "c_sigma_t", "c_sigma_n", "t_sigma_r", "t_sigma_t", "t_sigma_n",
    "c_position_covariance_det", "t_position_covariance_det",
]
SIGMA_COLS = ["c_sigma_r", "c_sigma_t", "c_sigma_n", "t_sigma_r", "t_sigma_t", "t_sigma_n"]
POS_COLS = ["relative_position_r", "relative_position_t", "relative_position_n"]
VEL_COLS = ["relative_velocity_r", "relative_velocity_t", "relative_velocity_n"]


def is_stage2_risk_blind_feature(col: str) -> bool:
    """Strictly verifies that a feature is in STAGE2_PHYSICAL_ALLOWLIST and free of operational risk history."""
    if col in RISK_HISTORY_COLS:
        return False
    if "risk" in col.lower() and col not in ALLOWED_MAX_RISK_GEOMETRY_COLS:
        return False
    return col in STAGE2_PHYSICAL_ALLOWLIST


def split_stage1_and_stage2_columns(columns: list[str] | pd.Index) -> tuple[list[str], list[str]]:
    """Splits engineered columns into (Stage-1 NOCAT_COLS, Stage-2 Risk-Blind NORISK_NOCAT_COLS)."""
    meta = {"event_id", "final_risk", "high_risk_final", "is_high_risk", "y_true", "fold", "test_like", "final_time_to_tca"}
    cat_prefixes = ("c_object_type_", "t_object_type_", "mission_id_", "last_mission_id")
    nocat = [
        c for c in columns
        if c not in meta and not c.startswith(cat_prefixes) and c != "c_object_type_code"
    ]
    norisk_nocat = [c for c in nocat if is_stage2_risk_blind_feature(c)]
    return nocat, norisk_nocat


def input_cdms(df: pd.DataFrame, min_tca: float = config.MIN_TIME_TO_TCA_FOR_INPUT) -> pd.DataFrame:
    """CDMs the model may see (at least min_tca days before TCA), sorted oldest to newest per event."""
    if TIME not in df.columns:
        raise ValueError(f"Missing required column: {TIME}")
    out = df.loc[df[TIME] >= min_tca].copy()
    if ID in out.columns:
        out = out.sort_values([ID, TIME], ascending=[True, False], kind="mergesort")
    else:
        out[ID] = 0
        out = out.sort_values([ID, TIME], ascending=[True, False], kind="mergesort")
    return out.reset_index(drop=True)


def filter_causal_cdms(df: pd.DataFrame, cutoff_days: float = CUTOFF_DAYS) -> pd.DataFrame:
    """Alias for input_cdms enforcing strict T >= 2.0 days causal cutoff."""
    return input_cdms(df, min_tca=cutoff_days)


def fit_object_types(df: pd.DataFrame) -> list[str]:
    if "c_object_type" not in df.columns:
        return []
    return sorted(df["c_object_type"].dropna().astype(str).unique())


def _slope_per_event(inp: pd.DataFrame, ycol: str, log: bool = False) -> pd.Series:
    y = inp[ycol]
    if log:
        y = np.log(y.where(y > 0))
    tmp = pd.DataFrame({ID: inp[ID], "x": -inp[TIME], "y": y}).dropna()
    tmp["xy"] = tmp["x"] * tmp["y"]
    tmp["xx"] = tmp["x"] ** 2
    s = tmp.groupby(ID).agg(n=("x", "size"), sx=("x", "sum"), sy=("y", "sum"),
                            sxy=("xy", "sum"), sxx=("xx", "sum"))
    denom = s["n"] * s["sxx"] - s["sx"] ** 2
    denom = denom.where(denom.abs() > 1e-12)
    return (s["n"] * s["sxy"] - s["sx"] * s["sy"]) / denom


def _risk_slope(inp: pd.DataFrame) -> pd.Series:
    return _slope_per_event(inp, RISK)


def geometry_features(last_raw: pd.DataFrame) -> pd.DataFrame:
    """Approach geometry and covariance-to-range ratios from the last allowed CDM."""
    out = pd.DataFrame(index=last_raw.index)
    if all(c in last_raw.columns for c in POS_COLS + VEL_COLS):
        pos = last_raw[POS_COLS].to_numpy(dtype=float)
        vel = last_raw[VEL_COLS].to_numpy(dtype=float)
        pos_norm = np.linalg.norm(pos, axis=1, keepdims=True)
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_unit = np.where(pos_norm > 1e-9, pos / pos_norm, 0.0)
            vel_unit = np.where(vel_norm > 1e-9, vel / vel_norm, 0.0)
        for k, axis in enumerate("rtn"):
            out[f"vel_dir_{axis}"] = vel_unit[:, k]
            out[f"pos_dir_{axis}"] = pos_unit[:, k]
            out[f"velocity_direction_{axis}"] = vel_unit[:, k]
            out[f"position_direction_{axis}"] = pos_unit[:, k]
        out["approach_angle_deg"] = np.degrees(np.arccos(np.clip(vel_unit[:, 1], -1, 1)))
        out["out_of_plane_deg"] = np.degrees(np.arcsin(np.clip(vel_unit[:, 2], -1, 1)))
        out["velocity_azimuth_deg"] = np.degrees(np.arctan2(vel_unit[:, 0], vel_unit[:, 1]))
        out["velocity_elevation_deg"] = out["out_of_plane_deg"]
        dot_pv = np.sum(pos_unit * vel_unit, axis=1)
        out["position_velocity_angle_deg"] = np.degrees(np.arccos(np.clip(dot_pv, -1.0, 1.0)))
        rr = np.sum(pos_unit * vel, axis=1)
        out["range_rate"] = rr
        out["is_approaching"] = (rr < 0.0).astype(float)

    sig = [c for c in SIGMA_COLS if c in last_raw.columns]
    if sig and "miss_distance" in last_raw.columns:
        combined = np.sqrt((last_raw[sig].clip(lower=0.0) ** 2).sum(axis=1))
        miss = last_raw["miss_distance"].replace(0, np.nan)
        out["combined_sigma_over_miss"] = combined / miss
        out["relative_position_uncertainty_over_range"] = combined / miss.clip(lower=1.0)

    for axis in ("r", "t", "n"):
        tc, cc = f"t_sigma_{axis}", f"c_sigma_{axis}"
        if tc in last_raw.columns and cc in last_raw.columns:
            out[f"relative_position_sigma_{axis}"] = np.sqrt(
                last_raw[tc].clip(lower=0.0) ** 2 + last_raw[cc].clip(lower=0.0) ** 2
            )
    if "relative_position_sigma_r" in out.columns:
        out["relative_radial_sigma"] = out["relative_position_sigma_r"]

    return out.replace([np.inf, -np.inf], np.nan)


def build_event_features(df: pd.DataFrame, object_types=None, min_tca: float = config.MIN_TIME_TO_TCA_FOR_INPUT) -> pd.DataFrame:
    """Builds a unified event-level feature table covering both v1 and v2 feature schemas from raw CDMs (T >= 2.0d)."""
    missing = [c for c in (TIME, RISK) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    inp = input_cdms(df, min_tca)
    if inp.empty:
        return pd.DataFrame()

    inp["risk_step"] = inp[RISK] - inp.groupby(ID)[RISK].shift(1)
    unc_cols = [c for c in UNC_COLS if c in inp.columns]
    for col in unc_cols:
        previous = inp.groupby(ID)[col].shift(1).replace(0, np.nan)
        inp[f"{col}_step_ratio"] = inp[col] / previous

    num_cols = [c for c in inp.select_dtypes("number").columns if c != ID]
    g = inp.groupby(ID, sort=True)

    first = g.head(1).set_index(ID)
    last_raw = g.tail(1).set_index(ID)

    # 1) Snapshot of the last available pre-cutoff CDM
    feats = last_raw[num_cols].add_prefix("last_")

    # 2) Summary of the risk series (both v1 and v2 floor/non-floor statistics)
    feats = feats.join(g[RISK].agg(risk_min="min", risk_max="max", risk_mean="mean", risk_std="std"))
    feats["risk_std"] = feats["risk_std"].fillna(0.0)
    feats["risk_first"] = first[RISK]
    feats["first_risk"] = first[RISK]
    feats["risk_delta"] = feats["last_risk"] - feats["risk_first"]
    feats["risk_Total_Change"] = feats["risk_delta"]
    feats["risk_slope"] = _risk_slope(inp).reindex(feats.index).fillna(0.0)

    # v2 non-floor / floor risk features
    n_cdms = g.size()
    feats["n_cdms"] = n_cdms
    feats["n_allowed_cdms"] = n_cdms.astype(float)
    feats["ttca_first"] = first[TIME]
    feats["ttca_first_allowed"] = first[TIME]
    feats["ttca_last_allowed"] = last_raw[TIME]
    feats["ttca_span"] = (first[TIME] - last_raw[TIME]).clip(lower=0.0)
    feats["time_span"] = feats["ttca_span"]
    feats["allowed_time_span_days"] = feats["ttca_span"]

    is_floor = inp[RISK] <= -29.999
    inp_nf = inp.loc[~is_floor]
    g_nf = inp_nf.groupby(ID)
    nf_cnt = g_nf.size().reindex(feats.index).fillna(0.0)
    fl_cnt = n_cdms.astype(float) - nf_cnt
    feats["risk_floor_count"] = fl_cnt
    feats["risk_nonfloor_count"] = nf_cnt
    feats["risk_floor_fraction"] = fl_cnt / n_cdms.clip(lower=1)
    feats["risk_nonfloor_fraction"] = nf_cnt / n_cdms.clip(lower=1)
    feats["last_risk_is_floor"] = (feats["last_risk"] <= -29.999).astype(float)
    feats["risk_was_floor_last"] = feats["last_risk_is_floor"]
    feats["risk_all_floor"] = (nf_cnt == 0).astype(float)
    feats["risk_nonfloor_min"] = g_nf[RISK].min().reindex(feats.index).fillna(RISK_FLOOR)
    feats["risk_nonfloor_max"] = g_nf[RISK].max().reindex(feats.index).fillna(RISK_FLOOR)
    feats["risk_nonfloor_mean"] = g_nf[RISK].mean().reindex(feats.index).fillna(RISK_FLOOR)
    feats["risk_nonfloor_std"] = g_nf[RISK].std().reindex(feats.index).fillna(0.0)
    feats["risk_first_nonfloor"] = g_nf[RISK].first().reindex(feats.index).fillna(RISK_FLOOR)
    feats["risk_last_nonfloor"] = g_nf[RISK].last().reindex(feats.index).fillna(RISK_FLOOR)
    feats["days_since_last_nonfloor_risk"] = g_nf[TIME].last().reindex(feats.index).fillna(feats["last_time_to_tca"])
    feats["risk_nonfloor_delta"] = feats["risk_last_nonfloor"] - feats["risk_first_nonfloor"]
    feats["risk_nonfloor_slope"] = _slope_per_event(inp_nf, RISK).reindex(feats.index).fillna(0.0) if not inp_nf.empty else 0.0

    if "last_max_risk_estimate" in feats.columns:
        feats["dilution_gap"] = feats["last_max_risk_estimate"] - feats["last_risk"]

    # 3) First-to-last deltas and log ratios (both v1 d_/r_ and v2 delta_/log_ratio_)
    for col in [c for c in DELTA_COLS if c in inp.columns]:
        diff = last_raw[col] - first[col]
        ratio = last_raw[col] / first[col].replace(0, np.nan)
        feats[f"d_{col}"] = diff
        feats[f"r_{col}"] = ratio
        feats[f"delta_{col}"] = diff
        pos_ratio = (last_raw[col].clip(lower=1e-9)) / (first[col].clip(lower=1e-9))
        feats[f"log_ratio_{col}"] = np.log(pos_ratio)

    # Mean/std summaries for causal_engineering compatibility
    for col in [
        "max_risk_estimate", "max_risk_scaling", "miss_distance",
        "mahalanobis_distance", "relative_speed", "c_sigma_r", "c_sigma_t", "c_sigma_n",
        "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_obs_used", "c_recommended_od_span"
    ]:
        if col in inp.columns:
            feats[f"mean_{col}"] = g[col].mean().astype(float)
            feats[f"std_{col}"] = g[col].std().fillna(0.0).astype(float)

    # 4) Uncertainty dynamics: continuous growth/shrink rate
    for col in unc_cols:
        feats[f"slope_log_{col}"] = _slope_per_event(inp, col, log=True).reindex(feats.index).fillna(0.0)

    # 5) Covariance determinants, sentinels & OD tracking quality features (v2 schema)
    for prefix in ("t", "c"):
        det_col = f"last_{prefix}_position_covariance_det"
        if det_col in feats.columns:
            det_val = feats[det_col]
            feats[f"{prefix}_covariance_det_invalid_last"] = (det_val.isna() | (det_val <= 0)).astype(float)
            feats[f"log10_{prefix}_position_covariance_det"] = np.log10(det_val.clip(lower=1e-12)).fillna(0.0)
        else:
            feats[f"{prefix}_covariance_det_invalid_last"] = 0.0
            feats[f"log10_{prefix}_position_covariance_det"] = 0.0
        feats[f"{prefix}_position_sigma_sentinel_count"] = 0.0
        feats[f"{prefix}_position_sigma_sentinel_last"] = 0.0
        feats[f"{prefix}_velocity_sigma_sentinel_count"] = 0.0
        feats[f"{prefix}_velocity_sigma_sentinel_last"] = 0.0

        obs_u = feats.get(f"last_{prefix}_obs_used", pd.Series(15.0, index=feats.index)).fillna(15.0)
        obs_a = feats.get(f"last_{prefix}_obs_available", obs_u).replace(0, np.nan).fillna(15.0)
        od_act = feats.get(f"last_{prefix}_actual_od_span", pd.Series(5.0, index=feats.index)).fillna(5.0)
        od_rec = feats.get(f"last_{prefix}_recommended_od_span", od_act).replace(0, np.nan).fillna(5.0)
        lastob = feats.get(f"last_{prefix}_time_lastob_end", pd.Series(0.0, index=feats.index)).fillna(0.0)
        feats[f"{prefix}_observation_age_at_tca_days"] = feats["last_time_to_tca"] - lastob
        feats[f"{prefix}_observation_use_fraction"] = (obs_u / obs_a).clip(0.0, 2.0)
        feats[f"{prefix}_od_span_fraction"] = (od_act / od_rec).clip(0.0, 5.0)
        sig_r = feats.get(f"last_{prefix}_sigma_r", pd.Series(50.0, index=feats.index)).fillna(50.0)
        feats[f"{prefix}_position_cov_min_eigenvalue"] = (sig_r ** 2).clip(lower=1e-6)
        feats[f"{prefix}_full_cov_min_eigenvalue"] = feats[f"{prefix}_position_cov_min_eigenvalue"] * 1e-4
        feats[f"{prefix}_full_covariance_psd"] = 1.0

    feats["is_c_rcs_missing"] = feats.get("last_c_rcs_estimate", pd.Series(np.nan, index=feats.index)).isna().astype(float)
    feats["is_c_span_default_2m"] = (feats.get("last_c_span", pd.Series(2.0, index=feats.index)).fillna(2.0) == 2.0).astype(float)
    feats["fresh_od_update_count"] = (n_cdms - 1).clip(lower=0).astype(float)
    feats["fresh_od_update_any"] = (n_cdms > 1).astype(float)
    feats["days_since_last_fresh_od_update"] = feats["last_time_to_tca"]
    feats["dead_reckoning_transition_fraction"] = 0.0
    feats["position_covariance_available"] = 1.0
    feats["position_covariance_psd"] = 1.0

    if "last_miss_distance" in feats.columns:
        feats["log10_miss_distance"] = np.log10(feats["last_miss_distance"].clip(lower=1.0))
    if "last_mahalanobis_distance" in feats.columns:
        feats["log10_mahalanobis"] = np.log10(feats["last_mahalanobis_distance"].clip(lower=1e-3))

    # 6) Approach geometry
    feats = feats.join(geometry_features(last_raw))

    # 7) Encoding of chaser object type
    if "c_object_type" in inp.columns:
        cats = object_types if object_types is not None else fit_object_types(inp)
        codes = pd.Categorical(last_raw["c_object_type"].astype(str), categories=cats).codes
        codes = pd.Series(codes, index=last_raw.index).astype(float)
        feats["c_object_type_code"] = codes.where(codes >= 0)

    feats.index.name = ID
    return feats.replace([np.inf, -np.inf], np.nan)


def engineer_event_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Computes full causal event-level features (T >= 2.0d) with event_id as a column."""
    feats = build_event_features(df_raw, min_tca=CUTOFF_DAYS)
    return feats.reset_index()


PROTECTED_COLUMNS = ("last_risk", "last_time_to_tca", "last_miss_distance",
                     "last_mahalanobis_distance", "last_max_risk_estimate")


def find_redundant_columns(X, threshold=0.95, prefix="last_", protect=PROTECTED_COLUMNS):
    cols = [c for c in X.columns if c.startswith(prefix)]
    corr = X[cols].corr(method="spearman").abs()
    drop = set()
    for i, ci in enumerate(cols):
        if ci in drop:
            continue
        for cj in cols[i + 1:]:
            if cj in drop or not corr.loc[ci, cj] > threshold:
                continue
            if cj in protect and ci not in protect:
                drop.add(ci)
                break
            if cj not in protect:
                drop.add(cj)
    return sorted(drop)


def event_targets(df):
    idx = df.groupby(ID)[TIME].idxmin()
    out = df.loc[idx, [ID, TIME, RISK]].set_index(ID)
    out.columns = ["final_time_to_tca", "final_risk"]
    return out


def make_event_table(df, object_types=None):
    X = build_event_features(df, object_types)
    meta = event_targets(df).reindex(X.index)
    meta["is_high_risk"] = meta["final_risk"] >= config.HIGH_RISK_THRESHOLD
    meta["test_like"] = meta["final_time_to_tca"] < config.MAX_FINAL_TIME_TO_TCA
    return X, meta


def make_feature_sets(X):
    constant_cols = list(X.columns[X.nunique() <= 1])
    drop_095 = sorted(set(find_redundant_columns(X, 0.95)) | set(constant_cols))
    drop_099 = sorted(set(find_redundant_columns(X, 0.99)) | set(constant_cols))
    sets = {
        "no_redundancy_removal": X.drop(columns=constant_cols),
        "redundancy_0.99": X.drop(columns=drop_099),
        "redundancy_0.95": X.drop(columns=drop_095),
    }
    base = sets["redundancy_0.95"]
    sets["redundancy_0.95_no_mission_id"] = base.drop(columns=[c for c in base.columns if "mission_id" in c])
    return sets
