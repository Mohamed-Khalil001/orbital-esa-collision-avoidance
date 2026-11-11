#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESA Kelvins Spacecraft Collision Avoidance Challenge
Script 25 — Master End-to-End Pipeline:
  1. Feature Engineering (from pre-cutoff CDM summaries & feature matrices)
  2. Training (Two-Stage Veto+Rescue Classifier + Two-Regime Shrinkage Residual Regressor)
  3. Cross-Validation (15-Fold Stratified CV on 8,892 training events)
  4. Prediction (Generating flag_high_risk and predicted_risk on all 2,167 test events)
  5. Evaluation & Leaderboard Comparison (Comparing against B0-B3, v1/v2 models, and Official ESA Top-5 Teams)
"""
import os, json, pickle
import numpy as np
import pandas as pd

WORKDIR = "/working_dir/c_9801697d79fff702"
DL = os.path.join(WORKDIR, "downloads")
OUT = "/tmp/final_process_out"
os.makedirs(OUT, exist_ok=True)

logit = lambda p: np.log(np.clip(p, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(p, 1e-4, 1.0 - 1e-4)))

def build_engineered_features_train():
    c01 = pd.read_csv(os.path.join(DL, "01_train_event_cohort_audit.csv"))
    c04 = pd.read_csv(os.path.join(DL, "04_cutoff_event_summary.csv"))
    el = c01[c01["cohort"] == "ELIGIBLE_COMPLETE"].sort_values("event_id").reset_index(drop=True)
    c04_el = c04[c04["event_id"].isin(el["event_id"])].sort_values("event_id").reset_index(drop=True)

    df = pd.DataFrame({
        "event_id": el["event_id"].values,
        "last_risk": el["last_allowed_risk"].values.astype(float),
        "final_risk": el["final_risk"].values.astype(float),
        "n_allowed": el["n_allowed"].values.astype(float),
        "first_tca": el["first_time_to_tca"].values.astype(float),
        "last_tca": el["last_allowed_time"].values.astype(float),
        "earliest_tca": c04_el["earliest_tca"].values.astype(float),
    })
    df["y"] = (df["final_risk"] >= -6.0).astype(int)
    df["allowed_span_days"] = (df["first_tca"] - df["last_tca"]).clip(lower=0.0)
    df["cutoff_gap"] = (df["last_tca"] - 2.0).clip(lower=0.0)
    df["log1p_n_allowed"] = np.log1p(df["n_allowed"])
    df["cdm_density"] = df["n_allowed"] / (df["allowed_span_days"] + 0.25)
    df["near_cutoff_flag"] = (df["cutoff_gap"] <= 0.25).astype(float)
    df["sparse_cdm_flag"] = (df["n_allowed"] <= 2).astype(float)
    df["lr_x_log1p_n"] = df["last_risk"] * df["log1p_n_allowed"]

    oof_full_raw = pd.read_csv(os.path.join(DL, "oof_full_v2.csv"))
    oof_nocat_raw = pd.read_csv(os.path.join(DL, "oof_without_categorical.csv"))
    df["p_lgbm_oof"] = oof_full_raw.groupby("row_index")["probability"].mean().sort_index().values
    df["logit_p_lgbm"] = logit(df["p_lgbm_oof"].values)

    rf_oof = pickle.load(open(os.path.join(DL, "v1_oof_RF_nocat_0-1-2.pkl"), "rb"))
    hgb_oof = pickle.load(open(os.path.join(DL, "v1_oof_HGB_0-1-2.pkl"), "rb"))
    brf_oof = pickle.load(open(os.path.join(DL, "v1_oof_BalancedRF_0-1-2.pkl"), "rb"))
    return df, oof_nocat_raw, rf_oof, hgb_oof, brf_oof

def build_engineered_features_test():
    Xt = pd.read_csv(os.path.join(DL, "X_test_fe_v2.csv")).sort_values("event_id").reset_index(drop=True)
    p_lgb_df = pd.read_csv(os.path.join(DL, "test_high_risk_predictions.csv")).sort_values("event_id").reset_index(drop=True)
    colab_ens = pd.read_csv(os.path.join(DL, "final_test_predictions.csv")).sort_values("event_id").reset_index(drop=True)
    colab_lck = pd.read_csv(os.path.join(DL, "final_test_predictions_LOCKED.csv")).sort_values("event_id").reset_index(drop=True)
    reg_v1 = pd.read_csv(os.path.join(DL, "test_predictions_v1.csv")).sort_values("event_id").reset_index(drop=True)
    raw_te = pd.read_csv(os.path.join(DL, "test_data_private.csv"))
    te_final = raw_te.sort_values(["event_id", "time_to_tca"]).groupby("event_id", as_index=False).first().sort_values("event_id").reset_index(drop=True)

    df_te = pd.DataFrame({
        "event_id": Xt["event_id"].values,
        "last_risk": Xt["last_risk"].values.astype(float),
        "final_risk": te_final["true_risk"].values.astype(float),
        "n_allowed": Xt["n_allowed_cdms"].values.astype(float),
        "first_tca": Xt["ttca_first_allowed"].values.astype(float),
        "last_tca": Xt["ttca_last_allowed"].values.astype(float),
        "risk_nonfloor_min": Xt["risk_nonfloor_min"].values.astype(float),
        "risk_nonfloor_max": Xt["risk_nonfloor_max"].values.astype(float),
        "p_lgbm": p_lgb_df["high_risk_probability"].values.astype(float),
        "p_lgbm_default_flag": p_lgb_df["predicted_is_high_risk"].values.astype(int),
        "v1_pred_risk": reg_v1["predicted_risk"].values.astype(float),
        "v1_adj_risk": reg_v1["adjusted_risk"].values.astype(float),
        "v1_alarm": reg_v1["alarm"].values.astype(int),
        "colab_ens_flag": colab_ens["flag_high_risk"].values.astype(int),
        "colab_ens_pred_risk": colab_ens["predicted_risk"].values.astype(float),
        "colab_lck_flag": colab_lck["flag_high_risk"].values.astype(int),
        "colab_lck_pred_risk": colab_lck["predicted_risk"].values.astype(float),
    })
    df_te["y"] = (df_te["final_risk"] >= -6.0).astype(int)
    df_te["allowed_span_days"] = Xt["allowed_time_span_days"].clip(lower=0.0).values.astype(float)
    df_te["cutoff_gap"] = (df_te["last_tca"] - 2.0).clip(lower=0.0)
    df_te["log1p_n_allowed"] = np.log1p(df_te["n_allowed"])
    df_te["cdm_density"] = df_te["n_allowed"] / (df_te["allowed_span_days"] + 0.25)
    df_te["near_cutoff_flag"] = (df_te["cutoff_gap"] <= 0.25).astype(float)
    df_te["sparse_cdm_flag"] = (df_te["n_allowed"] <= 2).astype(float)
    df_te["lr_x_log1p_n"] = df_te["last_risk"] * df_te["log1p_n_allowed"]
    df_te["logit_p_lgbm"] = logit(df_te["p_lgbm"].values)
    return df_te
