#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESA Kelvins Spacecraft Collision Avoidance Challenge
Script 20 — Phase 6 Single-Pass Final Test Evaluation Runner
Strictly loads pre-locked parameters from 19_Phase6_Frozen_PreTest_Configuration.json
and executes a single evaluation pass on the 2,167 test events (with explicit semi-exposed holdout disclosure).
"""
import os, json
import numpy as np
import pandas as pd

WORKDIR = "/working_dir/c_9801697d79fff702"
DL = os.path.join(WORKDIR, "downloads")
OUT = "/tmp/final_process_out"

cfg = json.load(open(os.path.join(OUT, "19_Phase6_Frozen_PreTest_Configuration.json")))
assert cfg["locked_before_test_evaluation"] is True

GATE = float(cfg["architecture_parameters"]["GATE"])
THR = float(cfg["architecture_parameters"]["BASELINE_THRESHOLD"])
HL = float(cfg["architecture_parameters"]["HardLock"])
w_distill = np.array(cfg["deployable_test_scoring_calibration"]["w_distill"], dtype=float)
tau_veto_dist = float(cfg["deployable_test_scoring_calibration"]["tau_veto_dist"])
tau_rescue_r100 = float(cfg["deployable_test_scoring_calibration"]["tau_rescue_dist_r100"])
tau_rescue_r135 = float(cfg["deployable_test_scoring_calibration"]["tau_rescue_dist_r135"])

Xt = pd.read_csv(os.path.join(DL, "X_test_fe_v2.csv")).sort_values("event_id").reset_index(drop=True)
p_lgb_df = pd.read_csv(os.path.join(DL, "test_high_risk_predictions.csv")).sort_values("event_id").reset_index(drop=True)
colab_ens = pd.read_csv(os.path.join(DL, "final_test_predictions.csv")).sort_values("event_id").reset_index(drop=True)
raw_te = pd.read_csv(os.path.join(DL, "test_data_private.csv"))
te_final = raw_te.sort_values(["event_id", "time_to_tca"], ascending=[True, True]).groupby("event_id", as_index=False).first()
te_final = te_final.sort_values("event_id").reset_index(drop=True)

FR_te = te_final["true_risk"].values.astype(float)
y_te = (FR_te >= THR).astype(int)
LR_te = Xt["last_risk"].values.astype(float)
n_all_te = Xt["n_allowed_cdms"].values.astype(float)
span_te = Xt["allowed_time_span_days"].clip(lower=0.0).values.astype(float)
cgap_te = (Xt["ttca_last_allowed"] - 2.0).clip(lower=0.0).values.astype(float)
p_lgb_te = p_lgb_df["high_risk_probability"].values.astype(float)

logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1.0 - np.clip(p, 1e-5, 1 - 1e-5)))

X_te_gate = np.column_stack([
    np.ones(len(Xt)),
    logit(p_lgb_te),
    LR_te,
    np.log1p(n_all_te),
    span_te,
    cgap_te,
    LR_te * np.log1p(n_all_te),
    (cgap_te <= 0.25).astype(float),
    (n_all_te <= 2).astype(float),
])
rf_logit_hat_te = X_te_gate @ w_distill
s1_te = rf_logit_hat_te + 0.25 * LR_te
s2_te = rf_logit_hat_te + 0.20 * LR_te
s_m3_te = rf_logit_hat_te + 0.35 * LR_te

et_reg_pred = np.where(colab_ens["predicted_risk"].values > -29.0, colab_ens["predicted_risk"].values, np.maximum(LR_te, -5.99))

keep_s1 = (LR_te >= -6.0) & ((s1_te >= tau_veto_dist) | (LR_te >= HL))
vet_mask_s1 = (LR_te >= -6.0) & (~keep_s1)
n_vet_te = int(vet_mask_s1.sum())
sub_te = (LR_te >= GATE) & (LR_te < -6.0)
s2_sub_sorted = np.sort(s2_te[sub_te])[::-1]

n_res_r100 = max(1, int(round(n_vet_te * 1.00)))
thr_res_r100 = float(s2_sub_sorted[min(n_res_r100 - 1, len(s2_sub_sorted) - 1)])
res_mask_r100 = sub_te & (s2_te >= thr_res_r100)
flag_r100 = keep_s1 | res_mask_r100

n_res_r135 = max(1, int(round(n_vet_te * 1.35)))
thr_res_r135 = float(s2_sub_sorted[min(n_res_r135 - 1, len(s2_sub_sorted) - 1)])
res_mask_r135 = sub_te & (s2_te >= thr_res_r135)
flag_r135 = keep_s1 | res_mask_r135
