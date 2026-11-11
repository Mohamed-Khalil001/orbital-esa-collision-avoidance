#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESA Kelvins Spacecraft Collision Avoidance Challenge
Script 16 — Phase 5 Physics Validation & Feature Family Attribution Standalone Runner
Validates:
1. Why GATE = -7.0 is the Physical Observability Horizon at T - 2.0 days (Z1..Z6 base rates & uA_nocat).
2. Why HardLock = -5.25 + buf = 0.10 achieves 100.0% Stage-1 Veto Purity (0 True Positives lost in G2).
3. Feature Family Logit Attribution (Covariance F6, Risk Dynamics F4, Geometry F2/F3/F5, OD Quality F6, Temporal Slopes F5).
"""
import os, pickle
import numpy as np
import pandas as pd

WORKDIR = "/working_dir/c_9801697d79fff702"
DL = os.path.join(WORKDIR, "downloads")
SUB_DL = "/working_dir/c_fdf58d6d93feebe0/downloads"
OUT = "/tmp/final_process_out"
os.makedirs(OUT, exist_ok=True)

c01 = pd.read_csv(os.path.join(DL, "01_train_event_cohort_audit.csv"))
c04 = pd.read_csv(os.path.join(DL, "04_cutoff_event_summary.csv"))
el = c01[c01["cohort"] == "ELIGIBLE_COMPLETE"].sort_values("event_id").reset_index(drop=True)
c04_el = c04[c04["event_id"].isin(el["event_id"])].sort_values("event_id").reset_index(drop=True)

df = el[["event_id", "total_cdms", "n_allowed", "n_late", "first_time_to_tca", "final_time_to_tca", "last_allowed_time", "last_allowed_risk", "final_risk"]].copy()
df["earliest_tca"] = c04_el["earliest_tca"].values
df["y"] = (df["final_risk"] >= -6.0).astype(int)
df["LR"] = df["last_allowed_risk"].astype(float)
df["FR"] = df["final_risk"].astype(float)
df["allowed_span_days"] = (df["first_time_to_tca"] - df["last_allowed_time"]).clip(lower=0.0)
df["cutoff_gap"] = (df["last_allowed_time"] - 2.0).clip(lower=0.0)
df["cdm_density"] = df["n_allowed"] / (df["allowed_span_days"] + 0.25)

rf_oof = pickle.load(open(os.path.join(DL, "v1_oof_RF_nocat_0-1-2.pkl"), "rb"))
hgb_oof = pickle.load(open(os.path.join(DL, "v1_oof_HGB_0-1-2.pkl"), "rb"))
brf_oof = pickle.load(open(os.path.join(DL, "v1_oof_BalancedRF_0-1-2.pkl"), "rb"))

def load_oof_mean(path):
    o = pd.read_csv(path)
    return o.groupby("row_index")["probability"].mean().sort_index().values

oof_nocat_raw = pd.read_csv(os.path.join(DL, "oof_without_categorical.csv"))
df["uA_nocat"] = load_oof_mean(os.path.join(DL, "oof_without_categorical.csv"))
df["uA_full"] = load_oof_mean(os.path.join(DL, "oof_full_v2.csv"))

logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1.0 - np.clip(p, 1e-5, 1 - 1e-5)))
df["logit_nocat"] = logit(df["uA_nocat"])
df["attr_Covariance_F6"] = df["logit_nocat"] - logit(load_oof_mean(os.path.join(SUB_DL, "oof_without_covariance_uncertainty.csv")))
df["attr_RiskDynamics_F4"] = df["logit_nocat"] - logit(load_oof_mean(os.path.join(SUB_DL, "oof_without_risk_history.csv")))
df["attr_Geometry_F2F3F5"] = df["logit_nocat"] - logit(load_oof_mean(os.path.join(SUB_DL, "oof_without_geometry.csv")))
df["attr_OD_Quality_F6"] = df["logit_nocat"] - logit(load_oof_mean(os.path.join(SUB_DL, "oof_without_observation_od.csv")))
df["attr_Temporal_Change_F5"] = df["logit_nocat"] - logit(load_oof_mean(os.path.join(SUB_DL, "oof_without_temporal_change.csv")))

df["rf_mean"] = (np.clip(rf_oof[0], 0, 1) + np.clip(rf_oof[1], 0, 1) + np.clip(rf_oof[2], 0, 1)) / 3.0
df["hgb_mean"] = (np.clip(hgb_oof[0], 0, 1) + np.clip(hgb_oof[1], 0, 1) + np.clip(hgb_oof[2], 0, 1)) / 3.0
df["brf_mean"] = (np.clip(brf_oof[0], 0, 1) + np.clip(brf_oof[1], 0, 1) + np.clip(brf_oof[2], 0, 1)) / 3.0

y = df["y"].values
LR = df["LR"].values
all_idx = np.arange(len(df))

vetoed_mask = np.zeros(len(df), dtype=bool)
rescued_mask = np.zeros(len(df), dtype=bool)
rf0 = np.clip(rf_oof[0], 0, 1)
s1_0 = logit(rf0) + 0.25 * LR
s2_0 = logit(rf0) + 0.20 * LR

for f_id in range(1, 6):
    te = np.sort(oof_nocat_raw.loc[oof_nocat_raw["fold"] == f_id, "row_index"].values)
    tr = np.setdiff1d(all_idx, te, assume_unique=True)
    b_tr = (LR[tr] >= -6.0) & (LR[tr] < -5.25)
    tau_veto = float(s1_0[tr][b_tr & (y[tr] == 1)].min()) - 0.10
    keep_tr = (LR[tr] >= -6.0) & ((s1_0[tr] >= tau_veto) | (LR[tr] >= -5.25))
    keep_te = (LR[te] >= -6.0) & ((s1_0[te] >= tau_veto) | (LR[te] >= -5.25))
    n_vetoed_tr = int((LR[tr] >= -6.0).sum() - keep_tr.sum())
    sub_tr = (LR[tr] >= -7.0) & (LR[tr] < -6.0)
    s2_sub_tr = np.sort(s2_0[tr][sub_tr])[::-1]
    tau_rescue = float(s2_sub_tr[min(n_vetoed_tr - 1, len(s2_sub_tr) - 1)])
    rescue_te = (LR[te] >= -7.0) & (LR[te] < -6.0) & (s2_0[te] >= tau_rescue)
    vetoed_mask[te] = (LR[te] >= -6.0) & (~keep_te)
    rescued_mask[te] = rescue_te
