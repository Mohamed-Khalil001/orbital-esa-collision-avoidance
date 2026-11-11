#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESA Kelvins Spacecraft Collision Avoidance Challenge
Script 12 — Phase 4B Forensic Reproducibility, Zero-Leakage & Operational Cost Standalone Runner
Strictly uses pre-cutoff files (01_train_event_cohort_audit.csv, 04_cutoff_event_summary.csv) and same-seed OOFs.
Enforces strict zero-training-TP-loss calibration with out-of-sample safety buffer (buf = 0.10) and HardLock = -5.25.
"""
import os, pickle, hashlib
import numpy as np
import pandas as pd

WORKDIR = "/working_dir/c_9801697d79fff702"
DL = os.path.join(WORKDIR, "downloads")
OUT = "/tmp/final_process_out"
os.makedirs(OUT, exist_ok=True)

FORBIDDEN_FILES = {"09_temporal_event_summary.csv", "test_data_private.csv", "X_test_fe_v2.csv"}

c01 = pd.read_csv(os.path.join(DL, "01_train_event_cohort_audit.csv"))
c04 = pd.read_csv(os.path.join(DL, "04_cutoff_event_summary.csv"))
el = c01[c01["cohort"] == "ELIGIBLE_COMPLETE"].sort_values("event_id").reset_index(drop=True)
c04_el = c04[c04["event_id"].isin(el["event_id"])].sort_values("event_id").reset_index(drop=True)

df = el[["event_id", "total_cdms", "n_allowed", "n_late", "first_time_to_tca", "final_time_to_tca", "last_allowed_time", "last_allowed_risk", "final_risk"]].copy()
df["earliest_tca"] = c04_el["earliest_tca"].values
df["y"] = (df["final_risk"] >= -6.0).astype(int)
df["LR"] = df["last_allowed_risk"].astype(float)
df["FR"] = df["final_risk"].astype(float)

rf_oof = pickle.load(open(os.path.join(DL, "v1_oof_RF_nocat_0-1-2.pkl"), "rb"))
hgb_oof = pickle.load(open(os.path.join(DL, "v1_oof_HGB_0-1-2.pkl"), "rb"))
brf_oof = pickle.load(open(os.path.join(DL, "v1_oof_BalancedRF_0-1-2.pkl"), "rb"))
oof_nocat_raw = pd.read_csv(os.path.join(DL, "oof_without_categorical.csv"))

logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1.0 - np.clip(p, 1e-5, 1 - 1e-5)))
y = df["y"].values
LR = df["LR"].values
all_idx = np.arange(len(df))

exact_folds = []
for f_id in range(1, 16):
    s = (f_id - 1) // 5
    k = (f_id - 1) % 5
    te = np.sort(oof_nocat_raw.loc[oof_nocat_raw["fold"] == f_id, "row_index"].values)
    tr = np.setdiff1d(all_idx, te, assume_unique=True)
    exact_folds.append((s, k, tr, te))

def calc_metrics(y_te, pred_high):
    tp = int(((y_te == 1) & pred_high).sum())
    fp = int(((y_te == 0) & pred_high).sum())
    fn = int(((y_te == 1) & (~pred_high)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2 = (5 * p * r / (4 * p + r)) if (p + r) > 0 else 0.0
    return dict(Flagged=tp+fp, TP=tp, FP=fp, FN=fn, P=p, R=r, F2=f2,
                Cost_R4=fp + 4*fn, Cost_R10=fp + 10*fn, Cost_R20=fp + 20*fn)

def eval_two_stage(tr, te, stage1_score, stage2_score, hard_lock=-5.25, rescue_mult=1.00, buf=0.10):
    lr_tr, lr_te = LR[tr], LR[te]
    y_tr, y_te = y[tr], y[te]
    b_tr = (lr_tr >= -6.0) & (lr_tr < hard_lock)
    tau_veto = float(stage1_score[tr][b_tr & (y_tr == 1)].min()) - buf
    keep_tr = (lr_tr >= -6.0) & ((stage1_score[tr] >= tau_veto) | (lr_tr >= hard_lock))
    keep_te = (lr_te >= -6.0) & ((stage1_score[te] >= tau_veto) | (lr_te >= hard_lock))
    n_vetoed_tr = int((lr_tr >= -6.0).sum() - keep_tr.sum())
    n_rescue_tr = max(1, int(round(n_vetoed_tr * rescue_mult)))

    sub_tr = (lr_tr >= -7.0) & (lr_tr < -6.0)
    s2_sub_tr = np.sort(stage2_score[tr][sub_tr])[::-1]
    tau_rescue = float(s2_sub_tr[min(n_rescue_tr - 1, len(s2_sub_tr) - 1)])

    rescue_te = (lr_te >= -7.0) & (lr_te < -6.0) & (stage2_score[te] >= tau_rescue)
    pred_te = keep_te | rescue_te
    m = calc_metrics(y_te, pred_te)
    m["tau_veto"] = round(tau_veto, 6)
    m["tau_rescue"] = round(tau_rescue, 6)
    m["Vetoed_Te"] = int(((lr_te >= -6.0) & (~keep_te)).sum())
    m["Vetoed_TP_Te"] = int(((lr_te >= -6.0) & (~keep_te) & (y_te == 1)).sum())
    m["Rescued_Te"] = int(rescue_te.sum())
    m["Rescued_TP_Te"] = int((rescue_te & (y_te == 1)).sum())
    return m, keep_te, rescue_te

fold_rows = []
for s, k, tr, te in exact_folds:
    rf_s = np.clip(rf_oof[s], 0, 1)
    hgb_s = np.clip(hgb_oof[s], 0, 1)
    brf_s = np.clip(brf_oof[s], 0, 1)
    f4_z = (LR + 6.5)
    s1_m6 = 0.45 * logit(rf_s) + 0.35 * logit(hgb_s) + 0.20 * f4_z
    s2_m6 = 0.40 * logit(rf_s) + 0.40 * logit(brf_s) + 0.20 * logit(hgb_s)

    m0 = calc_metrics(y[te], LR[te] >= -6.0)
    m0.update(Model="M0_Baseline_LastRisk", Seed=s, Fold=k, HardLock=-6.0, RescueMult=0.0, tau_veto=0.0, tau_rescue=0.0, Vetoed_Te=0, Vetoed_TP_Te=0, Rescued_Te=0, Rescued_TP_Te=0)
    fold_rows.append(m0)

    n_base_tr = int((LR[tr] >= -6.0).sum())
    for mname, sc in [
        ("M1_F4_Simple_Linear", LR + 0.15 * np.log1p(df["n_allowed"].values)),
        ("M3_SingleStage_RF_nocat", logit(rf_s) + 0.35 * LR),
        ("M4_SingleStage_HGB_nocat", logit(hgb_s) + 0.35 * LR),
    ]:
        in_gate_tr = LR[tr] >= -7.0
        s_tr_sorted = np.sort(sc[tr][in_gate_tr])[::-1]
        thr = float(s_tr_sorted[min(n_base_tr - 1, len(s_tr_sorted) - 1)])
        pred_te = (LR[te] >= -7.0) & (sc[te] >= thr)
        mm = calc_metrics(y[te], pred_te)
        mm.update(Model=mname, Seed=s, Fold=k, HardLock=-7.0, RescueMult=1.0, tau_veto=0.0, tau_rescue=round(thr, 6), Vetoed_Te=0, Vetoed_TP_Te=0, Rescued_Te=0, Rescued_TP_Te=0)
        fold_rows.append(mm)

    for mname, s1, s2, hl, rmult in [
        ("M5c_TwoStage_HGB_nocat_HL520_r100", logit(hgb_s) + 0.25*LR, logit(hgb_s) + 0.20*LR, -5.20, 1.00),
        ("M5d_TwoStage_RF_nocat_HL520_r100", logit(rf_s) + 0.25*LR, logit(rf_s) + 0.20*LR, -5.20, 1.00),
        ("M6_True_Ensemble_HL520_r100", s1_m6, s2_m6, -5.20, 1.00),
        ("M5c_TwoStage_HGB_nocat_HL525_r100", logit(hgb_s) + 0.25*LR, logit(hgb_s) + 0.20*LR, -5.25, 1.00),
        ("M5d_TwoStage_RF_nocat_HL525_r100", logit(rf_s) + 0.25*LR, logit(rf_s) + 0.20*LR, -5.25, 1.00),
        ("M5d_TwoStage_RF_nocat_HL525_r115", logit(rf_s) + 0.25*LR, logit(rf_s) + 0.20*LR, -5.25, 1.15),
        ("M5d_TwoStage_RF_nocat_HL525_r135", logit(rf_s) + 0.25*LR, logit(rf_s) + 0.20*LR, -5.25, 1.35),
        ("M6_True_Ensemble_HL525_r100", s1_m6, s2_m6, -5.25, 1.00),
    ]:
        mm, _, _ = eval_two_stage(tr, te, s1, s2, hard_lock=hl, rescue_mult=rmult, buf=0.10)
        mm.update(Model=mname, Seed=s, Fold=k, HardLock=hl, RescueMult=rmult)
        fold_rows.append(mm)

fold_df = pd.DataFrame(fold_rows)
summ_df = fold_df.groupby(["Model", "HardLock", "RescueMult"], as_index=False)[
    ["Flagged", "TP", "FP", "FN", "P", "R", "F2", "Cost_R4", "Cost_R10", "Cost_R20", "Vetoed_Te", "Vetoed_TP_Te", "Rescued_Te", "Rescued_TP_Te"]
].mean().round(4)
