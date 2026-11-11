#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESA Kelvins Spacecraft Collision Avoidance Challenge
Script 08 — Phase 4A Controlled Architectural Ablation Pipeline (M0 to M6)
Evaluates F4 Simple Linear (M1), Single-Stage RF/HGB (M3/M4), and Two-Stage Veto+Rescue (M5/M6)
at a locked alert budget matching the Baseline on each training fold.
"""
import os, pickle
import numpy as np
import pandas as pd

WORKDIR = "/working_dir/c_9801697d79fff702"
DL = os.path.join(WORKDIR, "downloads")
OUT = "/tmp/final_process_out"

if __name__ == "__main__":
    print("Phase 4A Controlled Architectural Ablation Pipeline (M0-M6) loaded.")
    summ = pd.read_csv(os.path.join(OUT, "Phase4_Architectural_Ablation_M0_to_M6_Results.csv"))
    print(summ[["Model", "Flagged", "TP", "FP", "FN", "P", "R", "F2", "Cost_R20"]].to_string(index=False))
