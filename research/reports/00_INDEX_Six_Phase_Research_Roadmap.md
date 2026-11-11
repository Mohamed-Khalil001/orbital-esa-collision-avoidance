# Six-Phase Research Roadmap & Milestone Index — ESA Spacecraft Collision Avoidance

This directory documents the complete six-phase scientific progression of the **ORBITAL-AI** project on the official [ESA Kelvins Spacecraft Collision Avoidance Challenge](https://kelvins.esa.int/collision-avoidance-challenge/) dataset (`162,634` training CDMs across `13,154` events; `24,484` test CDMs across `2,167` events).

## Phase Summary & Key Artifacts

1. **Phases 1 & 2 — Reference Baselines (`B0`–`B3`) & False Negative Forensic Audit (`01_Phase1_2_Baseline_and_FN_Audit_Report.md`)**:
   - Established the official Last-Risk Persistence (`B0: last_risk >= -6.0`) baseline (`15-Fold Train CV: F2 = 0.4912, ESA Loss L = 0.5597`; `Private Test: F2 = 0.7391, L = 0.6940`).
   - Cataloged all `23` training False Negatives and `35` test False Negatives (`results/phases/02_Phase2_False_Negative_Event_Level_Catalog.xlsx`), proving that `52.2%` (`12/23`) of missed training positives sit in the immediate sub-threshold rescue zone `Z3: [-7.0, -6.0)` due to **uncertainty dilution** (`max_risk_estimate >> last_risk`).
2. **Phase 3 — Physics-Guided Feature Engineering (`F1` to `F7`) (`02_Phase3_Physics_Feature_Engineering_F1_to_F7.md`)**:
   - Engineered `229`/`233` causal pre-cutoff features (`time_to_tca >= 2.0` days) across 7 physical families (`F1` Censoring-Floor History, `F2` 3D RTN Geometry, `F3` Cutoff-Window Dynamics, `F4` Dilution Gap, `F5` Orbit Determination Quality, `F6` 3x3 & 6x6 Covariance Eigenvalues/Slopes, and `F7` Categorical-Free `NOCAT` Matrix).
3. **Phase 4 (4A & 4B) — Controlled Architectural Ablation (`M0` to `M6`) & Operational Cost Audit (`03_Phase4_TwoStage_Architectural_Ablation.md`)**:
   - Proved that single-stage classifiers degrade Precision when pushed for higher Recall, whereas a **Two-Stage Veto + Rescue Architecture (`M5d` & `M6`)** simultaneously improves Precision, Recall, $F_2$, and asymmetric maneuver cost (`Cost_R20 = 20*FN + 1*FP`).
4. **Phase 5 — Physical Regime Validation (`Z1` to `Z6`) & Safety Lock Freeze (`04_Phase5_Physics_Regime_Validation.md`)**:
   - Partitioned the risk spectrum into 6 physical zones (`Z1: >= -5.25`, `Z2: [-6.0, -5.25)`, `Z3: [-7.0, -6.0)`, `Z4: [-10.0, -7.0)`, `Z5: (-30.0, -10.0)`, `Z6: == -30.0`), establishing the `HardLock = -5.25` safety lock (`100%` positive retention in `Z1`) and the `GATE = -7.0` rescue horizon.
5. **Phase 6 — One-Shot Private Test Evaluation & Post-Competition Upper-Bound Analysis (`05_Phase6_Final_Test_and_Leaderboard_Report.md`)**:
   - Evaluated our frozen two-stage `M6` pipeline (`#8` on the official ESA leaderboard in Loss `L = 0.6355` / `0.6377` and **#1 Global in Classification $F_2 = 0.7692$**), our exploratory three-stage `M7-V4` upper-bound (`L = 0.4853, F2 = 0.8065`, `#1` overall benchmark), and the 505 Monte Carlo split findings of [Uriot et al. (2022)](https://doi.org/10.1007/s42064-021-0101-5).
