# ORBITAL-AI Project Architecture & Scientific Engineering Plan

## 1. Mission & Objectives
**ORBITAL-AI** is a complete scientific research and operational decision-support platform for spacecraft collision avoidance in Low Earth Orbit (LEO), built on the [ESA Kelvins Spacecraft Collision Avoidance Challenge](https://kelvins.esa.int/collision-avoidance-challenge/) dataset (`162,634` training CDMs, `24,484` test CDMs).

The project bridges three pillars:
1. **Strict Causal Machine Learning (`src/features.py`, `src/modeling.py`, `notebooks/01`–`04`, `research/notebooks/28`–`30`)**: Enforcing the 48-hour operational cutoff (`time_to_tca >= 2.0 days`) and decoupling Stage-1 False-Alarm Veto (`NOCAT`) from Stage-2 Risk-Blind Physical Rescue (`NORISK_NOCAT`) and Three-Zone Conditional Regression.
2. **Physics-Accurate 3D Orbital & B-Plane Simulation (`src/simulation.py`, `app/`)**: Reconstructing 3D Earth LEO orbits, Clohessy–Wiltshire relative motion in the `RTN` frame, $3\times 3$ principal-axis covariance ellipsoids, 2D encounter B-plane collision probability contours, and $\Delta v$ Collision Avoidance Manoeuvre (`CAM`) trade-offs.
3. **Comprehensive Academic Textbook (`book/`)**: A 6-chapter reference text covering orbital debris mechanics, space surveillance (`SP` vs. `SGP4`), the 103-column CCSDS 508.0-B-1 CDM lineage, ESOC operational constraints, and physics-guided machine learning.

## 2. Summary of Milestones Achieved
- **Milestone 1 (Phases 1–2):** Baseline benchmarking (`B0`–`B3`) and event-level False Negative forensic catalog (`8,892` causal training events, `93` positives).
- **Milestone 2 (Phase 3):** 229/233-column physics-guided feature engineering (`F1`–`F7`) with automated future-tampering invariance tests.
- **Milestone 3 (Phases 4–5):** Two-Stage Veto + Rescue ablation (`M0`–`M6`) and physical regime boundary validation (`Z1`–`Z6`, `HardLock = -5.25`, `GATE = -7.0`).
- **Milestone 4 (Phase 6 — Initial Official Leaderboard Evaluation):** Achieved **8th Place Overall** on the official ESA Kelvins Leaderboard (`ESA Loss L = 0.6377 / 0.6433`, $F_2 = 0.7682$).
- **Milestone 5 (Post-Competition Exploratory Upper-Bound — `M7-V4`):** Designed the Three-Stage Covariance-Contraction Deep Rescue architecture (`M7_V4_ThreeStage_PeakF2_Champion`), achieving the **#1 Overall Benchmark Score (`ESA Loss L = 0.4853, F2 = 0.8065`)** and `L = 0.4996, F2 = 0.8005` in budget-matched mode.
- **Milestone 6 (Strict 100% Leak-Free Scientific Refactoring — `M6-LeakFree`):** Physically isolated `test_data_private.csv` until the final evaluation step, introduced the Stage-2 Risk-Blind Physical Classifier (`NORISK_NOCAT`, `+7.01%` OOF AUC in `Z3`), the Three-Zone Train-Fitted Conditional Regressor (`-33.5%` Z3 MSE), and Unsupervised EM Prior-Shift Adaptation ($ho_{\text{prior}} = 2.1546\times$), achieving **`15-Fold Train CV F2 = 0.5236, L = 0.5181`** and **`One-Shot Private Test F2 = 0.7692 (#1 Global across all competition teams), L = 0.6355 (#8 Official Top-10)`**.
