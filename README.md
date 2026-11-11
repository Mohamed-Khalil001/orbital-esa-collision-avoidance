# 🛰️ ORBITAL-AI: Physics-Guided Two-Stage Spacecraft Collision Avoidance & 6-DOF CW/RK4 Astrodynamics Workbench

🚀 **Live application:** [Open ORBITAL-AI on Streamlit](https://orbital-esa-collision-avoidance-3ewp7cecpuqb3kh6kmmbse.streamlit.app/)

[![Official Metric: ESA Loss](https://img.shields.io/badge/Official_ESA_Loss_L-0.6355_(~8th_Comparative)-0284c7?style=for-the-badge)](#1-executive-summary--benchmark-summary)
[![15-Fold Grouped OOF CV](https://img.shields.io/badge/15--Fold_Grouped_OOF_CV-L%3D0.5181_%7C_F2%3D0.5236-10b981?style=for-the-badge)](#3-methodological-integrity--why-train-cv-vs-test-differs)
[![Architecture](https://img.shields.io/badge/Engine-Unified_350--Tree_LightGBM_%2B_Risk--Blind_S2-8b5cf6?style=for-the-badge)](#2-unified-end-to-end-system-architecture)
[![Simulator](https://img.shields.io/badge/Simulator-WebGL_Three.js_%2B_6--State_CW%2FRK4_J2_(Zero_Plotly)-f59e0b?style=for-the-badge)](#4-coupled-6-state-cw--rk4-j2-astrodynamics--b-plane-workbench)

**ORBITAL-AI** is an end-to-end research and engineering workbench built on the [**ESA Kelvins Spacecraft Collision Avoidance Challenge**](https://kelvins.esa.int/collision-avoidance-challenge/) dataset (`13,154` training conjunction events / `162,634` CDMs and `2,167` test conjunction events / `24,484` CDMs) and grounded in the official post-competition findings of [**Uriot et al. (2022), Astrodynamics**](https://www.sciopen.com/article/10.1007/s42064-021-0101-5).

> **Operational & Scientific Scope Disclosure:** This repository is a **Research Prototype and Retrospective Benchmark**, not a flight-certified operational collision avoidance system. The 15-fold event-grouped out-of-fold cross-validation (`ESA Loss L = 0.5181, F2 = 0.5236`) serves as our primary unbiased scientific benchmark, while test set numbers (`ESA Loss L = 0.6355, F2 = 0.7684`) are reported with explicit disclosure of early Phase 1–2 error analysis and the synthetic Monte Carlo base-rate shift documented by ESA.

## 1. Executive Summary & Benchmark Summary

In the official ESA Kelvins competition, systems are ranked strictly by **ESA Asymmetric Loss** (lower is better), evaluated at the **48-hour causal decision cutoff (`time_to_tca >= 2.0 days`)**:

$$\mathcal{L}_{\text{ESA}}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\text{MSE}_{\text{HR}}(\mathbf{y}, \hat{\mathbf{y}})}{F_2(\mathbf{y}, \hat{\mathbf{y}})}, \quad \text{where } y_i = \mathbb{I}(\text{final\_risk}_i \ge -6.0)$$

| Evaluation Protocol / System | Cohort Size | Positive Base Rate | Official ESA Loss $\mathcal{L}$ (Lower = Better) | Classification $F_2$ | $\text{MSE}_{\text{HR}}$ | Notes & Governance Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **15-Fold Event-Grouped OOF CV (`M6-LeakFree`)** | 8,892 | 1.05% (93) | **0.51810** | 0.52360 | 0.27131 | **Primary Unbiased Scientific Benchmark** |
| **sesc** ([Official ESA 1st Place](https://kelvins.esa.int/collision-avoidance-challenge/results/)) | 2,167 | 3.18% | **0.55531** | — | — | Official ESA Competition Winner |
| **dietmarw** ([Official ESA 2nd Place](https://kelvins.esa.int/collision-avoidance-challenge/results/)) | 2,167 | 3.18% | **0.57446** | — | — | Official ESA 2nd Place |
| **Magpies** ([Official ESA 3rd Place](https://kelvins.esa.int/collision-avoidance-challenge/results/)) | 2,167 | 3.18% | **0.58489** | — | — | Official ESA 3rd Place |
| **DunderMifflin** ([Official ESA 7th Place](https://kelvins.esa.int/collision-avoidance-challenge/results/)) | 2,167 | 3.18% | **0.62758** | — | — | Official ESA 7th Place |
| **⭐ `M6-LeakFree` (Prior-Adapted 3-Zone)** | 2,167 | 3.18% | **0.63548** | 0.76835 | 0.48827 | **Retrospective Test Benchmark (~8th Comparative in Loss)** |
| **⭐ `M6-LeakFree` (Fixed Budget $r = 1.00$)** | 2,167 | 3.18% | **0.63650** | 0.76923 | 0.48961 | **Retrospective Test Benchmark (Zero Prior Shift Tuning)** |
| **Spacemeister** ([Official ESA 8th Place](https://kelvins.esa.int/collision-avoidance-challenge/results/)) | 2,167 | 3.18% | **0.65162** | — | — | Official ESA 8th Place |
| **B0 Last-Risk Persistence (`last_risk >= -6.0`)** | 2,167 | 3.18% | **0.69396** | 0.73300 | 0.50867 | Official Domain Heuristic Baseline |

## 2. Unified End-to-End System Architecture

Unlike disconnected prototypes where a saved `.joblib` model, a heuristic API script, and a static frontend operate independently, **ORBITAL-AI** executes a single unified pipeline from raw CDM time series to final predictions across the Python package, FastAPI server, and Streamlit console:

1. **Causal Cutoff Filter (`src/features/causal_engineering.py:filter_causal_cdms`)**:
   - Strictly discards any Conjunction Data Message (CDM) with `time_to_tca < 2.0 days`. Payloads lacking pre-cutoff CDMs are rejected by `POST /api/v1/predict-event` with `HTTP 422`.
2. **Unified Event-Level Feature Engineering (`src/features.build_event_features`)**:
   - Aggregates variable-length CDM histories per `event_id` into `306` features covering kinematics, 3D RTN covariance dynamics, OD tracking quality, and approach geometry.
3. **Explicit Stage-2 Physical Allowlist (`STAGE2_PHYSICAL_ALLOWLIST` & `RISK_HISTORY_COLS`)**:
   - Enforces a dual barrier for the Stage-2 Rescue classifier: every Stage-2 feature must belong to the explicit `184`-feature `STAGE2_PHYSICAL_ALLOWLIST` AND must not appear in `RISK_HISTORY_COLS` (which blocks all 39 operational risk-history columns, including `risk_floor_fraction`, `risk_floor_count`, `risk_all_floor`, `risk_nonfloor_std`, `risk_first_nonfloor`, `risk_last_nonfloor`, `days_since_last_nonfloor_risk`, `risk_nonfloor_delta`, `risk_nonfloor_slope`, and `dilution_gap`).
   - Verified by automated unit tests (`tests/test_pipeline.py`): randomizing all 39 operational risk-history columns produces **exact `0.000000` change** in `stage2_booster` output.
4. **Serialized LightGBM v4 Tree Ensembles (`models/final_model.joblib` & `models/final_collision_risk_model.joblib`)**:
   - Both `src/modeling.py` and `src/models/two_stage_m6.py` load `models/final_model.joblib` from disk and evaluate its **350 compiled Stage-1 `NOCAT` LightGBM decision trees** and **350 compiled Stage-2 Risk-Blind `STAGE2_PHYSICAL_ALLOWLIST` LightGBM decision trees**, followed by the train-fitted **Three-Zone Conditional Regressor** (`Z1 HardLock >= -5.25`, `Z2 Veto [-6.0, -5.25)`, `Z3 Rescue [-7.0, -6.0)`).

## 3. Methodological Integrity & Why Train CV vs. Test Differs

A critical scientific question when inspecting the results table is why Classification $F_2$ rises from **`0.5236`** on 15-Fold Out-of-Fold Training CV to **`0.7684`–`0.7692`** on the test cohort, while Official ESA Loss $\mathcal{L}$ increases from **`0.5181`** to **`0.6355`**:

1. **Positive Base-Rate Shift (505 Synthetic Monte Carlo Events)**:
   - As documented in Section 2.3 of [**Uriot et al. (2022)**](https://www.sciopen.com/article/10.1007/s42064-021-0101-5), the ESA organizers augmented real ESOC conjunction logs with `505` synthetic Monte Carlo close approaches and partitioned them disproportionately into the test set.
   - Consequently, the test-like training split has a positive prevalence of **$1.05\%$ (`93 / 8,892`)**, whereas the official test split has a positive prevalence of **$3.18\%$ (`69 / 2,167` strict target / `150 / 2,167` extended cohort)** — a **$3.0\times$ higher positive base rate**.
   - At a fixed false-positive rate, a $3\times$ higher positive base rate mechanically increases Precision and boosts $F_2$ (even the simple `B0` baseline jumps from `F2 = 0.502` in Train CV to `F2 = 0.733` on Test).
   - Simultaneously, late-escalating high-risk test events whose pre-cutoff risk is floored (`-30.0`) incur large squared errors on positive events, raising $\text{MSE}_{\text{HR}}$ from `0.2713` to `0.4883` and therefore increasing $\mathcal{L} = \text{MSE}_{\text{HR}} / F_2$ from `0.5181` to `0.6355`.
2. **Unsupervised EM Prior Adaptation Verification**:
   - `modeling.verify_and_estimate_em_prior_shift` implements the Saerens–Latinne–Decaestecker (2002) EM algorithm alongside a Two-Sample Kolmogorov–Smirnov (`ks_2samp`) diagnostic (`KS stat = 0.0503`) to verify that the unlabeled test score distribution is consistent with a prior mixture shift of the training class-conditional distributions.
3. **Strict Public Repository Data Governance**:
   - No private test ground-truth labels (`true_final_risk`, `true_high_risk`, `test_data_private.csv`) are shipped inside this public repository (`data/`, `reports/`, `research/results/`). Researchers with local access to the external ESA archive can run `python3 tools/evaluate_with_external_ground_truth.py --external-private-labels /path/to/test_data_private.csv`.

## 4. Coupled 6-State CW + RK4 J2 Astrodynamics & B-Plane Workbench

All Plotly charts have been completely removed and replaced with a hardware-accelerated **WebGL (`Three.js`) + HTML5 2D Engineering Canvas Workbench** (`web/index.html` and `src/simulator/orbital_mechanics.py`):
- **Coupled 6-State Clohessy–Wiltshire State-Transition Matrix $\Phi(t, t_0)$ & RK4 $J_2$ Numerical Integration**: Integrates relative position and velocity $\mathbf{x}(t) = [r_R, r_T, r_N, v_R, v_T, v_N]^T$ across $t \in [-600\text{ s}, +600\text{ s}]$ around TCA.
- **Akella & Alfriend (2000) 2D Encounter B-Plane Polar Gaussian Quadrature**: Projects the 3×3 combined RTN covariance onto the encounter plane $(\xi, \zeta) \perp \mathbf{v}_{\text{rel}}$ and numerically integrates $P_c$ over the Hard-Body Radius disk.
- **Explicit Analytical Boundary**: Simulator outputs are strictly separated from the ML risk classifier and labeled with `OPERATIONAL_DISCLAIMER`.

## 5. Repository Structure (Including Companion Book `book/`)

```text
orbital-esa-collision-avoidance/
├── src/
│   ├── config.py                        # Thresholds, cutoffs, and paths
│   ├── features/                        # Unified Causal Feature Engineering package
│   │   ├── __init__.py                  # build_event_features, RISK_HISTORY_COLS, STAGE2_PHYSICAL_ALLOWLIST
│   │   └── causal_engineering.py        # filter_causal_cdms (T >= 2.0d), split_stage1_and_stage2_columns
│   ├── modeling.py                      # Compiled LightGBM v4 tree evaluator & bundle manager
│   ├── models/two_stage_m6.py           # Unified Two-Stage M6 + LightGBM inference interface
│   ├── simulation.py                    # Coupled 6-State CW STM + RK4 J2 + 2D B-Plane quadrature (Zero Plotly)
│   ├── simulator/orbital_mechanics.py   # Astrodynamics simulator package interface
│   └── api/server.py                    # FastAPI production server with X-API-Key, rate limiting & /predict-event
├── models/
│   ├── final_model.joblib               # Active Unified 350-Tree S1 + 350-Tree Risk-Blind S2 LightGBM bundle
│   ├── final_collision_risk_model.joblib# Mirror unified bundle for API/CLI compatibility
│   ├── m6_leakfree_r100_budget.joblib   # Fixed-budget (r = 1.00) unified bundle
│   └── m7_v4_champion_exploratory.joblib# Three-stage exploratory upper-bound bundle
├── web/index.html                       # Standalone WebGL Three.js + RK4 + 2D B-Plane Workbench (Zero Plotly)
├── app/                                 # Streamlit Operations Console (Zero Plotly)
├── book/                                # Dedicated directory for the 6-Chapter Companion Book Manuscript
├── notebooks/                           # 4 Step-by-Step Educational Notebooks (100% English)
├── research/                            # 6-Phase Research Archive & Master Colab Notebooks (100% English)
├── tools/evaluate_with_external_ground_truth.py # Offline evaluator for external private test labels
└── tests/                               # End-to-End Integration, Risk-Blind, Data Governance & Zero-Plotly Tests
```
