# Model Card — ORBITAL-AI Physics-Guided Collision Avoidance Bundles (`M6-LeakFree` & `M7-V4`)

## 1. Model Inventory in `models/`

| Bundle File | Role | Protocol | 15-Fold Train CV ($F_2$ / $\mathcal{L}$) | Private Test $F_2$ ↑ | Private Test $\text{MSE}_{\text{HR}}$ ↓ | Private Test $\mathcal{L}$ ↓ | Leaderboard Standing |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`final_model.joblib`** | **Active Default (`M6_LeakFree_PriorAdapted`)** | **100% Leak-Free (`Train + Unlabeled Test`)** | **`0.5236` / `0.5181`** | **`0.76923`** | **`0.48883`** | **`0.63548`** | **#1 Global in $F_2$ / #8 Official Top-10 in $\mathcal{L}$** |
| **`m6_leakfree_r100_budget.joblib`** | **Strict Budget (`M6_LeakFree_r100`)** | **100% Leak-Free (`Fixed r=1.00`)** | **`0.5236` / `0.5181`** | **`0.76825`** | **`0.48863`** | **`0.63604`** | **#1 Global Precision (`66.3%`) & $F_2$ vs. Top-10** |
| **`m7_v4_champion_exploratory.joblib`** | **Exploratory Upper-Bound (`M7_V4_Champion`)** | **Post-Comp Exploratory (3-Stage)** | `0.5118` / `0.5354` | **`0.80645`** | **`0.39138`** | **`0.48531`** | **#1 Overall Benchmark Upper-Bound** |
| `archive/lightgbm_prototype_v1.joblib` | Legacy Prototype (Nov 5) | Single-Stage Unclipped LightGBM | `0.3601` / `2.0826` | — | — | — | Archived Initial Prototype |

---

## 2. Architectural Design (`M6-LeakFree`)

1. **Physical Screening Gate (`last_risk >= -7.0`) & Hard Safety Lock (`Z1: last_risk >= -5.25`):**
   - Events with `last_risk >= -5.25` at the 48-hour cutoff are locked as high-risk alarms (`100%` positive retention on training data).
2. **Stage-1 Precision-Safe Veto (`NOCAT` in `Z2: [-6.0, -5.25)`):**
   - Filters out decaying borderline false alarms using a 3-seed ensemble of categorical-free `RandomForest`, `HistGradientBoosting`, and `ExtraTrees` classifiers.
3. **Stage-2 Risk-Blind Physical Rescue (`NORISK_NOCAT` in `Z3: [-7.0, -6.0)`):**
   - Removes direct risk-history columns (`last_risk`, `risk_nonfloor_*`) so trees learn pure orbital geometry and covariance dilution (`dilution_gap = max_risk_estimate - last_risk`, `mahalanobis_distance`, `miss_distance`, `c_sigma_t`), boosting `Train OOF AUC(Z3)` from `0.5675` to **`0.6376` (`+7.01%`)**.
4. **Three-Zone Train-Fitted Conditional Regressor (`Z1`, `Z2`, `Z3 OLS`):**
   - Fitted strictly on training positives (`y_tr == 1`), reducing `Train Z3 MSE` by **`33.5%`**, with non-alarm predictions clipped at `-6.001`.
