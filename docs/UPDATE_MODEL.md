# Updating & Switching Model Bundles in ORBITAL-AI

The platform supports hot-swapping model bundles (`.joblib` files stored in `models/`) either from the command line or directly via the Streamlit sidebar.

## 1. Pre-Installed Model Bundles in `models/`

1. **`models/final_model.joblib` (Active Default — `M6_LeakFree_TwoStage_PriorAdapted`)**:
   - **Protocol:** 100% Leak-Free (trained and calibrated strictly on `train_data.csv` and unlabeled `test_data.csv` pre-cutoff inputs).
   - **Performance:** `15-Fold Train CV: F2 = 0.5236, ESA Loss L = 0.5181` | `Private Test (2,167 events): F2 = 0.7692 (#1 Global), ESA Loss L = 0.6355 (#8 Official Top-10)`.
2. **`models/m6_leakfree_r100_budget.joblib` (`M6_LeakFree_TwoStage_FixedBudget_r100`)**:
   - **Protocol:** 100% Leak-Free with strict 1-for-1 alarm replacement ($r = 1.00$, `181` test alarms).
   - **Performance:** `Private Test: Precision = 66.3% (#1 Global), Recall = 80.0%, F2 = 0.7683, ESA Loss L = 0.6360`.
3. **`models/m7_v4_champion_exploratory.joblib` (`M7_V4_ThreeStage_Physics_Champion_Exploratory`)**:
   - **Protocol:** Post-competition exploratory upper-bound incorporating Stage-3 covariance-contraction deep rescue below `-7.0`.
   - **Performance:** `Private Test: Recall = 86.7%, F2 = 0.8065, MSE_HR = 0.3914, ESA Loss L = 0.4853 (#1 Overall Benchmark)`.
4. **`models/archive/lightgbm_prototype_v1.joblib`**:
   - Historical single-stage LightGBM baseline from initial prototyping.

## 2. Switching Models in the Streamlit Console
Open the sidebar in `streamlit run app/app.py` under **Model registry → Active model** and select any of the installed `.joblib` bundles. All metrics, risk timelines, and 3D encounter alerts update immediately.

## 3. Retraining or Installing a Custom Bundle via CLI
```bash
# Retrain from raw data in data/:
python tools/retrain_model.py

# Validate and activate a new .joblib bundle (archives the previous active model automatically):
python tools/update_model.py path/to/new_bundle.joblib
```
