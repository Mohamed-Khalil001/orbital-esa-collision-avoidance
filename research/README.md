# Research Archive — Six-Phase Scientific Progression & Master Notebooks

This directory contains the complete six-phase research archive, master Google Colab / Jupyter notebooks, cross-validation tables, and phase workbooks behind **ORBITAL-AI**.

## Master Notebooks (`research/notebooks/`)

| Notebook | Protocol | Key Achievement |
| :--- | :--- | :--- |
| **`30_ESA_LeakFree_TrainAndTest_Master_Colab_T4_GPU.ipynb`** | **100% Leak-Free (`Train + Unlabeled Test Only`)** | Physically segregates `test_data_private.csv` until Step 5B; introduces Stage-2 `NORISK_NOCAT` (`Z3 AUC = 0.6376`), 3-Zone Regressor, and EM Prior-Shift Adaptation $\\rightarrow$ **`15-Fold CV F2 = 0.5236, L = 0.5181`** & **`Private Test F2 = 0.7692 (#1 Global), L = 0.6355 (#8 Official)`** |
| **`28_ESA_FromScratch_Master_Colab_T4_GPU_FIXED_v3.ipynb`** | **Two-Stage Nested OOF CV (`M0` to `M6_v3`)** | Rebuilds 229 physics features (`F1`–`F7`), runs 15-fold nested OOF CV, and evaluates `M5d` (`L = 0.6558`) and `M6_v3` (`L = 0.6433, F2 = 0.7668`) |
| **`29_ESA_FirstPlace_Champion_M7_V4_Colab_T4_GPU.ipynb`** | **Post-Competition Exploratory Upper-Bound (`M7-V4`)** | Adds Stage-3 Covariance-Contraction Deep Rescue (`< -7.0`) and 3-Regime Regression $\\rightarrow$ **`Private Test L = 0.4853 (#1 Benchmark), F2 = 0.8065`** |

## Directory Structure

```text
research/
├── README.md
├── notebooks/
│   ├── 28_ESA_FromScratch_Master_Colab_T4_GPU_FIXED_v3.ipynb
│   ├── 29_ESA_FirstPlace_Champion_M7_V4_Colab_T4_GPU.ipynb
│   ├── 30_ESA_LeakFree_TrainAndTest_Master_Colab_T4_GPU.ipynb
│   └── executed/                                                # Executed notebooks with saved outputs
├── reports/
│   └── 00_INDEX_Six_Phase_Research_Roadmap.md                   # English summary of Phases 1 through 6
├── results/
│   ├── colab_run_outputs/                                       # CSV/JSON/PNG outputs (CV summaries, 2,167 test predictions, leaderboard table)
│   └── phases/                                                  # Phase 1–6 Excel workbooks & frozen JSON configs
└── archive/phase_scripts/                                       # Historical standalone runners for Phases 3, 4, 5, and 6
```
