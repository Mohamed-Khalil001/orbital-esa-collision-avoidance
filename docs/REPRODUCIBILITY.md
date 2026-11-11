# Reproducing ORBITAL-AI research results

This guide separates **running the shipped application**, **retraining the packaged baseline**, and **reproducing the full research experiments**. They are not the same workflow.

## 1. Data provenance

The project is based on the ESA Advanced Concepts Team / ESA Space Debris Office Collision Avoidance Challenge dataset:

- Zenodo record: https://zenodo.org/records/4463683
- Challenge page: https://kelvins.esa.int/collision-avoidance-challenge/
- Publication context: Uriot et al. (2022), *Astrodynamics*: https://www.sciopen.com/article/10.1007/s42064-021-0101-5

Download and verify the archive using the repository helper:

```bash
python tools/download_data.py
```

Expected local files in `data/` include `train_data.csv`, `test_data.csv`, and, only for the separately authorized retrospective evaluation, `test_data_private.csv`. The full files are intentionally not committed. Read [data/README.md](../data/README.md) and check dataset redistribution terms before sharing any copy.

The committed `data/sample/demo_events.csv` is a small demonstration sample, not a replacement for the full training cohort.

## 2. Environment and smoke tests

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
pytest
```

Run the shipped Streamlit app:

```bash
streamlit run app/app.py
```

## 3. Retraining the packaged baseline

The supported retraining entry point is:

```bash
python tools/retrain_model.py --config models/model_config.json
```

Optionally preserve the configured alarm threshold:

```bash
python tools/retrain_model.py --config models/model_config.json --keep-tau
```

This script loads the full training CSV, builds event-level features and fits the configured model. Without `--keep-tau`, it performs event-level cross-validation to select the alarm threshold before fitting the final bundle. It requires the full training data and an existing compatible configuration file. It is not a one-command reproduction of the six-phase M6 research protocol.

## 4. Full research experiment notebooks

The full research sequence is archived in:

- [Research overview](README.md)
- [Notebook 28 — from-scratch feature and nested OOF experiments](notebooks/28_ESA_FromScratch_Master_Colab_T4_GPU_FIXED_v3.ipynb)
- [Notebook 29 — exploratory M7-V4 upper-bound study](notebooks/29_ESA_FirstPlace_Champion_M7_V4_Colab_T4_GPU.ipynb)
- [Notebook 30 — leak-free train/test master protocol](notebooks/30_ESA_LeakFree_TrainAndTest_Master_Colab_T4_GPU.ipynb)
- [Historical phase scripts and provenance note](archive/phase_scripts/README.md)

These notebooks were authored for Google Colab / GPU workflows. They may need path and runtime adjustments to run locally. The historical phase scripts are provenance artifacts and are explicitly **not standalone runnable scripts** because they depend on intermediate files from the original research workspace.

For the primary reported M6 benchmark, use Notebook 30 and preserve its separation between training data and unlabeled test inputs. Do not use private test labels to select features, thresholds, or model parameters.

## 5. Saved results and audit artifacts

Key outputs are stored in `research/results/`:

- `colab_run_outputs/02_CV_15Fold_NestedOOF_Summary.csv`
- `colab_run_outputs/03a_CV_Protocol_Audit.csv`
- `colab_run_outputs/04_Frozen_PreTest_Config.json`
- `colab_run_outputs/06_Final_Test_and_ESA_Leaderboard_Comparison.csv`
- `colab_run_outputs/08_Final_Evaluation_Report.md`
- `phases/` contains phase workbooks and frozen configurations.

Reported test metrics are retrospective benchmark results, not evidence of flight readiness or a new official ESA competition placement. Keep OOF CV metrics, public competition leaderboard values, and private-label retrospective metrics clearly distinguished.
