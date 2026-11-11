# Contributing to ORBITAL-AI

Thanks for helping improve this research prototype. Please keep scientific claims reproducible, preserve the causal data boundary, and avoid committing private or large source datasets.

## 1. Development setup

Requirements: Python 3.11 or 3.12, Git, and (for LightGBM on Linux) the OpenMP runtime.

```bash
git clone https://github.com/Mohamed-Khalil001/orbital-esa-collision-avoidance.git
cd orbital-esa-collision-avoidance
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The repository includes a small demo sample and pretrained model bundles, so the UI and most tests can run without downloading the full ESA dataset.

## 2. Run the application

```bash
streamlit run app/app.py
```

The FastAPI service is separate:

```bash
# Set a long, random API key before starting the server.
# PowerShell:
$env:ORBITAL_API_KEY = "replace-with-a-long-random-secret"
# macOS / Linux:
# export ORBITAL_API_KEY="replace-with-a-long-random-secret"
uvicorn src.api.server:app --host 127.0.0.1 --port 8000
```

Protected write endpoints under `/api/v1/` require `X-API-Key` by default. Do not disable this requirement in a public deployment. For isolated local tests only, `ORBITAL_API_KEY_REQUIRED=false` opts out when no key is configured.

## 3. Run checks

```bash
pytest
```

The GitHub Actions workflow runs the test suite on Python 3.11 and 3.12. Add regression tests for behavior changes, especially feature engineering, model validation, API security, and simulation physics.

## 4. Data and model handling

- See [data/README.md](data/README.md) for the official ESA dataset source and download instructions.
- Do not commit `train_data.csv`, `test_data.csv`, `test_data_private.csv`, secrets, or local environment files.
- Never load a `.joblib` / pickle model from an untrusted source; deserialization can execute arbitrary code.
- Keep model artifacts and their registry manifest consistent. Use `python tools/retrain_model.py` only after obtaining the full training data and a valid `models/model_config.json`.
- Preserve the 48-hour causal cutoff and Stage-2 risk-blind feature allowlist. Any change to either requires explicit tests and a scientific explanation.

## 5. Pull request expectations

- Explain the problem and the reason for the change.
- Include test output and, for model changes, the exact data split, seed, metric definition, and artifact provenance.
- Do not claim leaderboard rank or operational readiness from an internal benchmark alone.
- Keep documentation links relative where possible and verify that referenced files exist.
