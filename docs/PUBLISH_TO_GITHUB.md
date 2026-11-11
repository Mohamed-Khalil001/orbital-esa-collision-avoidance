# Publishing to GitHub: step by step

You need Git installed and a GitHub account. Replace `YOUR_USERNAME` with your GitHub user name.

## 1. Create the empty repository

On <https://github.com/new>: name it `orbital-esa-collision-avoidance`, choose Public or Private, and **do not** tick "Add a README", ".gitignore" or "license" (the project already has them).

## 2. Check locally first

```bash
cd orbital-esa-collision-avoidance
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
streamlit run app/app.py
```

All tests should pass and the app should open at <http://localhost:8501>. If a test fails, fix that before publishing.

## 3. Edit three things

- `LICENSE`: replace "ORBITAL project authors" with your name, or choose another license.
- `README.md`: replace `YOUR_USERNAME` in the clone command.
- Decide whether the Arabic reports in `research/reports_ar/` should be public (they are the full research write-up).

## 4. Push

```bash
git init -b main
git add .
git status          # read the list: no data/*.csv, no .venv
git commit -m "Initial commit: ORBITAL collision-avoidance platform and research archive"
git remote add origin https://github.com/YOUR_USERNAME/orbital-esa-collision-avoidance.git
git push -u origin main
```

If `git status` shows any file of 50 MB or more, stop and check `.gitignore`; GitHub rejects files above 100 MB.

## 5. After the push

- Open the **Actions** tab: the CI workflow runs the tests on Python 3.11 and 3.12. Fix anything red.
- Deploy the app: `docs/DEPLOYMENT.md`.
- Add the live app link and a screenshot of the 3D view to the top of `README.md`.
- In the repository settings, add the topics `collision-avoidance`, `space-debris`, `machine-learning`, `streamlit`.
