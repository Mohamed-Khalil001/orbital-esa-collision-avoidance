# Deploying the platform

The app is a standard Streamlit application (`app/app.py`, with a second page in `app/pages/`). It starts on the bundled demo sample, so no data download is needed for a first deployment.

## Option 1: Streamlit Community Cloud (free, easiest)

1. Push the repository to GitHub (see `docs/PUBLISH_TO_GITHUB.md`). The repository must be reachable by Streamlit (public, or private with access granted).
2. Go to <https://share.streamlit.io>, sign in with GitHub, and choose **Create app**.
3. Select the repository and branch `main`. Set **Main file path** to `app/app.py`.
4. In **Advanced settings** choose Python 3.11 or 3.12. Streamlit installs `requirements.txt` automatically.
5. Deploy. The first build takes a few minutes while LightGBM installs.

The free tier has limited memory. The demo sample is small, so it fits; loading the full 230 MB training file inside the hosted app is not advisable.

## Option 2: Docker

```bash
docker build -t orbital .
docker run --rm -p 8501:8501 orbital
```

Open <http://localhost:8501>. To use the full data, mount it:

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/data:/app/data" orbital
```

## Option 3: any machine with Python

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

On Windows you can double-click `INSTALL_AND_START.bat` instead.

## Things to know before sharing a public link

- The sidebar lets a visitor upload a `.joblib` model bundle. Joblib files are pickles and can run code when loaded. On a public deployment, anyone who can open the page could use that upload. **Do not expose the app publicly with the upload enabled on a machine you care about**; for a public demo, keep it on Streamlit Community Cloud (isolated container) or remove the "Install / update model" block from `app/app.py` first.
- The app is a research prototype. The home page and the research page state that it is not an operational conjunction assessment system; keep those statements in any fork.
