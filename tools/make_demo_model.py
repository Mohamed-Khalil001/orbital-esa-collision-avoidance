"""Create a DEMO model and a few sample events from synthetic data, to try the platform right away.

WARNING: the model is trained on fake data. Its predictions mean nothing. Use it only to check that
the platform runs on your machine. The app shows a warning whenever a demo model is active.

Usage (from the project folder):
    python tools/make_demo_model.py
    streamlit run app/app.py

Creates models/demo_model.joblib and sample_data/sample_events.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "tools"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from make_synthetic_data import make_events
from src import features, modeling


def main():
    rng = np.random.default_rng(7)

    train = make_events(1500, rng, first_event_id=0)
    object_types = features.fit_object_types(train)
    X, meta = features.make_event_table(train, object_types)
    X = X.drop(columns=list(X.columns[X.nunique() <= 1]))

    scope = meta["test_like"]
    model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=0)
    model.fit(X.loc[scope], meta.loc[scope, "final_risk"].clip(lower=-12))

    bundle = {
        "model": model, "model_name": "demo (sklearn HistGradientBoosting, synthetic data)",
        "feature_names": list(X.columns), "object_types": object_types, "to_drop": [],
        "alarm_tau": -6.5, "clip_low": -12.0, "train_scope": "test_like", "high_weight": None,
        "demo": True,
    }
    result = modeling.install_bundle(bundle, models_dir=ROOT / "models", activate=False, name="demo_model.joblib")
    print(f"demo model: {result['installed']}")

    sample = make_events(24, rng, first_event_id=900000, high_risk_share=0.3)
    sample_dir = ROOT / "sample_data"
    sample_dir.mkdir(exist_ok=True)
    sample.to_csv(sample_dir / "sample_events.csv", index=False)
    print(f"sample events: {sample_dir / 'sample_events.csv'} ({sample['event_id'].nunique()} events)")


if __name__ == "__main__":
    main()
