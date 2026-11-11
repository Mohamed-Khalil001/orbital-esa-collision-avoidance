"""Tests that load and verify the active and auxiliary model bundles in models/."""
from pathlib import Path
import pytest
from src import config, modeling

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return modeling.load_bundle(ROOT / "models" / "final_model.joblib")


def test_bundle_passes_validation(bundle):
    assert modeling.validate_bundle(bundle) == []


def test_auxiliary_bundles_pass_validation():
    for fname in ("m6_leakfree_r100_budget.joblib", "m7_v4_champion_exploratory.joblib"):
        b = modeling.load_bundle(ROOT / "models" / fname)
        assert modeling.validate_bundle(b) == []


def test_prediction_for_a_demo_event(bundle, demo_events):
    event_id = demo_events[config.ID_COL].iloc[0]
    cdms = demo_events[demo_events[config.ID_COL] == event_id]
    result = modeling.predict_event(bundle, cdms)
    assert result is not None
    assert set(result) >= {"raw", "adjusted", "is_alarm", "probability", "last_risk", "zone"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["coverage"] > 0.5
