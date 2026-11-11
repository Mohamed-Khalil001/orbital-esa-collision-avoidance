"""Tests for the competition metric and out-of-fold validation consistency."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src import config, metric

ROOT = Path(__file__).resolve().parents[1]


def test_perfect_prediction_gives_zero_score():
    y = np.array([-5.0, -4.0, -8.0, -12.0])
    result = metric.collision_score(y, y)
    assert result["mse_high"] == 0.0
    assert result["f2"] == 1.0
    assert result["score"] == 0.0


def test_missing_every_high_risk_event_gives_infinite_score():
    y_true = np.array([-5.0, -4.0, -8.0])
    y_pred = np.array([-7.0, -7.0, -7.0])
    assert metric.collision_score(y_true, y_pred)["score"] == np.inf


def test_nan_predictions_are_rejected():
    with pytest.raises(ValueError):
        metric.collision_score([-5.0, -8.0], [np.nan, -8.0])


def test_score_is_undefined_without_high_risk_events():
    with pytest.raises(ValueError):
        metric.collision_score([-8.0, -9.0], [-8.0, -9.0])


def test_clipping_reduces_the_penalty_for_very_low_predictions():
    y_true = np.array([-5.0, -20.0])
    y_pred = np.array([-30.0, -20.0])
    unclipped = metric.collision_score(y_true, y_pred)
    clipped = metric.esa_loss(y_true, y_pred)
    assert clipped["mse_high"] < unclipped["mse_high"]
    assert clipped["mse_high"] == pytest.approx((-5.0 - config.OFFICIAL_CLIP_LOW) ** 2)


def test_evaluate_reports_counts():
    y_true = np.array([-5.0, -4.0, -8.0, -9.0])
    y_pred = np.array([-5.0, -7.0, -5.5, -9.0])
    result = metric.evaluate(y_true, y_pred)
    assert result["n_events"] == 4 and result["n_high"] == 2
    assert result["recall"] == 0.5 and result["precision"] == 0.5


def test_oof_validation_score_is_finite_and_consistent():
    """Verifies metric.esa_loss on the training Out-of-Fold validation table (no test ground-truth leak)."""
    oof = pd.read_csv(ROOT / "reports" / "oof_validation.csv")
    res = metric.esa_loss(oof["actual_final_risk"], oof["oof_predicted_risk"])
    assert np.isfinite(res["score"]) and res["score"] > 0.0
    assert 0.0 < res["f2"] <= 1.0
