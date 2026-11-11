"""Tests for event-level feature engineering, including a leakage check."""
import numpy as np
import pandas as pd
import pytest

from src import config, features


def test_input_cdms_keeps_only_allowed_rows():
    df = pd.DataFrame({
        config.ID_COL: [1, 1, 1, 2],
        config.TIME_COL: [3.0, 2.0, 0.5, 1.9],
        config.RISK_COL: [-7.0, -6.5, -4.0, -5.0],
    })
    out = features.input_cdms(df)
    assert out[config.TIME_COL].min() >= config.MIN_TIME_TO_TCA_FOR_INPUT
    assert set(out[config.ID_COL]) == {1}
    assert list(out[config.TIME_COL]) == [3.0, 2.0]  # oldest CDM first


def test_one_feature_row_per_event(demo_events):
    X = features.build_event_features(demo_events)
    assert len(X) == demo_events[config.ID_COL].nunique()
    assert X.index.is_unique
    assert "last_risk" in X.columns and "n_cdms" in X.columns
    assert not np.isinf(X.to_numpy(dtype=float)).any()


def test_last_risk_matches_last_allowed_cdm(demo_events):
    X = features.build_event_features(demo_events)
    allowed = features.input_cdms(demo_events)
    expected = allowed.groupby(config.ID_COL).tail(1).set_index(config.ID_COL)[config.RISK_COL]
    pd.testing.assert_series_equal(X["last_risk"].sort_index(), expected.sort_index(), check_names=False)


def test_features_ignore_cdms_after_the_decision_point(demo_events):
    """Adding CDMs closer than 2 days to TCA must not change any feature (no leakage)."""
    late = demo_events.sort_values(config.TIME_COL).groupby(config.ID_COL).head(1).copy()
    late[config.TIME_COL] = 0.5
    late[config.RISK_COL] = -1.0
    with_late = pd.concat([demo_events, late], ignore_index=True)

    base = features.build_event_features(demo_events)
    augmented = features.build_event_features(with_late)
    pd.testing.assert_frame_equal(base, augmented)


def test_missing_required_columns_raise():
    with pytest.raises(ValueError):
        features.build_event_features(pd.DataFrame({config.ID_COL: [1]}))
