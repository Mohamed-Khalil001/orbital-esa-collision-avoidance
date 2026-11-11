"""Tests for data loading helpers and the bundled demo sample."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src import config, data


def test_check_schema_accepts_valid_table():
    df = pd.DataFrame({config.ID_COL: [1], config.TIME_COL: [2.5], config.RISK_COL: [-6.0]})
    assert data.check_schema(df) is df


def test_check_schema_rejects_missing_columns():
    with pytest.raises(ValueError):
        data.check_schema(pd.DataFrame({config.ID_COL: [1]}))


def test_test_files_do_not_need_the_target():
    df = pd.DataFrame({config.ID_COL: [1], config.TIME_COL: [2.5]})
    assert data.check_schema(df, need_risk=False) is df


def test_missing_training_file_gives_a_clear_error():
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(FileNotFoundError):
            data.resolve_train_path(Path(folder))


def test_demo_sample_is_valid(demo_events):
    data.check_schema(demo_events, need_risk=True)
    assert demo_events[config.ID_COL].nunique() == 150
    assert demo_events[config.TIME_COL].min() >= config.MIN_TIME_TO_TCA_FOR_INPUT
    assert demo_events.shape[1] == 103


def test_demo_sample_contains_alarming_and_quiet_events(demo_events):
    peak = demo_events.groupby(config.ID_COL)[config.RISK_COL].max()
    assert (peak >= config.HIGH_RISK_THRESHOLD).any()
    assert (peak < config.HIGH_RISK_THRESHOLD).any()
