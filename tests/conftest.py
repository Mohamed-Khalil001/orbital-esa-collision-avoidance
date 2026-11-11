"""Shared fixtures for the test suite."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def demo_events():
    """The bundled demo sample (150 events from the ESA test file)."""
    return pd.read_csv(ROOT / "data" / "sample" / "demo_events.csv")
