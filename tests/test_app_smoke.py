"""Run the Streamlit pages headlessly and check that they finish without an exception."""
from pathlib import Path
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_main_app_runs_on_the_demo_sample():
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


def test_research_results_page_runs():
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "1_Research_Results.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
