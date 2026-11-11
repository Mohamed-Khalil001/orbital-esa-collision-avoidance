"""Consistency and Data Governance checks on archived research results."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "research" / "results" / "colab_run_outputs"


@pytest.fixture(scope="module")
def board():
    return pd.read_csv(OUTPUTS / "06_Final_Test_and_ESA_Leaderboard_Comparison.csv")


def test_confusion_counts_are_consistent(board):
    assert ((board["TP"] + board["FP"]) == board["Flagged"]).all()
    assert ((board["TP"] + board["FN"]) == 150).all()


def test_every_system_has_a_known_category(board):
    allowed = {"1_Baseline", "2_Our_Pipeline", "2_Our_Champion", "3_ESA_Leaderboard"}
    assert set(board["Category"]) <= allowed


def test_leaderboard_is_sorted_by_loss(board):
    assert board["ESA_Loss_L"].is_monotonic_increasing


def test_prediction_file_covers_all_test_events_without_leaking_ground_truth():
    preds = pd.read_csv(OUTPUTS / "05_Final_Test_Predictions_2167.csv")
    assert len(preds) == 2167 and preds["event_id"].is_unique
    assert "true_final_risk" not in preds.columns
    assert "true_high_risk" not in preds.columns


def test_cv_summary_has_no_missing_values():
    cv = pd.read_csv(OUTPUTS / "02_CV_15Fold_Summary.csv")
    assert not cv.isna().any().any()
