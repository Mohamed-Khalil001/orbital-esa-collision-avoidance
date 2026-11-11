"""Offline External Ground-Truth Evaluation Utility.

To preserve strict repository data governance, private test ground-truth labels (`test_data_private.csv`,
`true_final_risk`, `true_high_risk`) are NOT stored inside this public repository.
Researchers with local access to the official ESA Kelvins post-competition test archive can run this script
to verify all retrospective test metrics locally.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src import metric


def evaluate_predictions_against_external_labels(predictions_csv: Path, private_labels_csv: Path) -> pd.DataFrame:
    preds = pd.read_csv(predictions_csv).sort_values("event_id").reset_index(drop=True)
    raw_priv = pd.read_csv(private_labels_csv)
    if "time_to_tca" in raw_priv.columns:
        priv_final = (
            raw_priv.sort_values(["event_id", "time_to_tca"], ascending=[True, True])
            .groupby("event_id", as_index=False)
            .first()
        )
    else:
        priv_final = raw_priv.copy()
    priv_final = priv_final.sort_values("event_id").reset_index(drop=True)
    risk_col = next(c for c in ("true_final_risk", "true_risk", "final_risk", "risk") if c in priv_final.columns)
    y_true = priv_final[risk_col].to_numpy(dtype=float)

    rows = []
    for col in [c for c in preds.columns if c.startswith("final_risk_")]:
        res = metric.esa_loss(y_true, preds[col].to_numpy(dtype=float))
        rows.append({"System_Column": col, **res})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved test predictions against external ESA private labels.")
    parser.add_argument("--predictions", type=Path, default=Path("research/results/colab_run_outputs/05_Final_Test_Predictions_2167.csv"))
    parser.add_argument("--external-private-labels", type=Path, required=True, help="Path to external test_data_private.csv (outside repository)")
    args = parser.parse_args()
    df_res = evaluate_predictions_against_external_labels(args.predictions, args.external_private_labels)
    print(df_res.to_string(index=False))
