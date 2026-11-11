"""Research results page: archived experiments, with an explicit evaluation-status label on every system (Zero-Plotly)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import config, operational_cost, simulation

st.set_page_config(page_title="ORBITAL | Research results", page_icon="📊", layout="wide")

RESULTS = config.RESEARCH_RESULTS_DIR / "colab_run_outputs"

STATUS_BY_CATEGORY = {
    "1_Baseline": "Baseline (last risk)",
    "3_ESA_Leaderboard": "Reported by ESA (external)",
    "2_Our_Pipeline": "Ours: designed with CV, test partly exposed",
    "2_Our_Champion": "Ours: tuned after inspecting test labels",
}


def read_csv(name):
    path = RESULTS / name
    if not path.exists():
        st.info(f"File not found: research/results/colab_run_outputs/{name}")
        return None
    return pd.read_csv(path)


def clean_name(text):
    return re.sub(r"[^\x00-\x7F]+", "", str(text)).strip()


st.title("📊 Research results")
st.caption("Archived experiments from the six-phase research process. Read the evaluation notes before comparing numbers.")

st.warning(
    "**How to read these results.** The ESA test labels were inspected during Phases 1-2, so every test number "
    "below is a retrospective post-competition benchmark rather than a blind operational holdout. The cross-validation "
    "tables use training events only and are the primary unbiased scientific evidence. Details: docs/EVALUATION_INTEGRITY.md."
)

st.subheader("1. Cross-validation on training events (primary evidence)")
cohort = read_csv("01_Cohort_Audit_Summary.csv")
if cohort is not None:
    st.dataframe(cohort, hide_index=True, use_container_width=True)

cv = read_csv("02_CV_15Fold_Summary.csv")
if cv is not None:
    st.caption("15 stratified folds (3 seeds x 5 folds). Counts are averages per fold. Lower ESA_Loss_L is better.")
    show_cols = [c for c in ["Model", "Flagged", "TP", "FP", "FN", "Precision", "Recall", "F2", "MSE_HR", "ESA_Loss_L"] if c in cv.columns]
    st.dataframe(cv[show_cols], hide_index=True, use_container_width=True)
    components.html(
        simulation.render_horizontal_bar_chart_html(
            cv, x_col="ESA_Loss_L", y_col="Model", title="ESA Loss L by Model (15-Fold CV, Lower is Better)", height=310
        ),
        height=325,
        scrolling=False,
    )

st.subheader("2. Test set comparison (2,167 events)")
board = read_csv("06_Final_Test_and_ESA_Leaderboard_Comparison.csv")
if board is not None:
    board = board.copy()
    board["System"] = board["System"].map(clean_name)
    board["Evaluation status"] = board["Category"].map(STATUS_BY_CATEGORY).fillna("Unknown")
    cols = [c for c in ["System", "Evaluation status", "Flagged", "TP", "FP", "FN", "Precision", "Recall", "F2", "MSE_HR", "ESA_Loss_L"] if c in board.columns]
    st.dataframe(board[cols], hide_index=True, use_container_width=True)
    components.html(
        simulation.render_horizontal_bar_chart_html(
            board.sort_values("ESA_Loss_L", ascending=True),
            x_col="ESA_Loss_L",
            y_col="System",
            title="ESA Loss L on Test Set (Lower is Better) Coloured by Evaluation Status",
            color_col="Evaluation status",
            height=520,
        ),
        height=535,
        scrolling=False,
    )

st.subheader("3. Physical zones of the last observed risk (training events)")
zones = read_csv("04a_Physics_Zones_Validation.csv")
if zones is not None:
    st.dataframe(zones, hide_index=True, use_container_width=True)
    st.caption("70 of the 93 high-risk training events already had a last observed risk of -6 or higher (zones Z1 and Z2).")

st.subheader("4. The model currently active in this platform")
val_path = config.REPORTS_DIR / "model_validation.json"
if val_path.exists():
    val = json.loads(val_path.read_text(encoding="utf-8"))
    clipped = val.get("official_metric_clipped")
    rows = []
    for key in ("baseline", "model_raw", "model_nested"):
        if key in val:
            entry = {"Method": val[key]["name"], "L (unclipped)": round(val[key]["score"], 3), "F2": round(val[key]["f2"], 3)}
            if clipped and key in clipped:
                entry["L (official, clipped)"] = round(clipped[key]["score"], 3)
            rows.append(entry)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.info("reports/model_validation.json not found.")


st.subheader("5. Phase 5: Operational Cost Evaluation (C = C_FP·FP + C_FN·FN + C_CAM·N_CAM)")
st.caption("Evaluates systems not on F2 alone, but on total operational decision and maneuver execution cost.")
cost_table = operational_cost.build_operational_cost_comparison_table(c_fp=1.0, c_fn=20.0, c_cam=2.0)
st.dataframe(cost_table, hide_index=True, use_container_width=True)
components.html(
    simulation.render_horizontal_bar_chart_html(
        cost_table.head(12),
        x_col="Total_Operational_Cost_C",
        y_col="System",
        title="Total Operational Cost C = 1.0*FP + 20.0*FN + 2.0*N_CAM (Lower is Better)",
        height=380,
    ),
    height=395,
    scrolling=False,
)

st.subheader("6. Phase 3: J2 Orbital Propagator Verification vs. 3D ECI Cowell (DOP853) & Orekit 12 / Ansys STK 12")
stk_df = simulation.validate_j2_against_high_precision_and_stk_orekit()
st.dataframe(stk_df, hide_index=True, use_container_width=True)

st.divider()
st.caption("Research-use prototype · Not an operational conjunction assessment system.")
