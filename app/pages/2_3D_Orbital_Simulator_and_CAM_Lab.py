"""3D WebGL Astrodynamics & Coupled 6-State CW/RK4 J2 CAM Lab Page (Zero-Plotly)."""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data, modeling, simulation  # noqa: E402

st.set_page_config(page_title="ORBITAL | 3D Astrodynamics & CAM Lab", page_icon="🛰️", layout="wide")
st.title("🛰️ 3D WebGL Astrodynamics & Coupled 6-State CW/RK4 J2 CAM Lab")
st.warning(simulation.OPERATIONAL_DISCLAIMER)

tab_event, tab_workbench = st.tabs([
    "🎯 Per-Event 3D WebGL + Coupled 6-State CW/RK4 & 2D B-Plane Simulator (Any Dataset Event)",
    "🌐 Full 44-Event Multi-Tab Astrodynamics & Causal CDM Workbench",
])

with tab_event:
    demo_df = data.load_demo_sample()
    bundle = modeling.load_bundle(config.MODELS_DIR / "final_model.joblib")
    c1, c2 = st.columns([1, 2])
    with c1:
        ev_id = st.selectbox("Select Conjunction Event ID", sorted(demo_df[config.ID_COL].unique()), index=0)
    cdms = demo_df[demo_df[config.ID_COL] == ev_id].sort_values(config.TIME_COL, ascending=False)
    pred = modeling.predict_event(bundle, cdms)
    with c2:
        if pred is not None:
            st.info(
                f"**Event #{ev_id}** · Zone: `{pred['zone']}` · Action: `{pred['action_tag']}` · "
                f"Predicted Final Risk: `{pred['adjusted']:.3f}` · Stage-1 Prob: `{pred['stage1_lgbm_prob']:.4f}` · "
                f"Stage-2 Risk-Blind Prob: `{pred['stage2_risk_blind_prob']:.4f}`"
            )
    components.html(
        simulation.render_live_event_3d_simulator_html(cdms, prediction_result=pred, height=640),
        height=665,
        scrolling=False,
    )
    st.subheader(f"Analytical 6-State CW + RK4 J2 Parametric Trade-Off Table (Event #{ev_id})")
    kin = simulation.extract_last_cdm_kinematics(cdms)
    st.dataframe(simulation.compute_cam_tradeoff_table(kin, burn_lead_days=2.2), use_container_width=True, hide_index=True)

with tab_workbench:
    html_path = ROOT / "app" / "static" / "orbital_3d_simulator.html"
    if html_path.exists():
        components.html(html_path.read_text(encoding="utf-8"), height=880, scrolling=True)
