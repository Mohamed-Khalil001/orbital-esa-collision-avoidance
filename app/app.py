"""Research dashboard for ESA Kelvins Collision Avoidance Challenge (Zero-Plotly WebGL & Canvas Edition).

Run from project root: streamlit run app/app.py
Connected end-to-end to `models/final_model.joblib` (350 Stage-1 NOCAT + 350 Stage-2 Risk-Blind LightGBM trees)
and the Coupled 6-State Clohessy-Wiltshire + RK4 J2 + 2D B-Plane Astrodynamics Simulator.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import joblib
except ImportError:
    joblib = None

from src import config, data, features, inference, modeling, monitoring, operational_cost, registry, simulation

st.set_page_config(page_title="ORBITAL | Collision Risk & Astrodynamics Lab", page_icon="🛰️", layout="wide")

TEST_SOURCE = "ESA test_data.csv — 2,167 events"
TRAIN_SOURCE = "Training data — full CDM history"
DEMO_SOURCE = "Bundled demo sample — 150 events from ESA test_data"


@st.cache_resource
def load_model(path_str, modified_time):
    return modeling.load_bundle(Path(path_str))


@st.cache_data(show_spinner="Loading CDM dataset…")
def load_table(path_str):
    return pd.read_csv(path_str)


def fmt_num(x, digits=2):
    return "—" if pd.isna(x) else f"{x:,.{digits}f}"


def show_model_controls():
    st.sidebar.markdown("### Model registry")
    model_files = modeling.list_model_files()
    bundle = None
    model_path = None
    if model_files:
        model_path = st.sidebar.selectbox(
            "Active model", model_files,
            index=next((i for i, p in enumerate(model_files) if p.name == "final_model.joblib"), 0),
            format_func=lambda p: p.name,
        )
        try:
            bundle = load_model(str(model_path), model_path.stat().st_mtime)
            problems = modeling.validate_bundle(bundle)
            if problems:
                st.sidebar.error("Invalid model: " + "; ".join(problems))
                bundle = None
            else:
                info = modeling.describe_bundle(bundle, model_path)
                st.sidebar.success(f"{info['model']} · {info['features']} features")
                st.sidebar.caption(f"Updated: {info['last modified']} · alarm τ: {info['alarm threshold tau']}")
                if info["demo model"]:
                    st.sidebar.warning("Synthetic demo model — predictions are not scientifically meaningful.")
                with st.sidebar.expander("Model details"):
                    st.json(info)
        except Exception as exc:
            st.sidebar.error(f"Could not load model: {type(exc).__name__}: {exc}")
            bundle = None
    else:
        st.sidebar.warning("No trained model installed. Dataset exploration and encounter geometry are available; ML predictions are disabled.")

    if st.sidebar.button("Reload model files"):
        load_model.clear()
        st.rerun()

    with st.sidebar.expander("Install / update model"):
        st.caption("Upload a trusted .joblib bundle exported by the project's training notebook. The previous active model is archived automatically.")
        uploaded_model = st.file_uploader("Model bundle (.joblib)", type=["joblib"], key="model_upload")
        trusted = st.checkbox("I trained or created this file and trust its source", key="model_trust")
        if st.button("Validate, install and activate", disabled=(uploaded_model is None or not trusted)):
            try:
                candidate = joblib.load(uploaded_model) if joblib is not None else pickle.load(uploaded_model)
                problems = modeling.validate_bundle(candidate)
                if problems:
                    raise ValueError("; ".join(problems))
                installed = modeling.install_bundle(candidate)
                load_model.clear()
                st.sidebar.success(f"Activated {installed['installed'].name}")
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Model was not installed: {type(exc).__name__}: {exc}")
    return bundle, model_path


st.sidebar.title("ORBITAL")
st.sidebar.caption("Collision Avoidance · Research Console")
bundle, model_path = show_model_controls()

st.sidebar.markdown("### Data source")
data_options = []
test_path = config.DATA_DIR / "test_data.csv"
if test_path.exists():
    data_options.append(TEST_SOURCE)
try:
    train_path = data.resolve_train_path()
    data_options.append(TRAIN_SOURCE)
except FileNotFoundError:
    train_path = None
demo_path = config.SAMPLE_DIR / "demo_events.csv"
if demo_path.exists():
    data_options.append(DEMO_SOURCE)

uploaded_csv = st.sidebar.file_uploader("Or load another CDM CSV", type=["csv"], key="cdm_csv")
if uploaded_csv is not None:
    table = pd.read_csv(uploaded_csv)
    source_label = uploaded_csv.name
elif data_options:
    selected_source = st.sidebar.selectbox("Dataset", data_options)
    selected_path = {TEST_SOURCE: test_path, TRAIN_SOURCE: train_path, DEMO_SOURCE: demo_path}[selected_source]
    table = load_table(str(selected_path))
    source_label = selected_source
else:
    st.title("ORBITAL | Collision Risk Lab")
    st.error("No dataset found. Run tools/download_data.py, place test_data.csv in data/, or upload a CDM CSV in the sidebar.")
    st.stop()

try:
    table = data.check_schema(table, need_risk=True)
except Exception as exc:
    st.error(f"Dataset schema error: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Header and data quality overview
st.markdown("# 🛰️ ORBITAL")
st.caption("ESA Kelvins Collision Avoidance Challenge · Unified Two-Stage LightGBM + M6 Risk Intelligence & 6-DOF CW/RK4 Astrodynamics Workbench")
st.markdown("**Dataset:** " + source_label)

validation_path = config.REPORTS_DIR / "model_validation.json"
if validation_path.exists():
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        with st.expander("Model validation · event-level out-of-fold results", expanded=False):
            st.caption(
                f"{validation['folds']}-fold validation · {validation['evaluation_cohort_events']:,} test-like events · "
                f"{validation['high_risk_events']} high-risk events · deployment τ = {validation['deployment_threshold']:.2f}"
            )
            compare = pd.DataFrame([
                {"Method": validation["baseline"]["name"], **{k: validation["baseline"][k] for k in ["score", "mse_high", "f2", "precision", "recall"]}},
                {"Method": validation["model_raw"]["name"], **{k: validation["model_raw"][k] for k in ["score", "mse_high", "f2", "precision", "recall"]}},
                {"Method": validation["model_nested"]["name"], **{k: validation["model_nested"][k] for k in ["score", "mse_high", "f2", "precision", "recall"]}},
            ])
            st.dataframe(compare, hide_index=True, use_container_width=True)
            st.warning(validation["caveat"])
    except Exception as exc:
        st.info(f"Validation summary could not be loaded: {type(exc).__name__}: {exc}")

n_events = int(table[config.ID_COL].nunique())
n_rows = int(len(table))
per_event = table.groupby(config.ID_COL).size()
min_tca = float(table[config.TIME_COL].min())
max_tca = float(table[config.TIME_COL].max())
high_cdm = int((table[config.RISK_COL] >= config.HIGH_RISK_THRESHOLD).sum())
missing_pct = float(table.isna().mean().mean() * 100)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Conjunction events", f"{n_events:,}")
m2.metric("CDM records", f"{n_rows:,}")
m3.metric("Median CDMs / event", f"{per_event.median():.0f}")
m4.metric("Time to TCA", f"{min_tca:.2f}–{max_tca:.2f} d")
m5.metric("Missing cells", f"{missing_pct:.1f}%")

if source_label in (TEST_SOURCE, DEMO_SOURCE) and max_tca >= 2 and min_tca >= 2:
    st.info("This test file contains only pre-decision CDMs (at least 2 days before TCA). It does not contain the final near-TCA outcome needed to score predictions. A trained model must come from the training dataset.")

# Dataset quality and event selection
left, right = st.columns([1.05, 1.45], gap="large")
with left:
    st.subheader("Dataset profile")
    components.html(simulation.render_risk_histogram_html(table, height=310), height=320, scrolling=False)
    st.caption(f"{high_cdm:,} of {n_rows:,} CDM rows ({high_cdm / max(n_rows, 1):.1%}) have risk ≥ −6. This is a row-level descriptive statistic, not the final event label.")

with right:
    st.subheader("Select conjunction event")
    event_ids = sorted(table[config.ID_COL].dropna().unique().tolist())
    only_high = st.checkbox("Show events with at least one high-risk CDM", value=False)
    if only_high:
        high_ids = set(table.loc[table[config.RISK_COL] >= config.HIGH_RISK_THRESHOLD, config.ID_COL].unique())
        event_ids = [x for x in event_ids if x in high_ids]
    if not event_ids:
        st.warning("No events match this filter.")
        st.stop()
    event_id = st.selectbox("Event ID", event_ids, index=0)
    cdms = table.loc[table[config.ID_COL] == event_id].sort_values(config.TIME_COL, ascending=False).copy()
    latest = cdms.iloc[-1]
    nearest = cdms.loc[cdms[config.TIME_COL].idxmin()]
    e1, e2, e3 = st.columns(3)
    e1.metric("CDMs in event", f"{len(cdms)}")
    e2.metric("Closest listed TCA", f"{nearest[config.TIME_COL]:.3f} d")
    e3.metric("Latest listed risk", fmt_num(nearest[config.RISK_COL]))
    st.caption(f"Object type: {latest.get('c_object_type', 'Unknown')} · Mission ID: {latest.get('mission_id', '—')}")

# ---------------------------------------------------------------------------
# Event risk timeline and ML prediction
st.divider()
st.subheader("Event analysis (Unified .joblib Two-Stage LightGBM + M6 Engine)")
result = None
if bundle is not None:
    try:
        engine = inference.get_shared_inference_engine()
        result = engine.predict_event_from_cdms(cdms, bundle_path=model_path)
    except Exception as exc:
        st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

if result is not None:
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Predicted final risk (log₁₀)", f"{result['raw']:.3f}", result.get("action_tag", ""))
    p2.metric("Implied probability (10ʳ)", f"{result['probability']:.3e}", result.get("zone", ""))
    p3.metric("Last observed CDM risk", fmt_num(result["last_risk"]), f"Dilution Gap: +{fmt_num(result.get('dilution_gap'))}")
    p4.metric("Stage-1 LightGBM Prob", f"{result.get('stage1_lgbm_prob', 0.0):.4f}", f"Score: {result.get('stage1_score', 0.0):.2f}")
    p5.metric("Stage-2 Risk-Blind Prob", f"{result.get('stage2_risk_blind_prob', 0.0):.4f}", "184 Physical Features")
    p6.metric("CDMs used (T >= 2.0d)", f"{result['n_cdms_used']}", f"Coverage: {result['coverage']:.0%}")
    if result["is_alarm"]:
        st.error(f"HIGH-RISK MODEL ALERT ({result.get('action_tag', 'ALARM')}) · predicted risk is at or above −6 ({result.get('zone', '')})")
    else:
        st.success(f"MODEL OUTPUT BELOW ALERT THRESHOLD ({result.get('action_tag', 'SAFE')}) · predicted risk is below −6 ({result.get('zone', '')})")
else:
    st.warning("ML prediction is currently inactive. Install a compatible trained model bundle to enable event-level predictions.")

show_running = st.checkbox("Show experimental rolling predictions after each CDM", value=False, disabled=(bundle is None))
running = modeling.running_predictions(bundle, cdms) if (bundle is not None and show_running) else None
components.html(
    simulation.render_risk_timeline_html(cdms, predicted=(result["raw"] if result else None), running=running, height=360),
    height=375,
    scrolling=False,
)

# ---------------------------------------------------------------------------
# 3D Encounter Geometry & Coupled 6-State CW/RK4 Astrodynamics Simulator
st.subheader("3D Encounter Geometry & Coupled 6-State CW/RK4 Astrodynamics Simulator")
st.info(simulation.OPERATIONAL_DISCLAIMER)
frames = simulation.encounter_frames(cdms)
if frames:
    components.html(
        simulation.render_live_event_3d_simulator_html(cdms, prediction_result=result, height=640),
        height=665,
        scrolling=False,
    )
    st.caption(
        "Interactive WebGL Three.js + Coupled 6-State Clohessy-Wiltshire STM Phi(t,t0) + RK4 J2 relative trajectory propagator "
        "and 2D Akella & Alfriend (2000) Encounter B-Plane polar quadrature reconstructed from the selected event's pre-cutoff CDM."
    )
else:
    st.warning("The selected event does not contain sufficient relative position/velocity data for the 3D view.")

# ---------------------------------------------------------------------------
# Detailed data and diagnostics
with st.expander("CDM records for selected event"):
    st.dataframe(cdms, use_container_width=True, hide_index=True)
with st.expander("Analytical 6-State CW + RK4 J2 Maneuver Trade-Off Table for Selected Event"):
    kin = simulation.extract_last_cdm_kinematics(cdms)
    st.dataframe(simulation.compute_cam_tradeoff_table(kin, burn_lead_days=2.2), use_container_width=True, hide_index=True)
with st.expander("Data quality diagnostics"):
    missing = table.isna().mean().mul(100).sort_values(ascending=False).rename("missing_%").to_frame()
    st.dataframe(missing.head(25), use_container_width=True)
    st.download_button("Download selected event CSV", cdms.to_csv(index=False).encode("utf-8"), file_name=f"event_{event_id}_cdms.csv", mime="text/csv")


with st.expander("Phase 4 & 5: Operational Cost Decision Analysis (C = C_FP·FP + C_FN·FN + C_CAM), SHA-256 Registry & Drift Monitor", expanded=False):
    c_col1, c_col2, c_col3 = st.columns(3)
    c_fp_val = c_col1.number_input("False Positive Cost (C_FP)", min_value=0.1, max_value=50.0, value=1.0, step=0.5)
    c_fn_val = c_col2.number_input("False Negative Cost (C_FN)", min_value=1.0, max_value=200.0, value=20.0, step=1.0)
    c_cam_val = c_col3.number_input("Maneuver Execution Cost (C_CAM)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
    if result is not None:
        ev_cost = operational_cost.compute_single_event_decision_cost(
            calibrated_collision_prob=float(max(result.get("probability", 0.0), 0.5 * (result.get("stage1_lgbm_prob", 0.0) + result.get("stage2_risk_blind_prob", 0.0)))),
            c_fp=c_fp_val, c_fn=c_fn_val, c_cam=c_cam_val,
        )
        st.markdown(
            f"**Event Operational Cost Recommendation:** `{ev_cost['recommended_operational_decision']}` · "
            f"E[Cost | No CAM] = `{ev_cost['expected_cost_no_action']:.3f}` vs. "
            f"E[Cost | Alarm + CAM] = `{ev_cost['expected_cost_alarm_and_cam']:.3f}` "
            f"(Bayes Optimal Threshold $p^* = {ev_cost['bayes_optimal_prob_threshold']:.4f}$)"
        )
    st.markdown("#### Cryptographic Model Registry (`models/registry_manifest.json` — 5-Stage Validation Gate)")
    manifest = registry.load_registry_manifest()
    st.dataframe(pd.DataFrame(manifest["models"]), use_container_width=True, hide_index=True)
    st.markdown("#### Production Model Monitoring (PSI & KS Feature Drift)")
    mon_summary = monitoring.get_model_monitor().get_summary()
    st.caption(
        f"Status: **{mon_summary['status']}** · Logged Inferences: `{mon_summary['total_inferences_logged']}` · "
        f"p50 Latency: `{mon_summary['p50_latency_ms']} ms` · p95 Latency: `{mon_summary['p95_latency_ms']} ms`"
    )
    st.dataframe(pd.DataFrame(mon_summary["drift_table"]), use_container_width=True, hide_index=True)

st.divider()
st.caption("Research-use prototype · Not an operational conjunction assessment system · Validate all physical assumptions and model calibration before interpreting alerts.")
