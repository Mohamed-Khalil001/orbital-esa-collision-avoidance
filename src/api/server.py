"""FastAPI Production Inference, Model Registry, Monitoring, Operational Cost & 6-DOF Astrodynamics API.

Unifies Streamlit (`app/app.py`) and FastAPI on `src.inference.get_shared_inference_engine()`.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .. import config, features, inference, modeling, monitoring, operational_cost, registry, simulation
from ..models.two_stage_m6 import evaluate_cdm_history, evaluate_single_cdm_event
from ..simulator.orbital_mechanics import simulate_along_track_cam

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_HTML_PATH = ROOT_DIR / "web" / "index.html"
FALLBACK_HTML_PATH = ROOT_DIR / "app" / "static" / "orbital_3d_simulator.html"
EVENTS_JSON_PATH = ROOT_DIR / "data" / "benchmark_44_conjunction_events.json"

app = FastAPI(
    title="ORBITAL-AI: Unified Conjunction Inference, Model Registry, Monitoring & Astrodynamics API",
    version="3.0.0",
    description=(
        "Unified Two-Stage LightGBM + M6 Inference Engine (shared with Streamlit), Cryptographic SHA-256 Model Registry "
        "with 5-Stage Validation Gate, PSI/KS Drift Monitor, Operational Cost (C = C_FP*FP + C_FN*FN + C_CAM*N_CAM) "
        "Evaluator, and Coupled 6-State CW/RK4 J2 + Full 3x3 Covariance B-Plane Simulator."
    ),
)

_RATE_LIMIT_MAX_REQ = int(os.environ.get("ORBITAL_RATE_LIMIT_PER_MIN", "120"))
_RATE_BUCKETS: dict[str, list[float]] = {}


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "local"
    now = time.time()
    window = [t for t in _RATE_BUCKETS.get(client_ip, []) if now - t < 60.0]
    if len(window) >= _RATE_LIMIT_MAX_REQ:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Maximum requests per minute: " + str(_RATE_LIMIT_MAX_REQ)},
        )
    window.append(now)
    _RATE_BUCKETS[client_ip] = window

    configured_key = os.environ.get("ORBITAL_API_KEY", "").strip()
    if configured_key and request.url.path.startswith("/api/v1/") and request.method == "POST":
        supplied_key = request.headers.get("X-API-Key", "").strip()
        if supplied_key != configured_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing X-API-Key header."},
            )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-RateLimit-Limit"] = str(_RATE_LIMIT_MAX_REQ)
    response.headers["X-RateLimit-Remaining"] = str(max(0, _RATE_LIMIT_MAX_REQ - len(window)))
    return response


class CDMRecordInput(BaseModel):
    event_id: int = Field(0, ge=0)
    time_to_tca: float = Field(..., gt=-1.0, le=30.0)
    risk: float = Field(..., ge=-30.0, le=0.0)
    max_risk_estimate: float = Field(..., ge=-30.0, le=0.0)
    max_risk_scaling: float = Field(10.0, ge=0.0)
    miss_distance: float = Field(..., gt=0.0, le=5e6)
    mahalanobis_distance: float = Field(..., gt=0.0, le=1e5)
    relative_speed: float = Field(..., gt=0.0, le=35000.0)
    relative_position_r: float = Field(-120.0)
    relative_position_t: float = Field(650.0)
    relative_position_n: float = Field(-310.0)
    relative_velocity_r: float = Field(15.0)
    relative_velocity_t: float = Field(-12500.0)
    relative_velocity_n: float = Field(4200.0)
    c_sigma_r: float = Field(180.0, gt=0.0)
    c_sigma_t: float = Field(3200.0, gt=0.0)
    c_sigma_n: float = Field(140.0, gt=0.0)
    c_ct_r: float = Field(-0.35, ge=-1.0, le=1.0)
    c_cn_r: float = Field(-0.15, ge=-1.0, le=1.0)
    c_cn_t: float = Field(-0.05, ge=-1.0, le=1.0)
    t_sigma_r: float = Field(5.0, gt=0.0)
    t_sigma_t: float = Field(45.0, gt=0.0)
    t_sigma_n: float = Field(3.0, gt=0.0)
    t_ct_r: float = Field(-0.10, ge=-1.0, le=1.0)
    t_cn_r: float = Field(0.0, ge=-1.0, le=1.0)
    t_cn_t: float = Field(0.0, ge=-1.0, le=1.0)
    c_obs_used: float = Field(22.0, ge=0.0)
    c_obs_available: float = Field(25.0, ge=0.0)
    c_recommended_od_span: float = Field(5.5, ge=0.0)
    c_actual_od_span: float = Field(5.5, ge=0.0)
    c_object_type: str = Field("DEBRIS")


class CDMEventHistoryRequest(BaseModel):
    cdms: list[CDMRecordInput] = Field(...)
    operating_mode: str = Field("prior")
    c_fp: float = Field(1.0, gt=0.0)
    c_fn: float = Field(20.0, gt=0.0)
    c_cam: float = Field(2.0, ge=0.0)


class CDMPredictRequest(BaseModel):
    last_risk: float = Field(..., ge=-30.0, le=0.0)
    max_risk_estimate: float = Field(..., ge=-30.0, le=0.0)
    miss_distance_m: float = Field(..., gt=0.0, le=5e6)
    mahalanobis_distance: float = Field(..., gt=0.0, le=1e5)
    c_sigma_t_m: float = Field(..., gt=0.0, le=1e7)
    relative_speed_ms: float = Field(..., gt=0.0, le=35000.0)
    n_allowed_cdms: int = Field(..., ge=1, le=200)
    operating_mode: str = Field("prior")


class CAMSimulateRequest(BaseModel):
    rel_pos_r_m: float = Field(...)
    rel_pos_t_m: float = Field(...)
    rel_pos_n_m: float = Field(...)
    c_sigma_t_m: float = Field(..., gt=0.0)
    initial_predicted_risk: float = Field(..., ge=-30.0, le=0.0)
    delta_v_cms: float = Field(..., ge=-50.0, le=50.0)
    delta_v_r_cms: float = Field(0.0, ge=-50.0, le=50.0)
    delta_v_n_cms: float = Field(0.0, ge=-50.0, le=50.0)
    lead_time_days: float = Field(2.0, ge=0.25, le=14.0)
    c_ct_r: float = Field(-0.35, ge=-1.0, le=1.0)
    c_cn_r: float = Field(-0.15, ge=-1.0, le=1.0)
    c_cn_t: float = Field(-0.05, ge=-1.0, le=1.0)


@app.get("/", response_class=HTMLResponse)
def serve_platform_ui():
    target = WEB_HTML_PATH if WEB_HTML_PATH.exists() else FALLBACK_HTML_PATH
    return target.read_text(encoding="utf-8")


@app.get("/api/v1/health")
def health_check():
    engine = inference.get_shared_inference_engine()
    bundle, file_sha, p = engine.load_verified_bundle()
    return {
        "status": "ok",
        "shared_engine": "UnifiedInferenceEngine (Streamlit + FastAPI Parity)",
        "bundle_file": p.name,
        "model_version": bundle.get("model_version"),
        "file_sha256": file_sha,
        "bundle_sha256": bundle.get("bundle_sha256"),
        "stage1_trees": getattr(bundle["model"], "stage1_booster", {}).get("n_trees", 0),
        "stage2_risk_blind_trees": getattr(bundle["model"], "stage2_booster", {}).get("n_trees", 0),
        "stage2_physical_allowlist_size": len(getattr(bundle["model"], "stage2_booster", {}).get("feature_names", [])),
    }


@app.get("/api/v1/model-info")
def get_model_info():
    engine = inference.get_shared_inference_engine()
    bundle, file_sha, p = engine.load_verified_bundle()
    return {
        "bundle_file": p.name,
        "file_sha256": file_sha,
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("model_version"),
        "architecture": bundle.get("architecture"),
        "bundle_sha256": bundle.get("bundle_sha256"),
        "stage1_feature_count": len(bundle.get("stage1_features", [])),
        "stage2_risk_blind_feature_count": len(bundle.get("stage2_features", [])),
        "em_prior_diagnostics": bundle.get("em_prior_diagnostics", {}),
    }


@app.get("/api/v1/registry")
def get_model_registry():
    """Returns the cryptographic SHA-256 Model Registry manifest and 5-Stage Validation Gate status."""
    return registry.load_registry_manifest()


@app.post("/api/v1/registry/validate")
def validate_registered_model(bundle_file: str = Query("final_model.joblib")):
    """Runs the 5-Stage Pre-Activation Validation Gate on a model bundle in models/."""
    target = config.MODELS_DIR / Path(bundle_file).name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {target.name}")
    b = modeling.load_bundle(target)
    return registry.run_pre_activation_validation_gate(b)


@app.get("/api/v1/monitoring")
def get_monitoring_telemetry():
    """Returns real-time PSI/KS covariate drift, zone alarm rates, feature coverage, and latency percentiles."""
    return monitoring.get_model_monitor().get_summary()


@app.get("/api/v1/cost-evaluation")
def get_operational_cost_evaluation(
    c_fp: float = Query(1.0, gt=0.0),
    c_fn: float = Query(20.0, gt=0.0),
    c_cam: float = Query(2.0, ge=0.0),
):
    """Evaluates all benchmark models using C = C_FP*FP + C_FN*FN + C_CAM*N_CAM."""
    df_cost = operational_cost.build_operational_cost_comparison_table(c_fp=c_fp, c_fn=c_fn, c_cam=c_cam)
    return {
        "formula": "C = C_FP * FP + C_FN * FN + C_CAM * N_CAM",
        "parameters": {"c_fp": c_fp, "c_fn": c_fn, "c_cam": c_cam},
        "ranking": df_cost.to_dict(orient="records"),
    }


@app.get("/api/v1/stk-orekit-benchmark")
def get_stk_orekit_propagator_validation():
    """Returns the J2 numerical verification table comparing CW, Schweighart-Sedwick J2, 3D ECI Cowell DOP853, and Orekit/STK."""
    df_val = simulation.validate_j2_against_high_precision_and_stk_orekit()
    return df_val.to_dict(orient="records")


@app.get("/api/v1/events")
def list_benchmark_events():
    return json.loads(EVENTS_JSON_PATH.read_text(encoding="utf-8"))


@app.post("/api/v1/predict-event")
def predict_from_cdm_history(req: CDMEventHistoryRequest):
    """Full end-to-end causal inference via the shared UnifiedInferenceEngine (identical to Streamlit)."""
    df_raw = pd.DataFrame([(r.model_dump() if hasattr(r, 'model_dump') else r.dict()) for r in req.cdms])
    allowed = features.filter_causal_cdms(df_raw, cutoff_days=2.0)
    if allowed.empty:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Causal Cutoff Violation: All supplied CDMs have time_to_tca < 2.0 days. At least 1 pre-cutoff CDM (time_to_tca >= 2.0d) is required.",
        )
    engine = inference.get_shared_inference_engine()
    res = engine.predict_event_from_cdms(
        df_raw, operating_mode=req.operating_mode, c_fp=req.c_fp, c_fn=req.c_fn, c_cam=req.c_cam
    )
    return res


@app.post("/api/v1/predict")
def predict_conjunction(req: CDMPredictRequest):
    res = evaluate_single_cdm_event(
        last_risk=req.last_risk,
        max_risk_estimate=req.max_risk_estimate,
        miss_distance_m=req.miss_distance_m,
        mahalanobis_distance=req.mahalanobis_distance,
        c_sigma_t_m=req.c_sigma_t_m,
        relative_speed_ms=req.relative_speed_ms,
        n_allowed_cdms=req.n_allowed_cdms,
        operating_mode=req.operating_mode,
    )
    return asdict(res)


@app.post("/api/v1/simulate-cam")
def simulate_cam_burn(req: CAMSimulateRequest):
    res = simulate_along_track_cam(
        rel_pos_r_m=req.rel_pos_r_m,
        rel_pos_t_m=req.rel_pos_t_m,
        rel_pos_n_m=req.rel_pos_n_m,
        c_sigma_t_m=req.c_sigma_t_m,
        initial_predicted_risk=req.initial_predicted_risk,
        delta_v_cms=req.delta_v_cms,
        delta_v_r_cms=req.delta_v_r_cms,
        delta_v_n_cms=req.delta_v_n_cms,
        lead_time_days=req.lead_time_days,
        c_ct_r=req.c_ct_r,
        c_cn_r=req.c_cn_r,
        c_cn_t=req.c_cn_t,
    )
    return asdict(res)
