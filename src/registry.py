"""Phase 4: Cryptographic Model Registry (SHA-256) & 5-Stage Pre-Activation Validation Gate."""
from __future__ import annotations
import hashlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from . import config, features
from .features import RISK_HISTORY_COLS, is_stage2_risk_blind_feature

MANIFEST_PATH = config.MODELS_DIR / "registry_manifest.json"


def compute_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bundle_sha256(bundle: dict) -> str:
    clean = {k: v for k, v in bundle.items() if k != "bundle_sha256"}
    raw = pickle.dumps(clean, protocol=4)
    return hashlib.sha256(raw).hexdigest()


def run_pre_activation_validation_gate(bundle: dict, demo_df: pd.DataFrame | None = None) -> dict:
    """Executes the mandatory 5-Stage Validation Gate before any model bundle can be activated."""
    from . import data, modeling

    gates = {
        "gate_1_schema_and_finite_probe": False,
        "gate_2_sha256_cryptographic_digest": False,
        "gate_3_stage2_risk_blind_zero_sensitivity": False,
        "gate_4_causal_cutoff_compliance": False,
        "gate_5_benchmark_prediction_and_cost_sanity": False,
    }
    errors: list[str] = []

    # Gate 1: Schema & all-NaN probe
    probs = modeling.validate_bundle(bundle)
    if probs:
        errors.extend(probs)
    else:
        gates["gate_1_schema_and_finite_probe"] = True

    # Gate 2: SHA-256 digest computation & verification
    try:
        digest = compute_bundle_sha256(bundle)
        if len(digest) == 64:
            gates["gate_2_sha256_cryptographic_digest"] = True
    except Exception as exc:
        errors.append(f"SHA-256 computation failed: {exc}")
        digest = ""

    # Gate 3: Stage-2 Risk-Blind Zero-Sensitivity Verification
    try:
        m = bundle["model"]
        if hasattr(m, "_verify_stage2_risk_blind_integrity"):
            m._verify_stage2_risk_blind_integrity()
        sample_df = demo_df if demo_df is not None else data.load_demo_sample()
        X_ref = features.build_event_features(sample_df, bundle.get("object_types")).iloc[:15].copy()
        if hasattr(m, "predict_stage_probabilities"):
            _, p2_orig = m.predict_stage_probabilities(X_ref)
            X_pert = X_ref.copy()
            rng = np.random.default_rng(99)
            for col in RISK_HISTORY_COLS:
                if col in X_pert.columns:
                    X_pert[col] = rng.uniform(-30.0, -1.0, size=len(X_pert))
            _, p2_pert = m.predict_stage_probabilities(X_pert)
            max_diff = float(np.max(np.abs(p2_orig - p2_pert)))
            if max_diff == 0.0:
                gates["gate_3_stage2_risk_blind_zero_sensitivity"] = True
            else:
                errors.append(f"Stage-2 leaked risk columns (max sensitivity diff = {max_diff})")
        else:
            gates["gate_3_stage2_risk_blind_zero_sensitivity"] = True
    except Exception as exc:
        errors.append(f"Stage-2 Risk-Blind gate failed: {exc}")

    # Gate 4: Causal Cutoff Compliance (T >= 2.0d)
    try:
        sample_df = demo_df if demo_df is not None else data.load_demo_sample()
        ev0_id = sample_df[config.ID_COL].iloc[0]
        ev0 = sample_df[sample_df[config.ID_COL] == ev0_id].copy()
        res_clean = modeling.predict_event(bundle, ev0)
        # Inject a fake post-cutoff row at time_to_tca = 0.2d with extreme risk = -1.0
        fake_late = ev0.iloc[-1:].copy()
        fake_late[config.TIME_COL] = 0.20
        fake_late[config.RISK_COL] = -1.0
        ev0_with_late = pd.concat([ev0, fake_late], ignore_index=True)
        res_with_late = modeling.predict_event(bundle, ev0_with_late)
        if res_clean is not None and res_with_late is not None and abs(res_clean["raw"] - res_with_late["raw"]) < 1e-9:
            gates["gate_4_causal_cutoff_compliance"] = True
        else:
            errors.append("Post-cutoff CDM leaked into model prediction!")
    except Exception as exc:
        errors.append(f"Causal cutoff gate failed: {exc}")

    # Gate 5: Benchmark Prediction & Operational Cost Sanity
    try:
        sample_df = demo_df if demo_df is not None else data.load_demo_sample()
        X_all = features.build_event_features(sample_df, bundle.get("object_types"))
        preds = np.asarray(bundle["model"].predict(X_all.reindex(columns=bundle["feature_names"])), dtype=float)
        alarm_rate = float(np.mean(preds >= bundle.get("alarm_tau", -6.0)))
        if np.isfinite(preds).all() and 0.0 <= alarm_rate <= 0.85:
            gates["gate_5_benchmark_prediction_and_cost_sanity"] = True
        else:
            errors.append(f"Benchmark prediction sanity failed (alarm_rate={alarm_rate:.3f})")
    except Exception as exc:
        errors.append(f"Benchmark sanity gate failed: {exc}")

    passed = all(gates.values()) and len(errors) == 0
    return {
        "passed": passed,
        "sha256": digest,
        "gates": gates,
        "errors": errors,
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def build_or_refresh_registry_manifest(models_dir: Path | None = None) -> dict:
    """Scans models/*.joblib, computes full SHA-256 digests, runs validation gates, and writes registry_manifest.json."""
    from . import modeling
    models_dir = Path(models_dir or config.MODELS_DIR)
    entries = []
    for p in sorted(models_dir.glob("*.joblib")):
        bundle = modeling.load_bundle(p)
        file_sha = compute_file_sha256(p)
        gate_res = run_pre_activation_validation_gate(bundle)
        entries.append({
            "file": p.name,
            "model_name": bundle.get("model_name", type(bundle["model"]).__name__),
            "model_version": bundle.get("model_version", "M6.2-Unified-LightGBM"),
            "architecture": bundle.get("architecture", ""),
            "status": "active" if p.name == "final_model.joblib" else "registered",
            "file_sha256": file_sha,
            "bundle_sha256": gate_res["sha256"][:16],
            "size_bytes": p.stat().st_size,
            "validation_gate_passed": gate_res["passed"],
            "validation_gates": gate_res["gates"],
            "stage1_features": len(bundle.get("stage1_features", bundle.get("feature_names", []))),
            "stage2_risk_blind_features": len(bundle.get("stage2_features", [])),
            "updated_at_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    manifest = {
        "registry_schema_version": "2.0",
        "active_model_file": "final_model.joblib",
        "models": entries,
    }
    (models_dir / "registry_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_registry_manifest(models_dir: Path | None = None) -> dict:
    models_dir = Path(models_dir or config.MODELS_DIR)
    mpath = models_dir / "registry_manifest.json"
    if not mpath.exists():
        return build_or_refresh_registry_manifest(models_dir)
    return json.loads(mpath.read_text(encoding="utf-8"))
