"""Phase 4: Unified Stateless Production Inference Engine Shared by Both Streamlit and FastAPI.

Guarantees that Streamlit (`app/app.py`) and FastAPI (`src/api/server.py`) execute the exact same
code path, SHA-256 verified model bundle, causal cutoff filter, Stage-2 Risk-Blind allowlist check,
Operational Cost calculation, and real-time ModelMonitor telemetry logging.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd

from . import config, features, modeling, monitoring, operational_cost, registry


class UnifiedInferenceEngine:
    """Single source of truth for conjunction risk inference across API, CLI, and Streamlit UI."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        self._cache: dict[str, tuple[float, dict, str]] = {}

    def load_verified_bundle(self, bundle_name_or_path: str | Path = "final_model.joblib") -> tuple[dict, str, Path]:
        p = Path(bundle_name_or_path)
        if not p.is_absolute() and not p.exists():
            p = self.models_dir / p.name
        if not p.exists():
            p = self.models_dir / "final_model.joblib"

        mtime = p.stat().st_mtime
        key = str(p.resolve())
        if key in self._cache and self._cache[key][0] == mtime:
            return self._cache[key][1], self._cache[key][2], p

        file_sha = registry.compute_file_sha256(p)
        bundle = modeling.load_bundle(p)
        probs = modeling.validate_bundle(bundle)
        if probs:
            raise ValueError(f"Bundle {p.name} failed validation check: {'; '.join(probs)}")
        bundle["file_sha256"] = file_sha
        bundle["bundle_file"] = p.name
        self._cache[key] = (mtime, bundle, file_sha)
        return bundle, file_sha, p

    def predict_event_from_cdms(
        self,
        cdms: pd.DataFrame,
        operating_mode: str = "prior",
        bundle_path: str | Path | None = None,
        c_fp: float = 1.0,
        c_fn: float = 20.0,
        c_cam: float = 2.0,
    ) -> dict | None:
        t0 = time.perf_counter()
        if bundle_path is not None:
            target = bundle_path
        elif operating_mode == "r100":
            target = self.models_dir / "m6_leakfree_r100_budget.joblib"
        else:
            target = self.models_dir / "final_model.joblib"

        bundle, file_sha, resolved_path = self.load_verified_bundle(target)
        X = features.build_event_features(cdms, bundle.get("object_types"))
        if X.empty:
            return None

        res = modeling.predict_event(bundle, cdms)
        if res is None:
            return None

        # Attach Phase 5 operational cost evaluation for this event
        cal_prob = float(max(res.get("probability", 0.0), 0.5 * (res.get("stage1_lgbm_prob", 0.0) + res.get("stage2_risk_blind_prob", 0.0))))
        cost_eval = operational_cost.compute_single_event_decision_cost(
            calibrated_collision_prob=cal_prob,
            c_fp=c_fp,
            c_fn=c_fn,
            c_cam=c_cam,
        )
        res["file_sha256"] = file_sha[:16]
        res["bundle_file"] = resolved_path.name
        res["operational_cost_analysis"] = cost_eval

        latency_ms = (time.perf_counter() - t0) * 1000.0
        res["inference_latency_ms"] = round(latency_ms, 3)

        ev_id = cdms[config.ID_COL].iloc[0] if config.ID_COL in cdms.columns else 0
        monitoring.get_model_monitor().record_inference(ev_id, X.iloc[:1], res, latency_ms)
        return res


_SHARED_ENGINE = UnifiedInferenceEngine()


def get_shared_inference_engine() -> UnifiedInferenceEngine:
    return _SHARED_ENGINE
