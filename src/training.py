"""Phase 4: Dedicated Model Training, Event-Grouped Cross-Validation & Pre-Validated Registration Pipeline.

Strictly separated from stateless production inference (`src/inference.py`).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from . import config, features, modeling, registry


def train_and_validate_bundle(
    train_cdms_df: pd.DataFrame,
    model_name: str = "m6_prior_adapted",
    train_scope: str = "test_like",
    alarm_tau: float = config.HIGH_RISK_THRESHOLD,
    activate: bool = False,
    output_path: Path | None = None,
) -> dict:
    """End-to-end training pipeline: builds causal features, fits the model, runs the 5-stage validation gate, and registers the artifact."""
    object_types = features.fit_object_types(train_cdms_df)
    X, meta = features.make_event_table(train_cdms_df, object_types=object_types)
    bundle = modeling.fit_final_model(
        X=X,
        meta=meta,
        model_name=model_name,
        train_scope=train_scope,
        alarm_tau=alarm_tau,
        object_types=object_types,
    )
    gate_report = registry.run_pre_activation_validation_gate(bundle, demo_df=train_cdms_df)
    if not gate_report["passed"]:
        raise ValueError(f"Trained model failed pre-activation validation gate: {gate_report['errors']}")

    bundle["bundle_sha256"] = gate_report["sha256"][:16]
    if output_path is not None:
        modeling.save_bundle(bundle, output_path)
    elif activate:
        modeling.install_bundle(bundle, activate=True)
    return {"bundle": bundle, "validation_gate": gate_report}


__all__ = [
    "train_and_validate_bundle",
]
