"""Retrain the final model on the current data with the saved configuration, then install it.

Usage (from the project folder):
    python tools/retrain_model.py
    python tools/retrain_model.py --keep-tau
    python tools/retrain_model.py --config models/model_config.json

Steps: build the events and features -> out-of-fold predictions to re-tune the alarm threshold ->
fit on all events -> install as models/final_model.joblib (the previous model is archived).
The configuration file models/model_config.json is written by block 12 of notebooks/04_modeling.ipynb.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import config, data, features, modeling


def main():
    parser = argparse.ArgumentParser(description="Retrain the final model.")
    parser.add_argument("--config", default=None, help="path to model_config.json")
    parser.add_argument("--keep-tau", action="store_true", help="keep the old alarm threshold instead of re-tuning it")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else config.MODELS_DIR / "model_config.json"
    if not cfg_path.exists():
        sys.exit(f"Configuration not found: {cfg_path}. Run block 12 of notebooks/04_modeling.ipynb first.")
    cfg = json.loads(cfg_path.read_text())

    print("loading data and building features ...")
    train = data.load_train()
    object_types = features.fit_object_types(train)
    X, meta = features.make_event_table(train, object_types)
    Xs = features.make_feature_sets(X)[cfg["feature_set"]]
    to_drop = sorted(set(X.columns) - set(Xs.columns))
    y_eval = meta.loc[meta["test_like"], "final_risk"]
    print(f"events: {len(X):,} | features: {Xs.shape[1]} | test-like: {len(y_eval):,}")

    tau = cfg["alarm_tau"]
    model_params = cfg.get("model_params", {})
    if not args.keep_tau:
        print("selecting the alarm threshold with nested event-level validation ...")
        oof = modeling.cross_validate_events(
            Xs, meta, cfg["model_name"], train_scope=cfg["train_scope"], clip_low=cfg["clip_low"],
            high_weight=cfg["high_weight"], seed=config.RANDOM_SEED, **model_params,
        ).loc[y_eval.index]
        _, nested_taus = modeling.nested_alarm_threshold(y_eval, oof, seed=config.RANDOM_SEED)
        tau = float(__import__("numpy").median(nested_taus))
        print(f"alarm threshold: old {cfg['alarm_tau']} -> nested median {tau}; fold values={nested_taus}")

    print("fitting the final model ...")
    bundle = modeling.fit_final_model(
        Xs, meta, cfg["model_name"], clip_low=cfg["clip_low"], train_scope=cfg["train_scope"],
        high_weight=cfg["high_weight"], alarm_tau=tau, object_types=object_types, to_drop=to_drop,
        **model_params,
    )
    result = modeling.install_bundle(bundle, models_dir=config.MODELS_DIR)
    print(f"installed: {result['installed']}")
    if result["archived"]:
        print(f"previous model archived as: {result['archived']}")

    cfg["alarm_tau"] = float(tau)
    cfg["retrained_at"] = datetime.now().isoformat(timespec="seconds")
    cfg_path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
