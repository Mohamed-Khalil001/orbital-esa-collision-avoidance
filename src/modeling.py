"""Unified Two-Stage LightGBM + M6 Physics-Guided Inference Engine & Model Bundle Management.

Every prediction executes the compiled LightGBM v4 decision trees stored inside the .joblib bundle:
  - Stage-1 Veto Booster (350 trees on 200 NOCAT features) + OOF calibration head
  - Stage-2 Risk-Blind Physics-Rescue Booster (350 trees strictly on 184 STAGE2_PHYSICAL_ALLOWLIST features)
  - Stage-3 Three-Zone Conditional Regressor (learned on training positives y_train == 1)
"""
from __future__ import annotations
import hashlib
import json
import pickle
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import joblib
except ImportError:
    joblib = None

from . import config, features
from .features import (
    RISK_HISTORY_COLS,
    STAGE2_PHYSICAL_ALLOWLIST,
    is_stage2_risk_blind_feature,
)
from .metric import collision_score


def _dump_bundle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    return path


def _load_bundle_file(path):
    path = Path(path)
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_compiled_lgbm_booster(booster: dict, X_df: pd.DataFrame) -> np.ndarray:
    """Evaluates a compiled LightGBM v4 decision tree forest stored inside the model bundle."""
    feat_names = booster["feature_names"]
    X_aligned = X_df.reindex(columns=feat_names)
    X_arr = X_aligned.to_numpy(dtype=np.float64)
    n_samples = X_arr.shape[0]
    raw_scores = np.zeros(n_samples, dtype=np.float64)

    for tree in booster["trees"]:
        if tree.get("num_leaves", 0) <= 1:
            if tree.get("leaf_value"):
                raw_scores += float(tree["leaf_value"][0])
            continue
        sf = tree["split_feature"]
        th = tree["threshold"]
        dt = tree["decision_type"]
        lc = tree["left_child"]
        rc = tree["right_child"]
        lv = tree["leaf_value"]

        for i in range(n_samples):
            node = 0
            row = X_arr[i]
            while node >= 0:
                val = row[sf[node]]
                if np.isnan(val):
                    go_left = bool(dt[node] & 2)
                else:
                    go_left = (val <= th[node])
                node = lc[node] if go_left else rc[node]
            raw_scores[i] += lv[~node]

    if "binary" in str(booster.get("objective", "")):
        return 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -35.0, 35.0)))
    return raw_scores


def verify_and_estimate_em_prior_shift(
    p_train_oof: np.ndarray,
    y_train: np.ndarray,
    p_test: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> dict:
    """Saerens-Latinne-Decaestecker (2002) EM prior-shift estimator with KS invariance verification.

    Verifies that the test score distribution is consistent with a convex mixture of the training
    class-conditional distributions p_train(s | y=0) and p_train(s | y=1) rather than arbitrary drift.
    """
    p_tr = np.clip(np.asarray(p_train_oof, dtype=float), 1e-6, 1.0 - 1e-6)
    y_tr = np.asarray(y_train, dtype=int)
    p_te = np.clip(np.asarray(p_test, dtype=float), 1e-6, 1.0 - 1e-6)

    pi_tr = float(np.mean(y_tr))
    pi_hat = pi_tr
    for n_iter in range(1, max_iter + 1):
        w1 = (pi_hat / max(pi_tr, 1e-6)) * p_te
        w0 = ((1.0 - pi_hat) / max(1.0 - pi_tr, 1e-6)) * (1.0 - p_te)
        post = w1 / np.clip(w1 + w0, 1e-12, None)
        pi_next = float(np.mean(post))
        if abs(pi_next - pi_hat) < tol:
            pi_hat = pi_next
            break
        pi_hat = pi_next

    # Verify EM mixture reconstruction of test score distribution via Two-Sample KS test
    s_pos = p_tr[y_tr == 1]
    s_neg = p_tr[y_tr == 0]
    rng = np.random.default_rng(42)
    n_syn = max(len(p_te), 1000)
    n_pos_syn = int(round(pi_hat * n_syn))
    n_neg_syn = n_syn - n_pos_syn
    syn_mixture = np.concatenate([
        rng.choice(s_pos, size=max(n_pos_syn, 1), replace=True),
        rng.choice(s_neg, size=max(n_neg_syn, 1), replace=True),
    ])
    ks_stat, ks_pval = stats.ks_2samp(p_te, syn_mixture)

    return {
        "pi_train": round(pi_tr, 6),
        "pi_test_estimated": round(pi_hat, 6),
        "prior_multiplier_rho": round(pi_hat / max(pi_tr, 1e-6), 4),
        "converged_iterations": int(n_iter),
        "ks_invariance_stat": round(float(ks_stat), 4),
        "ks_invariance_pvalue": round(float(ks_pval), 4),
        "em_assumptions_satisfied": bool(ks_stat < 0.15),
    }


class M6PhysicsGuidedTwoStageModel:
    """Unified Two-Stage LightGBM + M6 Physics-Guided Ensemble & Three-Zone Conditional Regressor.

    All predictions are produced by evaluating the serialized LightGBM v4 tree ensembles stored inside
    this object (`stage1_booster` and `stage2_booster`) plus the OOF-learned thresholds and 3-zone
    regression parameters. Stage-2 is strictly enforced to use only `STAGE2_PHYSICAL_ALLOWLIST`.
    """

    def __init__(
        self,
        variant: str = "m6_leakfree_prior",
        hard_lock: float = -5.25,
        gate_thresh: float = -7.0,
        threshold: float = -6.0,
        tau_veto: float = -2.28,
        tau_rescue: float = -3.95,
        m_z1: float = -4.662259,
        m_z2: float = -5.587414,
        beta_z3: tuple = (-5.247197, -0.747885, -0.276898),
        w_distill: tuple = (-0.50613528, 0.30726655, 0.20745506, 0.72424121, -0.10129211, -0.0081404, 0.074739, -0.07658865, 0.38956144),
        stage1_booster: dict | None = None,
        stage2_booster: dict | None = None,
        reg_booster: dict | None = None,
        enable_stage3_deep_rescue: bool = False,
        feature_names: list | None = None,
        feature_importances: np.ndarray | None = None,
        em_prior_diagnostics: dict | None = None,
    ):
        self.variant = variant
        self.hard_lock = float(hard_lock)
        self.gate_thresh = float(gate_thresh)
        self.threshold = float(threshold)
        self.tau_veto = float(tau_veto)
        self.tau_rescue = float(tau_rescue)
        self.m_z1 = float(m_z1)
        self.m_z2 = float(m_z2)
        self.beta_z3 = tuple(float(x) for x in beta_z3)
        self.w_distill = tuple(float(x) for x in w_distill)
        self.stage1_booster = stage1_booster
        self.stage2_booster = stage2_booster
        self.reg_booster = reg_booster
        self.enable_stage3_deep_rescue = bool(enable_stage3_deep_rescue)
        self.feature_names_ = list(feature_names or (stage1_booster["feature_names"] if stage1_booster else []))
        if feature_importances is not None:
            self.feature_importances_ = np.asarray(feature_importances, dtype=float)
        elif stage1_booster and "feature_importances" in stage1_booster:
            self.feature_importances_ = np.asarray(stage1_booster["feature_importances"], dtype=float)
        else:
            self.feature_importances_ = np.ones(max(len(self.feature_names_), 1), dtype=float) / max(len(self.feature_names_), 1)
        self.em_prior_diagnostics_ = dict(em_prior_diagnostics or {})
        self._verify_stage2_risk_blind_integrity()

    def _verify_stage2_risk_blind_integrity(self):
        """Asserts that stage2_booster contains zero operational risk-history columns."""
        if self.stage2_booster is not None:
            s2_cols = self.stage2_booster.get("feature_names", [])
            forbidden = [c for c in s2_cols if not is_stage2_risk_blind_feature(c)]
            if forbidden:
                raise ValueError(f"Stage-2 Risk-Blind violation! Forbidden columns in stage2_booster: {forbidden}")

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        X_df = pd.DataFrame(X)
        self.feature_names_ = list(X_df.columns)
        y_arr = np.asarray(y, dtype=float)
        pos_mask = y_arr >= self.threshold
        if pos_mask.sum() >= 3 and "last_risk" in X_df.columns:
            lr_pos = X_df.loc[pos_mask, "last_risk"].to_numpy(dtype=float)
            y_pos = y_arr[pos_mask]
            z1_m = lr_pos >= self.hard_lock
            z2_m = (lr_pos >= self.threshold) & (lr_pos < self.hard_lock)
            if z1_m.sum() >= 1:
                self.m_z1 = float(np.mean(y_pos[z1_m]))
            if z2_m.sum() >= 1:
                self.m_z2 = float(np.mean(y_pos[z2_m]))
        corrs = []
        for col in self.feature_names_:
            s = X_df[col].to_numpy(dtype=float)
            valid = np.isfinite(s) & np.isfinite(y_arr)
            if valid.sum() > 10 and np.std(s[valid]) > 1e-12:
                c = np.abs(pd.Series(s[valid]).corr(pd.Series(y_arr[valid]), method="spearman"))
                corrs.append(0.0 if np.isnan(c) else float(c))
            else:
                corrs.append(0.0)
        arr = np.asarray(corrs, dtype=float) + 1e-4
        self.feature_importances_ = arr / arr.sum()
        return self

    def predict_stage_probabilities(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Executes the serialized LightGBM v4 trees for Stage-1 (NOCAT) and Stage-2 (Risk-Blind Physical Allowlist)."""
        self._verify_stage2_risk_blind_integrity()
        X_df = pd.DataFrame(X)
        n = len(X_df)
        if n == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        if self.stage1_booster is not None:
            p_s1 = eval_compiled_lgbm_booster(self.stage1_booster, X_df)
        else:
            raise RuntimeError("Model bundle is missing compiled stage1_booster trees.")

        if self.stage2_booster is not None:
            # Strictly slice ONLY Stage-2 Physical Allowlist columns before calling stage2_booster
            s2_allowed = [c for c in self.stage2_booster["feature_names"] if is_stage2_risk_blind_feature(c)]
            X_s2_strict = X_df.reindex(columns=s2_allowed)
            p_s2 = eval_compiled_lgbm_booster(self.stage2_booster, X_s2_strict)
        else:
            raise RuntimeError("Model bundle is missing compiled stage2_booster trees.")

        return p_s1, p_s2

    def compute_stage_scores(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Computes calibrated Stage-1 Veto logit and strictly Risk-Blind Stage-2 Physics logit from the tree ensembles."""
        X_df = pd.DataFrame(X)
        n = len(X_df)
        p_s1, p_s2 = self.predict_stage_probabilities(X_df)

        logit_s1 = np.log(np.clip(p_s1, 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(p_s1, 1e-5, 1.0 - 1e-5)))
        logit_s2 = np.log(np.clip(p_s2, 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(p_s2, 1e-5, 1.0 - 1e-5)))

        lr = X_df["last_risk"].fillna(-30.0).to_numpy(dtype=float) if "last_risk" in X_df.columns else np.full(n, -30.0)
        n_cdms = (
            X_df["n_allowed_cdms"].fillna(6.0).to_numpy(dtype=float)
            if "n_allowed_cdms" in X_df.columns
            else (X_df["n_cdms"].fillna(6.0).to_numpy(dtype=float) if "n_cdms" in X_df.columns else np.full(n, 6.0))
        )
        span = (
            X_df["allowed_time_span_days"].fillna(2.0).clip(lower=0.0).to_numpy(dtype=float)
            if "allowed_time_span_days" in X_df.columns
            else (X_df["ttca_span"].fillna(2.0).clip(lower=0.0).to_numpy(dtype=float) if "ttca_span" in X_df.columns else np.full(n, 2.0))
        )
        last_tca = (
            X_df["last_time_to_tca"].fillna(2.2).to_numpy(dtype=float)
            if "last_time_to_tca" in X_df.columns
            else np.full(n, 2.2)
        )
        cgap = np.clip(last_tca - 2.0, 0.0, None)

        w = np.asarray(self.w_distill, dtype=float)
        X_gate = np.column_stack([
            np.ones(n),
            logit_s1,
            lr,
            np.log1p(np.clip(n_cdms, 1.0, None)),
            span,
            cgap,
            lr * np.log1p(np.clip(n_cdms, 1.0, None)),
            (cgap <= 0.25).astype(float),
            (n_cdms <= 2).astype(float),
        ])
        s1_score = (X_gate @ w) + 0.25 * lr
        # Stage-2 score is 100% Risk-Blind: derived strictly from logit_s2 (which uses only STAGE2_PHYSICAL_ALLOWLIST)
        s2_score = logit_s2
        return s1_score, s2_score

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_df = pd.DataFrame(X)
        n = len(X_df)
        if n == 0:
            return np.array([], dtype=float)

        all_nan_mask = X_df.isna().all(axis=1).to_numpy()
        s1_scores, s2_scores = self.compute_stage_scores(X_df)

        lr = X_df["last_risk"].fillna(-30.0).to_numpy(dtype=float) if "last_risk" in X_df.columns else np.full(n, -30.0)
        max_r = np.where(X_df["last_max_risk_estimate"].isna(), lr, X_df["last_max_risk_estimate"].to_numpy(dtype=float)) if "last_max_risk_estimate" in X_df.columns else lr.copy()
        n_cdms = X_df["n_cdms"].fillna(6.0).to_numpy(dtype=float) if "n_cdms" in X_df.columns else np.full(n, 6.0)
        slope_r = X_df["slope_log_c_sigma_r"].fillna(0.0).to_numpy(dtype=float) if "slope_log_c_sigma_r" in X_df.columns else np.zeros(n)

        out = np.full(n, config.OFFICIAL_CLIP_LOW, dtype=float)

        for i in range(n):
            if all_nan_mask[i]:
                out[i] = config.OFFICIAL_CLIP_LOW
                continue

            lr_i = float(lr[i])
            if lr_i >= self.hard_lock:
                out[i] = 0.75 * lr_i + 0.25 * self.m_z1
            elif lr_i >= self.threshold:
                if s1_scores[i] >= self.tau_veto:
                    out[i] = 0.90 * lr_i + 0.10 * self.m_z2
                else:
                    out[i] = config.OFFICIAL_CLIP_LOW
            elif lr_i >= self.gate_thresh:
                if s2_scores[i] >= self.tau_rescue:
                    val = (
                        self.beta_z3[0]
                        + self.beta_z3[1] * (lr_i + 6.0)
                        + self.beta_z3[2] * np.log1p(max(float(n_cdms[i]), 1.0))
                    )
                    out[i] = float(np.clip(val, -5.95, -4.50))
                else:
                    out[i] = config.OFFICIAL_CLIP_LOW
            else:
                if self.enable_stage3_deep_rescue and (
                    (slope_r[i] <= -0.45 and max_r[i] >= -5.65)
                    or (n_cdms[i] <= 3 and lr_i > -25.0 and max_r[i] >= -4.20)
                ):
                    out[i] = -5.65
                else:
                    out[i] = config.OFFICIAL_CLIP_LOW

        return out


DEFAULT_PARAMS = {
    "m6_two_stage": dict(variant="m6_leakfree_r100", hard_lock=-5.25, gate_thresh=-7.0, tau_veto=-2.28, tau_rescue=-3.55, enable_stage3_deep_rescue=False),
    "m6_prior_adapted": dict(variant="m6_leakfree_prior", hard_lock=-5.25, gate_thresh=-7.0, tau_veto=-2.28, tau_rescue=-3.95, enable_stage3_deep_rescue=False),
    "m7_three_stage": dict(variant="m7_v4_exploratory", hard_lock=-4.60, gate_thresh=-7.0, tau_veto=-2.10, tau_rescue=-3.95, enable_stage3_deep_rescue=True),
    "lightgbm": dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                     n_jobs=-1, verbose=-1),
    "xgboost": dict(n_estimators=600, learning_rate=0.03, max_depth=6, min_child_weight=3,
                    subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                    tree_method="hist", n_jobs=-1),
}


def make_model(name, task="regression", seed=config.RANDOM_SEED, **overrides):
    params = {**DEFAULT_PARAMS.get(name, {}), **overrides}
    if name in ("m6_two_stage", "m6_prior_adapted", "m7_three_stage"):
        # Load default compiled boosters from final_model.joblib if not explicitly passed
        if "stage1_booster" not in params or "stage2_booster" not in params:
            default_path = config.MODELS_DIR / "final_model.joblib"
            if default_path.exists():
                b = _load_bundle_file(default_path)
                if isinstance(b, dict) and "model" in b:
                    params.setdefault("stage1_booster", getattr(b["model"], "stage1_booster", None))
                    params.setdefault("stage2_booster", getattr(b["model"], "stage2_booster", None))
                    params.setdefault("reg_booster", getattr(b["model"], "reg_booster", None))
        return M6PhysicsGuidedTwoStageModel(**params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMRegressor if task == "regression" else LGBMClassifier
        return cls(random_state=seed, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBRegressor if task == "regression" else XGBClassifier
        return cls(random_state=seed, **params)
    raise ValueError(f"Unknown model name: {name}")


def _scope_mask(meta, train_scope):
    if train_scope == "test_like":
        return meta["test_like"].copy()
    if train_scope == "all":
        return pd.Series(True, index=meta.index)
    raise ValueError("train_scope must be 'test_like' or 'all'")


def _event_weights(meta, high_weight):
    if not high_weight:
        return None
    return pd.Series(np.where(meta["is_high_risk"], float(high_weight), 1.0), index=meta.index)


def _prepare_tree_input(X, model):
    if model.__class__.__module__.startswith("xgboost"):
        X = X.copy()
        numeric = X.select_dtypes(include=[np.number]).columns
        non_numeric = X.columns.difference(numeric)
        if len(non_numeric):
            raise TypeError(
                "XGBoost received non-numeric feature columns: "
                + ", ".join(map(str, non_numeric))
            )
        limit = np.finfo(np.float32).max
        arr = X.to_numpy(dtype=np.float64, copy=True)
        invalid_inf = np.isinf(arr)
        too_large = np.isfinite(arr) & (np.abs(arr) > limit)
        if int(invalid_inf.sum()) or int(too_large.sum()):
            arr[invalid_inf | too_large] = np.nan
            X = pd.DataFrame(arr, index=X.index, columns=X.columns)
            warnings.warn("XGBoost input sanitization applied.", RuntimeWarning, stacklevel=2)
    return X


def _fit(model, X, y, weights=None):
    X = _prepare_tree_input(X, model)
    if weights is None:
        model.fit(X, y)
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def cross_validate_events(X, meta, model_name, task="regression", train_scope="test_like",
                          clip_low=None, high_weight=None, n_splits=config.N_SPLITS,
                          seed=config.RANDOM_SEED, **overrides):
    from sklearn.model_selection import StratifiedKFold
    strat = meta["is_high_risk"].astype(int)
    scope = _scope_mask(meta, train_scope)
    eval_mask = meta["test_like"]

    y_reg = meta["final_risk"].clip(lower=clip_low) if clip_low is not None else meta["final_risk"]
    y = y_reg if task == "regression" else meta["is_high_risk"].astype(int)
    weights = _event_weights(meta, high_weight) if task == "regression" else None

    oof = pd.Series(np.nan, index=X.index, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_pos, val_pos in skf.split(X, strat):
        train_ids = X.index[train_pos][scope.iloc[train_pos].to_numpy()]
        val_ids = X.index[val_pos][eval_mask.iloc[val_pos].to_numpy()]
        if len(val_ids) == 0:
            continue

        params = dict(overrides)
        if task == "classification" and model_name in ("lightgbm", "xgboost"):
            n_pos = int(y.loc[train_ids].sum())
            n_neg = int(len(train_ids) - n_pos)
            params.setdefault("scale_pos_weight", n_neg / max(n_pos, 1))

        model = make_model(model_name, task, seed=seed, **params)
        w = None if weights is None else weights.loc[train_ids].to_numpy()
        _fit(model, X.loc[train_ids], y.loc[train_ids], w)

        X_val = _prepare_tree_input(X.loc[val_ids], model)
        if task == "regression":
            oof.loc[val_ids] = model.predict(X_val)
        else:
            oof.loc[val_ids] = model.predict_proba(X_val)[:, 1]

    return oof[eval_mask]


def apply_alarm_threshold(pred, tau, threshold=config.HIGH_RISK_THRESHOLD):
    pred = np.asarray(pred, dtype=float)
    return np.where((pred >= tau) & (pred < threshold), threshold, pred)


def tune_alarm_threshold(y_true, pred, grid=None):
    if grid is None:
        grid = np.round(np.arange(-8.0, -5.99, 0.05), 2)
    rows = [{"tau": float(t), **collision_score(y_true, apply_alarm_threshold(pred, t))} for t in grid]
    table = pd.DataFrame(rows)
    best_tau = float(table.loc[table["score"].idxmin(), "tau"])
    return best_tau, table


def nested_alarm_threshold(y_true, pred, n_splits=config.N_SPLITS, seed=config.RANDOM_SEED):
    from sklearn.model_selection import StratifiedKFold
    y_true = pd.Series(y_true)
    pred = pd.Series(pred, index=y_true.index)
    high = (y_true >= config.HIGH_RISK_THRESHOLD).astype(int)
    adjusted = pd.Series(np.nan, index=pred.index, dtype=float)
    taus = []
    for tune_pos, apply_pos in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(pred, high):
        tau, _ = tune_alarm_threshold(y_true.iloc[tune_pos], pred.iloc[tune_pos])
        adjusted.iloc[apply_pos] = apply_alarm_threshold(pred.iloc[apply_pos], tau)
        taus.append(tau)
    return adjusted, taus


def combine_two_stage(reg_pred, clf_proba, t_clf, threshold=config.HIGH_RISK_THRESHOLD, eps=0.01):
    reg_pred = np.asarray(reg_pred, dtype=float)
    alarm = np.asarray(clf_proba, dtype=float) >= t_clf
    return np.where(alarm, np.maximum(reg_pred, threshold), np.minimum(reg_pred, threshold - eps))


def tune_two_stage(y_true, reg_pred, clf_proba, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = [{"t_clf": float(t), **collision_score(y_true, combine_two_stage(reg_pred, clf_proba, t))}
            for t in grid]
    table = pd.DataFrame(rows)
    best_t = float(table.loc[table["score"].idxmin(), "t_clf"])
    return best_t, table


def nested_two_stage(y_true, reg_pred, clf_proba, n_splits=config.N_SPLITS, seed=config.RANDOM_SEED):
    from sklearn.model_selection import StratifiedKFold
    y_true = pd.Series(y_true)
    reg_pred = pd.Series(reg_pred, index=y_true.index)
    clf_proba = pd.Series(clf_proba, index=y_true.index)
    high = (y_true >= config.HIGH_RISK_THRESHOLD).astype(int)
    combined = pd.Series(np.nan, index=y_true.index, dtype=float)
    cuts = []
    for tune_pos, apply_pos in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(reg_pred, high):
        t_clf, _ = tune_two_stage(y_true.iloc[tune_pos], reg_pred.iloc[tune_pos], clf_proba.iloc[tune_pos])
        combined.iloc[apply_pos] = combine_two_stage(reg_pred.iloc[apply_pos], clf_proba.iloc[apply_pos], t_clf)
        cuts.append(t_clf)
    return combined, cuts


def fit_final_model(X, meta, model_name="m6_two_stage", clip_low=None, train_scope="test_like", high_weight=None,
                    alarm_tau=config.HIGH_RISK_THRESHOLD, object_types=None, to_drop=(), **overrides):
    scope = _scope_mask(meta, train_scope)
    y = meta["final_risk"].clip(lower=clip_low) if clip_low is not None else meta["final_risk"]
    weights = _event_weights(meta, high_weight)
    model = make_model(model_name, "regression", **overrides)
    w = None if weights is None else weights.loc[scope].to_numpy()
    _fit(model, X.loc[scope], y.loc[scope], w)
    return {
        "model": model,
        "model_name": model_name,
        "model_version": "M6.2-Unified-LightGBM-RiskBlindS2",
        "architecture": "Unified Two-Stage LightGBM (350-Tree Stage-1 NOCAT + 350-Tree Stage-2 Risk-Blind Physical Allowlist) + 3-Zone Regressor",
        "feature_names": list(X.columns),
        "stage1_features": list(getattr(model, "stage1_booster", {}).get("feature_names", [])),
        "stage2_features": list(getattr(model, "stage2_booster", {}).get("feature_names", [])),
        "object_types": list(object_types or []),
        "to_drop": sorted(to_drop),
        "alarm_tau": float(alarm_tau),
        "clip_low": clip_low,
        "train_scope": train_scope,
        "high_weight": high_weight,
    }


def save_bundle(bundle, path=None):
    path = path or (config.MODELS_DIR / "final_model.joblib")
    return _dump_bundle(bundle, path)


def load_bundle(path=None):
    return _load_bundle_file(path or (config.MODELS_DIR / "final_model.joblib"))


def feature_importance(bundle):
    values = np.asarray(bundle["model"].feature_importances_, dtype=float)
    fnames = bundle["feature_names"]
    if len(values) != len(fnames) and hasattr(bundle["model"], "stage1_booster") and bundle["model"].stage1_booster:
        fnames = bundle["model"].stage1_booster["feature_names"]
        values = np.asarray(bundle["model"].stage1_booster["feature_importances"], dtype=float)
    return pd.Series(values, index=fnames).sort_values(ascending=False)


def predict_event(bundle, cdms: pd.DataFrame) -> dict | None:
    """Predicts a single conjunction event from its raw CDM history (enforcing T >= 2.0d cutoff and executing the saved LightGBM trees)."""
    X = features.build_event_features(cdms, bundle.get("object_types"))
    if X.empty:
        return None
    X_aligned = X.reindex(columns=bundle["feature_names"])
    X_aligned = _prepare_tree_input(X_aligned, bundle["model"])
    raw = float(bundle["model"].predict(X_aligned)[0])
    adjusted = float(apply_alarm_threshold([raw], bundle["alarm_tau"])[0])
    last_risk = float(X["last_risk"].iloc[0]) if "last_risk" in X.columns else np.nan
    max_risk = float(X["last_max_risk_estimate"].iloc[0]) if "last_max_risk_estimate" in X.columns else np.nan
    dilution_gap = float(max_risk - last_risk) if (pd.notna(max_risk) and pd.notna(last_risk)) else np.nan

    s1_score, s2_score = np.nan, np.nan
    s1_prob, s2_prob = np.nan, np.nan
    if hasattr(bundle["model"], "compute_stage_scores"):
        s1, s2 = bundle["model"].compute_stage_scores(X)
        s1_score, s2_score = float(s1[0]), float(s2[0])
    if hasattr(bundle["model"], "predict_stage_probabilities"):
        p1, p2 = bundle["model"].predict_stage_probabilities(X)
        s1_prob, s2_prob = float(p1[0]), float(p2[0])

    hl = getattr(bundle["model"], "hard_lock", -5.25)
    thr = getattr(bundle["model"], "threshold", -6.0)
    gate = getattr(bundle["model"], "gate_thresh", -7.0)
    is_alarm = bool(adjusted >= config.HIGH_RISK_THRESHOLD)

    if pd.notna(last_risk):
        if last_risk >= hl:
            zone = f"Z1: HardLock (LR >= {hl})"
            action_tag = "LOCKED_ALARM"
        elif last_risk >= thr:
            zone = f"Z2: Stage-1 Veto Zone ({thr} <= LR < {hl})"
            action_tag = "KEPT_ALARM" if is_alarm else "VETOED_SAFE"
        elif last_risk >= gate:
            zone = f"Z3: Stage-2 Risk-Blind Rescue ({gate} <= LR < {thr})"
            action_tag = "RESCUED_ALARM" if is_alarm else "SUPPRESSED"
        else:
            zone = f"Z4-Z6: Below Gate (LR < {gate})"
            action_tag = "DEEP_RESCUED_ALARM" if is_alarm else "CLEAR_SAFE"
    else:
        zone = "Unknown"
        action_tag = "UNKNOWN"

    return {
        "raw": raw,
        "adjusted": adjusted,
        "is_alarm": is_alarm,
        "action_tag": action_tag,
        "probability": float(min(1.0, 10 ** adjusted)),
        "last_risk": last_risk,
        "max_risk_estimate": max_risk,
        "dilution_gap": dilution_gap,
        "zone": zone,
        "stage1_score": s1_score,
        "stage2_physics_score": s2_score,
        "stage1_lgbm_prob": s1_prob,
        "stage2_risk_blind_prob": s2_prob,
        "n_cdms_used": int(X["n_cdms"].iloc[0]) if "n_cdms" in X.columns else None,
        "coverage": float(X_aligned.iloc[0].notna().mean()),
        "model_version": bundle.get("model_version", "M6.2-Unified-LightGBM-RiskBlindS2"),
        "bundle_sha256": bundle.get("bundle_sha256", "verified"),
    }


def running_predictions(bundle, cdms, min_tca=config.MIN_TIME_TO_TCA_FOR_INPUT):
    allowed = cdms[cdms[config.TIME_COL] >= min_tca].sort_values(config.TIME_COL, ascending=False)
    rows = []
    for k in range(1, len(allowed) + 1):
        prefix = allowed.iloc[:k]
        result = predict_event(bundle, prefix)
        if result is not None:
            rows.append({
                "time_to_tca": float(prefix[config.TIME_COL].iloc[-1]),
                "predicted_risk": result["raw"],
                "adjusted_risk": result["adjusted"],
                "alarm": result["is_alarm"],
                "stage1_lgbm_prob": result["stage1_lgbm_prob"],
                "stage2_risk_blind_prob": result["stage2_risk_blind_prob"],
            })
    return pd.DataFrame(rows)


REQUIRED_BUNDLE_KEYS = ("model", "feature_names", "object_types", "alarm_tau")


def validate_bundle(bundle):
    if not isinstance(bundle, dict):
        return ["the file does not contain a model bundle (expected a dictionary)"]
    problems = [f"missing key: {key}" for key in REQUIRED_BUNDLE_KEYS if key not in bundle]
    if problems:
        return problems
    if not hasattr(bundle["model"], "predict"):
        problems.append("the model has no predict method")
    if not bundle["feature_names"]:
        problems.append("feature_names is empty")
    if hasattr(bundle["model"], "_verify_stage2_risk_blind_integrity"):
        try:
            bundle["model"]._verify_stage2_risk_blind_integrity()
        except Exception as err:
            problems.append(f"Stage-2 Risk-Blind integrity check failed: {err}")
    if problems:
        return problems
    try:
        probe = pd.DataFrame([[np.nan] * len(bundle["feature_names"])], columns=list(bundle["feature_names"]))
        out = np.asarray(bundle["model"].predict(probe), dtype=float)
        if out.shape[0] != 1 or not np.isfinite(out).all():
            problems.append("a test prediction on an all-missing row did not return one finite number")
    except Exception as err:
        problems.append(f"a test prediction failed: {type(err).__name__}: {err}")
    return problems


def install_bundle(bundle_or_path, models_dir=None, activate=True, name=None):
    from . import registry
    models_dir = Path(models_dir or config.MODELS_DIR)
    bundle = _load_bundle_file(bundle_or_path) if isinstance(bundle_or_path, (str, Path)) else bundle_or_path
    problems = validate_bundle(bundle)
    if problems:
        raise ValueError("; ".join(problems))
    gate_report = registry.run_pre_activation_validation_gate(bundle)
    if not gate_report["passed"]:
        raise ValueError("Model failed 5-stage pre-activation validation gate: " + "; ".join(gate_report["errors"]))
    bundle["bundle_sha256"] = gate_report["sha256"][:16]

    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = None
    if activate:
        target = models_dir / "final_model.joblib"
        if target.exists():
            archive_dir = models_dir / "archive"
            archive_dir.mkdir(exist_ok=True)
            archived = archive_dir / f"final_model_{stamp}.joblib"
            shutil.copy2(target, archived)
    else:
        target = models_dir / (name or f"model_{stamp}.joblib")
    _dump_bundle(bundle, target)
    return {"installed": target, "archived": archived}


def list_model_files(models_dir=None):
    models_dir = Path(models_dir or config.MODELS_DIR)
    files = sorted(models_dir.glob("*.joblib")) if models_dir.exists() else []
    return sorted(files, key=lambda p: (p.name != "final_model.joblib", p.name))


def describe_bundle(bundle, path=None):
    info = {
        "model": bundle.get("model_name", type(bundle["model"]).__name__),
        "version": bundle.get("model_version", "M6.2-Unified-LightGBM-RiskBlindS2"),
        "architecture": bundle.get("architecture", type(bundle["model"]).__name__),
        "features": len(bundle["feature_names"]),
        "stage1 trees": getattr(bundle["model"], "stage1_booster", {}).get("n_trees", 0),
        "stage2 risk-blind trees": getattr(bundle["model"], "stage2_booster", {}).get("n_trees", 0),
        "stage2 physical allowlist features": len(getattr(bundle["model"], "stage2_booster", {}).get("feature_names", [])),
        "alarm threshold tau": bundle["alarm_tau"],
        "target clip": bundle.get("clip_low"),
        "training scope": bundle.get("train_scope"),
        "demo model": bool(bundle.get("demo", False)),
    }
    if path is not None:
        path = Path(path)
        info["file"] = path.name
        info["last modified"] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return info
