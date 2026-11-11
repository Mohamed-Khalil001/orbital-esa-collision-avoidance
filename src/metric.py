"""Competition metric: L = MSE_high / F2 (lower is better)."""
import numpy as np

from . import config


def collision_score(y_true, y_pred, threshold=config.HIGH_RISK_THRESHOLD, clip_low=None):
    """Competition score L = MSE_high / F2 (lower is better)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if clip_low is not None:
        y_pred = np.maximum(y_pred, clip_low)
    if np.isnan(y_pred).any():
        raise ValueError("y_pred contains NaN: fill or fix the predictions first.")

    true_high = y_true >= threshold
    pred_high = y_pred >= threshold

    if true_high.sum() == 0:
        raise ValueError("No high-risk events in y_true: the score is undefined.")

    tp = int((true_high & pred_high).sum())
    fp = int((~true_high & pred_high).sum())
    fn = int((true_high & ~pred_high).sum())

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f2 = float((5.0 * prec * rec) / (4.0 * prec + rec)) if (4.0 * prec + rec) > 0 else 0.0
    mse_high = float(np.mean((y_true[true_high] - y_pred[true_high]) ** 2))

    score = float(mse_high / f2) if f2 > 0 else np.inf
    return {"score": score, "mse_high": mse_high, "f2": f2}


def esa_loss(y_true, y_pred, threshold=config.HIGH_RISK_THRESHOLD, clip_low=config.OFFICIAL_CLIP_LOW):
    """Official-convention score: predictions are clipped at -6.001 before MSE_high and F2 are computed."""
    return collision_score(y_true, y_pred, threshold, clip_low=clip_low)


def evaluate(y_true, y_pred, threshold=config.HIGH_RISK_THRESHOLD, clip_low=None):
    """Same as collision_score plus precision, recall and counts (for reports)."""
    result = collision_score(y_true, y_pred, threshold, clip_low=clip_low)
    if clip_low is not None:
        y_pred = np.maximum(np.asarray(y_pred, dtype=float), clip_low)
    true_high = (np.asarray(y_true, dtype=float) >= threshold)
    pred_high = (np.asarray(y_pred, dtype=float) >= threshold)
    tp = int((true_high & pred_high).sum())
    fp = int((~true_high & pred_high).sum())
    fn = int((true_high & ~pred_high).sum())
    result["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    result["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    result["n_events"] = int(len(true_high))
    result["n_high"] = int(true_high.sum())
    return result


def bootstrap_ci(y_true, y_pred, n_boot=300, seed=0, alpha=0.10, threshold=config.HIGH_RISK_THRESHOLD):
    """Resample events with replacement to see how much the score moves by chance alone."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if (y_true[idx] >= threshold).sum() == 0:
            continue
        scores.append(collision_score(y_true[idx], y_pred[idx], threshold)["score"])
    scores = np.asarray(scores, dtype=float)
    return {
        "median": float(np.quantile(scores, 0.5, method="nearest")),
        "low": float(np.quantile(scores, alpha / 2, method="nearest")),
        "high": float(np.quantile(scores, 1 - alpha / 2, method="nearest")),
        "n_valid": int(len(scores)),
    }
