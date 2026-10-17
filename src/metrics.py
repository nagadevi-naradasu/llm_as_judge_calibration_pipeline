"""Statistical metrics engine: TPR, TNR, agreement, Cohen's kappa + bootstrap CIs.

Class convention (see ``data/rubric.md``): **FAIL = positive class (label 0)**.
TPR therefore answers *"what share of the human-labelled failures did the judge
catch?"* and TNR answers *"what share of the human-labelled passes did the judge
leave alone?"*. Raw agreement is reported too, but it is the metric that hides
class imbalance, so it is never interpreted on its own.

Confidence intervals come from the non-parametric bootstrap (N=1000 by default)
because TPR/TNR/kappa on a ~110-row test split are not normally distributed.
Everything is seeded, so a given dataset always yields identical numbers.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

MetricFn = Callable[[np.ndarray, np.ndarray], float]

POS_LABEL = 0  # Fail is the positive class.
NEG_LABEL = 1  # Pass.


def tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Positive Rate for the FAIL class: fail detected as fail."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos = y_true == POS_LABEL
    if pos.sum() == 0:
        return float("nan")
    return float((y_pred[pos] == POS_LABEL).mean())


def tnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Negative Rate for the PASS class: pass left as pass."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    neg = y_true == NEG_LABEL
    if neg.sum() == 0:
        return float("nan")
    return float((y_pred[neg] == NEG_LABEL).mean())


def agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return the 2x2 confusion matrix named from the FAIL=positive view."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL]).ravel()
    return {"tn_pass_kept": int(tn), "fp_pass_called_fail": int(fp),
            "fn_fail_missed": int(fn), "tp_fail_caught": int(tp)}


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: MetricFn,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap a metric.

    Params:
        y_true: array of human labels (1 pass / 0 fail)
        y_pred: array of judge predictions (1 pass / 0 fail)
        metric_fn: callable(y_true, y_pred) -> float
        n_iterations: number of resamples (default 1000)
        seed: RNG seed for reproducibility
    Returns:
        (mean_score, ci_lower_2.5, ci_upper_97.5)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    scores = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)  # sample indices with replacement
        s = metric_fn(y_true[idx], y_pred[idx])
        scores[i] = s

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite))
    lower = float(np.percentile(finite, 2.5))
    upper = float(np.percentile(finite, 97.5))
    return mean, lower, upper


def compute_metrics(
    y_true,
    y_pred,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Full metric row (means + 95% CIs) ready for ``results/metrics.csv``."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tpr_mean, tpr_lo, tpr_hi = bootstrap_metric(y_true, y_pred, tpr, n_iterations, seed)
    tnr_mean, tnr_lo, tnr_hi = bootstrap_metric(y_true, y_pred, tnr, n_iterations, seed + 1)
    agr_mean, agr_lo, agr_hi = bootstrap_metric(y_true, y_pred, agreement, n_iterations, seed + 2)
    kap_mean, kap_lo, kap_hi = bootstrap_metric(y_true, y_pred, kappa, n_iterations, seed + 3)

    def clip(x: float) -> float:
        return float(min(1.0, max(0.0, x))) if np.isfinite(x) else float("nan")

    return {
        "n": int(len(y_true)),
        "n_fail": int((y_true == POS_LABEL).sum()),
        "tpr_mean": clip(tpr_mean),
        "tpr_ci_low": clip(tpr_lo),
        "tpr_ci_high": clip(tpr_hi),
        "tnr_mean": clip(tnr_mean),
        "tnr_ci_low": clip(tnr_lo),
        "tnr_ci_high": clip(tnr_hi),
        "agreement": clip(agr_mean),
        "agreement_ci_low": clip(agr_lo),
        "agreement_ci_high": clip(agr_hi),
        "kappa_mean": kap_mean,
        "kappa_ci_low": kap_lo,
        "kappa_ci_high": kap_hi,
        **confusion(y_true, y_pred),
    }
