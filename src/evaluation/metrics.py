"""
Streaming evaluation and metrics tracking.

Implements the prequential metric accumulator required by the thesis
evaluation framework (Source of Truth §12).

Primary metric:    PR-AUC / Average Precision (essential for severe class imbalance)
Secondary metrics: ROC-AUC, Precision, Recall, F1, F2 (beta=2)

Design
------
- Predictions and labels are accumulated in-memory as simple lists.
- Metrics are computed on-demand via ``compute()`` using scikit-learn.
- The tracker is STATEFUL — it accumulates across all streaming steps.
- ``reset()`` clears all state for a fresh independent seed run.
- Per-window metrics (rolling evaluation) are supported via ``compute_window()``.
- No plotting functions live here.

NOTE: This module tracks predictions and latencies for offline metric
computation.  Real-time streaming metrics (e.g. ADWIN-based performance
monitors) are separate from this tracker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MetricSnapshot:
    """Point-in-time metric values computed over an accumulated set of predictions.

    Attributes
    ----------
    n_samples:
        Number of predictions accumulated.
    n_positive:
        Number of positive (fraud) ground-truth labels.
    pr_auc:
        Precision-Recall AUC (Average Precision).  ``None`` if fewer than
        2 distinct labels have been seen.
    roc_auc:
        ROC AUC.  ``None`` if fewer than 2 distinct labels have been seen.
    precision:
        Precision at 0.5 threshold.
    recall:
        Recall at 0.5 threshold.
    f1:
        F1-score at 0.5 threshold.
    f2:
        F-beta score (beta=2) at 0.5 threshold.
    """

    n_samples: int = 0
    n_positive: int = 0
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    f2: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        """Serialise to plain dict."""
        return {
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f2": self.f2,
        }


class StreamingMetricsTracker:
    """Accumulates prequential predictions and labels for offline metric computation.

    Metrics are evaluated on all accumulated data since the last ``reset()``
    call.  No streaming approximation is used — this is an exact computation
    over the held prediction list, reflecting the prequential test-then-train
    protocol.

    Usage
    -----
    Call ``update(y_true, y_prob)`` immediately after obtaining a prediction
    and revealing the true label.  Call ``compute()`` at any time to obtain
    the current snapshot of metrics.

    Parameters
    ----------
    track_latency:
        If ``True``, record per-prediction inference latency via
        ``update_latency(duration_s)``.

    Examples
    --------
    >>> tracker = StreamingMetricsTracker()
    >>> tracker.update(y_true=1, y_prob=0.9)
    >>> tracker.update(y_true=0, y_prob=0.1)
    >>> snap = tracker.compute()
    >>> snap.n_samples
    2
    """

    def __init__(self, track_latency: bool = True) -> None:
        self._track_latency = track_latency
        self._y_true: List[int] = []
        self._y_prob: List[float] = []      # P(fraud=1)
        self._latencies_s: List[float] = []  # inference latency per prediction

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update(self, y_true: int, y_prob: float) -> None:
        """Record one prequential prediction result.

        Parameters
        ----------
        y_true:
            Ground-truth label (0 or 1), revealed AFTER the prediction.
        y_prob:
            Predicted probability of class 1 (fraud), produced BEFORE y_true
            was revealed.
        """
        self._y_true.append(int(y_true))
        self._y_prob.append(float(y_prob))

    def update_latency(self, duration_s: float) -> None:
        """Record inference latency for a single prediction.

        Parameters
        ----------
        duration_s:
            Wall-clock time (seconds) taken to produce one prediction.
        """
        if self._track_latency:
            self._latencies_s.append(float(duration_s))

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute(self) -> MetricSnapshot:
        """Compute metrics over all accumulated predictions.

        Returns
        -------
        :class:`MetricSnapshot` — ``None`` fields indicate insufficient data
        (e.g. only one class seen so far).
        """
        n = len(self._y_true)
        if n == 0:
            return MetricSnapshot()

        y_true_arr = np.array(self._y_true, dtype=int)
        y_prob_arr = np.array(self._y_prob, dtype=float)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)

        n_positive = int(y_true_arr.sum())

        snap = MetricSnapshot(n_samples=n, n_positive=n_positive)

        # Need at least one positive and one negative label for AUC metrics
        if n_positive == 0 or n_positive == n:
            # Can still compute precision/recall/f1 at threshold
            snap.precision, snap.recall, snap.f1, snap.f2 = self._threshold_metrics(
                y_true_arr, y_pred_arr
            )
            return snap

        try:
            from sklearn.metrics import (  # type: ignore[import]
                average_precision_score,
                roc_auc_score,
            )
            snap.pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
            snap.roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except Exception:  # pragma: no cover
            pass

        snap.precision, snap.recall, snap.f1, snap.f2 = self._threshold_metrics(
            y_true_arr, y_pred_arr
        )
        return snap

    def compute_window(self, last_n: int) -> MetricSnapshot:
        """Compute metrics over the most recent ``last_n`` predictions.

        Parameters
        ----------
        last_n:
            Number of most recent predictions to include.

        Returns
        -------
        :class:`MetricSnapshot` over the window.
        """
        if last_n <= 0:
            return MetricSnapshot()
        window_true = self._y_true[-last_n:]
        window_prob = self._y_prob[-last_n:]
        # Create a temporary tracker and delegate
        tmp = StreamingMetricsTracker(track_latency=False)
        for yt, yp in zip(window_true, window_prob):
            tmp.update(yt, yp)
        return tmp.compute()

    # ------------------------------------------------------------------
    # Latency percentiles
    # ------------------------------------------------------------------

    def latency_percentiles(self) -> Dict[str, float]:
        """Return inference latency percentiles (p50, p95, p99) in seconds.

        Returns an empty dict if no latencies have been recorded.
        """
        if not self._latencies_s:
            return {}
        arr = np.array(self._latencies_s)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "n": len(arr),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated predictions and latencies."""
        self._y_true = []
        self._y_prob = []
        self._latencies_s = []

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of predictions accumulated since last reset."""
        return len(self._y_true)

    @property
    def n_positive(self) -> int:
        """Number of positive (fraud) ground-truth labels accumulated."""
        return sum(self._y_true)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _threshold_metrics(
        y_true: "np.ndarray",
        y_pred: "np.ndarray",
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Compute precision, recall, F1, F2 from hard predictions."""
        try:
            from sklearn.metrics import (  # type: ignore[import]
                precision_score,
                recall_score,
                f1_score,
                fbeta_score,
            )
            zero_div = 0
            precision = float(precision_score(y_true, y_pred, zero_division=zero_div))
            recall = float(recall_score(y_true, y_pred, zero_division=zero_div))
            f1 = float(f1_score(y_true, y_pred, zero_division=zero_div))
            f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=zero_div))
            return precision, recall, f1, f2
        except Exception:  # pragma: no cover
            return None, None, None, None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StreamingMetricsTracker(n_samples={self.n_samples}, "
            f"n_positive={self.n_positive})"
        )
