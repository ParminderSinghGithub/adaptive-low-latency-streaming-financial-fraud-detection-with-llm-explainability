"""
Event-Centered Concept Drift Evaluation Metrics.

Computes discrete event-centered metrics around known ground-truth drift onset
t_drift for Objective 2 (Experiment E3) and Objective 3 (Experiment E4).

Metrics evaluated:
1. pre_drift_pr_auc:             Average Precision over [t_drift - horizon, t_drift)
2. post_drift_pr_auc:            Average Precision over [t_drift, t_drift + horizon)
3. delta_pr_auc:                 post_drift_pr_auc - pre_drift_pr_auc
4. degradation_slope:            Linear slope of rolling PR-AUC across post-drift horizon
5. detection_delay:              Stream steps from t_drift to first subsequent detector alarm
6. false_alarms_before_drift:    Drift detector alarm count strictly before t_drift
7. recovery_time:                Steps after t_drift to return within tolerance of pre_drift_pr_auc
8. adaptation_count_post_drift:  Number of model adaptation events triggered at or after t_drift
9. total_retrain_time_post_s:    Total retrain duration in seconds post-drift

Preserves existing O1 metrics semantics without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.drift.injection import E3DriftConfig


@dataclass
class EventCenteredMetrics:
    """Container for discrete event-centered evaluation metrics."""
    regime: str
    t_drift: int
    horizon: int
    pre_drift_pr_auc: Optional[float]
    post_drift_pr_auc: Optional[float]
    delta_pr_auc: Optional[float]
    degradation_slope: Optional[float]
    detection_delay: Optional[int]
    false_alarms_before_drift: int
    recovery_time: Optional[int]
    adaptation_count_post_drift: int
    total_retrain_time_post_s: float
    full_stream_pr_auc: Optional[float]
    n_pre_samples: int = 0
    n_post_samples: int = 0
    segment_evaluated: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "regime": self.regime,
            "t_drift": self.t_drift,
            "horizon": self.horizon,
            "pre_drift_pr_auc": self.pre_drift_pr_auc,
            "post_drift_pr_auc": self.post_drift_pr_auc,
            "delta_pr_auc": self.delta_pr_auc,
            "degradation_slope": self.degradation_slope,
            "detection_delay": self.detection_delay,
            "false_alarms_before_drift": self.false_alarms_before_drift,
            "recovery_time": self.recovery_time,
            "adaptation_count_post_drift": self.adaptation_count_post_drift,
            "total_retrain_time_post_s": round(self.total_retrain_time_post_s, 3),
            "full_stream_pr_auc": self.full_stream_pr_auc,
            "n_pre_samples": self.n_pre_samples,
            "n_post_samples": self.n_post_samples,
            "segment_evaluated": self.segment_evaluated,
        }


def _safe_ap(y_true: List[int], y_prob: List[float]) -> Optional[float]:
    """Calculate Average Precision safely, returning None if class diversity < 2."""
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return None


def compute_event_centered_metrics(
    records: List[Any],
    drift_event_log: List[Dict[str, Any]],
    adaptation_log: List[Dict[str, Any]],
    drift_config: E3DriftConfig,
    horizon: int = 5000,
    recovery_window: int = 1000,
    recovery_tolerance: float = 0.05,
    filter_segment: Optional[str] = None,
) -> EventCenteredMetrics:
    """Compute event-centered concept drift metrics from a completed streaming run.

    Parameters
    ----------
    records:
        List of StreamingRecord objects from RunResult.
    drift_event_log:
        List of drift alarm dictionaries from RunResult.
    adaptation_log:
        List of model adaptation dictionaries from RunResult.
    drift_config:
        E3DriftConfig object containing t_drift, regime, etc.
    horizon:
        Evaluation horizon H (in transactions) before and after t_drift.
    recovery_window:
        Window size for computing post-drift rolling PR-AUC for recovery tracking.
    recovery_tolerance:
        Allowed margin (pre_drift_pr_auc - tolerance) to declare recovery.
    filter_segment:
        Optional segment key string to restrict evaluation to a specific segment.

    Returns
    -------
    EventCenteredMetrics
    """
    t_drift = drift_config.t_drift
    regime = drift_config.regime.value if hasattr(drift_config.regime, "value") else str(drift_config.regime)

    # 1. Filter records by segment if specified
    if filter_segment is not None:
        eval_records = [r for r in records if getattr(r, "segment_key", None) == filter_segment]
    else:
        eval_records = records

    # 2. Pre-drift window slice: [max(0, t_drift - horizon), t_drift)
    t_pre_start = max(0, t_drift - horizon)
    pre_slice = [r for r in eval_records if t_pre_start <= r.tx_index < t_drift]
    y_pre_true = [r.y_true for r in pre_slice]
    y_pre_prob = [r.y_prob for r in pre_slice]
    pre_pr_auc = _safe_ap(y_pre_true, y_pre_prob)

    # 3. Post-drift window slice: [t_drift, t_drift + horizon)
    t_post_end = t_drift + horizon
    post_slice = [r for r in eval_records if t_drift <= r.tx_index < t_post_end]
    y_post_true = [r.y_true for r in post_slice]
    y_post_prob = [r.y_prob for r in post_slice]
    post_pr_auc = _safe_ap(y_post_true, y_post_prob)

    # 4. Delta PR-AUC
    delta_pr_auc = (
        round(post_pr_auc - pre_pr_auc, 5)
        if (post_pr_auc is not None and pre_pr_auc is not None)
        else None
    )

    # 5. Full-stream PR-AUC
    y_all_true = [r.y_true for r in eval_records]
    y_all_prob = [r.y_prob for r in eval_records]
    full_pr_auc = _safe_ap(y_all_true, y_all_prob)

    # 6. Degradation Slope (linear fit on rolling sub-windows across post-drift slice)
    degradation_slope = None
    if len(post_slice) >= 200:
        step = max(50, len(post_slice) // 20)
        rolling_x = []
        rolling_y = []
        for i in range(step, len(post_slice) + 1, step):
            sub_true = [r.y_true for r in post_slice[:i]]
            sub_prob = [r.y_prob for r in post_slice[:i]]
            ap_val = _safe_ap(sub_true, sub_prob)
            if ap_val is not None:
                rolling_x.append(post_slice[i - 1].tx_index - t_drift)
                rolling_y.append(ap_val)
        if len(rolling_x) >= 3:
            x_arr = np.array(rolling_x, dtype=float)
            y_arr = np.array(rolling_y, dtype=float)
            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)
            denom = np.sum((x_arr - x_mean) ** 2)
            if denom > 1e-12:
                slope = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / denom)
                degradation_slope = round(slope, 8)

    # 7. Detection Delay & False Alarms
    # Filter drift events to matching detector scope if segment specified
    filtered_drift_events = []
    for d in drift_event_log:
        seg = d.get("segment_key")
        if filter_segment is None or seg is None or seg == filter_segment:
            filtered_drift_events.append(d)

    false_alarms = sum(1 for d in filtered_drift_events if d.get("tx_index", 0) < t_drift)

    detection_delay: Optional[int] = None
    post_drift_alarms = [d for d in filtered_drift_events if d.get("tx_index", 0) >= t_drift]
    if post_drift_alarms:
        first_alarm_t = min(d["tx_index"] for d in post_drift_alarms)
        detection_delay = int(first_alarm_t - t_drift)

    # 8. Recovery Time
    # First transaction offset after t_drift where rolling PR-AUC reaches >= (pre_pr_auc - tolerance)
    recovery_time: Optional[int] = None
    if pre_pr_auc is not None and len(post_slice) >= recovery_window:
        target_threshold = pre_pr_auc - recovery_tolerance
        window_true: List[int] = []
        window_prob: List[float] = []

        for r in post_slice:
            window_true.append(r.y_true)
            window_prob.append(r.y_prob)
            if len(window_true) > recovery_window:
                window_true.pop(0)
                window_prob.pop(0)

            if len(window_true) >= recovery_window:
                cur_ap = _safe_ap(window_true, window_prob)
                if cur_ap is not None and cur_ap >= target_threshold:
                    recovery_time = int(r.tx_index - t_drift)
                    break

    # 9. Post-Drift Adaptation Metrics
    post_adaptations = [a for a in adaptation_log if a.get("tx_index", 0) >= t_drift]
    adapt_count_post = len(post_adaptations)
    total_retrain_time_post = float(sum(a.get("duration_s", 0.0) for a in post_adaptations))

    return EventCenteredMetrics(
        regime=regime,
        t_drift=t_drift,
        horizon=horizon,
        pre_drift_pr_auc=round(pre_pr_auc, 5) if pre_pr_auc is not None else None,
        post_drift_pr_auc=round(post_pr_auc, 5) if post_pr_auc is not None else None,
        delta_pr_auc=delta_pr_auc,
        degradation_slope=degradation_slope,
        detection_delay=detection_delay,
        false_alarms_before_drift=false_alarms,
        recovery_time=recovery_time,
        adaptation_count_post_drift=adapt_count_post,
        total_retrain_time_post_s=total_retrain_time_post,
        full_stream_pr_auc=round(full_pr_auc, 5) if full_pr_auc is not None else None,
        n_pre_samples=len(pre_slice),
        n_post_samples=len(post_slice),
        segment_evaluated=filter_segment,
    )
