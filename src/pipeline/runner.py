"""
Integrated Prequential Pipeline Runner.

Orchestrates the full streaming experiment according to the thesis prequential
semantics (Source of Truth §16.2).  This module is the single entry point for
running a complete streaming experiment: warmup → stream → results.

============================================================
Strict Prequential Execution Order (per transaction t)
============================================================

Phase 1  WARMUP  (t = 0 … t_warmup − 1)
    ↓  Preprocessor already fitted externally on warmup data.
    ↓  Learner is trained on each warmup observation (learn_one).
    ↓  Memory buffer is populated.
    ↓  No metrics, no drift detection, no policy triggers.

Phase 2  STREAM  (t = t_warmup … T − 1) — the prequential loop:

  Step 1  PREDICT
          t_start = now()
          ŷ_prob_t = model.predict_one(x_t)         ← uses model BEFORE y_t
          latency = now() − t_start

  Step 2  METRIC PRE-REVEAL
          metrics_tracker.update_latency(latency)
          # NOTE: y_t not yet used — metric will be finalized in Step 3.

  Step 3  LABEL REVELATION & METRIC UPDATE
          y_t revealed from stream
          metrics_tracker.update(y_t, ŷ_prob_t)    ← y_t now known

  Step 4  ONLINE LEARNER UPDATE
          model.learn_one(x_t, y_t)                 ← incremental update

  Step 5  MEMORY BUFFER UPDATE
          retrainer.add(x_t, y_t, segment_key_t)

  Step 6  DRIFT DETECTOR UPDATE
          e_t = |y_t − round(ŷ_prob_t)|
          drift_monitor.update(e_t, tx_index=t, segment_key=segment_key_t)

  Step 7  POLICY TRIGGER EVALUATION
          new_model = policy_manager.step(
              tx_index=t, current_model=model, segment_key=segment_key_t
          )

  Step 8  MODEL SWAP (if adaptation triggered)
          if new_model is not None:
              model = new_model
          → the NEW model is used from t+1 onward.

  Advance to t+1.

============================================================
Design constraints
============================================================
- The runner is dataset-agnostic: segment information is passed through
  the stream, not hardcoded (no ProductCD, no PaySim `type`).
- All components are injected by the caller (learner, retrainer, monitor,
  policy_manager, metrics_tracker).  The runner does NOT construct them.
- The runner is stateless between runs IF `reset()` is called.
- Online learner ``learn_one`` is ALWAYS called on every streaming sample.
  Policy-triggered ``retrain_window`` produces a NEW learner INSTANCE that
  replaces the current one; it does NOT suppress incremental updates.
- P3 retrains only the affected segment; the runner swaps the model
  globally (the retrained segment model IS the new active model) so that
  predictions for ALL future transactions use the segment-retrained model.
  This is scientifically conservative: future Notebook 03 may extend to
  per-segment model registries if E4 motivates it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class StreamingRecord:
    """One prequential observation record.

    Attributes
    ----------
    tx_index:
        0-based stream index (post-warmup).
    y_true:
        Ground-truth label (0 or 1).
    y_prob:
        Model-predicted probability of fraud (class 1), produced BEFORE
        y_true was revealed.
    y_pred:
        Hard prediction (round(y_prob) at 0.5 threshold).
    error:
        Binary prediction error e_t = |y_true − y_pred| ∈ {0, 1}.
    segment_key:
        Categorical segment value of this transaction (None if unavailable).
    inference_latency_s:
        Wall-clock seconds for the predict_one call.
    adapted:
        True if a policy-triggered adaptation event occurred at this step.
    adaptation_scope:
        ``"global"`` or segment key string if adaptation occurred; ``None``
        otherwise.
    drift_detected_global:
        True if the global drift detector was in a drift state at this step.
    drift_detected_segment:
        True if the segment-specific detector was in a drift state.
    """

    tx_index: int
    y_true: int
    y_prob: float
    y_pred: int
    error: int
    segment_key: Optional[str]
    inference_latency_s: float
    adapted: bool = False
    adaptation_scope: Optional[str] = None
    drift_detected_global: bool = False
    drift_detected_segment: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tx_index": self.tx_index,
            "y_true": self.y_true,
            "y_prob": self.y_prob,
            "y_pred": self.y_pred,
            "error": self.error,
            "segment_key": self.segment_key,
            "inference_latency_s": self.inference_latency_s,
            "adapted": self.adapted,
            "adaptation_scope": self.adaptation_scope,
            "drift_detected_global": self.drift_detected_global,
            "drift_detected_segment": self.drift_detected_segment,
        }


@dataclass
class RunResult:
    """Complete result of one prequential experiment run.

    Attributes
    ----------
    policy:
        Policy name string (``"P0"``–``"P3"``).
    detector_type:
        Drift detector type used (``"adwin"``, ``"hddm_w"``, ``"hddm_a"``).
    n_warmup:
        Number of observations used for warmup (not evaluated).
    n_stream:
        Number of observations processed in the prequential stream.
    records:
        Per-transaction :class:`StreamingRecord` list.
    adaptation_log:
        Ordered list of adaptation event dicts (from PolicyManager log).
    drift_event_log:
        Ordered list of drift event dicts (from DriftMonitor log).
    final_metrics:
        :class:`~src.evaluation.metrics.MetricSnapshot` computed over the
        complete stream.
    latency_percentiles:
        Dict of inference latency percentiles (p50, p95, p99, mean, n).
    total_wall_time_s:
        Total wall-clock seconds for the stream phase.
    """

    policy: str
    detector_type: str
    n_warmup: int
    n_stream: int
    records: List[StreamingRecord] = field(default_factory=list)
    adaptation_log: List[Dict[str, Any]] = field(default_factory=list)
    drift_event_log: List[Dict[str, Any]] = field(default_factory=list)
    final_metrics: Any = None            # MetricSnapshot (avoid circular import)
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    total_wall_time_s: float = 0.0

    @property
    def n_adaptations(self) -> int:
        return len(self.adaptation_log)

    @property
    def n_drift_events(self) -> int:
        return len(self.drift_event_log)

    def summary_dict(self) -> Dict[str, Any]:
        """Return a high-level summary without per-record detail."""
        m = self.final_metrics
        return {
            "policy": self.policy,
            "detector_type": self.detector_type,
            "n_warmup": self.n_warmup,
            "n_stream": self.n_stream,
            "n_adaptations": self.n_adaptations,
            "n_drift_events": self.n_drift_events,
            "total_wall_time_s": self.total_wall_time_s,
            "pr_auc": getattr(m, "pr_auc", None),
            "roc_auc": getattr(m, "roc_auc", None),
            "f1": getattr(m, "f1", None),
            "recall": getattr(m, "recall", None),
            "latency_p95_s": self.latency_percentiles.get("p95"),
        }


# ---------------------------------------------------------------------------
# Policy name mapping: ExperimentConfig vocabulary → PolicyManager vocabulary
# ---------------------------------------------------------------------------

_CONFIG_TO_POLICY = {
    "static": "P0",
    "periodic": "P1",
    "global_drift_triggered": "P2",
    "segment_drift_triggered": "P3",
    # Also accept direct P0–P3 labels
    "P0": "P0",
    "P1": "P1",
    "P2": "P2",
    "P3": "P3",
}

_CONFIG_TO_DETECTOR = {
    "adwin": "adwin",
    "hddm": "hddm_w",   # 'hddm' in config → primary HDDM variant (weighted-average)
    "hddm_w": "hddm_w",
    "hddm_a": "hddm_a",
    "none": "adwin",    # fallback; P0 monitor still initialised
}


# ---------------------------------------------------------------------------
# PrequentialRunner
# ---------------------------------------------------------------------------

class PrequentialRunner:
    """Integrated prequential pipeline runner.

    Orchestrates the eight-phase per-transaction loop (Source of Truth §16.2).
    All components are injected by the caller — the runner performs only
    orchestration, never reimplements drift detection, policy logic, or
    metric formulas.

    Parameters
    ----------
    learner:
        Fitted or unfitted :class:`~src.models.base_learner.HoeffdingTreeLearner`
        (or any object with ``predict_one``, ``learn_one``, ``fit_batch``).
    retrainer:
        :class:`~src.adaptation.retrainer.RetrainingEngine` instance with an
        already-configured learner_factory and window_size.
    drift_monitor:
        :class:`~src.drift.monitor.DriftMonitor` instance.
    policy_manager:
        :class:`~src.adaptation.policy.PolicyManager` instance wired to the
        same ``drift_monitor`` and the retrainer's ``as_retrain_fn()``.
    metrics_tracker:
        :class:`~src.evaluation.metrics.StreamingMetricsTracker` instance.
    policy_name:
        Human-readable policy label for result tagging (e.g. ``"P2"``).
    detector_type:
        Human-readable detector label for result tagging (e.g. ``"adwin"``).

    Examples
    --------
    See :meth:`run` for a complete usage example.
    """

    def __init__(
        self,
        learner: Any,
        retrainer: Any,
        drift_monitor: Any,
        policy_manager: Any,
        metrics_tracker: Any,
        policy_name: str = "P0",
        detector_type: str = "adwin",
    ) -> None:
        self._learner = learner
        self._retrainer = retrainer
        self._monitor = drift_monitor
        self._policy = policy_manager
        self._metrics = metrics_tracker
        self._policy_name = policy_name
        self._detector_type = detector_type

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        segment: Optional[pd.Series],
        warmup_size: int,
    ) -> RunResult:
        """Execute a complete prequential experiment.

        Parameters
        ----------
        X:
            Preprocessed feature DataFrame (full dataset, chronological order).
            MUST have been preprocessed with warmup-only statistics to avoid
            future leakage.
        y:
            Full target Series aligned with X.
        segment:
            Full segment-key Series aligned with X (None values are allowed;
            use None for the whole series if no segment feature is available).
        warmup_size:
            Number of leading rows to treat as warmup (train-only, not
            evaluated).  Must be < len(X).

        Returns
        -------
        :class:`RunResult`
        """
        n_total = len(X)
        if warmup_size < 0:
            raise ValueError(f"warmup_size must be >= 0, got {warmup_size}.")
        if warmup_size >= n_total:
            raise ValueError(
                f"warmup_size ({warmup_size}) must be < n_total ({n_total})."
            )

        # ----------------------------------------------------------------
        # Phase 1: WARMUP
        # ----------------------------------------------------------------
        self._run_warmup(X, y, segment, warmup_size)
        n_warmup = warmup_size

        # ----------------------------------------------------------------
        # Phase 2: PREQUENTIAL STREAM
        # ----------------------------------------------------------------
        records: List[StreamingRecord] = []
        stream_start = time.perf_counter()

        X_stream = X.iloc[warmup_size:].reset_index(drop=True)
        y_stream = y.iloc[warmup_size:].reset_index(drop=True)
        seg_stream = (
            segment.iloc[warmup_size:].reset_index(drop=True)
            if segment is not None else None
        )

        for local_idx in range(len(X_stream)):
            tx_index = warmup_size + local_idx

            x_t: Dict[str, Any] = {
                col: X_stream.iat[local_idx, X_stream.columns.get_loc(col)]
                for col in X_stream.columns
            }
            y_t: int = int(y_stream.iat[local_idx])
            seg_t: Optional[str] = (
                str(seg_stream.iat[local_idx]) if seg_stream is not None else None
            )

            record = self._step(
                tx_index=tx_index,
                x_t=x_t,
                y_t=y_t,
                seg_t=seg_t,
            )
            records.append(record)

        total_wall_time_s = time.perf_counter() - stream_start

        # ----------------------------------------------------------------
        # Collect results
        # ----------------------------------------------------------------
        from src.evaluation.metrics import StreamingMetricsTracker  # local import
        final_metrics = self._metrics.compute()
        latency_pcts = self._metrics.latency_percentiles()

        adaptation_log = [r.as_dict() for r in self._policy.adaptation_log]
        drift_event_log = [
            {
                "stream_key": ev.stream_key,
                "tx_index": ev.tx_index,
                "detector_type": ev.detector_type,
            }
            for ev in self._monitor.all_events()
        ]

        return RunResult(
            policy=self._policy_name,
            detector_type=self._detector_type,
            n_warmup=n_warmup,
            n_stream=len(records),
            records=records,
            adaptation_log=adaptation_log,
            drift_event_log=drift_event_log,
            final_metrics=final_metrics,
            latency_percentiles=latency_pcts,
            total_wall_time_s=total_wall_time_s,
        )

    # ------------------------------------------------------------------
    # Warmup phase
    # ------------------------------------------------------------------

    def _run_warmup(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        segment: Optional[pd.Series],
        warmup_size: int,
    ) -> None:
        """Train learner and populate buffer on warmup observations.

        NO predictions, NO metric updates, NO drift detection during warmup.
        """
        for i in range(warmup_size):
            x_i: Dict[str, Any] = {
                col: X.iat[i, X.columns.get_loc(col)] for col in X.columns
            }
            y_i: int = int(y.iat[i])
            seg_i: Optional[str] = (
                str(segment.iat[i]) if segment is not None else None
            )
            # Train learner
            self._learner.learn_one(x_i, y_i)
            # Populate memory buffer (retrainer)
            self._retrainer.add(x_i, y_i, seg_i)

    # ------------------------------------------------------------------
    # Single prequential step
    # ------------------------------------------------------------------

    def _step(
        self,
        tx_index: int,
        x_t: Dict[str, Any],
        y_t: int,
        seg_t: Optional[str],
    ) -> StreamingRecord:
        """Execute one full prequential step and return a StreamingRecord.

        Steps 1–8 from the module docstring.
        """
        # --- Step 1: PREDICT (before y_t) ---
        t_infer_start = time.perf_counter()
        y_prob_t: float = self._learner.predict_one(x_t)
        latency_s = time.perf_counter() - t_infer_start

        # --- Step 2: Record latency ---
        self._metrics.update_latency(latency_s)

        # --- Step 3: Reveal label & update metrics ---
        y_pred_t: int = int(y_prob_t >= 0.5)
        self._metrics.update(y_t, y_prob_t)

        # Binary prediction error for drift detector
        error_t: int = abs(y_t - y_pred_t)

        # --- Step 4: Online learner update ---
        self._learner.learn_one(x_t, y_t)

        # --- Step 5: Memory buffer update ---
        self._retrainer.add(x_t, y_t, seg_t)

        # --- Step 6: Drift detector update ---
        self._monitor.update(error=error_t, tx_index=tx_index, segment_key=seg_t)

        global_drift = self._monitor.global_drift_detected
        segment_drift = (
            self._monitor.segment_drift_detected(seg_t)
            if seg_t is not None else False
        )

        # --- Step 7: Policy trigger evaluation ---
        new_model = self._policy.step(
            tx_index=tx_index,
            current_model=self._learner,
            segment_key=seg_t,
        )

        # --- Step 8: Model swap ---
        adapted = False
        adaptation_scope: Optional[str] = None
        if new_model is not None:
            self._learner = new_model
            adapted = True
            # Infer scope from the last adaptation record
            log = self._policy.adaptation_log
            if log:
                adaptation_scope = log[-1].scope

        return StreamingRecord(
            tx_index=tx_index,
            y_true=y_t,
            y_prob=y_prob_t,
            y_pred=y_pred_t,
            error=error_t,
            segment_key=seg_t,
            inference_latency_s=latency_s,
            adapted=adapted,
            adaptation_scope=adaptation_scope,
            drift_detected_global=global_drift,
            drift_detected_segment=segment_drift,
        )

    # ------------------------------------------------------------------
    # Reset (for multi-seed experiments)
    # ------------------------------------------------------------------

    def reset(
        self,
        learner: Optional[Any] = None,
        reset_monitor: bool = True,
        reset_retrainer: bool = True,
        reset_policy: bool = True,
        reset_metrics: bool = True,
    ) -> None:
        """Reset all injected components for a fresh independent run.

        Parameters
        ----------
        learner:
            If provided, replace the learner entirely (recommended for
            multi-seed runs to get a fresh untrained model).  If ``None``,
            calls ``self._learner.reset()`` if the method exists.
        reset_monitor:
            If ``True``, call ``drift_monitor.reset()``.
        reset_retrainer:
            If ``True``, call ``retrainer.reset()`` (clears memory buffer).
        reset_policy:
            If ``True``, call ``policy_manager.reset()`` (clears tx_counter
            and adaptation log).
        reset_metrics:
            If ``True``, call ``metrics_tracker.reset()``.
        """
        if learner is not None:
            self._learner = learner
        elif hasattr(self._learner, "reset"):
            self._learner.reset()

        if reset_monitor:
            self._monitor.reset()
        if reset_retrainer:
            self._retrainer.reset()
        if reset_policy:
            self._policy.reset()
        if reset_metrics:
            self._metrics.reset()

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        policy_str: str,
        detector_str: str,
        grace_period: int = 200,
        delta: float = 1e-7,
        window_size: int = 5_000,
        n_interval: int = 10_000,
        detector_kwargs: Optional[Dict[str, Any]] = None,
        segment_aware: bool = False,
        seed: Optional[int] = None,
    ) -> "PrequentialRunner":
        """Convenience factory that wires all components together.

        Intended for testing and notebook usage.  Production experiments
        should construct components explicitly for full control.

        Parameters
        ----------
        policy_str:
            Policy name (``"P0"``–``"P3"`` or config vocabulary like
            ``"static"``, ``"periodic"``, etc.).
        detector_str:
            Detector name (``"adwin"``, ``"hddm_w"``, ``"hddm_a"``,
            ``"hddm"`` as alias for ``"hddm_w"``).
        grace_period:
            Hoeffding Tree grace_period.
        delta:
            Hoeffding Tree Hoeffding bound delta (River 0.21+, formerly
            ``split_confidence``).
        window_size:
            Retraining window size (N_window).
        n_interval:
            Periodic retraining interval for P1.
        detector_kwargs:
            Extra kwargs for the drift detector (e.g. ``{"delta": 0.002}``).
        segment_aware:
            Whether to enable per-segment drift monitoring (required for P3).
        seed:
            Random seed (currently informational; Hoeffding Tree is
            deterministic given data order).

        Returns
        -------
        A fully wired :class:`PrequentialRunner`.
        """
        from src.models.base_learner import HoeffdingTreeLearner
        from src.adaptation.retrainer import RetrainingEngine
        from src.drift.monitor import DriftMonitor
        from src.adaptation.policy import PolicyManager
        from src.evaluation.metrics import StreamingMetricsTracker

        # Resolve policy name
        policy_key = _CONFIG_TO_POLICY.get(policy_str, policy_str.upper())
        if policy_key not in {"P0", "P1", "P2", "P3"}:
            raise ValueError(
                f"Unknown policy '{policy_str}'. "
                f"Accepted: {list(_CONFIG_TO_POLICY.keys())}"
            )

        # Resolve detector name
        detector_key = _CONFIG_TO_DETECTOR.get(detector_str, detector_str)

        # Build components
        def _learner_factory() -> HoeffdingTreeLearner:
            return HoeffdingTreeLearner(
                grace_period=grace_period,
                delta=delta,
            )

        learner = _learner_factory()
        retrainer = RetrainingEngine(
            learner_factory=_learner_factory,
            window_size=window_size,
        )
        monitor = DriftMonitor(
            detector_type=detector_key,
            segment_aware=segment_aware or (policy_key == "P3"),
            detector_kwargs=detector_kwargs or {},
        )
        policy = PolicyManager(
            policy=policy_key,
            retrain_fn=retrainer.as_retrain_fn(),
            drift_monitor=monitor,
            n_interval=n_interval,
            memory_buffer=None,  # retrainer owns buffer
        )
        metrics = StreamingMetricsTracker(track_latency=True)

        return cls(
            learner=learner,
            retrainer=retrainer,
            drift_monitor=monitor,
            policy_manager=policy,
            metrics_tracker=metrics,
            policy_name=policy_key,
            detector_type=detector_key,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PrequentialRunner(policy={self._policy_name!r}, "
            f"detector={self._detector_type!r})"
        )
