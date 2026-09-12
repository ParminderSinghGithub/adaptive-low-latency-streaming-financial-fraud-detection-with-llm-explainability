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
    ↓  Per-segment learners trained on corresponding segment warmup samples (P3).
    ↓  Memory buffer is populated with warmup observations.
    ↓  No metrics, no drift detection, no policy triggers.

Phase 2  STREAM  (t = t_warmup … T − 1) — the prequential loop:

  Step 1  ACTIVE MODEL SELECTION
          Select active model:
          - P3: self._segment_models[seg_t] (if seg_t present, else global)
          - P0, P1, P2: self._learner (global)

  Step 2  PREDICT (strictly before y_t revelation)
          t_start = now()
          ŷ_prob_t = active_model.predict_one(x_t)
          latency = now() − t_start

  Step 3  METRIC ACCUMULATION & LATENCY LOGGING
          metrics_tracker.update_latency(latency)
          metrics_tracker.update(y_t, ŷ_prob_t)

  Step 4  LABEL REVELATION & ERROR SIGNAL
          y_t revealed from stream
          ŷ_pred_t = int(ŷ_prob_t >= 0.5)
          e_t = |y_t − ŷ_pred_t| ∈ {0, 1}

  Step 5  INCREMENTAL ONLINE UPDATE (P0–P3)
          active_model.learn_one(x_t, y_t)
          (Executed by default under p0_mode="incremental" for all policies;
           skipped only if P0 auxiliary p0_mode="frozen" is active).

  Step 6  CONCEPT DRIFT MONITOR UPDATE
          drift_monitor.update(e_t, tx_index=t, segment_key=seg_t)

  Step 7  MEMORY BUFFER UPDATE
          retrainer.add(x_t, y_t, seg_t)
          (Current sample enters W_adapt before adaptation evaluation)

  Step 8  POLICY TRIGGER EVALUATION & MODEL SWAP
          new_model = policy_manager.step(
              tx_index=t, current_model=active_model, segment_key=seg_t
          )
          if new_model is not None:
              - P3 (segment scope): self._segment_models[seg_t] = new_model
                (Only the affected sub-population model is replaced; all other
                 segment models are completely untouched, preserving isolation).
              - P1 / P2 (global scope): self._learner = new_model
                (The retrained model has been fitted on W_adapt including (x_t, y_t)).

  Advance to t+1.

============================================================
Design constraints
============================================================
- The runner is dataset-agnostic: segment information is passed through
  the stream, not hardcoded (no ProductCD, no PaySim `type`).
- All components are injected by the caller (learner, retrainer, monitor,
  policy_manager, metrics_tracker).  The runner does NOT construct them.
- The runner is stateless between runs IF `reset()` is called.
- Online incremental learning: In all policies P0–P3, ``learn_one(x_t, y_t)``
  is executed after each label revelation by default (``p0_mode="incremental"``)
  to simulate realistic continuous streaming learning; ``retrain_window(W_adapt)``
  is an additional, heavier adaptation event governed by P1–P3 triggers and
  omitted in P0. An optional ``p0_mode="frozen"`` is supported for isolating
  pure concept drift degradation against an un-updated model.
- P3 maintains independent per-segment learners. Restricting retraining to
  the affected segment does NOT replace or degrade other segment models.
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

    def to_records_dataframe(self) -> pd.DataFrame:
        """Convert per-transaction records to a pandas DataFrame for analysis/plotting."""
        return pd.DataFrame([r.as_dict() for r in self.records])


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
        learner_factory: Optional[Any] = None,
        p0_mode: str = "incremental",
        diagnostic_horizon: int = 500,
        no_swap: bool = False,
    ) -> None:
        self._learner = learner
        self._retrainer = retrainer
        self._monitor = drift_monitor
        self._policy = policy_manager
        self._metrics = metrics_tracker
        self._policy_name = policy_name
        self._detector_type = detector_type
        self._learner_factory = learner_factory or getattr(retrainer, "learner_factory", None)
        self._p0_mode = p0_mode.lower()
        self._diagnostic_horizon = diagnostic_horizon
        self._no_swap = no_swap
        if hasattr(self._policy, "_no_swap"):
            self._policy._no_swap = no_swap
        # Dedicated per-segment model registry (active under P3)
        self._segment_models: Dict[str, Any] = {}
        self._pending_diagnostics: List[Dict[str, Any]] = []

    @property
    def diagnostic_horizon(self) -> int:
        """Diagnostic evaluation horizon H (in transactions) for M2 pre/post metrics."""
        return self._diagnostic_horizon

    @property
    def no_swap(self) -> bool:
        """Whether M3 Trigger/No-Swap ablation mode is active."""
        return self._no_swap

    @property
    def p0_mode(self) -> str:
        """Operating mode for P0 baseline ('incremental' vs 'frozen')."""
        return self._p0_mode

    @property
    def segment_models(self) -> Dict[str, Any]:
        """Dictionary of per-segment model instances (active under P3)."""
        return self._segment_models

    @staticmethod
    def _compute_slice_ap(records_slice: List[StreamingRecord]) -> Tuple[Optional[float], int]:
        """Compute Average Precision over a fixed horizon slice of recorded stream predictions."""
        if not records_slice:
            return None, 0
        y_true = [r.y_true for r in records_slice]
        y_prob = [r.y_prob for r in records_slice]
        n = len(records_slice)
        if sum(y_true) > 0:
            try:
                from sklearn.metrics import average_precision_score
                ap = float(average_precision_score(y_true, y_prob))
                return round(ap, 6), n
            except Exception:
                return None, n
        return 0.0, n

    def _make_fresh_learner(self) -> Any:
        """Instantiate a fresh learner via learner_factory or clone."""
        if self._learner_factory is not None:
            return self._learner_factory()
        import copy
        clone = copy.deepcopy(self._learner)
        if hasattr(clone, "reset"):
            clone.reset()
        return clone

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

        col_names = X_stream.columns.tolist()
        X_stream_arr = X_stream.to_numpy()
        y_stream_arr = y_stream.to_numpy()
        seg_stream_arr = seg_stream.to_numpy() if seg_stream is not None else None
        n_stream = len(X_stream)

        for local_idx in range(n_stream):
            tx_index = warmup_size + local_idx

            x_t: Dict[str, Any] = dict(zip(col_names, X_stream_arr[local_idx]))
            y_t: int = int(y_stream_arr[local_idx])
            seg_t: Optional[str] = (
                str(seg_stream_arr[local_idx]) if seg_stream_arr is not None else None
            )

            record = self._step(
                tx_index=tx_index,
                x_t=x_t,
                y_t=y_t,
                seg_t=seg_t,
                records=records,
            )
            records.append(record)

            # Update pending forward-horizon diagnostics
            if self._pending_diagnostics:
                still_pending = []
                for item in self._pending_diagnostics:
                    rec = item["record"]
                    start_idx = item["post_start_idx"]
                    target_h = item["target_horizon"]
                    available = len(records) - start_idx
                    if available >= target_h:
                        post_slice = records[start_idx : start_idx + target_h]
                        post_ap, post_n = self._compute_slice_ap(post_slice)
                        rec.post_ap = post_ap
                        rec.post_horizon_n = post_n
                        rec.delta_ap = round(post_ap - rec.pre_ap, 6) if (post_ap is not None and rec.pre_ap is not None) else None
                        rec.horizon_complete = True
                    else:
                        still_pending.append(item)
                self._pending_diagnostics = still_pending

        total_wall_time_s = time.perf_counter() - stream_start

        # Finalize any pending diagnostics where stream finished before H observations
        for item in self._pending_diagnostics:
            rec = item["record"]
            start_idx = item["post_start_idx"]
            post_slice = records[start_idx : len(records)]
            post_ap, post_n = self._compute_slice_ap(post_slice)
            rec.post_ap = post_ap
            rec.post_horizon_n = post_n
            rec.delta_ap = round(post_ap - rec.pre_ap, 6) if (post_ap is not None and rec.pre_ap is not None) else None
            rec.horizon_complete = False
        self._pending_diagnostics.clear()

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
    # Reference extraction helper (for equivalence testing)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_row_reference(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Reference extraction using pandas .iat for equivalence verification."""
        return {
            col: df.iat[idx, df.columns.get_loc(col)]
            for col in df.columns
        }

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
        For P3, also populates per-segment initial models.
        """
        is_p3 = (self._policy_name == "P3")
        X_warmup = X.iloc[:warmup_size]
        col_names = X_warmup.columns.tolist()
        X_warmup_arr = X_warmup.to_numpy()
        y_warmup_arr = y.iloc[:warmup_size].to_numpy()
        seg_warmup_arr = (
            segment.iloc[:warmup_size].to_numpy() if segment is not None else None
        )

        for i in range(warmup_size):
            x_i: Dict[str, Any] = dict(zip(col_names, X_warmup_arr[i]))
            y_i: int = int(y_warmup_arr[i])
            seg_i: Optional[str] = (
                str(seg_warmup_arr[i]) if seg_warmup_arr is not None else None
            )
            # Global learner always trained on warmup (baseline / fallback)
            self._learner.learn_one(x_i, y_i)

            # For P3: populate per-segment learners during warmup
            if is_p3 and seg_i is not None:
                if seg_i not in self._segment_models:
                    self._segment_models[seg_i] = self._make_fresh_learner()
                self._segment_models[seg_i].learn_one(x_i, y_i)

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
        records: Optional[List[StreamingRecord]] = None,
    ) -> StreamingRecord:
        """Execute one full prequential step and return a StreamingRecord."""
        is_p3 = (self._policy_name == "P3")

        # --- Step 1: Active model resolution ---
        if is_p3 and seg_t is not None:
            if seg_t not in self._segment_models:
                self._segment_models[seg_t] = self._make_fresh_learner()
            active_model = self._segment_models[seg_t]
        else:
            active_model = self._learner

        # --- Step 2: PREDICT (strictly before y_t revelation) ---
        t_infer_start = time.perf_counter()
        y_prob_t: float = active_model.predict_one(x_t)
        latency_s = time.perf_counter() - t_infer_start

        # --- Step 3: Record latency & prequential prediction ---
        self._metrics.update_latency(latency_s)
        self._metrics.update(y_t, y_prob_t)

        y_pred_t: int = int(y_prob_t >= 0.5)

        # --- Step 4: True label revelation & error calculation ---
        error_t: int = abs(y_t - y_pred_t)

        # --- Step 5: Incremental online learning update ---
        # Under default p0_mode="incremental", all policies P0–P3 call learn_one(x_t, y_t).
        # Under auxiliary p0_mode="frozen", P0 skips learn_one to isolate raw drift degradation.
        if self._policy_name != "P0" or self._p0_mode == "incremental":
            active_model.learn_one(x_t, y_t)

        # --- Step 6: Drift detector update ---
        self._monitor.update(error=error_t, tx_index=tx_index, segment_key=seg_t)
        global_drift = self._monitor.global_drift_detected
        segment_drift = (
            self._monitor.segment_drift_detected(seg_t)
            if seg_t is not None else False
        )

        # --- Step 7: Memory buffer update (current sample enters W_adapt) ---
        self._retrainer.add(x_t, y_t, seg_t)

        # Pre-trigger model complexity capture
        comp_pre = getattr(active_model, "model_complexity", {})
        if callable(comp_pre):
            comp_pre = comp_pre()
        model_complexity_pre = dict(comp_pre) if isinstance(comp_pre, dict) else {}

        # --- Step 8: Policy trigger evaluation ---
        new_model = self._policy.step(
            tx_index=tx_index,
            current_model=active_model,
            segment_key=seg_t,
        )

        # --- Step 9: Model adaptation swap if triggered ---
        adapted = False
        adaptation_scope: Optional[str] = None
        if new_model is not None:
            log = self._policy.adaptation_log
            if log:
                rec = log[-1]
                adaptation_scope = rec.scope
                # Populate M2 window statistics (strictly segment-specific for P3)
                if hasattr(self._retrainer, "get_window_stats"):
                    stats = self._retrainer.get_window_stats(segment_key=seg_t if is_p3 else None)
                    rec.window_size = stats.get("window_capacity")
                    rec.n_samples_window = stats.get("n_samples_window")
                    rec.n_fraud_window = stats.get("n_fraud_window")
                    rec.fraud_prevalence_window = stats.get("fraud_prevalence_window")
                rec.model_complexity_pre = model_complexity_pre

                # Pre-adaptation AP from already recorded predictions (strictly pre-adaptation)
                history = records if records is not None else []
                pre_start = max(0, len(history) - self._diagnostic_horizon)
                pre_slice = history[pre_start : len(history)]
                pre_ap, pre_n = self._compute_slice_ap(pre_slice)
                rec.pre_ap = pre_ap
                rec.pre_horizon_n = pre_n

                if rec.replacement_performed:
                    adapted = True
                    if is_p3 and seg_t is not None:
                        self._segment_models[seg_t] = new_model
                    else:
                        self._learner = new_model
                    comp_post = getattr(new_model, "model_complexity", {})
                    if callable(comp_post):
                        comp_post = comp_post()
                    rec.model_complexity_post = dict(comp_post) if isinstance(comp_post, dict) else {}
                else:
                    # M3 no-swap ablation: trigger observed, active model retained, replacement suppressed
                    adapted = False
                    rec.model_complexity_post = dict(model_complexity_pre)

                # Queue forward horizon diagnostic tracking
                self._pending_diagnostics.append({
                    "record": rec,
                    "post_start_idx": len(history) + 1,  # next record is the first post-adaptation observation
                    "target_horizon": self._diagnostic_horizon,
                })

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
        """Reset all injected components for a fresh independent run."""
        if learner is not None:
            self._learner = learner
        elif hasattr(self._learner, "reset"):
            self._learner.reset()

        if self._segment_models:
            self._segment_models.clear()

        if reset_monitor:
            self._monitor.reset()
        if reset_retrainer:
            self._retrainer.reset()
        if reset_policy:
            self._policy.reset()
        if reset_metrics:
            self._metrics.reset()
        self._pending_diagnostics.clear()

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
        p0_mode: str = "incremental",
        diagnostic_horizon: int = 500,
        no_swap: bool = False,
    ) -> "PrequentialRunner":
        """Convenience factory that wires all components together.

        Intended for testing and notebook usage.  Production experiments
        should construct components explicitly for full control.
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
            no_swap=no_swap,
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
            learner_factory=_learner_factory,
            p0_mode=p0_mode,
            diagnostic_horizon=diagnostic_horizon,
            no_swap=no_swap,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PrequentialRunner(policy={self._policy_name!r}, "
            f"detector={self._detector_type!r})"
        )


# Alias for backward compatibility
StreamingPipelineRunner = PrequentialRunner

