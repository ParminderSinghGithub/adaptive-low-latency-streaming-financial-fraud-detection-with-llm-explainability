"""
Adaptation Policy State Machine (P0 – P3).

Implements the locked policy matrix from the thesis Source of Truth (§8):

    P0  Static            — Never retrain.  Baseline.
    P1  Periodic          — Retrain every N_interval transactions (global).
    P2  Global Drift      — Retrain when global drift detector fires.
    P3  Segment Drift     — Retrain only the affected segment(s) when their
                            individual drift detector fires.

Design constraints
------------------
- Deterministic rule-based control logic only.  No RL, no Q-learning.
- Shared retraining engine: ``retrain_fn`` is called identically by P1/P2/P3.
  Only the *trigger condition* and *scope* differ (confound control §8).
- P3 segmentation key is caller-supplied (not hardcoded to ``ProductCD``)
  so the same engine works for IEEE-CIS, PaySim, and future datasets.
- ``retrain_fn`` is injected by the caller so this module has zero dependency
  on the base learner or data format — it only controls *when* and *what* to
  retrain.

Typical ``retrain_fn`` signature (windowed re-instantiation, DEC-02)
----------------------------------------------------------------------
    def retrain_fn(memory_buffer, segment_key=None):
        '''
        Fit a fresh Hoeffding Tree on the last W_adapt rows of memory_buffer.
        If segment_key is not None, filter buffer to that segment first.
        Returns the new model instance.
        '''
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.drift.monitor import DriftMonitor, DriftEvent


# ---------------------------------------------------------------------------
# AdaptationRecord  — lightweight log entry per retraining event
# ---------------------------------------------------------------------------

class AdaptationRecord:
    """Record of a single adaptation event.

    Attributes
    ----------
    policy:
        Policy name string (``"P0"``–``"P3"``).
    tx_index:
        Transaction index that triggered the adaptation (0-based).
    scope:
        ``"global"`` or the segment key string (e.g. ``"W"``).
    duration_s:
        Wall-clock seconds taken by ``retrain_fn``.
    trigger:
        Human-readable trigger description.
    adaptation_event_index:
        0-based index of this adaptation event within the run.
    trigger_occurred:
        Whether the adaptation policy trigger condition fired (True).
    replacement_performed:
        Whether model replacement was actually performed (True normally, False under M3 no_swap).
    """

    __slots__ = (
        "policy",
        "tx_index",
        "scope",
        "duration_s",
        "trigger",
        "adaptation_event_index",
        "trigger_occurred",
        "replacement_performed",
        "window_size",
        "n_samples_window",
        "n_fraud_window",
        "fraud_prevalence_window",
        "model_complexity_pre",
        "model_complexity_post",
        "pre_ap",
        "post_ap",
        "delta_ap",
        "pre_horizon_n",
        "post_horizon_n",
        "horizon_complete",
    )

    def __init__(
        self,
        policy: str,
        tx_index: int,
        scope: str,
        duration_s: float,
        trigger: str,
        adaptation_event_index: int = 0,
        trigger_occurred: bool = True,
        replacement_performed: bool = True,
        window_size: Optional[int] = None,
        n_samples_window: Optional[int] = None,
        n_fraud_window: Optional[int] = None,
        fraud_prevalence_window: Optional[float] = None,
        model_complexity_pre: Optional[Dict[str, Any]] = None,
        model_complexity_post: Optional[Dict[str, Any]] = None,
        pre_ap: Optional[float] = None,
        post_ap: Optional[float] = None,
        delta_ap: Optional[float] = None,
        pre_horizon_n: int = 0,
        post_horizon_n: int = 0,
        horizon_complete: bool = False,
    ) -> None:
        self.policy = policy
        self.tx_index = tx_index
        self.scope = scope
        self.duration_s = duration_s
        self.trigger = trigger
        self.adaptation_event_index = adaptation_event_index
        self.trigger_occurred = trigger_occurred
        self.replacement_performed = replacement_performed
        self.window_size = window_size
        self.n_samples_window = n_samples_window
        self.n_fraud_window = n_fraud_window
        self.fraud_prevalence_window = fraud_prevalence_window
        self.model_complexity_pre = model_complexity_pre or {}
        self.model_complexity_post = model_complexity_post or {}
        self.pre_ap = pre_ap
        self.post_ap = post_ap
        self.delta_ap = delta_ap
        self.pre_horizon_n = pre_horizon_n
        self.post_horizon_n = post_horizon_n
        self.horizon_complete = horizon_complete

    def as_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (e.g. for JSON logging)."""
        return {
            "policy": self.policy,
            "tx_index": self.tx_index,
            "scope": self.scope,
            "duration_s": self.duration_s,
            "trigger": self.trigger,
            "adaptation_event_index": self.adaptation_event_index,
            "trigger_occurred": self.trigger_occurred,
            "replacement_performed": self.replacement_performed,
            "window_size": self.window_size,
            "n_samples_window": self.n_samples_window,
            "n_fraud_window": self.n_fraud_window,
            "fraud_prevalence_window": self.fraud_prevalence_window,
            "model_complexity_pre": self.model_complexity_pre,
            "model_complexity_post": self.model_complexity_post,
            "pre_ap": self.pre_ap,
            "post_ap": self.post_ap,
            "delta_ap": self.delta_ap,
            "pre_horizon_n": self.pre_horizon_n,
            "post_horizon_n": self.post_horizon_n,
            "horizon_complete": self.horizon_complete,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AdaptationRecord(policy={self.policy!r}, tx_index={self.tx_index}, "
            f"scope={self.scope!r}, duration_s={self.duration_s:.4f}, "
            f"replaced={self.replacement_performed})"
        )


# ---------------------------------------------------------------------------
# PolicyManager
# ---------------------------------------------------------------------------

class PolicyManager:
    """Adaptation policy state machine implementing P0 – P3.

    The caller drives the state machine by calling :meth:`step` once per
    transaction, **after** label revelation and drift-detector update.

    Parameters
    ----------
    policy:
        One of ``"P0"``, ``"P1"``, ``"P2"``, ``"P3"`` (case-insensitive).
    retrain_fn:
        Callable with signature
        ``retrain_fn(memory_buffer, *, segment_key=None) -> model``.
        Called by P1/P2/P3 when the trigger condition fires.
        Must return the new model.  Not called by P0.
    drift_monitor:
        A :class:`~src.drift.monitor.DriftMonitor` instance.
        Required (and consumed) for P2 and P3.
        For P0/P1 a monitor must still be provided (it is updated externally
        by the caller before :meth:`step` is invoked) but its signals are
        not acted upon.
    n_interval:
        Transaction count between periodic retrains for P1 (``N_interval``,
        OP-02).  Ignored for other policies.
    memory_buffer:
        Any mutable container holding the rolling transaction memory window.
        Passed verbatim to ``retrain_fn`` on each trigger.  The caller is
        responsible for maintaining this buffer.

    Examples
    --------
    Minimal P1 usage::

        pm = PolicyManager(
            policy="P1",
            retrain_fn=my_retrain,
            drift_monitor=my_monitor,
            n_interval=10_000,
            memory_buffer=deque(maxlen=5_000),
        )
        for t, (x, y, y_hat, seg) in enumerate(stream):
            error = abs(y - y_hat)
            my_monitor.update(error, tx_index=t, segment_key=seg)
            new_model = pm.step(tx_index=t, current_model=model, segment_key=seg)
            if new_model is not None:
                model = new_model
    """

    _VALID_POLICIES = {"P0", "P1", "P2", "P3"}

    def __init__(
        self,
        policy: str,
        retrain_fn: Optional[Callable[..., Any]],
        drift_monitor: DriftMonitor,
        n_interval: int = 10_000,
        memory_buffer: Optional[Any] = None,
        no_swap: bool = False,
    ) -> None:
        policy = policy.upper()
        if policy not in self._VALID_POLICIES:
            raise ValueError(
                f"Unknown policy '{policy}'. Choose from {self._VALID_POLICIES}."
            )

        self._policy = policy
        self._retrain_fn = retrain_fn
        self._monitor = drift_monitor
        self._n_interval = n_interval
        self._memory_buffer = memory_buffer
        self._no_swap = no_swap

        # Internal state
        self._tx_counter: int = 0          # global transaction counter
        self._adaptation_log: List[AdaptationRecord] = []

    # ------------------------------------------------------------------
    # Main step interface
    # ------------------------------------------------------------------

    def step(
        self,
        tx_index: int,
        current_model: Any,
        segment_key: Optional[str] = None,
    ) -> Optional[Any]:
        """Evaluate trigger conditions and adapt if necessary.

        Call this **once per transaction**, in temporal order, **after** the
        drift monitor has already been updated for this transaction.

        Parameters
        ----------
        tx_index:
            0-based index of the current transaction in the stream.
        current_model:
            The model currently in production.
        segment_key:
            Categorical segment value of this transaction (e.g. ``"W"``).

        Returns
        -------
        - If trigger condition fires and no_swap=False: the new model returned by ``retrain_fn``.
        - If trigger condition fires and no_swap=True (M3 ablation): ``current_model`` is returned
          (trigger is logged, active model is retained, replacement is suppressed).
        - If no adaptation trigger fired: ``None``.
        """
        self._tx_counter += 1

        if self._policy == "P0":
            return self._step_p0()
        elif self._policy == "P1":
            return self._step_p1(tx_index, current_model=current_model)
        elif self._policy == "P2":
            return self._step_p2(tx_index, current_model=current_model)
        elif self._policy == "P3":
            return self._step_p3(tx_index, segment_key, current_model=current_model)
        return None  # unreachable

    # ------------------------------------------------------------------
    # Per-policy step implementations
    # ------------------------------------------------------------------

    def _step_p0(self) -> None:
        """P0 Static — never triggers adaptation."""
        return None

    def _step_p1(self, tx_index: int, current_model: Optional[Any] = None) -> Optional[Any]:
        """P1 Periodic — retrain every N_interval transactions (global scope)."""
        if self._tx_counter % self._n_interval == 0:
            return self._do_retrain(
                tx_index=tx_index,
                scope="global",
                trigger=f"P1:periodic:every_{self._n_interval}_tx",
                current_model=current_model,
            )
        return None

    def _step_p2(self, tx_index: int, current_model: Optional[Any] = None) -> Optional[Any]:
        """P2 Global Drift — retrain when the global drift detector fires."""
        if self._monitor.global_drift_detected:
            return self._do_retrain(
                tx_index=tx_index,
                scope="global",
                trigger="P2:global_drift_detected",
                current_model=current_model,
            )
        return None

    def _step_p3(
        self, tx_index: int, segment_key: Optional[str], current_model: Optional[Any] = None
    ) -> Optional[Any]:
        """P3 Segment Drift — retrain only the segment whose detector fired."""
        if segment_key is None:
            return None
        if self._monitor.segment_drift_detected(segment_key):
            return self._do_retrain(
                tx_index=tx_index,
                scope=segment_key,
                trigger=f"P3:segment_drift:{segment_key}",
                segment_key=segment_key,
                current_model=current_model,
            )
        return None

    # ------------------------------------------------------------------
    # Shared retraining executor
    # ------------------------------------------------------------------

    def _do_retrain(
        self,
        tx_index: int,
        scope: str,
        trigger: str,
        segment_key: Optional[str] = None,
        current_model: Optional[Any] = None,
    ) -> Any:
        """Record the trigger and execute retraining (or suppress replacement under no_swap)."""
        event_idx = len(self._adaptation_log)
        if self._no_swap:
            # M3 Trigger/No-Swap Ablation:
            # Trigger is evaluated and logged; retraining/replacement is suppressed.
            # The active model is retained and returned unchanged.
            record = AdaptationRecord(
                policy=self._policy,
                tx_index=tx_index,
                scope=scope,
                duration_s=0.0,
                trigger=trigger,
                adaptation_event_index=event_idx,
                trigger_occurred=True,
                replacement_performed=False,
            )
            self._adaptation_log.append(record)
            return current_model

        t_start = time.perf_counter()
        new_model = self._retrain_fn(
            self._memory_buffer, segment_key=segment_key
        )
        duration_s = time.perf_counter() - t_start

        record = AdaptationRecord(
            policy=self._policy,
            tx_index=tx_index,
            scope=scope,
            duration_s=duration_s,
            trigger=trigger,
            adaptation_event_index=event_idx,
            trigger_occurred=True,
            replacement_performed=True,
        )
        self._adaptation_log.append(record)
        return new_model

    # ------------------------------------------------------------------
    # Inspection API
    # ------------------------------------------------------------------

    @property
    def no_swap(self) -> bool:
        """Whether M3 Trigger/No-Swap ablation mode is enabled."""
        return self._no_swap

    @property
    def policy(self) -> str:
        """The active policy name (``"P0"`` – ``"P3"``)."""
        return self._policy

    @property
    def tx_counter(self) -> int:
        """Total transactions processed since creation (or last :meth:`reset`)."""
        return self._tx_counter

    @property
    def n_interval(self) -> int:
        """Periodic retraining interval for P1 (``N_interval``)."""
        return self._n_interval

    @property
    def adaptation_log(self) -> List[AdaptationRecord]:
        """Ordered log of all adaptation events."""
        return list(self._adaptation_log)

    def adaptation_count(self) -> int:
        """Total number of adaptation events triggered so far."""
        return len(self._adaptation_log)

    def total_adaptation_time(self) -> float:
        """Cumulative wall-clock seconds spent in ``retrain_fn`` calls."""
        return sum(r.duration_s for r in self._adaptation_log)

    def reset(self) -> None:
        """Reset internal counters and log.

        Does NOT reset the drift monitor or memory buffer — those are
        externally managed.  Call before each independent seed run.
        """
        self._tx_counter = 0
        self._adaptation_log = []

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PolicyManager(policy={self._policy!r}, "
            f"tx_counter={self._tx_counter}, "
            f"n_adaptations={self.adaptation_count()}, "
            f"n_interval={self._n_interval})"
        )
