"""
Multi-stream concept drift monitor.

Provides a unified abstraction over River's ADWIN and alternative drift
detectors, supporting both a single global stream and per-segment (P3)
sub-streams.

Design constraints (DEC-01 – DEC-06 / Source of Truth §7 & §8):
- Drift is detected on the binary 0-1 prediction error: e_t = |y_t - y_hat_t|.
- Global monitor: one detector over the entire stream (used by P2).
- Segment monitors: one detector per categorical segment key value (used by P3).
- No RL, no ML-based detection — pure statistical change-point detectors.
- Deterministic given the same input sequence.

Supported detector types
------------------------
- ``"adwin"``       → river.drift.ADWIN       (primary, DEC-03 / OP-03)
- ``"kswin"``       → river.drift.KSWIN       (secondary, E5)
- ``"page_hinkley"``→ river.drift.PageHinkley (secondary, E5 variant)

Note: River ≥ 0.21 ships ADWIN, KSWIN, and PageHinkley.  HDDM_W/HDDM_A
are not present in this release; KSWIN and PageHinkley are used instead for
the E5 detector-sensitivity comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_detector(detector_type: str, **kwargs: Any):
    """Instantiate a River drift detector by name.

    Parameters
    ----------
    detector_type:
        One of ``"adwin"`` (default), ``"kswin"``, or ``"page_hinkley"``.
    **kwargs:
        Keyword arguments forwarded directly to the River detector constructor.
        Typical: ``delta=0.002`` for ADWIN.

    Returns
    -------
    A River drift detector instance with an ``update(value)`` method and a
    ``drift_detected`` boolean property.
    """
    dt = detector_type.lower()
    if dt == "adwin":
        from river.drift import ADWIN  # type: ignore[import]
        return ADWIN(**kwargs)
    elif dt == "kswin":
        from river.drift import KSWIN  # type: ignore[import]
        return KSWIN(**kwargs)
    elif dt == "page_hinkley":
        from river.drift import PageHinkley  # type: ignore[import]
        return PageHinkley(**kwargs)
    else:
        raise ValueError(
            f"Unsupported detector type '{detector_type}'. "
            "Choose from: 'adwin', 'kswin', 'page_hinkley'."
        )


# ---------------------------------------------------------------------------
# DriftEvent dataclass (lightweight, no dataclasses import needed)
# ---------------------------------------------------------------------------

class DriftEvent:
    """Immutable record of a detected drift signal.

    Attributes
    ----------
    stream_key:
        ``"global"`` for a global drift event, or the segment value string
        (e.g. ``"W"``, ``"TRANSFER"``) for a segment-level event.
    tx_index:
        The transaction index (0-based) at which drift was detected.
    detector_type:
        The detector type string (``"adwin"``, ``"hddm_w"``, ``"hddm_a"``).
    """

    __slots__ = ("stream_key", "tx_index", "detector_type")

    def __init__(self, stream_key: str, tx_index: int, detector_type: str) -> None:
        self.stream_key = stream_key
        self.tx_index = tx_index
        self.detector_type = detector_type

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DriftEvent(stream_key={self.stream_key!r}, "
            f"tx_index={self.tx_index}, detector_type={self.detector_type!r})"
        )


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Multi-stream concept drift monitor.

    Maintains a single *global* detector (used by P2) and, optionally, one
    *segment* detector per categorical key value seen so far (used by P3).

    All detectors share the same detector type and constructor kwargs so that
    detector sensitivity is held constant during policy comparisons (confound
    control rule §8 of the Source of Truth).

    Parameters
    ----------
    detector_type:
        Drift detector to use.  One of ``"adwin"`` (default), ``"hddm_w"``,
        ``"hddm_a"``.
    segment_aware:
        If ``True``, maintain per-segment sub-stream detectors (required for
        P3).  If ``False``, only the global detector is active.
    detector_kwargs:
        Extra keyword arguments forwarded to every detector constructor.
        Example: ``{"delta": 0.002}`` for ADWIN.

    Examples
    --------
    >>> monitor = DriftMonitor(detector_type="adwin", segment_aware=True,
    ...                        detector_kwargs={"delta": 0.002})
    >>> event = monitor.update(error=0, tx_index=0, segment_key="W")
    >>> monitor.global_drift_detected
    False
    """

    def __init__(
        self,
        detector_type: str = "adwin",
        segment_aware: bool = False,
        detector_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._detector_type = detector_type
        self._segment_aware = segment_aware
        self._detector_kwargs: Dict[str, Any] = detector_kwargs or {}

        # Global detector — always active.
        self._global_detector = _make_detector(detector_type, **self._detector_kwargs)

        # Per-segment detectors — created lazily on first observation.
        self._segment_detectors: Dict[str, Any] = {}

        # Event log: list of DriftEvent (ordered by tx_index).
        self._events: List[DriftEvent] = []

        # Running counters per stream key.
        self._update_counts: Dict[str, int] = {"global": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        error: float,
        tx_index: int,
        segment_key: Optional[str] = None,
    ) -> Optional[DriftEvent]:
        """Feed a single prediction error to the monitor.

        Must be called AFTER label revelation, strictly in temporal order.
        ``error`` should be ``|y_t - y_hat_t|`` in {0, 1}.

        Parameters
        ----------
        error:
            Binary prediction error for transaction t.
        tx_index:
            0-based position of transaction t in the stream.
        segment_key:
            Categorical segment value of transaction t (e.g. ``"W"``,
            ``"TRANSFER"``).  Required when ``segment_aware=True``; ignored
            otherwise.

        Returns
        -------
        A :class:`DriftEvent` if **global** drift was detected on this update,
        ``None`` otherwise.  Segment drift events are stored internally and
        retrievable via :meth:`segment_drift_detected` or
        :meth:`pending_segment_events`.
        """
        # ---- Global detector ----
        self._global_detector.update(error)
        self._update_counts["global"] = self._update_counts.get("global", 0) + 1
        global_event: Optional[DriftEvent] = None
        if self._global_detector.drift_detected:
            global_event = DriftEvent(
                stream_key="global",
                tx_index=tx_index,
                detector_type=self._detector_type,
            )
            self._events.append(global_event)

        # ---- Segment detector (lazy creation) ----
        if self._segment_aware and segment_key is not None:
            if segment_key not in self._segment_detectors:
                self._segment_detectors[segment_key] = _make_detector(
                    self._detector_type, **self._detector_kwargs
                )
                self._update_counts[segment_key] = 0

            seg_det = self._segment_detectors[segment_key]
            seg_det.update(error)
            self._update_counts[segment_key] = (
                self._update_counts.get(segment_key, 0) + 1
            )
            if seg_det.drift_detected:
                seg_event = DriftEvent(
                    stream_key=segment_key,
                    tx_index=tx_index,
                    detector_type=self._detector_type,
                )
                self._events.append(seg_event)

        return global_event

    @property
    def global_drift_detected(self) -> bool:
        """``True`` if the global detector is currently in a drift state."""
        return bool(self._global_detector.drift_detected)

    def segment_drift_detected(self, segment_key: str) -> bool:
        """``True`` if the named segment detector is currently in a drift state.

        Returns ``False`` if the segment has never been observed.
        """
        det = self._segment_detectors.get(segment_key)
        if det is None:
            return False
        return bool(det.drift_detected)

    def pending_segment_events(self) -> Iterator[DriftEvent]:
        """Iterate over all segment drift events in the internal event log."""
        for ev in self._events:
            if ev.stream_key != "global":
                yield ev

    def all_events(self) -> List[DriftEvent]:
        """Return a copy of the full event log (global + segment)."""
        return list(self._events)

    def drift_counts(self) -> Dict[str, int]:
        """Return the total number of drift events per stream key."""
        counts: Dict[str, int] = {}
        for ev in self._events:
            counts[ev.stream_key] = counts.get(ev.stream_key, 0) + 1
        return counts

    def update_counts(self) -> Dict[str, int]:
        """Return the number of ``update()`` calls per stream key."""
        return dict(self._update_counts)

    @property
    def known_segments(self) -> List[str]:
        """Return a sorted list of segment keys seen so far."""
        return sorted(self._segment_detectors.keys())

    @property
    def segment_aware(self) -> bool:
        """``True`` when the monitor maintains per-segment sub-streams."""
        return self._segment_aware

    @property
    def detector_type(self) -> str:
        """The detector type string used by all sub-detectors."""
        return self._detector_type

    def reset(self) -> None:
        """Reset all detectors and clear the event log.

        Useful between independent experimental runs (10-seed replication).
        """
        self._global_detector = _make_detector(
            self._detector_type, **self._detector_kwargs
        )
        self._segment_detectors = {}
        self._events = []
        self._update_counts = {"global": 0}

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DriftMonitor(detector_type={self._detector_type!r}, "
            f"segment_aware={self._segment_aware}, "
            f"n_events={len(self._events)}, "
            f"known_segments={self.known_segments})"
        )
