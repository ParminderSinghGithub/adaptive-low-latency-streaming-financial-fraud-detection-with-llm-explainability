"""
Windowed re-instantiation retraining engine (DEC-02).

Implements the ``retrain_window`` mechanism shared by adaptation policies P1,
P2, and P3 (Source of Truth §8, §16.2 Phase 7):

    W_adapt extracted from memory buffer
    → fresh Hoeffding Tree initialised
    → fitted on W_adapt (or segment-filtered W_adapt for P3)
    → new model returned

Design constraints
------------------
- The SAME ``retrain_window`` function is shared by P1, P2, and P3.
  Only the *trigger condition* and *segment scope* differ between policies.
  This is a confound-control requirement (DEC-02).
- For P3, only the subset of ``W_adapt`` rows belonging to the affected
  segment is used for retraining.
- The memory buffer is a ``collections.deque`` of ``(x_dict, label,
  segment_key)`` triples maintained by the runner; the retrainer does NOT
  own it.
- A fresh learner is constructed on each retrain call (re-instantiation,
  not incremental update of the existing model) to avoid stale split nodes.
- The learner_factory callable is injected to keep the retrainer independent
  of the specific learner implementation.
"""

from __future__ import annotations

import collections
from typing import Any, Callable, Deque, List, Optional, Tuple


# Type alias: one memory-buffer entry
BufferEntry = Tuple[Any, int, Optional[str]]  # (x_dict, label, segment_key)


class RetrainingEngine:
    """Windowed re-instantiation retraining engine (DEC-02).

    Parameters
    ----------
    learner_factory:
        Zero-argument callable that returns a fresh, untrained learner
        instance (e.g. ``lambda: HoeffdingTreeLearner(grace_period=200)``).
        Called on every retraining event.
    window_size:
        Maximum number of recent observations to retain in the memory buffer.
        Corresponds to ``W_adapt`` / ``N_window`` in the Source of Truth (OP-01).

    Examples
    --------
    >>> engine = RetrainingEngine(lambda: HoeffdingTreeLearner(), window_size=500)
    >>> engine.add(x_dict, label=0, segment_key="W")
    >>> new_model = engine.retrain_window()          # global retrain
    >>> new_model = engine.retrain_window("W")       # segment-scoped retrain
    """

    def __init__(
        self,
        learner_factory: Callable[[], Any],
        window_size: int = 5_000,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}.")
        self._learner_factory = learner_factory
        self._window_size = window_size
        self._buffer: Deque[BufferEntry] = collections.deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def add(
        self,
        x: Any,
        label: int,
        segment_key: Optional[str] = None,
    ) -> None:
        """Append one observation to the sliding memory buffer.

        Must be called AFTER label revelation (Phase 4 in the prequential
        loop) to preserve strict test-then-train semantics.

        Parameters
        ----------
        x:
            Feature dictionary for this transaction.
        label:
            True label (0 or 1).
        segment_key:
            Categorical segment value (e.g. ``"W"``).  ``None`` if the
            dataset has no segment feature.
        """
        self._buffer.append((x, label, segment_key))

    # ------------------------------------------------------------------
    # Core retraining
    # ------------------------------------------------------------------

    def retrain_window(
        self,
        segment_key: Optional[str] = None,
    ) -> Any:
        """Fit a fresh learner on the sliding memory window and return it.

        This is the function injected into ``PolicyManager`` as ``retrain_fn``.
        The PolicyManager calls it as::

            retrain_fn(memory_buffer, segment_key=None)

        but this engine ignores the positional ``memory_buffer`` argument
        (it maintains its own internal buffer) for compatibility with the
        ``PolicyManager`` injection interface.  See :meth:`as_retrain_fn`
        for the adapter.

        Parameters
        ----------
        segment_key:
            If ``None``, retrain on all buffered observations (P1, P2).
            If a segment string, retrain only on buffer entries matching
            that segment (P3).

        Returns
        -------
        A freshly trained learner instance.

        Raises
        ------
        RuntimeError
            If the buffer is empty (no data available for retraining).
        """
        rows = self._get_rows(segment_key)
        if not rows:
            raise RuntimeError(
                f"Cannot retrain: memory buffer is empty"
                + (f" for segment '{segment_key}'" if segment_key else "")
                + f". Buffer size: {len(self._buffer)}."
            )

        new_learner = self._learner_factory()
        for x, y, _ in rows:
            new_learner.learn_one(x, y)
        return new_learner

    def as_retrain_fn(self) -> Callable[..., Any]:
        """Return a ``retrain_fn`` compatible with ``PolicyManager``.

        The ``PolicyManager`` calls ``retrain_fn(memory_buffer, segment_key=None)``.
        This adapter ignores the positional buffer argument (the engine owns
        its own buffer) and delegates to :meth:`retrain_window`.

        Returns
        -------
        Callable matching the signature expected by ``PolicyManager``.
        """
        def _fn(_buffer: Any, *, segment_key: Optional[str] = None) -> Any:
            return self.retrain_window(segment_key=segment_key)
        return _fn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_rows(self, segment_key: Optional[str]) -> List[BufferEntry]:
        """Return buffer entries, optionally filtered to a segment."""
        if segment_key is None:
            return list(self._buffer)
        return [
            entry for entry in self._buffer if entry[2] == segment_key
        ]

    def reset(self) -> None:
        """Clear the memory buffer (call before each independent seed run)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def buffer_size(self) -> int:
        """Current number of entries in the memory buffer."""
        return len(self._buffer)

    @property
    def window_size(self) -> int:
        """Maximum capacity of the sliding memory window."""
        return self._window_size

    @property
    def learner_factory(self) -> Callable[[], Any]:
        """Callable returning a fresh, untrained learner instance."""
        return self._learner_factory

    def segment_counts(self) -> dict:
        """Return count of buffer entries per segment key."""
        counts: dict = {}
        for _, _, seg in self._buffer:
            k = seg if seg is not None else "__none__"
            counts[k] = counts.get(k, 0) + 1
        return counts

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RetrainingEngine(window_size={self._window_size}, "
            f"buffer_size={self.buffer_size})"
        )
