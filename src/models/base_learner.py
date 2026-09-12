"""
Hoeffding Tree base learner wrapper (DEC-01).

Wraps ``river.tree.HoeffdingTreeClassifier`` to expose a clean, dataset-agnostic
interface used by the prequential runner and the retraining engine.

Locked methodological decision DEC-01:
    Standard Hoeffding Tree Classifier (River).
    - Supports ``predict_one(x)`` and ``learn_one(x, y)`` for incremental streaming.
    - Does NOT contain internal drift detection, background trees, or automated
      ensemble replacement.  This ensures that any observed performance changes
      can be attributed entirely to the external policy trigger, not to the learner
      itself.
    - This wrapper provides deterministic behaviour given the same seed.

Notes
-----
ARF (Adaptive Random Forest) is used only in experiment E10 as a benchmark
learner and is implemented separately.  Do NOT add ARF here.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class HoeffdingTreeLearner:
    """Thin wrapper around ``river.tree.HoeffdingTreeClassifier``.

    Exposes only the interface required by the prequential runner:

    - ``predict_one(x)``  → float probability of class 1
    - ``learn_one(x, y)`` → online update (in-place)
    - ``reset()``         → reinitialise to a fresh untrained state
    - ``fit_batch(rows)`` → batch-fit on a list of (x_dict, label) pairs
                            (used by ``retrain_window``)

    Parameters
    ----------
    grace_period:
        Minimum number of samples a leaf must see before it is allowed to
        split (``grace_period`` in River, OP-01 calibration target).
    delta:
        Hoeffding bound confidence parameter (``delta`` in River 0.21+,
        formerly called ``split_confidence`` in older River versions).
        Smaller values → more conservative splitting.
    **kwargs:
        Additional keyword arguments forwarded verbatim to
        ``HoeffdingTreeClassifier``.

    Examples
    --------
    >>> learner = HoeffdingTreeLearner(grace_period=200)
    >>> learner.predict_one({"x1": 1.0, "x2": 0.5})
    0.0
    >>> learner.learn_one({"x1": 1.0, "x2": 0.5}, 1)
    """

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        **kwargs: Any,
    ) -> None:
        if "split_confidence" in kwargs:
            delta = kwargs.pop("split_confidence")
        self._grace_period = grace_period
        self._delta = delta
        self._extra_kwargs = kwargs
        self._model = self._build_model()

    # ------------------------------------------------------------------
    # Internal factory
    # ------------------------------------------------------------------

    def _build_model(self) -> Any:
        """Instantiate a fresh River HoeffdingTreeClassifier."""
        from river.tree import HoeffdingTreeClassifier  # type: ignore[import]

        # River 0.21+: split_confidence was renamed to delta
        kwargs = dict(
            grace_period=self._grace_period,
            delta=self._delta,
            **self._extra_kwargs,
        )
        return HoeffdingTreeClassifier(**kwargs)

    # ------------------------------------------------------------------
    # Prequential interface
    # ------------------------------------------------------------------

    def predict_one(self, x: Dict[str, Any]) -> float:
        """Return P(fraud=1 | x) using the current model state.

        Parameters
        ----------
        x:
            Feature dictionary for a single transaction.

        Returns
        -------
        float in [0.0, 1.0].  Returns 0.0 before the tree has seen any
        training data (untrained state).
        """
        proba: Dict[int, float] = self._model.predict_proba_one(x)
        return float(proba.get(1, 0.0))

    def predict_label(self, x: Dict[str, Any]) -> int:
        """Return the hard predicted label (0 or 1)."""
        return int(self._model.predict_one(x) or 0)

    def learn_one(self, x: Dict[str, Any], y: int) -> None:
        """Perform a single online update (learn_one in River).

        Parameters
        ----------
        x:
            Feature dictionary.
        y:
            True label (0 or 1).
        """
        self._model.learn_one(x, y)

    def fit_batch(self, rows: list) -> None:
        """Fit on a list of ``(x_dict, label)`` pairs (batch warm-start).

        Used by ``retrain_window`` to reinitialise the model on a sliding
        memory window.

        Parameters
        ----------
        rows:
            List of ``(x_dict, int)`` tuples in chronological order.
        """
        self._model = self._build_model()
        for x, y in rows:
            self._model.learn_one(x, y)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinitialise the tree to a fresh untrained state.

        Call before each independent seed run to prevent cross-run state
        contamination.
        """
        self._model = self._build_model()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def grace_period(self) -> int:
        return self._grace_period

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def model_complexity(self) -> Dict[str, Any]:
        """Return tree structural complexity metrics for adaptation diagnostics.

        Extracts summary metrics from River's HoeffdingTreeClassifier if available,
        with graceful fallback across River versions.
        """
        complexity: Dict[str, Any] = {}
        if hasattr(self._model, "summary") and isinstance(self._model.summary, dict):
            complexity.update(self._model.summary)
        else:
            for attr in ("n_nodes", "height", "n_leaves", "n_active_leaves", "n_inactive_leaves", "n_branches"):
                val = getattr(self._model, attr, None)
                if val is not None:
                    complexity[attr] = val
        return complexity

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HoeffdingTreeLearner(grace_period={self._grace_period}, "
            f"delta={self._delta})"
        )

