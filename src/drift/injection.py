"""
Synthetic Concept Drift Injection Engine for Objective 2 (Experiment E3).

Implements controlled exogenous concept drift injection over streaming data
according to thesis methodology (Source of Truth §6, §7 DEC-05, §11 E3).

Supported drift regimes:
1. Sudden:    Abrupt permutation of discriminative feature mappings at stream index t_drift.
2. Gradual:   Sigmoidal transition between Concept A and Concept B over transition_width.
3. Recurring: Periodic oscillation between Concept A and Concept B with recurrence_period.
4. Localized: Perturbation applied strictly to target segment (e.g. ProductCD == 'C');
              all other segments remain unaffected.

Guarantees:
- Strict determinism given random seed.
- Zero future-data or ground-truth label leakage (perturbations modify feature space only).
- Complete preservation of transaction ordering, timestamps, and row indices.
- Rich ground-truth metadata emitted per event for event-centered evaluation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DriftRegime(str, Enum):
    """Supported concept drift regimes for Experiment E3."""
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    RECURRING = "recurring"
    LOCALIZED = "localized"


@dataclass
class DriftEventMetadata:
    """Ground-truth metadata emitted for every stream event under synthetic drift.

    Attributes
    ----------
    stream_idx:
        0-based index within the streaming slice (post-warmup).
    regime:
        The active drift regime identifier (sudden, gradual, recurring, localized).
    event_id:
        Descriptive event label for the current regime/cycle state.
    t_drift:
        Configured onset index of the drift event.
    is_affected:
        True if this specific transaction was altered under Concept B;
        False if it represents unaltered Concept A baseline.
    active_concept:
        "concept_a" (unperturbed baseline) or "concept_b" (perturbed drifting).
    transition_interval:
        (t_start, t_end) range for gradual transitions, or None.
    recurrence_interval:
        Length of period in transactions if recurring, or None.
    target_segment:
        Categorical segment key targeted under localized drift, or None.
    segment_key:
        Observed segment key of this transaction.
    drift_prob:
        Sigmoidal probability of sampling Concept B for gradual drift;
        1.0 for active sudden/recurring/localized, 0.0 for baseline.
    """
    stream_idx: int
    regime: str
    event_id: str
    t_drift: int
    is_affected: bool
    active_concept: str
    transition_interval: Optional[Tuple[int, int]] = None
    recurrence_interval: Optional[int] = None
    target_segment: Optional[str] = None
    segment_key: Optional[str] = None
    drift_prob: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        """Convert metadata to plain dictionary for logging and serialization."""
        return {
            "stream_idx": self.stream_idx,
            "regime": self.regime,
            "event_id": self.event_id,
            "t_drift": self.t_drift,
            "is_affected": self.is_affected,
            "active_concept": self.active_concept,
            "transition_interval": self.transition_interval,
            "recurrence_interval": self.recurrence_interval,
            "target_segment": self.target_segment,
            "segment_key": self.segment_key,
            "drift_prob": self.drift_prob,
        }


@dataclass
class E3DriftConfig:
    """Configuration specification for Experiment E3 drift injection.

    Attributes
    ----------
    regime:
        Drift regime name ("sudden", "gradual", "recurring", "localized").
    t_drift:
        Stream index (0-based post-warmup) where drift onset commences.
    transition_width:
        Number of transactions over which gradual sigmoid transition occurs.
        Default = 5,000 transactions.
    recurrence_period:
        Number of transactions per concept cycle under recurring drift.
        Default = 25,000 transactions.
    target_segment:
        Target segment key for localized drift (default "C" for ProductCD=='C').
    segment_col:
        Column name used for segmentation (default "ProductCD").
    feature_pairs:
        List of 2-tuples of feature names to permute/swap under Concept B.
        Defaults to high-importance velocity and card features:
        [("C1", "C2"), ("C3", "C4"), ("card1", "card2")].
    window_size:
        Retraining window size locked to 10,000 for E3.
    seed:
        Random seed for reproducible stochastic transitions.
    """
    regime: Union[DriftRegime, str] = DriftRegime.SUDDEN
    t_drift: int = 10000
    transition_width: int = 5000
    recurrence_period: int = 25000
    target_segment: Optional[str] = "C"
    segment_col: str = "ProductCD"
    feature_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [("C1", "C2"), ("C3", "C4"), ("card1", "card2")]
    )
    window_size: int = 10000
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.regime, str):
            self.regime = DriftRegime(self.regime.lower())
        if self.t_drift < 0:
            raise ValueError(f"t_drift must be non-negative, got {self.t_drift}.")
        if self.transition_width <= 0:
            raise ValueError(f"transition_width must be > 0, got {self.transition_width}.")
        if self.recurrence_period <= 0:
            raise ValueError(f"recurrence_period must be > 0, got {self.recurrence_period}.")
        if self.window_size != 10000:
            raise ValueError(f"E3 adaptation window size must be exactly 10,000, got {self.window_size}.")
        if not self.feature_pairs:
            raise ValueError("feature_pairs cannot be empty; specify at least one feature pair to swap.")

    def as_dict(self) -> Dict[str, Any]:
        """Serialize configuration parameters."""
        return {
            "regime": self.regime.value if isinstance(self.regime, DriftRegime) else str(self.regime),
            "t_drift": self.t_drift,
            "transition_width": self.transition_width,
            "recurrence_period": self.recurrence_period,
            "target_segment": self.target_segment,
            "segment_col": self.segment_col,
            "feature_pairs": self.feature_pairs,
            "window_size": self.window_size,
            "seed": self.seed,
        }


class SyntheticDriftInjector:
    """Stateful, deterministic concept drift injector for streaming experiments.

    Transforms streaming feature vectors according to the active drift regime
    and emits authoritative ground-truth metadata per transaction.
    """

    def __init__(self, config: E3DriftConfig) -> None:
        self.config = config
        self._rng = np.random.RandomState(config.seed)
        self._pairs = list(config.feature_pairs)

    @property
    def regime(self) -> DriftRegime:
        return DriftRegime(self.config.regime)

    def apply_perturbation(self, x_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Concept B feature mapping permutation to a feature dictionary.

        Swaps values between configured feature pairs without touching ground-truth
        labels, preserving original dictionary types and missingness profiles.
        """
        x_perturbed = copy.copy(x_dict)
        for feat_a, feat_b in self._pairs:
            if feat_a in x_perturbed and feat_b in x_perturbed:
                val_a = x_perturbed[feat_a]
                val_b = x_perturbed[feat_b]
                x_perturbed[feat_a] = val_b
                x_perturbed[feat_b] = val_a
        return x_perturbed

    def evaluate_drift_state(
        self, stream_idx: int, segment_key: Optional[str] = None
    ) -> DriftEventMetadata:
        """Compute the ground-truth concept state and metadata for a given stream index."""
        regime = self.regime
        t_drift = self.config.t_drift
        seg_str = str(segment_key) if segment_key is not None else None
        target_seg = str(self.config.target_segment) if self.config.target_segment is not None else None

        # -------------------------------------------------------------
        # 1. Sudden Drift
        # -------------------------------------------------------------
        if regime == DriftRegime.SUDDEN:
            is_affected = stream_idx >= t_drift
            prob = 1.0 if is_affected else 0.0
            event_id = "sudden_concept_b" if is_affected else "baseline_concept_a"
            return DriftEventMetadata(
                stream_idx=stream_idx,
                regime=regime.value,
                event_id=event_id,
                t_drift=t_drift,
                is_affected=is_affected,
                active_concept="concept_b" if is_affected else "concept_a",
                segment_key=seg_str,
                drift_prob=prob,
            )

        # -------------------------------------------------------------
        # 2. Gradual Drift (Sigmoid Transition)
        # -------------------------------------------------------------
        elif regime == DriftRegime.GRADUAL:
            w_trans = self.config.transition_width
            t_end = t_drift + w_trans

            if stream_idx < t_drift:
                prob = 0.0
                is_affected = False
                event_id = "gradual_pre_drift_a"
            elif stream_idx >= t_end:
                prob = 1.0
                is_affected = True
                event_id = "gradual_post_drift_b"
            else:
                # Sigmoidal probability curve over [t_drift, t_drift + w_trans]
                # Center sigmoid at midpoint; scale exponent so curve spans ~0.007 to ~0.993
                t_mid = t_drift + (w_trans / 2.0)
                k = 10.0 / float(w_trans)
                exponent = -k * (float(stream_idx) - t_mid)
                # Clip to prevent numerical overflow in exp
                exponent = float(np.clip(exponent, -50.0, 50.0))
                prob = 1.0 / (1.0 + np.exp(exponent))

                # Deterministic pseudo-random Bernoulli sampling per stream index and seed
                # Hash stream_idx and seed to guarantee seed-invariant stream replay
                draw_rng = np.random.RandomState(self.config.seed + stream_idx * 1009)
                is_affected = bool(draw_rng.uniform(0.0, 1.0) < prob)
                event_id = "gradual_transition_window"

            return DriftEventMetadata(
                stream_idx=stream_idx,
                regime=regime.value,
                event_id=event_id,
                t_drift=t_drift,
                is_affected=is_affected,
                active_concept="concept_b" if is_affected else "concept_a",
                transition_interval=(t_drift, t_end),
                segment_key=seg_str,
                drift_prob=float(prob),
            )

        # -------------------------------------------------------------
        # 3. Recurring Drift (Periodic Alternation)
        # -------------------------------------------------------------
        elif regime == DriftRegime.RECURRING:
            period = self.config.recurrence_period
            if stream_idx < t_drift:
                is_affected = False
                event_id = "recurring_pre_drift_a"
                prob = 0.0
            else:
                cycle_idx = (stream_idx - t_drift) // period
                # Cycle 0: Concept B, Cycle 1: Concept A, Cycle 2: Concept B, ...
                is_concept_b = bool(cycle_idx % 2 == 0)
                is_affected = is_concept_b
                prob = 1.0 if is_affected else 0.0
                event_id = f"recurring_cycle_{cycle_idx}_{'b' if is_concept_b else 'a'}"

            return DriftEventMetadata(
                stream_idx=stream_idx,
                regime=regime.value,
                event_id=event_id,
                t_drift=t_drift,
                is_affected=is_affected,
                active_concept="concept_b" if is_affected else "concept_a",
                recurrence_interval=period,
                segment_key=seg_str,
                drift_prob=prob,
            )

        # -------------------------------------------------------------
        # 4. Localized Drift (Target Segment Only)
        # -------------------------------------------------------------
        elif regime == DriftRegime.LOCALIZED:
            is_post_drift = stream_idx >= t_drift
            is_target = bool(seg_str == target_seg)
            is_affected = is_post_drift and is_target
            prob = 1.0 if is_affected else 0.0

            if not is_post_drift:
                event_id = "localized_pre_drift_a"
            elif is_target:
                event_id = f"localized_target_segment_{target_seg}_b"
            else:
                event_id = f"localized_unaffected_segment_{seg_str}_a"

            return DriftEventMetadata(
                stream_idx=stream_idx,
                regime=regime.value,
                event_id=event_id,
                t_drift=t_drift,
                is_affected=is_affected,
                active_concept="concept_b" if is_affected else "concept_a",
                target_segment=target_seg,
                segment_key=seg_str,
                drift_prob=prob,
            )

        else:
            raise ValueError(f"Unrecognized drift regime: {regime}")

    def inject_event(
        self,
        x_dict: Dict[str, Any],
        y: int,
        segment_key: Optional[str],
        stream_idx: int,
    ) -> Tuple[Dict[str, Any], int, Optional[str], DriftEventMetadata]:
        """Process a single prequential event and return perturbed features + metadata."""
        meta = self.evaluate_drift_state(stream_idx, segment_key)
        if meta.is_affected:
            x_out = self.apply_perturbation(x_dict)
        else:
            x_out = x_dict
        return x_out, y, segment_key, meta

    def inject_stream(
        self,
        stream_events: Iterable[Tuple[Dict[str, Any], int, Optional[str]]],
    ) -> Generator[Tuple[Dict[str, Any], int, Optional[str], DriftEventMetadata], None, None]:
        """Wrap an existing prequential generator, yielding perturbed events with metadata."""
        for stream_idx, (x_dict, y, seg_key) in enumerate(stream_events):
            yield self.inject_event(x_dict, y, seg_key, stream_idx)

    def inject_dataframe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        segment: Optional[pd.Series] = None,
        warmup_size: int = 0,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series], List[DriftEventMetadata]]:
        """Apply drift injection directly to pandas streaming dataframes.

        Warmup rows (indices 0 .. warmup_size-1) are preserved strictly unaltered
        (Concept A baseline) so warmup statistics represent uncorrupted state.
        The post-warmup stream slice (indices warmup_size .. N-1) is evaluated and
        perturbed according to the configured drift regime.

        Returns
        -------
        Tuple of (X_drift, y, segment, metadata_list)
        """
        n_total = len(X)
        if warmup_size > n_total:
            raise ValueError(f"warmup_size ({warmup_size}) exceeds dataframe length ({n_total}).")

        X_drift = X.copy()
        metadata_list: List[DriftEventMetadata] = []

        # Warmup slice receives neutral baseline metadata
        for w_idx in range(warmup_size):
            seg_val = str(segment.iloc[w_idx]) if segment is not None else None
            metadata_list.append(
                DriftEventMetadata(
                    stream_idx=w_idx - warmup_size,  # negative stream index indicates warmup
                    regime=self.regime.value,
                    event_id="warmup_baseline",
                    t_drift=self.config.t_drift,
                    is_affected=False,
                    active_concept="concept_a",
                    target_segment=self.config.target_segment,
                    segment_key=seg_val,
                    drift_prob=0.0,
                )
            )

        # Stream slice
        n_stream = n_total - warmup_size
        col_names = X.columns.tolist()
        pairs = self._pairs

        # Precompute feature column swap pairs present in X
        pair_indices: List[Tuple[str, str]] = [
            (f_a, f_b) for f_a, f_b in pairs if f_a in col_names and f_b in col_names
        ]

        affected_global_indices: List[int] = []
        for local_idx in range(n_stream):
            global_idx = warmup_size + local_idx
            seg_val = str(segment.iloc[global_idx]) if segment is not None else None
            meta = self.evaluate_drift_state(local_idx, seg_val)
            metadata_list.append(meta)
            if meta.is_affected:
                affected_global_indices.append(global_idx)

        # Vectorized batch swapping for affected rows
        if affected_global_indices and pair_indices:
            affected_idx_arr = np.array(affected_global_indices, dtype=int)
            for f_a, f_b in pair_indices:
                col_a_vals = X_drift.iloc[affected_idx_arr, X_drift.columns.get_loc(f_a)].to_numpy()
                col_b_vals = X_drift.iloc[affected_idx_arr, X_drift.columns.get_loc(f_b)].to_numpy()
                X_drift.iloc[affected_idx_arr, X_drift.columns.get_loc(f_a)] = col_b_vals
                X_drift.iloc[affected_idx_arr, X_drift.columns.get_loc(f_b)] = col_a_vals

        return X_drift, y, segment, metadata_list
