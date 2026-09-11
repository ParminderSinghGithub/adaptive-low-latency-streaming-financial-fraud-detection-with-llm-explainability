"""
Chronological stream generator for prequential evaluation.

Converts a pandas DataFrame (already preprocessed and temporally sorted)
into a row-by-row iterator that yields (x_dict, y, segment_key) tuples.

Design constraints (Source of Truth §16.2):
- No shuffling: strict chronological order is preserved.
- Labels are NEVER yielded before the corresponding prediction step.
- Segment key is passed through transparently — the generator does not
  interpret it.
- Warmup slice and streaming slice are separated by the caller.
"""

from __future__ import annotations

from typing import Any, Dict, Generator, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Type alias for a single prequential event
# ---------------------------------------------------------------------------

# (feature_dict, label, segment_key_or_None)
StreamEvent = Tuple[Dict[str, Any], int, Optional[str]]


def chronological_stream(
    X: pd.DataFrame,
    y: pd.Series,
    segment: Optional[pd.Series] = None,
) -> Generator[StreamEvent, None, None]:
    """Yield one prequential event per row, in original (chronological) order.

    Parameters
    ----------
    X:
        Feature DataFrame (already preprocessed; index must align with y).
    y:
        Integer target Series (0 = legitimate, 1 = fraud).
    segment:
        Optional Series of categorical segment key strings (e.g. ProductCD
        values for IEEE-CIS, transaction type for PaySim).  When ``None``,
        the yielded segment_key will be ``None`` for every row.

    Yields
    ------
    (x_dict, label, segment_key) where:
        x_dict      — dict mapping feature name → value for one transaction
        label       — int (0 or 1)
        segment_key — str or None

    Notes
    -----
    The generator does NOT split warmup from stream.  That responsibility
    belongs to the caller (typically :class:`~src.pipeline.runner.PrequentialRunner`).
    """
    col_names = X.columns.tolist()
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    seg_arr = segment.to_numpy() if segment is not None else None

    for i in range(len(X_arr)):
        x_dict: Dict[str, Any] = dict(zip(col_names, X_arr[i]))
        label: int = int(y_arr[i])
        seg_key: Optional[str] = str(seg_arr[i]) if seg_arr is not None else None
        yield x_dict, label, seg_key
