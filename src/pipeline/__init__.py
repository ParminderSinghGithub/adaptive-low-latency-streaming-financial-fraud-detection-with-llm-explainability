"""
Prequential pipeline orchestration layer.

Exports the runner, result structures, and policy/detector name mappings.
"""

from src.pipeline.runner import (
    PrequentialRunner,
    RunResult,
    StreamingRecord,
    _CONFIG_TO_DETECTOR,
    _CONFIG_TO_POLICY,
)

__all__ = [
    "PrequentialRunner",
    "RunResult",
    "StreamingRecord",
    "_CONFIG_TO_POLICY",
    "_CONFIG_TO_DETECTOR",
]
