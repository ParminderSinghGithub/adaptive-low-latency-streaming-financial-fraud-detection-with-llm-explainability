"""
Data ingestion, raw dataset loading, schema validation, and streaming-safe preprocessing modules.
"""

from src.data.loader import (
    DATASET_METADATA,
    SUPPORTED_DATASETS,
    get_dataset_metadata,
    load_dataset,
    load_ieee_cis,
    load_paysim,
    load_ulb,
)
from src.data.preprocessing import StreamingPreprocessor

__all__ = [
    "SUPPORTED_DATASETS",
    "DATASET_METADATA",
    "get_dataset_metadata",
    "load_dataset",
    "load_ieee_cis",
    "load_paysim",
    "load_ulb",
    "StreamingPreprocessor",
]
