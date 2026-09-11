"""
Streaming-safe feature preprocessing module for financial fraud detection benchmarks.
Enforces prequential test-then-train protocol with zero future-data or label leakage.
All statistics (imputation medians, category encodings, scaling parameters) are fitted
exclusively on historical warmup data (t_0 -> t_warmup).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

from src.data.loader import get_dataset_metadata


class StreamingPreprocessor:
    """
    Stateful streaming-safe preprocessor for benchmark datasets (IEEE-CIS, PaySim, ULB).
    
    Fits imputation statistics, categorical index encodings, and scaling parameters
    exclusively on warmup historical data. Transforms streaming samples prequentially
    without leaking future observations or ground-truth labels.
    """

    def __init__(
        self,
        dataset_name: str,
        scale_features: bool = False,
        custom_categorical_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize StreamingPreprocessor.

        Args:
            dataset_name: Supported dataset identifier ('ieee_cis', 'paysim', 'ulb').
            scale_features: If True, applies standard scaling (z-score) using warmup mean & std.
            custom_categorical_cols: Optional list of explicit categorical column names.
        """
        self.dataset_name = dataset_name.lower()
        self.scale_features = scale_features
        self.metadata = get_dataset_metadata(self.dataset_name)

        self.target_col: str = self.metadata["target_col"]
        self.temporal_col: str = self.metadata["temporal_col"]
        self.id_col: Optional[str] = self.metadata.get("id_col")

        # Determine default segment column
        if self.dataset_name == "ieee_cis":
            self.segment_col: Optional[str] = "ProductCD"
        elif self.dataset_name == "paysim":
            self.segment_col: Optional[str] = "type"
        else:
            self.segment_col = None

        self.custom_categorical_cols = custom_categorical_cols or []

        # Fitted state attributes
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.imputation_medians: Dict[str, float] = {}
        self.scaling_means: Dict[str, float] = {}
        self.scaling_stds: Dict[str, float] = {}
        self.category_mappings: Dict[str, Dict[str, int]] = {}

    def fit(self, df_warmup: pd.DataFrame) -> "StreamingPreprocessor":
        """
        Fit preprocessing statistics exclusively on warmup historical dataset.

        Args:
            df_warmup: Historical warmup dataframe (t_0 -> t_warmup).

        Returns:
            Self (fitted preprocessor instance).
        """
        if not isinstance(df_warmup, pd.DataFrame):
            raise ValueError("df_warmup must be a pandas DataFrame.")
        if df_warmup.empty:
            raise ValueError("df_warmup cannot be empty.")

        # Exclude non-feature columns (target, ID, temporal)
        excluded_cols = {self.target_col, self.temporal_col}
        if self.id_col:
            excluded_cols.add(self.id_col)

        # Identify candidate feature columns in original order
        candidate_cols = [c for c in df_warmup.columns if c not in excluded_cols]

        self.categorical_cols = []
        self.numerical_cols = []
        self.feature_names = candidate_cols.copy()

        # Categorize columns based on dtypes and custom specs
        for col in candidate_cols:
            is_custom_cat = col in self.custom_categorical_cols
            is_dtype_cat = (
                isinstance(df_warmup[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(df_warmup[col])
                or pd.api.types.is_string_dtype(df_warmup[col])
            )

            if is_custom_cat or is_dtype_cat:
                self.categorical_cols.append(col)
            else:
                self.numerical_cols.append(col)

        # 1. Fit Imputation Medians & Scaling for Numerical Features
        self.imputation_medians = {}
        self.scaling_means = {}
        self.scaling_stds = {}

        for col in self.numerical_cols:
            series = df_warmup[col].dropna()
            if len(series) > 0:
                median_val = float(series.median())
                mean_val = float(series.mean())
                std_val = float(series.std(ddof=0))
            else:
                median_val = 0.0
                mean_val = 0.0
                std_val = 1.0

            # Guard against zero variance std
            if std_val == 0.0 or np.isnan(std_val):
                std_val = 1.0

            self.imputation_medians[col] = median_val
            self.scaling_means[col] = mean_val
            self.scaling_stds[col] = std_val

        # 2. Fit Category Mappings for Categorical Features
        # Reserved index 0: [UNKNOWN] or missing category
        self.category_mappings = {}

        for col in self.categorical_cols:
            unique_cats = df_warmup[col].dropna().unique()
            # Sort categories for deterministic mapping
            sorted_cats = sorted([str(c) for c in unique_cats])
            mapping = {cat_str: idx + 1 for idx, cat_str in enumerate(sorted_cats)}
            self.category_mappings[col] = mapping

        self.is_fitted = True
        return self

    def fit_transform(
        self, df_warmup: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """Fit on warmup data and return transformed warmup features, labels, and segment tags."""
        self.fit(df_warmup)
        return self.transform(df_warmup)

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
        """
        Transform streaming sample dataframe using pre-fitted warmup statistics.

        Args:
            df: Input dataframe to transform.

        Returns:
            Tuple of (X_transformed, y_target, segment_series):
            - X_transformed: Processed numerical/encoded feature matrix.
            - y_target: Extracted target Series (if present in df), else None.
            - segment_series: Preserved raw domain categorical segment Series (if present), else None.
        """
        if not self.is_fitted:
            raise RuntimeError("StreamingPreprocessor must be fitted before transform can be called.")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df must be a pandas DataFrame.")

        # 1. Extract Target Label (if present)
        y_target: Optional[pd.Series] = None
        if self.target_col in df.columns:
            y_target = df[self.target_col].copy()

        # 2. Preserve Raw Segment Feature (if present) for P3 Segmentation Module
        segment_series: Optional[pd.Series] = None
        if self.segment_col and self.segment_col in df.columns:
            segment_series = df[self.segment_col].astype(str).copy()
        elif self.segment_col is None:
            segment_series = pd.Series(["all"] * len(df), index=df.index, dtype=str)

        # 3. Construct Feature Matrix X
        feature_dict: Dict[str, Any] = {}

        # Process Numerical Features
        for col in self.numerical_cols:
            if col in df.columns:
                series = df[col].astype(float).copy()
                # Fill missing NaNs using pre-fitted warmup median
                series = series.fillna(self.imputation_medians[col])

                if self.scale_features:
                    mean_val = self.scaling_means[col]
                    std_val = self.scaling_stds[col]
                    series = (series - mean_val) / std_val

                feature_dict[col] = series
            else:
                # Missing column in input stream: fill with warmup median
                default_val = self.imputation_medians[col]
                feature_dict[col] = default_val

        # Process Categorical Features
        for col in self.categorical_cols:
            if col in df.columns:
                mapping = self.category_mappings[col]
                # Convert input series to string and map to pre-fitted index (unseen -> 0)
                series_str = df[col].astype(str)
                # Map categories: missing/NaN or unseen categories become 0 [UNKNOWN]
                mapped_series = series_str.map(mapping).fillna(0).astype(int)
                # Ensure null inputs in original data map to 0
                mapped_series[df[col].isna()] = 0
                feature_dict[col] = mapped_series
            else:
                feature_dict[col] = 0

        # Construct DataFrame in one go to prevent memory fragmentation
        X_df = pd.DataFrame(feature_dict, index=df.index)

        # Enforce exact feature order established during fit
        X_df = X_df[self.feature_names]

        return X_df, y_target, segment_series


    def to_dict(self) -> Dict[str, Any]:
        """Serialize preprocessor state to dictionary for logging/reproducibility."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted StreamingPreprocessor.")

        return {
            "dataset_name": self.dataset_name,
            "scale_features": self.scale_features,
            "target_col": self.target_col,
            "temporal_col": self.temporal_col,
            "id_col": self.id_col,
            "segment_col": self.segment_col,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "categorical_cols": self.categorical_cols,
            "numerical_cols": self.numerical_cols,
            "imputation_medians": self.imputation_medians,
            "scaling_means": self.scaling_means,
            "scaling_stds": self.scaling_stds,
            "category_mappings": self.category_mappings,
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "StreamingPreprocessor":
        """Reconstruct StreamingPreprocessor instance from state dictionary."""
        instance = cls(
            dataset_name=state_dict["dataset_name"],
            scale_features=state_dict["scale_features"],
        )
        instance.target_col = state_dict["target_col"]
        instance.temporal_col = state_dict["temporal_col"]
        instance.id_col = state_dict["id_col"]
        instance.segment_col = state_dict["segment_col"]
        instance.is_fitted = state_dict["is_fitted"]
        instance.feature_names = state_dict["feature_names"]
        instance.categorical_cols = state_dict["categorical_cols"]
        instance.numerical_cols = state_dict["numerical_cols"]
        instance.imputation_medians = state_dict["imputation_medians"]
        instance.scaling_means = state_dict["scaling_means"]
        instance.scaling_stds = state_dict["scaling_stds"]
        instance.category_mappings = state_dict["category_mappings"]
        return instance

    def save_state(self, filepath: Union[str, Path]) -> None:
        """Save preprocessor state to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_state(cls, filepath: Union[str, Path]) -> "StreamingPreprocessor":
        """Load preprocessor state from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor state file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            state_dict = json.load(f)
        return cls.from_dict(state_dict)
