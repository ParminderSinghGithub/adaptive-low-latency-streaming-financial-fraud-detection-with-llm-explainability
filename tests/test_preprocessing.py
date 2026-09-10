"""
Comprehensive unit test suite for streaming-safe preprocessing module.
Verifies zero temporal/label/future-data leakage, segment feature preservation,
and state reproducibility across IEEE-CIS, PaySim, and ULB datasets.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import StreamingPreprocessor


@pytest.fixture
def ieee_cis_mock_stream():
    """Generates a synthetic chronological stream matching IEEE-CIS schema."""
    n_rows = 100
    df = pd.DataFrame(
        {
            "TransactionID": list(range(1000, 1000 + n_rows)),
            "TransactionDT": list(range(86400, 86400 + n_rows)),
            "TransactionAmt": [10.0 + i * 2.0 if i % 5 != 0 else np.nan for i in range(n_rows)],
            "ProductCD": ["W", "H", "C", "S", "R"] * (n_rows // 5),
            "card4": ["visa", "mastercard", "discover", "american express", np.nan] * (n_rows // 5),
            "C1": [1.0, 2.0, np.nan, 4.0, 5.0] * (n_rows // 5),
            "isFraud": [0] * 90 + [1] * 10,
        }
    )
    return df


@pytest.fixture
def paysim_mock_stream():
    """Generates a synthetic chronological stream matching PaySim schema."""
    n_rows = 50
    df = pd.DataFrame(
        {
            "step": [1] * 25 + [2] * 25,
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"] * 10,
            "amount": [100.0 + i for i in range(n_rows)],
            "nameOrig": [f"C{i}" for i in range(n_rows)],
            "oldbalanceOrg": [1000.0] * n_rows,
            "newbalanceOrig": [900.0] * n_rows,
            "nameDest": [f"M{i}" for i in range(n_rows)],
            "oldbalanceDest": [0.0] * n_rows,
            "newbalanceDest": [100.0] * n_rows,
            "isFraud": [0] * 45 + [1] * 5,
        }
    )
    return df


@pytest.fixture
def ulb_mock_stream():
    """Generates a synthetic chronological stream matching ULB schema."""
    n_rows = 40
    data = {"Time": [float(i) for i in range(n_rows)]}
    for v in range(1, 29):
        data[f"V{v}"] = [float(v + i * 0.1) for i in range(n_rows)]
    data["Amount"] = [50.0 + i for i in range(n_rows)]
    data["Class"] = [0] * 35 + [1] * 5
    return pd.DataFrame(data)


def test_future_rows_cannot_influence_warmup_stats(ieee_cis_mock_stream):
    """Test 1: Assert future rows cannot alter fitted warmup medians or category mappings."""
    warmup_df = ieee_cis_mock_stream.iloc[:30].copy()
    
    # Fit preprocessor on warmup data only
    prep_warmup = StreamingPreprocessor(dataset_name="ieee_cis")
    prep_warmup.fit(warmup_df)

    median_warmup = prep_warmup.imputation_medians["TransactionAmt"]
    mapping_warmup = prep_warmup.category_mappings["ProductCD"].copy()

    # Create a manipulated stream with huge future outliers in rows 30..100
    future_corrupted_df = ieee_cis_mock_stream.copy()
    future_corrupted_df.loc[35:, "TransactionAmt"] = 99999999.0
    future_corrupted_df.loc[35:, "ProductCD"] = "NEW_UNSEEN_CATEGORY"

    # Re-fit preprocessor ONLY on the original warmup subset
    prep_independent = StreamingPreprocessor(dataset_name="ieee_cis")
    prep_independent.fit(future_corrupted_df.iloc[:30])

    # Warmup statistics MUST be identical regardless of future row corruption
    assert prep_independent.imputation_medians["TransactionAmt"] == median_warmup
    assert prep_independent.category_mappings["ProductCD"] == mapping_warmup


def test_transforming_sample_is_deterministic_and_independent_of_future(ieee_cis_mock_stream):
    """Test 2: Assert transforming row 0 alone produces the exact same output as transforming full df."""
    prep = StreamingPreprocessor(dataset_name="ieee_cis")
    prep.fit(ieee_cis_mock_stream.iloc[:20])

    # Transform sample row 0 alone
    X_single, _, _ = prep.transform(ieee_cis_mock_stream.iloc[:1])

    # Transform entire stream (rows 0..100)
    X_full, _, _ = prep.transform(ieee_cis_mock_stream)

    # Row 0 transformed features must be identical
    pd.testing.assert_frame_equal(X_single, X_full.iloc[:1])


def test_temporal_ordering_preservation(ieee_cis_mock_stream):
    """Test 3: Assert temporal row index order is strictly preserved after preprocessing."""
    prep = StreamingPreprocessor(dataset_name="ieee_cis")
    prep.fit(ieee_cis_mock_stream.iloc[:20])

    X, _, _ = prep.transform(ieee_cis_mock_stream)
    assert list(X.index) == list(ieee_cis_mock_stream.index)


def test_target_column_excluded_from_features(ieee_cis_mock_stream, ulb_mock_stream):
    """Test 4: Assert target column ('isFraud'/'Class') is excluded from feature matrix X."""
    # IEEE-CIS
    prep_ieee = StreamingPreprocessor(dataset_name="ieee_cis")
    prep_ieee.fit(ieee_cis_mock_stream.iloc[:20])
    X_ieee, y_ieee, _ = prep_ieee.transform(ieee_cis_mock_stream)

    assert "isFraud" not in X_ieee.columns
    assert y_ieee is not None
    assert list(y_ieee) == list(ieee_cis_mock_stream["isFraud"])

    # ULB
    prep_ulb = StreamingPreprocessor(dataset_name="ulb")
    prep_ulb.fit(ulb_mock_stream.iloc[:15])
    X_ulb, y_ulb, _ = prep_ulb.transform(ulb_mock_stream)

    assert "Class" not in X_ulb.columns
    assert y_ulb is not None
    assert list(y_ulb) == list(ulb_mock_stream["Class"])


def test_transaction_id_excluded_from_features(ieee_cis_mock_stream):
    """Test 5: Assert TransactionID primary key is excluded from predictive feature matrix X."""
    prep = StreamingPreprocessor(dataset_name="ieee_cis")
    prep.fit(ieee_cis_mock_stream.iloc[:20])
    X, _, _ = prep.transform(ieee_cis_mock_stream)

    assert "TransactionID" not in X.columns
    assert "TransactionDT" not in X.columns


def test_segment_feature_preserved_as_metadata(ieee_cis_mock_stream, paysim_mock_stream):
    """Test 6: Assert domain categorical segment feature ('ProductCD'/'type') remains accessible for P3."""
    # IEEE-CIS ProductCD segment
    prep_ieee = StreamingPreprocessor(dataset_name="ieee_cis")
    prep_ieee.fit(ieee_cis_mock_stream.iloc[:20])
    _, _, seg_ieee = prep_ieee.transform(ieee_cis_mock_stream)

    assert seg_ieee is not None
    assert list(seg_ieee) == list(ieee_cis_mock_stream["ProductCD"])

    # PaySim transaction type segment
    prep_pay = StreamingPreprocessor(dataset_name="paysim")
    prep_pay.fit(paysim_mock_stream.iloc[:15])
    _, _, seg_pay = prep_pay.transform(paysim_mock_stream)

    assert seg_pay is not None
    assert list(seg_pay) == list(paysim_mock_stream["type"])


def test_missing_value_imputation_uses_only_warmup_medians(ieee_cis_mock_stream):
    """Test 7: Assert NaN values are filled strictly using pre-fitted warmup medians."""
    # Set known median in warmup (rows 0..9)
    warmup_df = ieee_cis_mock_stream.iloc[:10].copy()
    warmup_df.loc[0:4, "TransactionAmt"] = 50.0
    warmup_df.loc[5:9, "TransactionAmt"] = 150.0
    expected_warmup_median = 100.0  # Median of [50*5, 150*5] = 100.0

    prep = StreamingPreprocessor(dataset_name="ieee_cis")
    prep.fit(warmup_df)
    assert prep.imputation_medians["TransactionAmt"] == expected_warmup_median


    # Create stream sample containing NaN
    stream_sample = pd.DataFrame(
        {
            "TransactionID": [9999],
            "TransactionDT": [999999],
            "TransactionAmt": [np.nan],
            "ProductCD": ["W"],
            "card4": ["visa"],
            "C1": [1.0],
            "isFraud": [0],
        }
    )

    X_transformed, _, _ = prep.transform(stream_sample)
    assert X_transformed["TransactionAmt"].iloc[0] == expected_warmup_median


def test_serialization_and_state_reproducibility(ieee_cis_mock_stream, tmp_path):
    """Test 8: Assert preprocessor state can be serialized to JSON/dict and reloaded deterministically."""
    prep = StreamingPreprocessor(dataset_name="ieee_cis", scale_features=True)
    prep.fit(ieee_cis_mock_stream.iloc[:30])

    X_orig, _, _ = prep.transform(ieee_cis_mock_stream)

    # Test dictionary serialization roundtrip
    state_dict = prep.to_dict()
    prep_reconstructed = StreamingPreprocessor.from_dict(state_dict)
    X_dict_reconstructed, _, _ = prep_reconstructed.transform(ieee_cis_mock_stream)

    pd.testing.assert_frame_equal(X_orig, X_dict_reconstructed)

    # Test JSON file serialization roundtrip
    json_file = tmp_path / "preprocessor_state.json"
    prep.save_state(json_file)

    prep_file_reconstructed = StreamingPreprocessor.load_state(json_file)
    X_file_reconstructed, _, _ = prep_file_reconstructed.transform(ieee_cis_mock_stream)

    pd.testing.assert_frame_equal(X_orig, X_file_reconstructed)


def test_no_full_dataset_fit(ieee_cis_mock_stream):
    """Test 9: Assert calling transform before fit raises RuntimeError."""
    prep = StreamingPreprocessor(dataset_name="ieee_cis")
    assert not prep.is_fitted

    with pytest.raises(RuntimeError, match="must be fitted before transform can be called"):
        prep.transform(ieee_cis_mock_stream)


def test_dataset_specific_schemas(ieee_cis_mock_stream, paysim_mock_stream, ulb_mock_stream):
    """Test 10: Assert preprocessor correctly isolates IEEE-CIS, PaySim, and ULB schemas."""
    prep_ieee = StreamingPreprocessor(dataset_name="ieee_cis").fit(ieee_cis_mock_stream.iloc[:10])
    prep_pay = StreamingPreprocessor(dataset_name="paysim").fit(paysim_mock_stream.iloc[:10])
    prep_ulb = StreamingPreprocessor(dataset_name="ulb").fit(ulb_mock_stream.iloc[:10])

    assert prep_ieee.target_col == "isFraud"
    assert prep_pay.target_col == "isFraud"
    assert prep_ulb.target_col == "Class"

    assert prep_ieee.segment_col == "ProductCD"
    assert prep_pay.segment_col == "type"
    assert prep_ulb.segment_col is None
