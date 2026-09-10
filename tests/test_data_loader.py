"""
Unit tests for raw dataset loader module (IEEE-CIS, PaySim, ULB).
Verifies schema validation, error handling, temporal ordering preservation, and identity merging.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.loader import (
    get_dataset_metadata,
    load_dataset,
    load_ieee_cis,
    load_paysim,
    load_ulb,
)


@pytest.fixture
def ieee_cis_fixture_dir(tmp_path):
    """Creates a temporary synthetic IEEE-CIS directory structure."""
    dir_path = tmp_path / "ieee_cis"
    dir_path.mkdir()

    # Train transaction CSV
    tx_df = pd.DataFrame(
        {
            "TransactionID": [100, 101, 102, 103, 104],
            "isFraud": [0, 0, 1, 0, 0],
            "TransactionDT": [86400, 86401, 86402, 86405, 86410],
            "TransactionAmt": [29.5, 43.0, 15.0, 100.0, 250.0],
            "ProductCD": ["W", "W", "C", "W", "H"],
        }
    )
    tx_df.to_csv(dir_path / "train_transaction.csv", index=False)

    # Train identity CSV
    id_df = pd.DataFrame(
        {
            "TransactionID": [102, 104],
            "id_01": [-5.0, 0.0],
            "id_02": [1000.0, 2000.0],
            "DeviceType": ["mobile", "desktop"],
        }
    )
    id_df.to_csv(dir_path / "train_identity.csv", index=False)

    return dir_path


@pytest.fixture
def paysim_fixture_dir(tmp_path):
    """Creates a temporary synthetic PaySim dataset directory."""
    dir_path = tmp_path / "paysim"
    dir_path.mkdir()

    df = pd.DataFrame(
        {
            "step": [1, 1, 2, 2, 3],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "TRANSFER"],
            "amount": [100.0, 5000.0, 200.0, 50.0, 10000.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
            "oldbalanceOrg": [100.0, 5000.0, 200.0, 50.0, 10000.0],
            "newbalanceOrig": [0.0, 0.0, 0.0, 0.0, 0.0],
            "nameDest": ["M1", "M2", "M3", "M4", "M5"],
            "oldbalanceDest": [0.0, 0.0, 0.0, 0.0, 0.0],
            "newbalanceDest": [100.0, 5000.0, 200.0, 50.0, 10000.0],
            "isFraud": [0, 0, 0, 0, 1],
            "isFlaggedFraud": [0, 0, 0, 0, 0],
        }
    )
    df.to_csv(dir_path / "PS_sample_log.csv", index=False)
    return dir_path


@pytest.fixture
def ulb_fixture_dir(tmp_path):
    """Creates a temporary synthetic ULB credit card dataset directory."""
    dir_path = tmp_path / "ulb"
    dir_path.mkdir()

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = {c: [0.0] * 5 for c in cols}
    data["Time"] = [0.0, 10.0, 20.0, 30.0, 40.0]
    data["Amount"] = [1.0, 25.5, 100.0, 0.5, 500.0]
    data["Class"] = [0, 0, 1, 0, 0]

    df = pd.DataFrame(data)
    df.to_csv(dir_path / "creditcard.csv", index=False)
    return dir_path


def test_metadata_retrieval():
    """Test get_dataset_metadata returns correct metadata structure."""
    meta_ieee = get_dataset_metadata("ieee_cis")
    assert meta_ieee["temporal_col"] == "TransactionDT"
    assert meta_ieee["target_col"] == "isFraud"
    assert meta_ieee["id_col"] == "TransactionID"

    meta_paysim = get_dataset_metadata("paysim")
    assert meta_paysim["temporal_col"] == "step"
    assert meta_paysim["target_col"] == "isFraud"

    meta_ulb = get_dataset_metadata("ulb")
    assert meta_ulb["temporal_col"] == "Time"
    assert meta_ulb["target_col"] == "Class"

    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_dataset_metadata("unknown_dataset")


def test_ieee_cis_loading_and_joining(ieee_cis_fixture_dir):
    """Test IEEE-CIS loading with and without identity join."""
    # Test joined loading
    df_joined = load_ieee_cis(ieee_cis_fixture_dir, split="train", join_identity=True)
    assert len(df_joined) == 5
    assert "TransactionID" in df_joined.columns
    assert "isFraud" in df_joined.columns
    assert "id_01" in df_joined.columns
    # Verify left-join behavior: non-matching rows have NaN for id_01
    assert df_joined.loc[df_joined["TransactionID"] == 100, "id_01"].isna().all()
    assert df_joined.loc[df_joined["TransactionID"] == 102, "id_01"].values[0] == -5.0

    # Test unjoined loading
    df_unjoined = load_ieee_cis(ieee_cis_fixture_dir, split="train", join_identity=False)
    assert len(df_unjoined) == 5
    assert "id_01" not in df_unjoined.columns


def test_paysim_loading(paysim_fixture_dir):
    """Test PaySim raw dataset loading and schema validation."""
    df = load_paysim(paysim_fixture_dir)
    assert len(df) == 5
    assert list(df["step"]) == [1, 1, 2, 2, 3]
    assert "isFraud" in df.columns
    assert "type" in df.columns


def test_ulb_loading(ulb_fixture_dir):
    """Test ULB credit card dataset loading and schema validation."""
    df = load_ulb(ulb_fixture_dir)
    assert len(df) == 5
    assert list(df["Time"]) == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert "V1" in df.columns


def test_unified_dispatcher(ieee_cis_fixture_dir, paysim_fixture_dir, ulb_fixture_dir):
    """Test unified load_dataset dispatcher function for all datasets."""
    df_ieee = load_dataset("ieee_cis", ieee_cis_fixture_dir)
    assert len(df_ieee) == 5

    df_pay = load_dataset("paysim", paysim_fixture_dir)
    assert len(df_pay) == 5

    df_ulb = load_dataset("ulb", ulb_fixture_dir)
    assert len(df_ulb) == 5

    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_dataset("unknown_dataset", ieee_cis_fixture_dir)


def test_missing_directory_failure():
    """Test clear FileNotFoundError when data directory does not exist."""
    fake_path = Path("/non_existent_dataset_directory_12345")
    with pytest.raises(FileNotFoundError, match="IEEE-CIS data directory does not exist"):
        load_ieee_cis(fake_path)

    with pytest.raises(FileNotFoundError, match="PaySim data directory does not exist"):
        load_paysim(fake_path)

    with pytest.raises(FileNotFoundError, match="ULB data directory does not exist"):
        load_ulb(fake_path)


def test_missing_required_column_failure(tmp_path):
    """Test ValueError when loaded dataset CSV is missing required schema columns."""
    dir_path = tmp_path / "corrupted_ieee"
    dir_path.mkdir()

    # Missing 'isFraud' column in train_transaction.csv
    bad_df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 101],
            # 'isFraud' is missing
        }
    )
    bad_df.to_csv(dir_path / "train_transaction.csv", index=False)

    with pytest.raises(ValueError, match="missing required column"):
        load_ieee_cis(dir_path, split="train")


def test_temporal_ordering_preservation(ieee_cis_fixture_dir, paysim_fixture_dir, ulb_fixture_dir):
    """Test that loading preserves exact raw row ordering without shuffling."""
    df_ieee = load_ieee_cis(ieee_cis_fixture_dir)
    assert list(df_ieee["TransactionDT"]) == [86400, 86401, 86402, 86405, 86410]

    df_pay = load_paysim(paysim_fixture_dir)
    assert list(df_pay["amount"]) == [100.0, 5000.0, 200.0, 50.0, 10000.0]

    df_ulb = load_ulb(ulb_fixture_dir)
    assert list(df_ulb["Time"]) == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_no_scientific_preprocessing(ieee_cis_fixture_dir):
    """Assert that raw missing values (NaNs) and unencoded categorical features are untouched."""
    df = load_ieee_cis(ieee_cis_fixture_dir)

    # Missing identity attributes should remain NaNs
    assert df["id_01"].isna().sum() > 0

    # Categorical ProductCD should remain raw strings (unencoded)
    assert isinstance(df["ProductCD"].iloc[0], str)
    assert pd.api.types.is_string_dtype(df["ProductCD"])

