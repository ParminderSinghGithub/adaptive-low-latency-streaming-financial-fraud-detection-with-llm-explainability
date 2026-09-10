"""
Raw dataset loader module for financial fraud detection benchmarks.
Provides clean, explicit loading interfaces for IEEE-CIS, PaySim, and ULB datasets.
Preserves raw features, missing values, class distributions, and temporal ordering.
No scientific preprocessing, feature selection, scaling, or imputation is performed here.
"""

from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd

SUPPORTED_DATASETS = {"ieee_cis", "paysim", "ulb"}

# Metadata definitions required by downstream streaming and preprocessing modules
DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "ieee_cis": {
        "name": "IEEE-CIS Fraud Detection",
        "temporal_col": "TransactionDT",
        "target_col": "isFraud",
        "id_col": "TransactionID",
        "dataset_type": "real_world",
        "default_split": "train",
    },
    "paysim": {
        "name": "PaySim Financial Logs",
        "temporal_col": "step",
        "target_col": "isFraud",
        "id_col": None,
        "dataset_type": "synthetic",
        "default_split": "full",
    },
    "ulb": {
        "name": "ULB Credit Card Fraud",
        "temporal_col": "Time",
        "target_col": "Class",
        "id_col": None,
        "dataset_type": "prototype",
        "default_split": "full",
    },
}

# Required raw schema columns for validation
REQUIRED_COLUMNS: Dict[str, list[str]] = {
    "ieee_cis_train": ["TransactionID", "TransactionDT", "isFraud"],
    "ieee_cis_test": ["TransactionID", "TransactionDT"],
    "paysim": [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ],
    "ulb": ["Time", "Amount", "Class"],
}


def get_dataset_metadata(name: str) -> Dict[str, Any]:
    """
    Retrieve metadata for a supported benchmark dataset.

    Args:
        name: Name of the dataset ('ieee_cis', 'paysim', 'ulb').

    Returns:
        Dict containing dataset metadata (temporal column, target column, etc.).

    Raises:
        ValueError: If dataset name is not supported.
    """
    key = name.lower()
    if key not in DATASET_METADATA:
        raise ValueError(
            f"Unsupported dataset '{name}'. Allowed datasets: {sorted(list(SUPPORTED_DATASETS))}"
        )
    return DATASET_METADATA[key].copy()


def load_ieee_cis(
    data_dir: Union[str, Path],
    split: str = "train",
    join_identity: bool = True,
) -> pd.DataFrame:
    """
    Load raw IEEE-CIS Fraud Detection transaction (and identity) data.

    Args:
        data_dir: Directory containing raw IEEE-CIS CSV files.
        split: 'train' or 'test'.
        join_identity: If True, left-joins transaction data with identity data on TransactionID.

    Returns:
        pd.DataFrame containing raw IEEE-CIS transaction features.

    Raises:
        FileNotFoundError: If required transaction or identity CSV file is missing.
        ValueError: If split is invalid or required schema columns are missing.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"IEEE-CIS data directory does not exist: {dir_path}")

    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split '{split}'. Allowed splits for IEEE-CIS: 'train', 'test'")

    tx_filename = f"{split}_transaction.csv"
    tx_path = dir_path / tx_filename

    # If file is not in dir_path directly, search subdirectories (e.g. dir_path / 'ieee_cis')
    if not tx_path.exists():
        candidates = list(dir_path.rglob(tx_filename))
        if candidates:
            tx_path = candidates[0]
            dir_path = tx_path.parent
        else:
            raise FileNotFoundError(f"Missing required transaction file: {tx_path}")

    df_tx = pd.read_csv(tx_path)

    # Validate required columns
    required_cols = REQUIRED_COLUMNS[f"ieee_cis_{split}"]
    missing_cols = [c for c in required_cols if c not in df_tx.columns]
    if missing_cols:
        raise ValueError(
            f"IEEE-CIS {split} transaction file is missing required column(s): {missing_cols}"
        )

    if not join_identity:
        return df_tx

    id_filename = f"{split}_identity.csv"
    id_path = dir_path / id_filename

    if not id_path.exists():
        id_candidates = list(dir_path.rglob(id_filename))
        if id_candidates:
            id_path = id_candidates[0]

    if id_path.exists():
        df_id = pd.read_csv(id_path)
        if "TransactionID" not in df_id.columns:
            raise ValueError(f"IEEE-CIS {split} identity file missing 'TransactionID' column.")

        # Left join on TransactionID to preserve all transactions and original row count
        df_joined = pd.merge(df_tx, df_id, on="TransactionID", how="left")
        return df_joined

    return df_tx


def load_paysim(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw PaySim financial transaction log CSV.

    Args:
        data_dir: Directory containing raw PaySim CSV file.

    Returns:
        pd.DataFrame containing raw PaySim transactions.

    Raises:
        FileNotFoundError: If data directory or PaySim CSV file is missing.
        ValueError: If required PaySim schema columns are missing.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"PaySim data directory does not exist: {dir_path}")

    csv_files = sorted(dir_path.glob("*.csv"))
    if not csv_files:
        csv_files = sorted(dir_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in PaySim directory: {dir_path}")

    # Select the largest CSV file in the directory if multiple exist
    paysim_file = max(csv_files, key=lambda f: f.stat().st_size)
    df = pd.read_csv(paysim_file)

    # Validate schema
    missing_cols = [c for c in REQUIRED_COLUMNS["paysim"] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"PaySim dataset in {paysim_file.name} is missing required column(s): {missing_cols}"
        )

    return df


def load_ulb(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw ULB Credit Card Fraud dataset CSV.

    Args:
        data_dir: Directory containing raw creditcard.csv file.

    Returns:
        pd.DataFrame containing raw ULB transactions.

    Raises:
        FileNotFoundError: If data directory or creditcard.csv file is missing.
        ValueError: If required ULB schema columns are missing.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"ULB data directory does not exist: {dir_path}")

    ulb_path = dir_path / "creditcard.csv"
    if not ulb_path.exists():
        candidates = list(dir_path.rglob("creditcard.csv"))
        if candidates:
            ulb_path = candidates[0]
        else:
            raise FileNotFoundError(f"Missing required ULB CSV file: {ulb_path}")

    df = pd.read_csv(ulb_path)

    # Validate schema
    missing_cols = [c for c in REQUIRED_COLUMNS["ulb"] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"ULB dataset in {ulb_path.name} is missing required column(s): {missing_cols}"
        )

    return df


def load_dataset(
    name: str,
    data_dir: Union[str, Path],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Unified dataset dispatcher for loading raw benchmark datasets.

    Args:
        name: Name of dataset ('ieee_cis', 'paysim', 'ulb').
        data_dir: Directory path containing the raw dataset files.
        **kwargs: Additional dataset-specific loading options (e.g. split='train', join_identity=True).

    Returns:
        pd.DataFrame containing the un-preprocessed raw dataset.
    """
    key = name.lower()
    if key == "ieee_cis":
        return load_ieee_cis(data_dir, **kwargs)
    elif key == "paysim":
        return load_paysim(data_dir)
    elif key == "ulb":
        return load_ulb(data_dir)
    else:
        raise ValueError(
            f"Unsupported dataset '{name}'. Allowed datasets: {sorted(list(SUPPORTED_DATASETS))}"
        )
