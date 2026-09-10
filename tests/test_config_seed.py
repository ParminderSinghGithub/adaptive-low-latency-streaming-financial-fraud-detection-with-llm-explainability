"""
Unit tests for configuration loading, schema validation, and seed reproducibility.
"""

import random
from pathlib import Path
import numpy as np
import pytest
import yaml

from src.utils.config import ExperimentConfig
from src.utils.seed import set_seed


@pytest.fixture
def valid_config_dict():
    """Returns a valid baseline experiment configuration dictionary."""
    return {
        "experiment_id": "E2_IEEE_CIS_P2_ADWIN_01",
        "seed": 42,
        "dataset": {
            "name": "ieee_cis",
            "path": "datasets/ieee_cis/",
            "warmup_fraction": 0.15,
        },
        "learner": {
            "type": "hoeffding_tree",
            "params": {"grace_period": 200},
        },
        "detector": {
            "type": "adwin",
            "delta": 0.002,
        },
        "policy": {
            "type": "global_drift_triggered",
            "adaptation_window_size": 5000,
        },
        "output": {
            "artifact_dir": "results/ieee_cis/E2/P2_ADWIN/",
        },
    }


def test_valid_yaml_loads_successfully(tmp_path, valid_config_dict):
    """Test that a valid YAML configuration file loads and parses successfully."""
    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)

    config = ExperimentConfig.from_yaml(yaml_file)
    assert config.experiment_id == "E2_IEEE_CIS_P2_ADWIN_01"
    assert config.seed == 42
    assert config.dataset["name"] == "ieee_cis"
    assert config.learner["type"] == "hoeffding_tree"
    assert config.detector["type"] == "adwin"
    assert config.policy["type"] == "global_drift_triggered"
    assert config.output["artifact_dir"] == "results/ieee_cis/E2/P2_ADWIN/"


def test_missing_required_fields_fail_clearly(valid_config_dict):
    """Test that missing required top-level configuration keys raise clear ValueError."""
    invalid_dict = valid_config_dict.copy()
    del invalid_dict["seed"]

    with pytest.raises(ValueError, match="Missing required top-level configuration key"):
        ExperimentConfig(invalid_dict)


def test_invalid_policy_name_fails(valid_config_dict):
    """Test that an invalid policy name raises ValueError."""
    invalid_dict = valid_config_dict.copy()
    invalid_dict["policy"] = {
        "type": "reinforcement_learning_agent",
        "adaptation_window_size": 5000,
    }

    with pytest.raises(ValueError, match="Invalid policy type"):
        ExperimentConfig(invalid_dict)


def test_invalid_learner_name_fails(valid_config_dict):
    """Test that an invalid learner name raises ValueError."""
    invalid_dict = valid_config_dict.copy()
    invalid_dict["learner"] = {
        "type": "deep_neural_network",
        "params": {},
    }

    with pytest.raises(ValueError, match="Invalid learner type"):
        ExperimentConfig(invalid_dict)


def test_invalid_detector_name_fails(valid_config_dict):
    """Test that an invalid drift detector name raises ValueError."""
    invalid_dict = valid_config_dict.copy()
    invalid_dict["detector"] = {
        "type": "custom_untested_detector",
    }

    with pytest.raises(ValueError, match="Invalid detector type"):
        ExperimentConfig(invalid_dict)


def test_invalid_seed_value_fails(valid_config_dict):
    """Test that a negative or boolean seed value raises ValueError."""
    invalid_dict = valid_config_dict.copy()
    invalid_dict["seed"] = -10

    with pytest.raises(ValueError, match="Seed must be a non-negative integer"):
        ExperimentConfig(invalid_dict)

    with pytest.raises(ValueError, match="Seed must be a non-negative integer"):
        set_seed(-5)


def test_python_random_seed_reproducibility():
    """Test that set_seed produces reproducible Python standard random values."""
    set_seed(12345)
    seq_a = [random.random() for _ in range(10)]

    set_seed(12345)
    seq_b = [random.random() for _ in range(10)]

    assert seq_a == seq_b


def test_numpy_random_seed_reproducibility():
    """Test that set_seed produces reproducible NumPy random values."""
    set_seed(999)
    arr_a = np.random.randn(10)

    set_seed(999)
    arr_b = np.random.randn(10)

    np.testing.assert_array_equal(arr_a, arr_b)


def test_config_values_preserved_and_serialized(tmp_path, valid_config_dict):
    """Test that configuration dictionary is preserved and can be saved to YAML."""
    config = ExperimentConfig(valid_config_dict)
    dict_out = config.to_dict()
    assert dict_out == valid_config_dict

    out_yaml = tmp_path / "saved_config.yaml"
    config.save_yaml(out_yaml)

    reloaded_config = ExperimentConfig.from_yaml(out_yaml)
    assert reloaded_config.to_dict() == valid_config_dict
