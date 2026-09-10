"""
Configuration management and validation for Phase 2 experiments.
Loads, validates, and serializes YAML experiment configurations.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml

# Allowed configuration values based on Implementation Contract
ALLOWED_POLICIES = {
    "static",
    "periodic",
    "global_drift_triggered",
    "segment_drift_triggered",
}

ALLOWED_LEARNERS = {
    "hoeffding_tree",
    "adaptive_random_forest",
    "logistic_regression",
    "sgd_classifier",
}

ALLOWED_DETECTORS = {
    "adwin",
    "hddm",
    "none",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment_id",
    "seed",
    "dataset",
    "learner",
    "detector",
    "policy",
    "output",
}


class ExperimentConfig:
    """
    Validated experiment configuration wrapper.
    Ensures all parameters conform to the Phase 2 Implementation Contract.
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.validate(config_dict)
        self._raw_config = config_dict.copy()

        # Top-level properties
        self.experiment_id: str = str(config_dict["experiment_id"])
        self.seed: int = int(config_dict["seed"])

        # Sub-configs
        self.dataset: Dict[str, Any] = config_dict["dataset"]
        self.learner: Dict[str, Any] = config_dict["learner"]
        self.detector: Dict[str, Any] = config_dict["detector"]
        self.policy: Dict[str, Any] = config_dict["policy"]
        self.output: Dict[str, Any] = config_dict["output"]

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        """Load and validate an ExperimentConfig from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {path}: {e}") from e

        if not isinstance(content, dict):
            raise ValueError(f"Configuration YAML in {path} must contain a top-level dictionary.")

        return cls(content)

    @staticmethod
    def validate(config: Dict[str, Any]) -> None:
        """
        Validate experiment configuration against schema and contract rules.
        Raises ValueError if any validation check fails.
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary.")

        # Check required top-level keys
        missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
        if missing_keys:
            raise ValueError(
                f"Missing required top-level configuration key(s): {sorted(list(missing_keys))}"
            )

        # Validate seed
        seed = config.get("seed")
        if not isinstance(seed, int) or seed < 0 or isinstance(seed, bool):
            raise ValueError(f"Seed must be a non-negative integer, got: {seed}")

        # Validate dataset
        dataset = config.get("dataset", {})
        if not isinstance(dataset, dict):
            raise ValueError("Dataset configuration must be a dictionary.")
        for key in ["name", "path", "warmup_fraction"]:
            if key not in dataset:
                raise ValueError(f"Missing required key in dataset configuration: '{key}'")
        
        warmup_frac = dataset.get("warmup_fraction")
        if isinstance(warmup_frac, bool) or not isinstance(warmup_frac, (int, float)) or not (0.0 < warmup_frac < 1.0):
            raise ValueError(
                f"Dataset warmup_fraction must be a float between 0 and 1 (exclusive), got: {warmup_frac}"
            )

        # Validate learner
        learner = config.get("learner", {})
        if not isinstance(learner, dict):
            raise ValueError("Learner configuration must be a dictionary.")
        learner_type = learner.get("type")
        if not learner_type:
            raise ValueError("Learner configuration missing 'type' key.")
        if learner_type not in ALLOWED_LEARNERS:
            raise ValueError(
                f"Invalid learner type '{learner_type}'. Allowed learners: {sorted(list(ALLOWED_LEARNERS))}"
            )
        if "params" not in learner or not isinstance(learner.get("params"), dict):
            raise ValueError("Learner configuration missing 'params' dictionary.")

        # Validate detector
        detector = config.get("detector", {})
        if not isinstance(detector, dict):
            raise ValueError("Detector configuration must be a dictionary.")
        detector_type = detector.get("type")
        if not detector_type:
            raise ValueError("Detector configuration missing 'type' key.")
        if detector_type not in ALLOWED_DETECTORS:
            raise ValueError(
                f"Invalid detector type '{detector_type}'. Allowed detectors: {sorted(list(ALLOWED_DETECTORS))}"
            )

        # Validate policy
        policy = config.get("policy", {})
        if not isinstance(policy, dict):
            raise ValueError("Policy configuration must be a dictionary.")
        policy_type = policy.get("type")
        if not policy_type:
            raise ValueError("Policy configuration missing 'type' key.")
        if policy_type not in ALLOWED_POLICIES:
            raise ValueError(
                f"Invalid policy type '{policy_type}'. Allowed policies: {sorted(list(ALLOWED_POLICIES))}"
            )

        # Policy-specific checks
        if policy_type == "periodic":
            interval = policy.get("periodic_interval")
            if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
                raise ValueError(
                    f"Periodic policy requires a positive integer 'periodic_interval', got: {interval}"
                )

        if policy_type == "segment_drift_triggered":
            seg_feature = policy.get("segment_feature")
            if not seg_feature or not isinstance(seg_feature, str):
                raise ValueError(
                    f"Segment drift-triggered policy requires a non-empty string 'segment_feature', got: {seg_feature}"
                )

        if policy_type in {"periodic", "global_drift_triggered", "segment_drift_triggered"}:
            window_size = policy.get("adaptation_window_size")
            if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
                raise ValueError(
                    f"Adaptive policy '{policy_type}' requires a positive integer 'adaptation_window_size', got: {window_size}"
                )

        # Validate output
        output = config.get("output", {})
        if not isinstance(output, dict):
            raise ValueError("Output configuration must be a dictionary.")
        if "artifact_dir" not in output or not output.get("artifact_dir"):
            raise ValueError("Output configuration missing 'artifact_dir' key.")

    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the raw configuration dictionary."""
        return self._raw_config.copy()

    def save_yaml(self, output_path: Union[str, Path]) -> None:
        """Save the configuration dictionary to a YAML file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._raw_config, f, default_flow_style=False, sort_keys=False)
