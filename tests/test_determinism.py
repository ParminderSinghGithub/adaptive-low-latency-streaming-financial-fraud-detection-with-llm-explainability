"""
Explicit Determinism and Reproducibility Verification Tests for Objective 1.

Verifies that for all four adaptation policies (P0, P1, P2, P3) on a deterministic
chronological stream, varying the nominal random seed setting (e.g. seed 42 vs 101)
produces bit-for-bit identical prequential execution traces:
- identical probability sequences: y_prob_t
- identical hard predictions: y_pred_t
- identical error sequences: e_t
- identical drift alarm triggers: drift_detected_global / drift_detected_segment
- identical adaptation events and timestamps
- identical final metrics: PR-AUC, ROC-AUC, F1, Recall, Precision.

METHODOLOGICAL NOTE (Thesis Source of Truth §13):
These tests prove that nominal seeds in the natural IEEE-CIS streaming pipeline
serve strictly as REPRODUCIBILITY CHECKS, confirming the determinism of the
streaming state machine. They do NOT constitute independent statistical degrees
of freedom, and must NOT be pooled for paired inferential statistical tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.runner import PrequentialRunner
from src.utils.seed import set_seed


def _generate_synthetic_stream(n_samples: int = 150, n_features: int = 20) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate a deterministic synthetic streaming dataset with multiple segments."""
    rng = np.random.default_rng(98765)
    data = {}
    for j in range(n_features):
        data[f"feat_{j:02d}"] = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    
    # Introduce clear class separation so trees make splits
    labels = (data["feat_00"] + data["feat_01"] > 0.5).astype(int)
    segments = (["W", "H", "C"] * ((n_samples // 3) + 1))[:n_samples]

    X = pd.DataFrame(data)
    y = pd.Series(labels, name="isFraud")
    seg = pd.Series(segments, name="segment")
    return X, y, seg


@pytest.mark.parametrize("policy", ["P0", "P1", "P2", "P3"])
def test_seed_determinism_trace_equality(policy: str):
    """Verify that seed 42 vs seed 101 produces bit-for-bit identical execution traces."""
    X, y, seg = _generate_synthetic_stream(n_samples=180, n_features=15)
    warmup_size = 40

    def _run_with_seed(seed_val: int):
        set_seed(seed_val)
        runner = PrequentialRunner.from_config(
            policy_str=policy,
            detector_str="adwin",
            window_size=30,
            n_interval=20,
            grace_period=10,
            delta=0.002,
            seed=seed_val,
        )
        return runner.run(X=X, y=y, segment=seg, warmup_size=warmup_size)

    res_42 = _run_with_seed(42)
    res_101 = _run_with_seed(101)

    # 1. Verify stream size and records count
    assert res_42.n_stream == res_101.n_stream == (180 - warmup_size)
    assert len(res_42.records) == len(res_101.records)

    # 2. Verify bit-for-bit trace equality across all prequential steps
    for i, (rec_a, rec_b) in enumerate(zip(res_42.records, res_101.records)):
        assert rec_a.tx_index == rec_b.tx_index, f"Step {i}: tx_index mismatch"
        assert rec_a.y_prob == rec_b.y_prob, f"Step {i}: y_prob mismatch {rec_a.y_prob} != {rec_b.y_prob}"
        assert rec_a.y_pred == rec_b.y_pred, f"Step {i}: y_pred mismatch"
        assert rec_a.error == rec_b.error, f"Step {i}: error mismatch"
        assert rec_a.drift_detected_global == rec_b.drift_detected_global, f"Step {i}: drift mismatch"
        assert rec_a.adapted == rec_b.adapted, f"Step {i}: adaptation mismatch"
        assert rec_a.adaptation_scope == rec_b.adaptation_scope, f"Step {i}: adaptation scope mismatch"

    # 3. Verify identical adaptation logs (comparing logical event properties, excluding wall-clock duration)
    def _strip_duration(log_list):
        return [{k: v for k, v in entry.items() if k != "duration_s"} for entry in log_list]

    assert _strip_duration(res_42.adaptation_log) == _strip_duration(res_101.adaptation_log)
    assert res_42.drift_event_log == res_101.drift_event_log


    # 4. Verify identical final metrics
    metrics_42 = res_42.final_metrics.as_dict()
    metrics_101 = res_101.final_metrics.as_dict()
    for metric_name in ["pr_auc", "roc_auc", "f1", "recall", "precision"]:
        val_42 = metrics_42.get(metric_name)
        val_101 = metrics_101.get(metric_name)
        assert val_42 == val_101, f"{metric_name} mismatch: {val_42} != {val_101}"
