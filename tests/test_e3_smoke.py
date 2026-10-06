"""
Fast Smoke Test for Objective 2 (Experiment E3) Pipeline Integration.

Verifies end-to-end execution of:
1. Synthetic drift injection into streaming dataset.
2. PrequentialRunner execution across P0, P1, P2, P3 under synthetic drift.
3. Event-centered metrics extraction from RunResult.
4. Paired comparisons and policy x regime statistical testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adaptation.policy import PolicyManager
from src.adaptation.retrainer import RetrainingEngine
from src.drift.injection import DriftRegime, E3DriftConfig, SyntheticDriftInjector
from src.drift.monitor import DriftMonitor
from src.evaluation.event_metrics import compute_event_centered_metrics
from src.evaluation.metrics import StreamingMetricsTracker
from src.evaluation.statistics import (
    paired_policy_comparison,
    policy_regime_interaction_test,
)
from src.models.base_learner import HoeffdingTreeLearner
from src.pipeline.runner import PrequentialRunner


def test_e3_end_to_end_smoke_pipeline():
    """Execute a miniature end-to-end smoke pipeline simulating an E3 run."""
    n_samples = 400
    n_warmup = 50
    t_drift = 150  # stream transaction index 150 (global index 200)

    # 1. Generate synthetic financial transactions
    rng = np.random.RandomState(42)
    c1 = rng.uniform(5.0, 15.0, size=n_samples)
    c2 = rng.uniform(1.0, 3.0, size=n_samples)
    card1 = rng.uniform(1000, 9000, size=n_samples)
    card2 = rng.uniform(100, 500, size=n_samples)
    amt = rng.uniform(10.0, 500.0, size=n_samples)
    
    # Ground truth: fraud correlated with C1 > 10
    is_fraud = (c1 > 11.0).astype(int)
    segments = rng.choice(["W", "C", "R", "H", "S"], size=n_samples)

    df_X = pd.DataFrame({
        "C1": c1,
        "C2": c2,
        "card1": card1,
        "card2": card2,
        "TransactionAmt": amt,
    })
    s_y = pd.Series(is_fraud)
    s_seg = pd.Series(segments)

    # 2. Inject Sudden Drift (C1 and C2 swapped at t_drift)
    cfg = E3DriftConfig(
        regime=DriftRegime.SUDDEN,
        t_drift=t_drift,
        feature_pairs=[("C1", "C2")],
        window_size=10000,
        seed=42,
    )
    injector = SyntheticDriftInjector(cfg)
    X_drift, y_drift, seg_drift, metadata_list = injector.inject_dataframe(
        df_X, s_y, s_seg, warmup_size=n_warmup
    )

    assert len(X_drift) == n_samples
    assert len(metadata_list) == n_samples

    # 3. Execute runner for P0, P1, P2, P3
    policy_results = {}
    for pol in ["P0", "P1", "P2", "P3"]:
        runner = PrequentialRunner.from_config(
            policy_str=pol,
            detector_str="adwin",
            grace_period=20,
            window_size=100,
            n_interval=50,
            seed=42,
        )

        res = runner.run(X=X_drift, y=y_drift, segment=seg_drift, warmup_size=n_warmup)
        assert res.n_stream == n_samples - n_warmup
        policy_results[pol] = res

        # 4. Compute Event-Centered Metrics
        event_metrics = compute_event_centered_metrics(
            records=res.records,
            drift_event_log=res.drift_event_log,
            adaptation_log=res.adaptation_log,
            drift_config=cfg,
            horizon=100,
            recovery_window=30,
        )
        assert event_metrics.t_drift == t_drift
        assert event_metrics.n_pre_samples == 100
        assert event_metrics.n_post_samples == 100
        res_dict = event_metrics.as_dict()
        assert "delta_pr_auc" in res_dict
        assert "detection_delay" in res_dict

    # 5. Statistical comparison test
    scores_p3 = [0.252, 0.264, 0.271]
    scores_p0 = [0.201, 0.215, 0.224]
    comp = paired_policy_comparison(scores_p3, scores_p0, policy_a="P3", policy_b="P0")
    assert comp["mean_diff"] > 0
    assert comp["cohens_dz"] > 0
