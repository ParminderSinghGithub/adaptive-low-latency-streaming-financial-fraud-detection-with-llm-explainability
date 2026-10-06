"""
Unit and Integration Tests for Objective 2 (Experiment E3) Infrastructure.

Covers:
1. Sudden drift occurs exactly at the configured onset t_drift.
2. Gradual transition follows the configured sigmoid transition window.
3. Recurring regime alternates correctly between Concept A and Concept B.
4. Localized drift affects only target segment (ProductCD == 'C') while others remain stable.
5. Ground-truth metadata aligns exactly with injected events.
6. Zero target leakage (labels are never modified or used in perturbation).
7. Bit-for-bit deterministic output for identical seed/config.
8. Event-centered metrics computation (detection delay, pre/post PR-AUC, recovery time).
9. Inferential statistical functions (Cohen's dz, Hedges' gz, Holm-Sidak, Benjamini-Hochberg, ANOVA).
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from src.drift.injection import DriftRegime, E3DriftConfig, SyntheticDriftInjector
from src.evaluation.event_metrics import (
    EventCenteredMetrics,
    compute_event_centered_metrics,
)
from src.evaluation.statistics import (
    benjamini_hochberg_fdr,
    cohens_dz,
    hedges_gz,
    holm_sidak_correction,
    paired_policy_comparison,
    policy_regime_interaction_test,
)
from src.pipeline.runner import StreamingRecord


@pytest.fixture
def sample_feature_dict() -> dict:
    return {
        "C1": 10.0,
        "C2": 2.0,
        "C3": 15.0,
        "C4": 1.0,
        "card1": 5000.0,
        "card2": 150.0,
        "TransactionAmt": 120.50,
        "ProductCD": "W",
    }


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    n = 200
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "C1": rng.uniform(5.0, 15.0, size=n),
        "C2": rng.uniform(1.0, 3.0, size=n),
        "C3": rng.uniform(10.0, 20.0, size=n),
        "C4": rng.uniform(0.5, 2.0, size=n),
        "card1": rng.uniform(1000, 9000, size=n),
        "card2": rng.uniform(100, 500, size=n),
        "TransactionAmt": rng.uniform(10.0, 500.0, size=n),
    })


# ---------------------------------------------------------------------------
# Test Group 1: Sudden Drift
# ---------------------------------------------------------------------------

def test_sudden_drift_onset_exactness(sample_feature_dict):
    """Test that sudden drift triggers exactly at t_drift, not a step earlier or later."""
    t_drift = 50
    cfg = E3DriftConfig(regime=DriftRegime.SUDDEN, t_drift=t_drift, seed=42)
    injector = SyntheticDriftInjector(cfg)

    # Before t_drift: strictly Concept A (unperturbed)
    for idx in range(t_drift):
        x_out, y_out, seg, meta = injector.inject_event(sample_feature_dict, y=0, segment_key="W", stream_idx=idx)
        assert not meta.is_affected
        assert meta.active_concept == "concept_a"
        assert x_out["C1"] == sample_feature_dict["C1"]
        assert x_out["C2"] == sample_feature_dict["C2"]

    # At and after t_drift: strictly Concept B (perturbed: C1 and C2 swapped)
    for idx in range(t_drift, t_drift + 20):
        x_out, y_out, seg, meta = injector.inject_event(sample_feature_dict, y=0, segment_key="W", stream_idx=idx)
        assert meta.is_affected
        assert meta.active_concept == "concept_b"
        assert x_out["C1"] == sample_feature_dict["C2"]
        assert x_out["C2"] == sample_feature_dict["C1"]
        assert x_out["C3"] == sample_feature_dict["C4"]
        assert x_out["C4"] == sample_feature_dict["C3"]


# ---------------------------------------------------------------------------
# Test Group 2: Gradual Drift (Sigmoid Transition)
# ---------------------------------------------------------------------------

def test_gradual_drift_transition_window(sample_feature_dict):
    """Test that gradual drift stays Concept A before t_drift, Concept B after window,

    and exhibits monotonic progression through the sigmoid window.
    """
    t_drift = 100
    w_trans = 500
    cfg = E3DriftConfig(
        regime=DriftRegime.GRADUAL,
        t_drift=t_drift,
        transition_width=w_trans,
        seed=42,
    )
    injector = SyntheticDriftInjector(cfg)

    # 1. Before t_drift: probability is strictly 0.0
    for idx in range(0, t_drift):
        meta = injector.evaluate_drift_state(idx, "W")
        assert not meta.is_affected
        assert meta.drift_prob == 0.0
        assert meta.active_concept == "concept_a"

    # 2. After t_drift + w_trans: probability is strictly 1.0
    for idx in range(t_drift + w_trans, t_drift + w_trans + 50):
        meta = injector.evaluate_drift_state(idx, "W")
        assert meta.is_affected
        assert meta.drift_prob == 1.0
        assert meta.active_concept == "concept_b"

    # 3. Inside transition window: probability increases monotonically
    probs = [
        injector.evaluate_drift_state(idx, "W").drift_prob
        for idx in range(t_drift, t_drift + w_trans, 50)
    ]
    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1], f"Monotonicity violated: {probs[i]} > {probs[i+1]}"

    # Midpoint of sigmoid must be approx 0.5
    t_mid = t_drift + w_trans // 2
    meta_mid = injector.evaluate_drift_state(t_mid, "W")
    assert pytest.approx(meta_mid.drift_prob, abs=0.05) == 0.5


# ---------------------------------------------------------------------------
# Test Group 3: Recurring Drift
# ---------------------------------------------------------------------------

def test_recurring_drift_alternation(sample_feature_dict):
    """Test that recurring drift alternates cycles between Concept B and Concept A."""
    t_drift = 20
    period = 10
    cfg = E3DriftConfig(
        regime=DriftRegime.RECURRING,
        t_drift=t_drift,
        recurrence_period=period,
        seed=42,
    )
    injector = SyntheticDriftInjector(cfg)

    # Pre-drift: Concept A
    for idx in range(t_drift):
        assert not injector.evaluate_drift_state(idx, "W").is_affected

    # Cycle 0 (20..29): Concept B
    for idx in range(t_drift, t_drift + period):
        meta = injector.evaluate_drift_state(idx, "W")
        assert meta.is_affected
        assert meta.active_concept == "concept_b"

    # Cycle 1 (30..39): Concept A
    for idx in range(t_drift + period, t_drift + 2 * period):
        meta = injector.evaluate_drift_state(idx, "W")
        assert not meta.is_affected
        assert meta.active_concept == "concept_a"

    # Cycle 2 (40..49): Concept B
    for idx in range(t_drift + 2 * period, t_drift + 3 * period):
        meta = injector.evaluate_drift_state(idx, "W")
        assert meta.is_affected
        assert meta.active_concept == "concept_b"


# ---------------------------------------------------------------------------
# Test Group 4: Localized Drift
# ---------------------------------------------------------------------------

def test_localized_drift_segment_isolation(sample_feature_dict):
    """Test that localized drift strictly perturbs ProductCD == 'C' and leaves others unaffected."""
    t_drift = 50
    cfg = E3DriftConfig(
        regime=DriftRegime.LOCALIZED,
        t_drift=t_drift,
        target_segment="C",
        seed=42,
    )
    injector = SyntheticDriftInjector(cfg)

    # Before t_drift: no segment affected
    for seg in ["W", "C", "R", "H", "S"]:
        assert not injector.evaluate_drift_state(10, seg).is_affected

    # After t_drift: ONLY segment 'C' is affected
    for idx in range(t_drift, t_drift + 20):
        # Target segment C
        meta_c = injector.evaluate_drift_state(idx, segment_key="C")
        assert meta_c.is_affected
        assert meta_c.active_concept == "concept_b"
        x_c, _, _, _ = injector.inject_event(sample_feature_dict, 1, "C", idx)
        assert x_c["C1"] == sample_feature_dict["C2"]

        # Non-target segments (W, R, H, S)
        for other_seg in ["W", "R", "H", "S"]:
            meta_other = injector.evaluate_drift_state(idx, segment_key=other_seg)
            assert not meta_other.is_affected
            assert meta_other.active_concept == "concept_a"
            x_other, _, _, _ = injector.inject_event(sample_feature_dict, 1, other_seg, idx)
            assert x_other["C1"] == sample_feature_dict["C1"]


# ---------------------------------------------------------------------------
# Test Group 5: Zero Target Leakage & Determinism
# ---------------------------------------------------------------------------

def test_zero_target_leakage(sample_feature_dict):
    """Test that ground-truth label y is never inspected or modified during injection."""
    cfg = E3DriftConfig(regime=DriftRegime.SUDDEN, t_drift=10, seed=42)
    injector = SyntheticDriftInjector(cfg)

    # Process label 0 and label 1: perturbation on X must be identical
    x0, y0, _, _ = injector.inject_event(sample_feature_dict, y=0, segment_key="W", stream_idx=15)
    x1, y1, _, _ = injector.inject_event(sample_feature_dict, y=1, segment_key="W", stream_idx=15)

    assert y0 == 0
    assert y1 == 1
    assert x0 == x1  # Feature mapping permutation depends purely on (x, stream_idx), not y


def test_seed_determinism(sample_feature_dict):
    """Test that two injector instances with identical seed yield bit-for-bit identical outputs."""
    cfg1 = E3DriftConfig(regime=DriftRegime.GRADUAL, t_drift=50, transition_width=100, seed=123)
    cfg2 = E3DriftConfig(regime=DriftRegime.GRADUAL, t_drift=50, transition_width=100, seed=123)
    inj1 = SyntheticDriftInjector(cfg1)
    inj2 = SyntheticDriftInjector(cfg2)

    for i in range(200):
        x1, _, _, m1 = inj1.inject_event(sample_feature_dict, 0, "W", i)
        x2, _, _, m2 = inj2.inject_event(sample_feature_dict, 0, "W", i)
        assert x1 == x2
        assert m1.is_affected == m2.is_affected
        assert m1.drift_prob == m2.drift_prob


def test_inject_dataframe_warmup_preservation(sample_dataframe):
    """Test that inject_dataframe keeps warmup rows untouched and injects post-warmup."""
    warmup_n = 50
    t_drift = 20  # relative to stream, i.e. global row 70
    cfg = E3DriftConfig(regime=DriftRegime.SUDDEN, t_drift=t_drift, seed=42)
    injector = SyntheticDriftInjector(cfg)

    y = pd.Series(np.zeros(len(sample_dataframe), dtype=int))
    seg = pd.Series(["W"] * len(sample_dataframe))

    X_drift, _, _, meta = injector.inject_dataframe(sample_dataframe, y, seg, warmup_size=warmup_n)

    # Warmup slice (0..49) must be identical
    pd.testing.assert_frame_equal(X_drift.iloc[:warmup_n], sample_dataframe.iloc[:warmup_n])

    # Pre-drift stream slice (50..69) must be identical
    pd.testing.assert_frame_equal(
        X_drift.iloc[warmup_n : warmup_n + t_drift],
        sample_dataframe.iloc[warmup_n : warmup_n + t_drift],
    )

    # Post-drift stream slice (70..end) must have C1 and C2 swapped
    drift_row = warmup_n + t_drift
    assert X_drift.iat[drift_row, X_drift.columns.get_loc("C1")] == sample_dataframe.iat[drift_row, sample_dataframe.columns.get_loc("C2")]
    assert X_drift.iat[drift_row, X_drift.columns.get_loc("C2")] == sample_dataframe.iat[drift_row, sample_dataframe.columns.get_loc("C1")]


# ---------------------------------------------------------------------------
# Test Group 6: Event-Centered Metrics
# ---------------------------------------------------------------------------

def test_event_centered_metrics_computation():
    """Test event-centered metrics calculation with synthetic StreamingRecord fixture."""
    t_drift = 500
    horizon = 200
    cfg = E3DriftConfig(regime=DriftRegime.SUDDEN, t_drift=t_drift)

    records = []
    # Build 800 synthetic records
    for i in range(800):
        # Pre-drift: high discriminative accuracy
        if i < t_drift:
            y_t = 1 if (i % 10 == 0) else 0
            y_prob = 0.9 if y_t == 1 else 0.1
        # Post-drift: degraded accuracy initially
        else:
            y_t = 1 if (i % 10 == 0) else 0
            y_prob = 0.3 if y_t == 1 else 0.4  # degraded

        records.append(
            StreamingRecord(
                tx_index=i,
                y_true=y_t,
                y_prob=y_prob,
                y_pred=int(y_prob >= 0.5),
                error=abs(y_t - int(y_prob >= 0.5)),
                segment_key="W",
                inference_latency_s=0.0001,
            )
        )

    drift_log = [
        {"tx_index": 100, "detector_type": "adwin", "segment_key": None},  # false alarm before t_drift
        {"tx_index": 540, "detector_type": "adwin", "segment_key": None},  # alarm 40 steps post drift
    ]
    adapt_log = [
        {"tx_index": 540, "policy": "P2", "duration_s": 1.25},
    ]

    metrics = compute_event_centered_metrics(
        records=records,
        drift_event_log=drift_log,
        adaptation_log=adapt_log,
        drift_config=cfg,
        horizon=horizon,
    )

    assert metrics.t_drift == t_drift
    assert metrics.pre_drift_pr_auc is not None and metrics.pre_drift_pr_auc > 0.8
    assert metrics.post_drift_pr_auc is not None and metrics.post_drift_pr_auc < metrics.pre_drift_pr_auc
    assert metrics.delta_pr_auc < 0  # degradation observed
    assert metrics.detection_delay == 40  # 540 - 500
    assert metrics.false_alarms_before_drift == 1
    assert metrics.adaptation_count_post_drift == 1
    assert metrics.total_retrain_time_post_s == 1.25


# ---------------------------------------------------------------------------
# Test Group 7: Inferential Statistical Functions
# ---------------------------------------------------------------------------

def test_cohens_dz_and_hedges_gz():
    """Test paired effect size calculations."""
    x = [0.25, 0.26, 0.24, 0.27, 0.25]
    y = [0.20, 0.21, 0.19, 0.22, 0.20]
    dz = cohens_dz(x, y)
    gz = hedges_gz(x, y)

    assert dz > 0
    # Hedges' gz has sample-size attenuation (gz <= dz for positive dz)
    assert 0 < gz <= dz


def test_paired_policy_comparison():
    """Test paired hypothesis test between two policy score arrays."""
    # Policy A clearly beats Policy B across 10 seeds
    p_a = [0.261, 0.274, 0.252, 0.283, 0.265, 0.271, 0.258, 0.292, 0.263, 0.277]
    p_b = [0.201, 0.212, 0.194, 0.221, 0.205, 0.210, 0.189, 0.231, 0.202, 0.215]

    res = paired_policy_comparison(p_a, p_b, policy_a="P3", policy_b="P1")
    assert res["policy_a"] == "P3"
    assert res["policy_b"] == "P1"
    assert res["n_seeds"] == 10
    assert res["mean_diff"] > 0
    assert res["t_pvalue"] < 0.001
    assert res["significant_at_alpha"] is True
    assert res["cohens_dz"] > 2.0


def test_holm_sidak_and_fdr_corrections():
    """Test Holm-Sidak and Benjamini-Hochberg multiplicity correction functions."""
    raw_p = [0.001, 0.012, 0.035, 0.048, 0.150, 0.850]

    adj_sidak, rej_sidak = holm_sidak_correction(raw_p, alpha=0.05)
    adj_fdr, rej_fdr = benjamini_hochberg_fdr(raw_p, q=0.05)

    assert len(adj_sidak) == len(raw_p)
    assert len(adj_fdr) == len(raw_p)

    # First p-value should easily be rejected in both
    assert rej_sidak[0] is True
    assert rej_fdr[0] is True

    # Monotonicity check
    assert adj_sidak[0] <= adj_sidak[1] <= adj_sidak[2]
    assert adj_fdr[0] <= adj_fdr[1] <= adj_fdr[2]


def test_policy_regime_interaction_anova():
    """Test two-way ANOVA decomposition for Policy x Regime interaction."""
    data = []
    # 4 policies x 4 regimes x 5 seeds
    for pol in ["P0", "P1", "P2", "P3"]:
        for reg in ["sudden", "gradual", "recurring", "localized"]:
            for seed in [42, 101, 123, 256, 314]:
                # In recurring regime, P3 excels; in sudden, P2 excels (simulated interaction)
                base = 0.20
                if pol == "P3" and reg == "localized":
                    score = base + 0.08 + (seed % 10) * 0.001
                elif pol == "P2" and reg == "sudden":
                    score = base + 0.05 + (seed % 10) * 0.001
                else:
                    score = base + 0.02 + (seed % 10) * 0.001

                data.append({"policy": pol, "regime": reg, "seed": seed, "pr_auc": score})

    df = pd.DataFrame(data)
    res = policy_regime_interaction_test(df)

    assert "main_effect_policy" in res
    assert "main_effect_regime" in res
    assert "interaction_policy_x_regime" in res
    assert "rank_concordance" in res
    assert res["interaction_policy_x_regime"]["df"] == 3 * 3  # (4-1)*(4-1) = 9


def test_scientific_validation_properties():
    """Lock validated scientific properties: label invariance, 100% post-drift coverage,

    and seed stochasticity divergence between gradual and sudden regimes.
    """
    n = 100
    df = pd.DataFrame({
        "C1": np.arange(n, dtype=float),
        "C2": np.arange(n, dtype=float) + 50.0,
        "C3": np.ones(n, dtype=float),
        "C4": np.zeros(n, dtype=float),
        "card1": np.arange(n, dtype=float) + 1000.0,
        "card2": np.arange(n, dtype=float) + 200.0,
    })
    y = pd.Series((np.arange(n) % 5 == 0).astype(int))
    seg = pd.Series(["W"] * n)

    cfg = E3DriftConfig(regime=DriftRegime.SUDDEN, t_drift=30, seed=42)
    inj = SyntheticDriftInjector(cfg)
    X_drift, y_drift, _, metas = inj.inject_dataframe(df, y, seg, warmup_size=10)

    # Property 1: Exact label preservation
    assert np.array_equal(y.to_numpy(), y_drift.to_numpy())
    assert y.mean() == y_drift.mean()

    # Property 2: 100% post-drift row modification in Concept B
    post_drift_diffs = (X_drift.iloc[40:] != df.iloc[40:]).any(axis=1)
    assert post_drift_diffs.all()

    # Property 3: Sudden is deterministic across seeds; Gradual is stochastic
    s1_sudden = [SyntheticDriftInjector(E3DriftConfig(regime=DriftRegime.SUDDEN, t_drift=30, seed=42)).evaluate_drift_state(i).is_affected for i in range(50)]
    s2_sudden = [SyntheticDriftInjector(E3DriftConfig(regime=DriftRegime.SUDDEN, t_drift=30, seed=101)).evaluate_drift_state(i).is_affected for i in range(50)]
    assert s1_sudden == s2_sudden  # Deterministic

    s1_gradual = [SyntheticDriftInjector(E3DriftConfig(regime=DriftRegime.GRADUAL, t_drift=10, transition_width=50, seed=42)).evaluate_drift_state(i).is_affected for i in range(10, 60)]
    s2_gradual = [SyntheticDriftInjector(E3DriftConfig(regime=DriftRegime.GRADUAL, t_drift=10, transition_width=50, seed=101)).evaluate_drift_state(i).is_affected for i in range(10, 60)]
    assert s1_gradual != s2_gradual  # Genuine stochastic divergence across seeds
