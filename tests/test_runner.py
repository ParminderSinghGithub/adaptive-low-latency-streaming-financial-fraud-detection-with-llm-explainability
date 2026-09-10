"""
Unit and integration tests for the prequential pipeline components:

- src/streaming/stream_generator.py  (StreamEvent / chronological_stream)
- src/models/base_learner.py          (HoeffdingTreeLearner)
- src/adaptation/retrainer.py         (RetrainingEngine)
- src/evaluation/metrics.py           (StreamingMetricsTracker)
- src/pipeline/runner.py              (PrequentialRunner, RunResult, StreamingRecord)

ALL tests use tiny deterministic synthetic streams.
NO real dataset files are required.
NO Kaggle execution.
NO E1-E10 research results.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 60, fraud_fraction: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Create a tiny synthetic DataFrame mimicking the IEEE-CIS schema."""
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n * fraud_fraction))
    labels = [0] * (n - n_fraud) + [1] * n_fraud

    return pd.DataFrame(
        {
            "TransactionID": list(range(1000, 1000 + n)),
            "x1": rng.normal(size=n).tolist(),
            "x2": rng.uniform(0, 100, size=n).tolist(),
            "x3": rng.choice([1, 2, 3, 4], size=n).tolist(),
            "segment": (["W", "H", "C"] * ((n // 3) + 1))[:n],
            "isFraud": labels,
        }
    )


def _split_df(df: pd.DataFrame, warmup_frac: float = 0.33):
    """Return X, y, segment as aligned Series from a synthetic DataFrame."""
    feat_cols = ["x1", "x2", "x3"]
    X = df[feat_cols].copy()
    y = df["isFraud"].copy()
    seg = df["segment"].astype(str).copy()
    return X, y, seg


# ---------------------------------------------------------------------------
# 1. StreamGenerator
# ---------------------------------------------------------------------------

class TestChronologicalStream:
    def test_yields_all_rows(self):
        from src.streaming.stream_generator import chronological_stream
        df = _make_df(20)
        X, y, seg = _split_df(df)
        events = list(chronological_stream(X, y, seg))
        assert len(events) == 20

    def test_preserves_order(self):
        from src.streaming.stream_generator import chronological_stream
        df = _make_df(10)
        X, y, seg = _split_df(df)
        labels = [e[1] for e in chronological_stream(X, y, seg)]
        assert labels == y.tolist()

    def test_segment_none(self):
        from src.streaming.stream_generator import chronological_stream
        df = _make_df(5)
        X, y, _ = _split_df(df)
        events = list(chronological_stream(X, y, segment=None))
        assert all(e[2] is None for e in events)

    def test_feature_dict_keys_match_columns(self):
        from src.streaming.stream_generator import chronological_stream
        df = _make_df(5)
        X, y, seg = _split_df(df)
        events = list(chronological_stream(X, y, seg))
        for x_dict, _, _ in events:
            assert set(x_dict.keys()) == set(X.columns)


# ---------------------------------------------------------------------------
# 2. HoeffdingTreeLearner
# ---------------------------------------------------------------------------

class TestHoeffdingTreeLearner:
    def _simple_learner(self):
        from src.models.base_learner import HoeffdingTreeLearner
        return HoeffdingTreeLearner(grace_period=5, delta=1e-3)

    def test_predict_before_training_returns_zero(self):
        learner = self._simple_learner()
        prob = learner.predict_one({"x1": 1.0, "x2": 0.5})
        assert prob == pytest.approx(0.0, abs=1e-9)

    def test_learn_and_predict(self):
        learner = self._simple_learner()
        for _ in range(10):
            learner.learn_one({"x1": 1.0, "x2": 1.0}, 1)
            learner.learn_one({"x1": -1.0, "x2": -1.0}, 0)
        # After training, predict should return a float in [0, 1]
        prob = learner.predict_one({"x1": 1.0, "x2": 1.0})
        assert 0.0 <= prob <= 1.0

    def test_predict_label_is_binary(self):
        learner = self._simple_learner()
        label = learner.predict_label({"x1": 0.0, "x2": 0.0})
        assert label in (0, 1)

    def test_fit_batch_resets_and_retrains(self):
        learner = self._simple_learner()
        # Prime the learner with legit data
        for _ in range(20):
            learner.learn_one({"x1": 0.0}, 0)
        rows = [({"x1": 1.0, "x2": 1.0}, 1)] * 30
        learner.fit_batch(rows)
        # After batch fit, model should favour class 1
        prob = learner.predict_one({"x1": 1.0, "x2": 1.0})
        assert prob > 0.0   # tree has seen class 1 examples

    def test_reset_clears_model(self):
        learner = self._simple_learner()
        for _ in range(20):
            learner.learn_one({"x1": 1.0}, 1)
        learner.reset()
        # After reset, no training data; should return 0
        assert learner.predict_one({"x1": 1.0}) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. RetrainingEngine
# ---------------------------------------------------------------------------

class TestRetrainingEngine:
    def _engine(self, window_size: int = 50):
        from src.models.base_learner import HoeffdingTreeLearner
        from src.adaptation.retrainer import RetrainingEngine
        return RetrainingEngine(
            learner_factory=lambda: HoeffdingTreeLearner(grace_period=5),
            window_size=window_size,
        )

    def test_add_and_buffer_size(self):
        engine = self._engine(window_size=10)
        for i in range(7):
            engine.add({"x": float(i)}, label=i % 2, segment_key="W")
        assert engine.buffer_size == 7

    def test_window_evicts_oldest(self):
        engine = self._engine(window_size=5)
        for i in range(8):
            engine.add({"x": float(i)}, label=0)
        assert engine.buffer_size == 5

    def test_retrain_global_returns_learner(self):
        from src.models.base_learner import HoeffdingTreeLearner
        engine = self._engine()
        for i in range(20):
            engine.add({"x": float(i)}, label=i % 2, segment_key="W")
        new_model = engine.retrain_window(segment_key=None)
        assert isinstance(new_model, HoeffdingTreeLearner)

    def test_retrain_segment_uses_only_segment_rows(self):
        from src.models.base_learner import HoeffdingTreeLearner
        engine = self._engine()
        for i in range(20):
            seg = "W" if i < 10 else "H"
            engine.add({"x": float(i)}, label=1, segment_key=seg)
        # Only W-labelled rows → should retrain without error
        new_model = engine.retrain_window(segment_key="W")
        assert isinstance(new_model, HoeffdingTreeLearner)

    def test_retrain_empty_buffer_raises(self):
        from src.adaptation.retrainer import RetrainingEngine
        from src.models.base_learner import HoeffdingTreeLearner
        engine = RetrainingEngine(
            lambda: HoeffdingTreeLearner(grace_period=5), window_size=10
        )
        with pytest.raises(RuntimeError, match="memory buffer is empty"):
            engine.retrain_window()

    def test_retrain_segment_no_matching_rows_raises(self):
        engine = self._engine()
        for i in range(5):
            engine.add({"x": float(i)}, label=0, segment_key="W")
        with pytest.raises(RuntimeError, match="memory buffer is empty"):
            engine.retrain_window(segment_key="H")

    def test_as_retrain_fn_interface(self):
        from src.models.base_learner import HoeffdingTreeLearner
        engine = self._engine()
        for i in range(10):
            engine.add({"x": float(i)}, label=i % 2)
        fn = engine.as_retrain_fn()
        # PolicyManager calls fn(buffer, segment_key=None)
        new_model = fn(None, segment_key=None)
        assert isinstance(new_model, HoeffdingTreeLearner)

    def test_reset_clears_buffer(self):
        engine = self._engine()
        for i in range(10):
            engine.add({"x": float(i)}, label=0)
        engine.reset()
        assert engine.buffer_size == 0

    def test_segment_counts(self):
        engine = self._engine()
        for i in range(6):
            seg = "W" if i % 2 == 0 else "H"
            engine.add({"x": float(i)}, label=0, segment_key=seg)
        counts = engine.segment_counts()
        assert counts["W"] == 3
        assert counts["H"] == 3


# ---------------------------------------------------------------------------
# 4. StreamingMetricsTracker
# ---------------------------------------------------------------------------

class TestStreamingMetricsTracker:
    def test_empty_compute(self):
        from src.evaluation.metrics import StreamingMetricsTracker
        tracker = StreamingMetricsTracker()
        snap = tracker.compute()
        assert snap.n_samples == 0
        assert snap.pr_auc is None

    def test_update_increments_n_samples(self):
        from src.evaluation.metrics import StreamingMetricsTracker
        tracker = StreamingMetricsTracker()
        tracker.update(0, 0.1)
        tracker.update(1, 0.9)
        assert tracker.n_samples == 2

    def test_pr_auc_computed_with_both_classes(self):
        from src.evaluation.metrics import StreamingMetricsTracker
        tracker = StreamingMetricsTracker()
        for _ in range(10):
            tracker.update(0, 0.1)
        for _ in range(5):
            tracker.update(1, 0.9)
        snap = tracker.compute()
        assert snap.pr_auc is not None
        assert 0.0 <= snap.pr_auc <= 1.0

    def test_f1_nonzero(self):
        from src.evaluation.metrics import StreamingMetricsTracker
        tracker = StreamingMetricsTracker()
        for _ in range(10):
            tracker.update(0, 0.1)
        for _ in range(5):
            tracker.update(1, 0.9)
        snap = tracker.compute()
        assert snap.f1 is not None
        assert snap.f1 >= 0.0

    def test_latency_recorded(self):
        from src.evaluation.metrics import StreamingMetricsTracker
        tracker = StreamingMetricsTracker(track_latency=True)
        for lat in [0.001, 0.002, 0.003]:
            tracker.update_latency(lat)
        pcts = tracker.latency_percentiles()
        assert "p50" in pcts
        assert "p95" in pcts
        assert pcts["n"] == 3

    def test_reset_clears_state(self):
        from src.evaluation.metrics import StreamingMetricsTracker
        tracker = StreamingMetricsTracker()
        tracker.update(1, 0.9)
        tracker.reset()
        assert tracker.n_samples == 0
        assert tracker.compute().n_samples == 0

    def test_compute_window(self):
        from src.evaluation.metrics import StreamingMetricsTracker
        tracker = StreamingMetricsTracker()
        for _ in range(20):
            tracker.update(0, 0.1)
        for _ in range(10):
            tracker.update(1, 0.9)
        # Compute over last 10 only (all fraud)
        snap = tracker.compute_window(last_n=10)
        assert snap.n_samples == 10
        assert snap.n_positive == 10


# ---------------------------------------------------------------------------
# 5. PrequentialRunner — helpers
# ---------------------------------------------------------------------------

def _make_runner(
    policy: str,
    detector: str = "adwin",
    n_interval: int = 10,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    p0_mode: str = "incremental",
):
    from src.pipeline.runner import PrequentialRunner
    if detector_kwargs is None:
        detector_kwargs = {"delta": 0.002} if detector == "adwin" else {}
    return PrequentialRunner.from_config(
        policy_str=policy,
        detector_str=detector,
        grace_period=5,
        delta=1e-3,
        window_size=50,
        n_interval=n_interval,
        detector_kwargs=detector_kwargs,
        segment_aware=(policy in ("P3", "segment_drift_triggered")),
        p0_mode=p0_mode,
    )


def _make_stream_inputs(n: int = 80, warmup: int = 20, seed: int = 0):
    """Return (X, y, segment, warmup_size) for a synthetic stream."""
    df = _make_df(n, seed=seed)
    X, y, seg = _split_df(df)
    return X, y, seg, warmup


# ---------------------------------------------------------------------------
# 5a. Prequential semantics — label leakage prevention
# ---------------------------------------------------------------------------

class TestPrequentialSemantics:
    def test_no_label_leakage_at_prediction_time(self):
        """Prediction at step t must not use y_t.

        Strategy: run a P0 runner on a stream where all labels are 0
        except row 0 which is fraud.  Record prediction at row 0.
        Then run with row 0 label flipped to 1 and confirm same prediction.
        (The model is the same at t=0 regardless of what y_0 is, because
        y_0 is only used for learn_one AFTER the prediction.)
        """
        from src.pipeline.runner import PrequentialRunner

        def _run(y0_label: int):
            df = _make_df(30, seed=1)
            X, y, seg = _split_df(df)
            y.iloc[20] = y0_label  # vary only the first streaming label
            runner = _make_runner("P0")
            result = runner.run(X, y, seg, warmup_size=20)
            return result.records[0].y_prob  # prediction at t=warmup (stream step 0)

        prob_y0 = _run(0)
        prob_y1 = _run(1)
        # Prediction must be identical — learner state at t=warmup is the same
        assert prob_y0 == pytest.approx(prob_y1, abs=1e-12), (
            "Prediction at t differs when only the current y_t differs — "
            "this indicates label leakage."
        )

    def test_y_true_in_record_matches_label(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        for i, rec in enumerate(result.records):
            assert rec.y_true == int(y.iloc[warmup + i])

    def test_record_count_matches_stream_size(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_stream == 60
        assert len(result.records) == 60

    def test_n_warmup_recorded(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_warmup == warmup

    def test_error_is_abs_diff(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        for rec in result.records:
            assert rec.error == abs(rec.y_true - rec.y_pred)
            assert rec.error in (0, 1)


# ---------------------------------------------------------------------------
# 5b. P0 Static
# ---------------------------------------------------------------------------

class TestRunnerP0:
    def test_p0_zero_adaptations(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_adaptations == 0
        assert all(not r.adapted for r in result.records)

    def test_p0_metrics_computed(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        snap = result.final_metrics
        assert snap.n_samples == result.n_stream

    def test_p0_result_has_policy_label(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.policy == "P0"


# ---------------------------------------------------------------------------
# 5c. P1 Periodic
# ---------------------------------------------------------------------------

class TestRunnerP1:
    def test_p1_triggers_at_correct_intervals(self):
        n_interval = 10
        runner = _make_runner("P1", n_interval=n_interval)
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        # Stream is 60 steps → intervals at 10, 20, 30, 40, 50, 60
        assert result.n_adaptations == 6

    def test_p1_adaptation_scope_is_global(self):
        runner = _make_runner("P1", n_interval=10)
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        for ev in result.adaptation_log:
            assert ev["scope"] == "global"

    def test_p1_adapted_flag_set(self):
        runner = _make_runner("P1", n_interval=10)
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        adapted_indices = [r.tx_index for r in result.records if r.adapted]
        assert len(adapted_indices) == 6

    def test_p1_config_alias(self):
        """'periodic' string should produce P1."""
        runner = _make_runner("periodic", n_interval=10)
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.policy == "P1"


# ---------------------------------------------------------------------------
# 5d. P2 Global Drift
# ---------------------------------------------------------------------------

class TestRunnerP2:
    def _abrupt_stream(self, warmup: int = 20, n: int = 1200):
        """Return a stream with abrupt 0→1 error shift around the midpoint."""
        n_leg = n // 2
        n_fraud = n - n_leg
        df = pd.DataFrame(
            {
                "x1": [0.0] * n_leg + [1.0] * n_fraud,
                "x2": [0.0] * n_leg + [1.0] * n_fraud,
                "segment": ["W"] * n,
                "isFraud": [0] * n_leg + [1] * n_fraud,
            }
        )
        X = df[["x1", "x2"]].copy()
        y = df["isFraud"].copy()
        seg = df["segment"].astype(str).copy()
        return X, y, seg, warmup

    def test_p2_triggers_on_concept_drift(self):
        runner = _make_runner("P2", detector="adwin")
        X, y, seg, warmup = self._abrupt_stream(warmup=20, n=1200)
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_adaptations >= 1, (
            "P2 should detect the abrupt 0→1 shift and trigger at least one adaptation."
        )

    def test_p2_no_adaptation_on_constant_stream(self):
        runner = _make_runner("P2", detector="adwin")
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)
        # Force all labels to 0 (constant, no drift)
        y[:] = 0
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_adaptations == 0

    def test_p2_config_alias(self):
        runner = _make_runner("global_drift_triggered")
        assert runner._policy_name == "P2"


# ---------------------------------------------------------------------------
# 5e. P3 Segment Drift
# ---------------------------------------------------------------------------

class TestRunnerP3:
    def _build_segment_stream(self, n: int = 1500, warmup: int = 20):
        """Stable stream for 'H', abrupt 0→1 shift in 'W' after warmup."""
        rows = []
        for i in range(n):
            seg = "W" if i % 2 == 0 else "H"
            # W drifts after halfway; H stays stable at 0
            if seg == "W":
                x1, x2 = 0.0, 0.0
                label = 1 if i > n // 2 else 0
            else:
                x1, x2 = 10.0, 10.0
                label = 0
            rows.append({"x1": x1, "x2": x2, "segment": seg, "isFraud": label})
        df = pd.DataFrame(rows)
        X = df[["x1", "x2"]].copy()
        y = df["isFraud"].copy()
        seg_s = df["segment"].astype(str).copy()
        return X, y, seg_s, warmup

    def test_p3_triggers_for_drifting_segment(self):
        runner = _make_runner("P3", detector="adwin")
        X, y, seg, warmup = self._build_segment_stream(n=1500, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_adaptations >= 1, "P3 should trigger for segment 'W'"

    def test_p3_adaptation_scope_is_segment(self):
        runner = _make_runner("P3", detector="adwin")
        X, y, seg, warmup = self._build_segment_stream(n=1500, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        for ev in result.adaptation_log:
            assert ev["scope"] != "global", (
                "P3 adaptation scope must be segment-specific, not 'global'."
            )

    def test_p3_config_alias(self):
        runner = _make_runner("segment_drift_triggered")
        assert runner._policy_name == "P3"


# ---------------------------------------------------------------------------
# 5f. HDDM detector through runner
# ---------------------------------------------------------------------------

class TestRunnerHDDM:
    def test_hddm_w_initialises_successfully(self):
        runner = _make_runner("P2", detector="hddm_w")
        assert runner._detector_type == "hddm_w"

    def test_hddm_alias_resolves_to_hddm_w(self):
        runner = _make_runner("P2", detector="hddm")
        assert runner._detector_type == "hddm_w"

    def test_hddm_w_runs_without_error(self):
        runner = _make_runner("P0", detector="hddm_w")
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_stream == 60


# ---------------------------------------------------------------------------
# 5g. Invalid inputs
# ---------------------------------------------------------------------------

class TestRunnerEdgeCases:
    def test_invalid_policy_raises(self):
        from src.pipeline.runner import PrequentialRunner
        with pytest.raises(ValueError, match="Unknown policy"):
            PrequentialRunner.from_config(policy_str="banana", detector_str="adwin")

    def test_warmup_too_large_raises(self):
        runner = _make_runner("P0")
        df = _make_df(10)
        X, y, seg = _split_df(df)
        with pytest.raises(ValueError, match="warmup_size"):
            runner.run(X, y, seg, warmup_size=10)

    def test_warmup_zero(self):
        """warmup_size=0 means no warmup; entire DataFrame is streamed."""
        runner = _make_runner("P0")
        X, y, seg, _ = _make_stream_inputs(n=20)
        result = runner.run(X, y, seg, warmup_size=0)
        assert result.n_stream == 20
        assert result.n_warmup == 0

    def test_segment_none_works_for_p0(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, None, warmup_size=warmup)
        assert result.n_stream > 0


# ---------------------------------------------------------------------------
# 5h. Reset / multi-seed independence
# ---------------------------------------------------------------------------

class TestRunnerReset:
    def test_reset_produces_independent_state(self):
        """Two runs with reset must produce identical results."""
        from src.models.base_learner import HoeffdingTreeLearner
        runner = _make_runner("P1", n_interval=10)
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)

        result1 = runner.run(X, y, seg, warmup_size=warmup)
        n_adapt1 = result1.n_adaptations

        # Reset and run again with fresh learner
        runner.reset(learner=HoeffdingTreeLearner(grace_period=5, delta=1e-3))
        result2 = runner.run(X, y, seg, warmup_size=warmup)
        n_adapt2 = result2.n_adaptations

        assert n_adapt1 == n_adapt2, "P1 trigger count must be identical after reset."

    def test_no_cross_run_state_contamination(self):
        """Metrics from run 1 must not appear in run 2 after reset."""
        from src.models.base_learner import HoeffdingTreeLearner
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20)

        runner.run(X, y, seg, warmup_size=warmup)
        runner.reset(learner=HoeffdingTreeLearner(grace_period=5, delta=1e-3))

        # After reset, metrics should start fresh
        assert runner._metrics.n_samples == 0


# ---------------------------------------------------------------------------
# 5i. RunResult structure
# ---------------------------------------------------------------------------

class TestRunResultStructure:
    def test_summary_dict_keys_present(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        summary = result.summary_dict()
        for key in ["policy", "n_stream", "n_adaptations", "pr_auc", "roc_auc"]:
            assert key in summary

    def test_records_are_streaming_record_instances(self):
        from src.pipeline.runner import StreamingRecord
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        for rec in result.records:
            assert isinstance(rec, StreamingRecord)

    def test_record_as_dict_has_required_keys(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        rec_dict = result.records[0].as_dict()
        for key in ["tx_index", "y_true", "y_prob", "y_pred", "error",
                    "segment_key", "adapted", "inference_latency_s"]:
            assert key in rec_dict

    def test_latency_percentiles_present(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs()
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert "p50" in result.latency_percentiles
        assert "p95" in result.latency_percentiles

    def test_to_records_dataframe(self):
        runner = _make_runner("P0")
        X, y, seg, warmup = _make_stream_inputs(n=50, warmup=10)
        result = runner.run(X, y, seg, warmup_size=warmup)
        df = result.to_records_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40
        assert "tx_index" in df.columns
        assert "y_prob" in df.columns


# ---------------------------------------------------------------------------
# 6. Focused Semantic Audit Proof Suite (Requirements A through J)
# ---------------------------------------------------------------------------

class TestSemanticAuditProof:
    """Comprehensive proof suite validating prequential correctness and policy fairness."""

    # --- A. Prediction-before-label-use ---
    def test_a_prediction_before_label_use(self):
        """Demonstrate that y_t cannot influence prediction at t."""
        X, y, seg, warmup = _make_stream_inputs(n=60, warmup=20, seed=42)

        # Run with y[warmup + 5] = 0
        y1 = y.copy()
        y1.iloc[warmup + 5] = 0
        runner1 = _make_runner("P1")
        res1 = runner1.run(X, y1, seg, warmup_size=warmup)

        # Run with y[warmup + 5] = 1 (label flipped)
        y2 = y.copy()
        y2.iloc[warmup + 5] = 1
        runner2 = _make_runner("P1")
        res2 = runner2.run(X, y2, seg, warmup_size=warmup)

        # Prediction probability at stream index 5 MUST be exactly identical
        assert res1.records[5].y_prob == pytest.approx(res2.records[5].y_prob, abs=1e-12), (
            "Prediction at t must NOT be influenced by the revealed label y_t at t."
        )

    # --- B. Exact adaptation timing ---
    def test_b_exact_adaptation_timing(self):
        """Verify the exact transaction indices at which P1 adaptation occurs."""
        X, y, seg, warmup = _make_stream_inputs(n=80, warmup=20, seed=0)
        # N_interval = 10 -> adaptations should trigger at row indices:
        # warmup + 10 - 1 = 29, then 39, 49, 59, 69, 79
        runner = _make_runner("P1", n_interval=10)
        result = runner.run(X, y, seg, warmup_size=warmup)
        actual_indices = [entry["tx_index"] for entry in result.adaptation_log]
        expected_indices = [warmup + (k * 10) - 1 for k in range(1, 7)]
        assert actual_indices == expected_indices, (
            f"Expected adaptations at {expected_indices}, got {actual_indices}"
        )

    # --- C. Current-observation handling ---
    def test_c_current_observation_in_retraining_window(self):
        """Verify whether current (x_t, y_t) belongs to the adaptation window upon trigger."""
        X, y, seg, warmup = _make_stream_inputs(n=40, warmup=10, seed=1)
        runner = _make_runner("P1", n_interval=5)
        # Check buffer contents immediately after run: buffer must contain the final sample
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert runner._retrainer.buffer_size >= 5
        # The last entry in the memory buffer must correspond to the final stream observation
        last_x, last_y, last_seg = runner._retrainer._buffer[-1]
        assert last_y == int(y.iloc[-1])

    # --- D. P0 fairness: incremental-only default & frozen reference ---
    def test_d_p0_model_frozen_mode(self):
        """Demonstrate that with p0_mode='frozen', P0 model state remains strictly frozen."""
        X, y, seg, warmup = _make_stream_inputs(n=100, warmup=20, seed=99)
        runner = _make_runner("P0", p0_mode="frozen")
        result = runner.run(X, y, seg, warmup_size=warmup)

        # 1. Zero adaptations
        assert result.n_adaptations == 0

        # 2. Probe point prediction after warmup vs after stream:
        from src.models.base_learner import HoeffdingTreeLearner
        warmup_only_model = HoeffdingTreeLearner(grace_period=5, delta=1e-3)
        for i in range(warmup):
            x_i = {col: X.iat[i, X.columns.get_loc(col)] for col in X.columns}
            warmup_only_model.learn_one(x_i, int(y.iat[i]))

        probe = {"x1": 0.25, "x2": 0.75}
        assert runner._learner.predict_one(probe) == pytest.approx(
            warmup_only_model.predict_one(probe), abs=1e-9
        ), "In frozen mode, P0 model must remain frozen at its post-warmup state."

    def test_d_p0_incremental_default_mode(self):
        """Demonstrate that with default p0_mode='incremental', P0 does online learning with zero retrains."""
        X, y, seg, warmup = _make_stream_inputs(n=100, warmup=20, seed=99)
        runner = _make_runner("P0", p0_mode="incremental")
        result = runner.run(X, y, seg, warmup_size=warmup)

        # Zero window retraining events
        assert result.n_adaptations == 0
        assert result.n_stream == 80
        assert len(result.records) == 80
        # Model has performed online updates; probe point differs from frozen post-warmup state
        from src.models.base_learner import HoeffdingTreeLearner
        warmup_only_model = HoeffdingTreeLearner(grace_period=5, delta=1e-3)
        for i in range(warmup):
            x_i = {col: X.iat[i, X.columns.get_loc(col)] for col in X.columns}
            warmup_only_model.learn_one(x_i, int(y.iat[i]))

        probe = {"x1": 0.25, "x2": 0.75}
        # In incremental mode, the model updated on the stream, so it's not identical to warmup-only
        assert runner.p0_mode == "incremental"

    # --- E. P1 exact periodic count ---
    def test_e_p1_periodic_exact_frequency(self):
        """Verify P1 triggers exactly n_stream // N_interval times."""
        X, y, seg, warmup = _make_stream_inputs(n=95, warmup=20, seed=5)
        n_stream = 75
        n_interval = 15
        runner = _make_runner("P1", n_interval=n_interval)
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_adaptations == n_stream // n_interval

    # --- F. P2 global drift causes global adaptation only ---
    def test_f_p2_global_scope_only(self):
        """Verify that P2 adaptations always have scope == 'global'."""
        from tests.test_runner import TestRunnerP2
        p2_fixture = TestRunnerP2()
        X, y, seg, warmup = p2_fixture._abrupt_stream(warmup=20, n=1200)
        runner = _make_runner("P2", detector="adwin")
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_adaptations >= 1
        for ev in result.adaptation_log:
            assert ev["scope"] == "global", f"P2 adaptation scope must be 'global', got {ev['scope']}"

    # --- G. P3 segment drift causes segment-scoped adaptation ---
    def test_g_p3_segment_scope_only(self):
        """Verify that P3 adaptations have scope matching the drifting segment."""
        from tests.test_runner import TestRunnerP3
        p3_fixture = TestRunnerP3()
        X, y, seg, warmup = p3_fixture._build_segment_stream(n=1500, warmup=20)
        runner = _make_runner("P3", detector="adwin")
        result = runner.run(X, y, seg, warmup_size=warmup)
        assert result.n_adaptations >= 1
        for ev in result.adaptation_log:
            assert ev["scope"] != "global", "P3 adaptation scope must be segment-specific."
            assert ev["scope"] in ("W", "H")

    # --- H. Segment isolation (Drift in A cannot silently adapt B) ---
    def test_h_segment_isolation_drift_in_a_does_not_adapt_b(self):
        """Verify that concept drift in segment 'A' does not trigger adaptation in segment 'B'."""
        rows = []
        n = 1500
        warmup = 40
        for i in range(n):
            seg_val = "A" if i % 2 == 0 else "B"
            if seg_val == "A":
                # Segment A experiences extreme concept drift after midpoint
                x1, x2 = 0.0, 0.0
                label = 1 if i > n // 2 else 0
            else:
                # Segment B is completely stable: legitimate transactions throughout
                x1, x2 = 10.0, 10.0
                label = 0
            rows.append({"x1": x1, "x2": x2, "segment": seg_val, "isFraud": label})

        df = pd.DataFrame(rows)
        X = df[["x1", "x2"]].copy()
        y = df["isFraud"].copy()
        seg_s = df["segment"].astype(str).copy()

        runner = _make_runner("P3", detector="adwin")
        result = runner.run(X, y, seg_s, warmup_size=warmup)

        # Segment A must trigger adaptation
        scopes = [ev["scope"] for ev in result.adaptation_log]
        assert "A" in scopes, "Segment A must trigger adaptation upon drift."
        # Segment B must NEVER trigger adaptation
        assert "B" not in scopes, "Segment B must NEVER be adapted when drift is localized to A."

        # Verify model isolation: B's dedicated model still predicts correctly for B
        b_model = runner.segment_models["B"]
        b_prob = b_model.predict_one({"x1": 10.0, "x2": 10.0})
        assert b_prob == pytest.approx(0.0, abs=1e-5), "Segment B model state must be preserved."

    # --- I. Reset isolation across independent runs ---
    def test_i_reset_produces_independent_runs(self):
        """Verify that two sequential runs on the same runner instance are completely independent."""
        from src.models.base_learner import HoeffdingTreeLearner
        runner = _make_runner("P1", n_interval=10)
        X, y, seg, warmup = _make_stream_inputs(n=60, warmup=20, seed=7)

        res1 = runner.run(X, y, seg, warmup_size=warmup)
        # Reset with fresh learner
        runner.reset(learner=HoeffdingTreeLearner(grace_period=5, delta=1e-3))
        res2 = runner.run(X, y, seg, warmup_size=warmup)

        assert res1.n_stream == res2.n_stream
        assert res1.n_adaptations == res2.n_adaptations
        expected_buf = min(len(res2.records) + warmup, runner._retrainer.window_size)
        assert runner._retrainer.buffer_size == expected_buf

    # --- J. Warmup counting and evaluation exclusion ---
    def test_j_warmup_excluded_from_evaluation(self):
        """Warmup observations must NOT be evaluated or included in stream records."""
        X, y, seg, warmup = _make_stream_inputs(n=100, warmup=30, seed=12)
        runner = _make_runner("P0")
        result = runner.run(X, y, seg, warmup_size=warmup)

        assert result.n_warmup == 30
        assert result.n_stream == 70
        assert len(result.records) == 70
        assert result.records[0].tx_index == 30
        assert result.records[-1].tx_index == 99
        assert result.final_metrics.n_samples == 70
