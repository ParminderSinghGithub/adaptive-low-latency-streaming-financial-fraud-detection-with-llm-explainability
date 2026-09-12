"""
Tests for Objective 1 Extensions:
- M1: Window sensitivity configuration (W=10000, W=20000)
- M2: Event mechanism diagnostics (window stats, model complexity, pre/post AP horizons)
- M3: Trigger/no-swap ablation semantics (trigger logged, active model retained, replacement suppressed)
- Manifest W-aware IDs & legacy alias compatibility
"""

import collections
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.base_learner import HoeffdingTreeLearner
from src.adaptation.retrainer import RetrainingEngine
from src.drift.monitor import DriftMonitor
from src.adaptation.policy import PolicyManager, AdaptationRecord
from src.pipeline.runner import PrequentialRunner, StreamingPipelineRunner
from src.pipeline.manifest import ExperimentManifest, JobRecord, validate_run_artifact


# ---------------------------------------------------------------------------
# Tests for RetrainingEngine M2 Window Stats
# ---------------------------------------------------------------------------

class TestRetrainingEngineStats:
    def test_window_stats_empty_buffer(self):
        engine = RetrainingEngine(lambda: HoeffdingTreeLearner(), window_size=5000)
        stats = engine.get_window_stats()
        assert stats["n_samples_window"] == 0
        assert stats["n_fraud_window"] == 0
        assert stats["fraud_prevalence_window"] == 0.0
        assert stats["window_capacity"] == 5000

    def test_window_stats_global(self):
        engine = RetrainingEngine(lambda: HoeffdingTreeLearner(), window_size=1000)
        # Add 80 legit, 20 fraud
        for _ in range(80):
            engine.add({"f1": 1.0}, label=0, segment_key="W")
        for _ in range(20):
            engine.add({"f1": 2.0}, label=1, segment_key="C")

        stats = engine.get_window_stats()
        assert stats["n_samples_window"] == 100
        assert stats["n_fraud_window"] == 20
        assert abs(stats["fraud_prevalence_window"] - 0.20) < 1e-5

    def test_window_stats_segment_filtered(self):
        engine = RetrainingEngine(lambda: HoeffdingTreeLearner(), window_size=1000)
        # Segment W: 90 legit, 10 fraud (prevalence = 10%)
        for _ in range(90):
            engine.add({"f1": 1.0}, label=0, segment_key="W")
        for _ in range(10):
            engine.add({"f1": 2.0}, label=1, segment_key="W")

        # Segment C: 20 legit, 30 fraud (prevalence = 60%)
        for _ in range(20):
            engine.add({"f1": 1.0}, label=0, segment_key="C")
        for _ in range(30):
            engine.add({"f1": 2.0}, label=1, segment_key="C")

        # Query segment W strictly
        stats_w = engine.get_window_stats(segment_key="W")
        assert stats_w["n_samples_window"] == 100
        assert stats_w["n_fraud_window"] == 10
        assert abs(stats_w["fraud_prevalence_window"] - 0.10) < 1e-5

        # Query segment C strictly
        stats_c = engine.get_window_stats(segment_key="C")
        assert stats_c["n_samples_window"] == 50
        assert stats_c["n_fraud_window"] == 30
        assert abs(stats_c["fraud_prevalence_window"] - 0.60) < 1e-5

        # Query non-existent segment safely
        stats_h = engine.get_window_stats(segment_key="H")
        assert stats_h["n_samples_window"] == 0
        assert stats_h["n_fraud_window"] == 0
        assert stats_h["fraud_prevalence_window"] == 0.0


# ---------------------------------------------------------------------------
# Tests for HoeffdingTreeLearner Complexity
# ---------------------------------------------------------------------------

class TestHoeffdingTreeLearnerComplexity:
    def test_model_complexity_keys(self):
        learner = HoeffdingTreeLearner(grace_period=50)
        complexity = learner.model_complexity
        assert isinstance(complexity, dict)

        # Update with some samples
        for i in range(100):
            learner.learn_one({"a": float(i % 5), "b": float(i % 2)}, y=i % 2)

        complexity = learner.model_complexity
        assert "n_nodes" in complexity or "height" in complexity or "n_leaves" in complexity


# ---------------------------------------------------------------------------
# Tests for PolicyManager M3 No-Swap Semantics
# ---------------------------------------------------------------------------

class TestPolicyManagerNoSwap:
    def test_no_swap_retains_active_model_and_suppresses_replacement(self):
        retrain_called = [False]

        def _mock_retrain(*args, **kwargs):
            retrain_called[0] = True
            return "NEW_MODEL"

        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain,
            drift_monitor=monitor,
            n_interval=5,
            no_swap=True,
        )

        active_model = "INITIAL_MODEL"
        for i in range(5):
            monitor.update(error=0, tx_index=i)
            res = pm.step(tx_index=i, current_model=active_model)

        # Trigger fired at step 5!
        # In no_swap mode: retrain_fn must NOT have been called
        assert retrain_called[0] is False
        # Active model must be returned (NOT None, NOT NEW_MODEL)
        assert res == "INITIAL_MODEL"
        # Adaptation log must record trigger_occurred=True, replacement_performed=False
        assert pm.adaptation_count() == 1
        rec = pm.adaptation_log[0]
        assert rec.trigger_occurred is True
        assert rec.replacement_performed is False
        assert rec.duration_s == 0.0

    def test_normal_swap_invokes_retrainer(self):
        retrain_called = [False]

        def _mock_retrain(*args, **kwargs):
            retrain_called[0] = True
            return "NEW_MODEL"

        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain,
            drift_monitor=monitor,
            n_interval=5,
            no_swap=False,
        )

        active_model = "INITIAL_MODEL"
        for i in range(5):
            monitor.update(error=0, tx_index=i)
            res = pm.step(tx_index=i, current_model=active_model)

        assert retrain_called[0] is True
        assert res == "NEW_MODEL"
        rec = pm.adaptation_log[0]
        assert rec.trigger_occurred is True
        assert rec.replacement_performed is True


# ---------------------------------------------------------------------------
# Tests for Runner M2 Diagnostics & M3 Ablation
# ---------------------------------------------------------------------------

class TestRunnerM2AndM3:
    def test_runner_m2_pre_post_horizons(self):
        # Create small stream of 150 rows
        n_rows = 150
        np.random.seed(42)
        df_x = pd.DataFrame({"f1": np.random.randn(n_rows), "f2": np.random.randn(n_rows)})
        y = pd.Series((df_x["f1"] > 0).astype(int))

        # Setup runner with diagnostic_horizon=20, n_interval=30
        runner = StreamingPipelineRunner.from_config(
            policy_str="P1",
            detector_str="adwin",
            window_size=50,
            n_interval=30,
            diagnostic_horizon=20,
        )

        result = runner.run(X=df_x, y=y, segment=None, warmup_size=30)
        assert len(result.adaptation_log) > 0

        # Check diagnostic fields in adaptation log
        first_event = result.adaptation_log[0]
        assert "pre_ap" in first_event
        assert "post_ap" in first_event
        assert "delta_ap" in first_event
        assert "n_samples_window" in first_event
        assert "n_fraud_window" in first_event
        assert "fraud_prevalence_window" in first_event
        assert "model_complexity_pre" in first_event
        assert "model_complexity_post" in first_event
        assert first_event["pre_horizon_n"] > 0
        assert first_event["trigger_occurred"] is True
        assert first_event["replacement_performed"] is True

    def test_runner_no_swap_ablation_mode(self):
        n_rows = 100
        df_x = pd.DataFrame({"f1": np.random.randn(n_rows), "f2": np.random.randn(n_rows)})
        y = pd.Series((df_x["f1"] > 0).astype(int))

        runner = StreamingPipelineRunner.from_config(
            policy_str="P1",
            detector_str="adwin",
            window_size=50,
            n_interval=25,
            no_swap=True,
        )

        result = runner.run(X=df_x, y=y, segment=None, warmup_size=25)
        assert len(result.adaptation_log) > 0
        for rec in result.adaptation_log:
            assert rec["trigger_occurred"] is True
            assert rec["replacement_performed"] is False
            assert rec["duration_s"] == 0.0

        # Records should indicate adapted=False since replacement was suppressed
        adapted_flags = [r.adapted for r in result.records]
        assert all(flag is False for flag in adapted_flags)


# ---------------------------------------------------------------------------
# Tests for ExperimentManifest W-Aware & Legacy Compatibility
# ---------------------------------------------------------------------------

class TestManifestCompatibility:
    def test_w_aware_job_id_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m_path = Path(tmpdir) / "manifest.json"
            art_dir = Path(tmpdir) / "runs"
            manifest = ExperimentManifest(
                manifest_path=m_path,
                artifacts_dir=art_dir,
                experiment_name="Objective1_E1_E2",
                dataset_name="IEEE-CIS",
            )

            # Default / W=5000: legacy style
            id_legacy = manifest.make_job_id("P1", seed=42)
            assert id_legacy == "IEEE-CIS_Objective1_E1_E2_P1_seed42"

            id_w5k = manifest.make_job_id("P1", seed=42, window_size=5000)
            assert id_w5k == "IEEE-CIS_Objective1_E1_E2_P1_seed42"

            # W=10000: canonical W-aware
            id_w10k = manifest.make_job_id("P1", seed=42, window_size=10000)
            assert id_w10k == "IEEE-CIS_Objective1_E1_E2_P1_W10000_seed42"

            # W=20000: canonical W-aware
            id_w20k = manifest.make_job_id("P2", seed=101, window_size=20000)
            assert id_w20k == "IEEE-CIS_Objective1_E1_E2_P2_W20000_seed101"

            # no_swap flag
            id_noswap = manifest.make_job_id("P1", seed=42, no_swap=True)
            assert id_noswap == "IEEE-CIS_Objective1_E1_E2_P1_noswap_seed42"

    def test_legacy_w5000_alias_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m_path = Path(tmpdir) / "manifest.json"
            art_dir = Path(tmpdir) / "runs"
            art_dir.mkdir(parents=True, exist_ok=True)

            # Create a mock legacy run artifact: run_P1_seed42.json
            legacy_art = art_dir / "run_P1_seed42.json"
            mock_data = {
                "policy": "P1",
                "final_metrics": {"pr_auc": 0.2022, "roc_auc": 0.721},
                "latency_percentiles": {"p50_ms": 0.034},
                "adaptation_log": [],
                "n_warmup": 100,
                "n_stream": 500,
            }
            with open(legacy_art, "w") as f:
                json.dump(mock_data, f)

            manifest = ExperimentManifest(
                manifest_path=m_path,
                artifacts_dir=art_dir,
                experiment_name="Objective1_E1_E2",
                dataset_name="IEEE-CIS",
            )

            # Checking is_job_completed with window_size=5000 should recognize legacy artifact!
            assert manifest.is_job_completed("P1", seed=42, window_size=5000) is True
            # Querying W=10000 should be False
            assert manifest.is_job_completed("P1", seed=42, window_size=10000) is False


# ---------------------------------------------------------------------------
# Tests for Horizon Edge Cases, Leakage Safety & Existing Policy Semantics
# ---------------------------------------------------------------------------

class TestHorizonEdgeCasesAndPolicySemantics:
    def test_incomplete_post_horizon_at_stream_end(self):
        # Stream ends before H=50 forward samples can be collected
        n_rows = 60
        df_x = pd.DataFrame({"f1": np.random.randn(n_rows), "f2": np.random.randn(n_rows)})
        y = pd.Series((df_x["f1"] > 0).astype(int))

        runner = PrequentialRunner.from_config(
            policy_str="P1",
            detector_str="adwin",
            window_size=30,
            n_interval=20,
            diagnostic_horizon=50,  # H=50 > remaining stream length after step 40
        )
        result = runner.run(X=df_x, y=y, segment=None, warmup_size=20)
        assert len(result.adaptation_log) > 0
        last_event = result.adaptation_log[-1]
        assert last_event["horizon_complete"] is False
        assert last_event["post_horizon_n"] < 50

    def test_no_leakage_in_pre_and_post_horizons(self):
        # Verify pre_ap uses only tx strictly preceding trigger, post_ap uses strictly following
        n_rows = 120
        df_x = pd.DataFrame({"f1": np.linspace(0, 1, n_rows), "f2": np.linspace(1, 0, n_rows)})
        y = pd.Series([0] * 60 + [1] * 60)

        runner = PrequentialRunner.from_config(
            policy_str="P1",
            detector_str="adwin",
            window_size=30,
            n_interval=30,
            diagnostic_horizon=15,
        )
        result = runner.run(X=df_x, y=y, segment=None, warmup_size=30)
        # Event triggers at warmup + 30 = tx_index 30
        first_event = result.adaptation_log[0]
        assert first_event["pre_horizon_n"] == 15
        assert first_event["post_horizon_n"] == 15
        assert first_event["horizon_complete"] is True

    def test_runner_supports_w10000_w20000_instantiation(self):
        for w in [10000, 20000]:
            runner = PrequentialRunner.from_config(
                policy_str="P2",
                detector_str="adwin",
                window_size=w,
            )
            assert runner._retrainer.window_size == w

    def test_p0_incremental_only_semantics_preserved(self):
        # P0 should never trigger adaptation, remain incremental
        n_rows = 50
        df_x = pd.DataFrame({"f1": np.random.randn(n_rows)})
        y = pd.Series([0] * 25 + [1] * 25)
        runner = PrequentialRunner.from_config(policy_str="P0", detector_str="none")
        res = runner.run(X=df_x, y=y, segment=None, warmup_size=10)
        assert len(res.adaptation_log) == 0
        assert res.n_adaptations == 0

