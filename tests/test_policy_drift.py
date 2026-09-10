"""
Unit tests for DriftMonitor (src/drift/monitor.py) and PolicyManager
(src/adaptation/policy.py).

All tests use tiny synthetic streams (< 1 000 errors) to guarantee
fast CI execution.  No real dataset files are required.

Test strategy
-------------
- DriftMonitor: verify global and segment detector instantiation, lazy
  segment creation, event logging, reset, and error on invalid detector type.
- PolicyManager: verify P0 never triggers, P1 triggers at exactly the right
  intervals, P2 triggers only when global drift fires, P3 triggers only for
  the affected segment, and adaptation logs are correct.
- Smoke test: drive a 200-step synthetic stream through the full
  (DriftMonitor + PolicyManager) stack for each policy.
"""

from __future__ import annotations

import collections
from typing import Any, Deque, List, Optional

import pytest

from src.drift.monitor import DriftEvent, DriftMonitor
from src.adaptation.policy import AdaptationRecord, PolicyManager


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_buffer() -> Deque:
    """Return a small memory buffer (deque) pre-filled with synthetic rows."""
    buf: Deque = collections.deque(maxlen=200)
    for i in range(50):
        buf.append({"x": float(i), "label": i % 2, "segment": "W"})
    return buf


def _mock_retrain_fn(memory_buffer: Any, *, segment_key: Optional[str] = None) -> str:
    """Simulated retrain function that returns a string sentinel model."""
    if segment_key is not None:
        return f"model_segment_{segment_key}"
    return "model_global"


# ---------------------------------------------------------------------------
# DriftMonitor — construction
# ---------------------------------------------------------------------------

class TestDriftMonitorConstruction:
    def test_global_only_default(self):
        monitor = DriftMonitor()
        assert monitor.detector_type == "adwin"
        assert not monitor.segment_aware
        assert monitor.known_segments == []
        assert monitor.all_events() == []

    def test_segment_aware_flag(self):
        monitor = DriftMonitor(segment_aware=True)
        assert monitor.segment_aware is True

    def test_kswin_detector(self):
        monitor = DriftMonitor(detector_type="kswin")
        assert monitor.detector_type == "kswin"

    def test_page_hinkley_detector(self):
        monitor = DriftMonitor(detector_type="page_hinkley")
        assert monitor.detector_type == "page_hinkley"

    def test_invalid_detector_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported detector type"):
            DriftMonitor(detector_type="unknown_detector")

    def test_detector_kwargs_forwarded(self):
        # Just check construction doesn't raise with a valid kwarg for ADWIN
        monitor = DriftMonitor(detector_type="adwin", detector_kwargs={"delta": 0.001})
        assert monitor.detector_type == "adwin"


# ---------------------------------------------------------------------------
# DriftMonitor — update & global detection
# ---------------------------------------------------------------------------

class TestDriftMonitorUpdate:
    def test_no_drift_on_zeros(self):
        """Constant zero error stream should not trigger ADWIN."""
        monitor = DriftMonitor(detector_type="adwin")
        for i in range(200):
            monitor.update(error=0.0, tx_index=i)
        assert monitor.global_drift_detected is False
        assert monitor.all_events() == []

    def test_drift_on_abrupt_shift(self):
        """Abrupt error shift (0→1) on a long stream must trigger ADWIN drift."""
        monitor = DriftMonitor(detector_type="adwin", detector_kwargs={"delta": 0.002})
        # Stable phase: all zeros
        for i in range(500):
            monitor.update(error=0, tx_index=i)
        # Drift phase: all ones
        detected = False
        for i in range(500, 1000):
            monitor.update(error=1, tx_index=i)
            if monitor.global_drift_detected:
                detected = True
                break
        assert detected, "ADWIN should detect the abrupt 0→1 shift within 500 steps"

    def test_global_event_logged(self):
        monitor = DriftMonitor(detector_type="adwin", detector_kwargs={"delta": 0.002})
        for i in range(500):
            monitor.update(error=0, tx_index=i)
        events_before = len(monitor.all_events())
        for i in range(500, 1000):
            monitor.update(error=1, tx_index=i)
            if monitor.global_drift_detected:
                break
        events = monitor.all_events()
        assert len(events) > events_before
        global_events = [e for e in events if e.stream_key == "global"]
        assert len(global_events) >= 1
        assert global_events[0].detector_type == "adwin"

    def test_update_counts_tracked(self):
        monitor = DriftMonitor()
        for i in range(10):
            monitor.update(error=0, tx_index=i)
        assert monitor.update_counts()["global"] == 10


# ---------------------------------------------------------------------------
# DriftMonitor — segment-aware mode
# ---------------------------------------------------------------------------

class TestDriftMonitorSegments:
    def test_segments_created_lazily(self):
        monitor = DriftMonitor(segment_aware=True)
        assert monitor.known_segments == []
        monitor.update(error=0, tx_index=0, segment_key="W")
        assert "W" in monitor.known_segments
        monitor.update(error=0, tx_index=1, segment_key="H")
        assert sorted(monitor.known_segments) == ["H", "W"]

    def test_segment_ignored_when_not_segment_aware(self):
        monitor = DriftMonitor(segment_aware=False)
        monitor.update(error=0, tx_index=0, segment_key="W")
        assert monitor.known_segments == []

    def test_segment_drift_unknown_returns_false(self):
        monitor = DriftMonitor(segment_aware=True)
        assert monitor.segment_drift_detected("TRANSFER") is False

    def test_segment_drift_detected_after_abrupt_shift(self):
        monitor = DriftMonitor(
            segment_aware=True,
            detector_kwargs={"delta": 0.002},
        )
        # Warm up segment "W" with zeros
        for i in range(500):
            monitor.update(error=0, tx_index=i, segment_key="W")
        # Abrupt shift in segment "W"
        detected = False
        for i in range(500, 1000):
            monitor.update(error=1, tx_index=i, segment_key="W")
            if monitor.segment_drift_detected("W"):
                detected = True
                break
        assert detected, "Per-segment ADWIN should detect abrupt shift in segment 'W'"

    def test_unaffected_segment_stays_clean(self):
        """Drift in W should not contaminate segment H."""
        monitor = DriftMonitor(
            segment_aware=True,
            detector_kwargs={"delta": 0.002},
        )
        for i in range(500):
            monitor.update(error=0, tx_index=i, segment_key="W")
            monitor.update(error=0, tx_index=i, segment_key="H")
        for i in range(500, 1000):
            monitor.update(error=1, tx_index=i, segment_key="W")
            monitor.update(error=0, tx_index=i, segment_key="H")  # H stays clean
        # H should still be clean (no drift)
        assert not monitor.segment_drift_detected("H"), (
            "Segment H should not be flagged when only W has drifted"
        )

    def test_pending_segment_events_excludes_global(self):
        monitor = DriftMonitor(
            segment_aware=True,
            detector_kwargs={"delta": 0.002},
        )
        for i in range(500):
            monitor.update(error=0, tx_index=i, segment_key="W")
        for i in range(500, 1000):
            monitor.update(error=1, tx_index=i, segment_key="W")
            if monitor.segment_drift_detected("W"):
                break
        seg_events = list(monitor.pending_segment_events())
        assert all(e.stream_key != "global" for e in seg_events)

    def test_segment_update_counts(self):
        monitor = DriftMonitor(segment_aware=True)
        for i in range(5):
            monitor.update(error=0, tx_index=i, segment_key="W")
        for i in range(5, 8):
            monitor.update(error=0, tx_index=i, segment_key="H")
        counts = monitor.update_counts()
        assert counts["W"] == 5
        assert counts["H"] == 3


# ---------------------------------------------------------------------------
# DriftMonitor — reset
# ---------------------------------------------------------------------------

class TestDriftMonitorReset:
    def test_reset_clears_events(self):
        monitor = DriftMonitor(
            segment_aware=True,
            detector_kwargs={"delta": 0.002},
        )
        for i in range(500):
            monitor.update(error=0, tx_index=i, segment_key="W")
        for i in range(500, 1000):
            monitor.update(error=1, tx_index=i, segment_key="W")
            if monitor.global_drift_detected:
                break

        monitor.reset()

        assert monitor.all_events() == []
        assert monitor.known_segments == []
        assert monitor.update_counts() == {"global": 0}
        assert not monitor.global_drift_detected

    def test_reset_allows_fresh_run(self):
        monitor = DriftMonitor(detector_kwargs={"delta": 0.002})
        for i in range(500):
            monitor.update(error=0, tx_index=i)
        for i in range(500, 1000):
            monitor.update(error=1, tx_index=i)
            if monitor.global_drift_detected:
                break
        monitor.reset()
        # After reset, constant zeros should produce no events
        for i in range(100):
            monitor.update(error=0, tx_index=i)
        assert monitor.all_events() == []


# ---------------------------------------------------------------------------
# PolicyManager — P0 Static
# ---------------------------------------------------------------------------

class TestPolicyP0:
    def test_p0_never_retrains(self):
        monitor = DriftMonitor(segment_aware=False)
        pm = PolicyManager(
            policy="P0",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        for i in range(200):
            monitor.update(error=i % 2, tx_index=i)  # alternating — may cause drift
            result = pm.step(tx_index=i, current_model="model_v0", segment_key="W")
            assert result is None, f"P0 should never retrain (step {i})"
        assert pm.adaptation_count() == 0

    def test_p0_tx_counter_increments(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P0",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        for i in range(7):
            pm.step(tx_index=i, current_model="m")
        assert pm.tx_counter == 7


# ---------------------------------------------------------------------------
# PolicyManager — P1 Periodic
# ---------------------------------------------------------------------------

class TestPolicyP1:
    def test_p1_triggers_at_exact_intervals(self):
        n_interval = 10
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            n_interval=n_interval,
            memory_buffer=_make_buffer(),
        )
        trigger_steps = []
        for i in range(50):
            monitor.update(error=0, tx_index=i)
            result = pm.step(tx_index=i, current_model="m")
            if result is not None:
                trigger_steps.append(pm.tx_counter)

        # Should trigger at tx_counter == 10, 20, 30, 40, 50
        expected = list(range(n_interval, 51, n_interval))
        assert trigger_steps == expected, (
            f"P1 triggered at {trigger_steps}, expected {expected}"
        )

    def test_p1_returns_new_model(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            n_interval=5,
            memory_buffer=_make_buffer(),
        )
        result = None
        for i in range(5):
            monitor.update(error=0, tx_index=i)
            result = pm.step(tx_index=i, current_model="old_model")
        assert result == "model_global"

    def test_p1_adaptation_log_entries(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            n_interval=10,
            memory_buffer=_make_buffer(),
        )
        for i in range(30):
            monitor.update(error=0, tx_index=i)
            pm.step(tx_index=i, current_model="m")

        assert pm.adaptation_count() == 3
        for record in pm.adaptation_log:
            assert record.policy == "P1"
            assert record.scope == "global"
            assert "periodic" in record.trigger

    def test_p1_total_adaptation_time_nonnegative(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            n_interval=5,
            memory_buffer=_make_buffer(),
        )
        for i in range(15):
            monitor.update(error=0, tx_index=i)
            pm.step(tx_index=i, current_model="m")
        assert pm.total_adaptation_time() >= 0.0


# ---------------------------------------------------------------------------
# PolicyManager — P2 Global Drift
# ---------------------------------------------------------------------------

class TestPolicyP2:
    def _run_until_global_drift(self, pm: PolicyManager, monitor: DriftMonitor):
        """Drive stream until global drift fires; return triggering tx_index."""
        for i in range(500):
            monitor.update(error=0, tx_index=i)
            pm.step(tx_index=i, current_model="m")
        for i in range(500, 2000):
            monitor.update(error=1, tx_index=i)
            result = pm.step(tx_index=i, current_model="m")
            if result is not None:
                return i
        return None  # drift never triggered (test will fail)

    def test_p2_triggers_on_global_drift(self):
        monitor = DriftMonitor(detector_kwargs={"delta": 0.002})
        pm = PolicyManager(
            policy="P2",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        trigger_idx = self._run_until_global_drift(pm, monitor)
        assert trigger_idx is not None, "P2 should trigger on global drift"
        assert pm.adaptation_count() >= 1

    def test_p2_log_scope_is_global(self):
        monitor = DriftMonitor(detector_kwargs={"delta": 0.002})
        pm = PolicyManager(
            policy="P2",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        self._run_until_global_drift(pm, monitor)
        for record in pm.adaptation_log:
            assert record.scope == "global"
            assert record.policy == "P2"

    def test_p2_no_trigger_on_constant_stream(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P2",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        for i in range(300):
            monitor.update(error=0, tx_index=i)
            pm.step(tx_index=i, current_model="m")
        assert pm.adaptation_count() == 0


# ---------------------------------------------------------------------------
# PolicyManager — P3 Segment Drift
# ---------------------------------------------------------------------------

class TestPolicyP3:
    def test_p3_triggers_only_for_drifting_segment(self):
        """P3 must trigger for 'W' (drifting) but not for 'H' (stable)."""
        monitor = DriftMonitor(
            segment_aware=True,
            detector_kwargs={"delta": 0.002},
        )
        pm = PolicyManager(
            policy="P3",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )

        # Warmup: both segments stable
        for i in range(500):
            seg = "W" if i % 2 == 0 else "H"
            monitor.update(error=0, tx_index=i, segment_key=seg)
            pm.step(tx_index=i, current_model="m", segment_key=seg)

        assert pm.adaptation_count() == 0, "No adaptation expected during stable phase"

        # Inject drift into W only, keep H stable
        w_triggers = 0
        for i in range(500, 2000):
            seg = "W"
            monitor.update(error=1, tx_index=i, segment_key="W")
            monitor.update(error=0, tx_index=i, segment_key="H")
            result = pm.step(tx_index=i, current_model="m", segment_key=seg)
            if result is not None:
                w_triggers += 1
                break

        assert w_triggers >= 1, "P3 should trigger for drifting segment 'W'"

    def test_p3_model_scoped_to_segment(self):
        """retrain_fn receives the correct segment_key for P3."""
        monitor = DriftMonitor(
            segment_aware=True,
            detector_kwargs={"delta": 0.002},
        )
        pm = PolicyManager(
            policy="P3",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        # Cause drift in segment "C"
        for i in range(500):
            monitor.update(error=0, tx_index=i, segment_key="C")
            pm.step(tx_index=i, current_model="m", segment_key="C")

        result = None
        for i in range(500, 2000):
            monitor.update(error=1, tx_index=i, segment_key="C")
            result = pm.step(tx_index=i, current_model="m", segment_key="C")
            if result is not None:
                break

        assert result == "model_segment_C", (
            f"Expected 'model_segment_C' but got {result!r}"
        )

    def test_p3_none_segment_key_is_noop(self):
        monitor = DriftMonitor(segment_aware=True)
        pm = PolicyManager(
            policy="P3",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        for i in range(10):
            monitor.update(error=1, tx_index=i)  # no segment_key
            result = pm.step(tx_index=i, current_model="m", segment_key=None)
            assert result is None

    def test_p3_log_scope_is_segment(self):
        monitor = DriftMonitor(
            segment_aware=True,
            detector_kwargs={"delta": 0.002},
        )
        pm = PolicyManager(
            policy="P3",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        for i in range(500):
            monitor.update(error=0, tx_index=i, segment_key="W")
            pm.step(tx_index=i, current_model="m", segment_key="W")
        for i in range(500, 2000):
            monitor.update(error=1, tx_index=i, segment_key="W")
            result = pm.step(tx_index=i, current_model="m", segment_key="W")
            if result is not None:
                break
        for record in pm.adaptation_log:
            assert record.scope == "W"
            assert record.policy == "P3"


# ---------------------------------------------------------------------------
# PolicyManager — invalid policy name
# ---------------------------------------------------------------------------

class TestPolicyManagerValidation:
    def test_invalid_policy_raises(self):
        monitor = DriftMonitor()
        with pytest.raises(ValueError, match="Unknown policy"):
            PolicyManager(
                policy="P99",
                retrain_fn=_mock_retrain_fn,
                drift_monitor=monitor,
                memory_buffer=_make_buffer(),
            )

    def test_case_insensitive_policy_name(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="p0",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            memory_buffer=_make_buffer(),
        )
        assert pm.policy == "P0"


# ---------------------------------------------------------------------------
# PolicyManager — reset
# ---------------------------------------------------------------------------

class TestPolicyManagerReset:
    def test_reset_clears_counter_and_log(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            n_interval=5,
            memory_buffer=_make_buffer(),
        )
        for i in range(15):
            monitor.update(error=0, tx_index=i)
            pm.step(tx_index=i, current_model="m")

        pm.reset()
        assert pm.tx_counter == 0
        assert pm.adaptation_count() == 0
        assert pm.adaptation_log == []


# ---------------------------------------------------------------------------
# AdaptationRecord
# ---------------------------------------------------------------------------

class TestAdaptationRecord:
    def test_as_dict(self):
        record = AdaptationRecord(
            policy="P2",
            tx_index=1500,
            scope="global",
            duration_s=0.003,
            trigger="P2:global_drift_detected",
        )
        d = record.as_dict()
        assert d["policy"] == "P2"
        assert d["tx_index"] == 1500
        assert d["scope"] == "global"
        assert d["duration_s"] == pytest.approx(0.003)
        assert d["trigger"] == "P2:global_drift_detected"


# ---------------------------------------------------------------------------
# Full-stack smoke test: 200-step stream through each policy
# ---------------------------------------------------------------------------

class TestFullStackSmoke:
    """Drive a synthetic 200-step stream through the full stack for each policy.

    The stream has no injected drift so P2 and P3 should remain quiet.
    P0 should never trigger; P1 with n_interval=50 should trigger 4 times.
    """

    SEGMENTS = ["W", "H", "C", "S", "R"]

    def _run_stream(
        self,
        policy: str,
        n_steps: int = 200,
        n_interval: int = 50,
        error_sequence: Optional[List[int]] = None,
        segment_aware: bool = False,
    ) -> PolicyManager:
        monitor = DriftMonitor(
            segment_aware=segment_aware,
            detector_kwargs={"delta": 0.002},
        )
        pm = PolicyManager(
            policy=policy,
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            n_interval=n_interval,
            memory_buffer=_make_buffer(),
        )
        if error_sequence is None:
            error_sequence = [0] * n_steps

        for i, err in enumerate(error_sequence):
            seg = self.SEGMENTS[i % len(self.SEGMENTS)]
            monitor.update(error=err, tx_index=i, segment_key=seg)
            pm.step(tx_index=i, current_model=f"model_v{i}", segment_key=seg)
        return pm

    def test_p0_smoke(self):
        pm = self._run_stream("P0")
        assert pm.adaptation_count() == 0
        assert pm.tx_counter == 200

    def test_p1_smoke(self):
        pm = self._run_stream("P1", n_steps=200, n_interval=50)
        # Triggers at tx_counter = 50, 100, 150, 200
        assert pm.adaptation_count() == 4

    def test_p2_smoke_no_drift(self):
        pm = self._run_stream("P2")
        assert pm.adaptation_count() == 0

    def test_p3_smoke_no_drift(self):
        pm = self._run_stream("P3", segment_aware=True)
        assert pm.adaptation_count() == 0

    def test_adaptation_record_fields_populated(self):
        monitor = DriftMonitor()
        pm = PolicyManager(
            policy="P1",
            retrain_fn=_mock_retrain_fn,
            drift_monitor=monitor,
            n_interval=10,
            memory_buffer=_make_buffer(),
        )
        for i in range(10):
            monitor.update(error=0, tx_index=i)
            pm.step(tx_index=i, current_model="m")

        assert pm.adaptation_count() == 1
        record = pm.adaptation_log[0]
        assert record.policy == "P1"
        assert record.scope == "global"
        assert record.tx_index == 9  # 0-based index of the 10th tx
        assert record.duration_s >= 0.0
        assert "periodic" in record.trigger
