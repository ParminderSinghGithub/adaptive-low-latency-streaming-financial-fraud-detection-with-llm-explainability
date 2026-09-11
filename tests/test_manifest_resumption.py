"""
Unit tests for the ExperimentManifest and atomic checkpoint system.

Verifies:
1. Atomic JSON writing behavior.
2. Artifact validation (detecting valid and corrupted files).
3. Manifest lifecycle: start -> complete -> skip on re-run.
4. Automatic recovery of completed jobs from disk artifacts if manifest is reset.
5. Zip archive mirroring.
"""

import json
from pathlib import Path
import pytest

from src.pipeline.manifest import (
    ExperimentManifest,
    atomic_write_json,
    validate_run_artifact,
)


class DummyMetrics:
    def as_dict(self):
        return {"pr_auc": 0.75, "roc_auc": 0.88, "f1": 0.65}


class DummyLatencies:
    def as_dict(self):
        return {"p50_ms": 1.2, "p95_ms": 3.5, "p99_ms": 8.0}


class DummyRunResult:
    def __init__(self):
        self.n_warmup = 100
        self.n_stream = 900
        self.final_metrics = DummyMetrics()
        self.latency_percentiles = DummyLatencies()
        self.adaptation_log = [{"tx": 500, "scope": "global"}]
        self.drift_event_log = [{"tx": 480, "type": "adwin"}]


def test_atomic_write_json(tmp_path: Path):
    target = tmp_path / "subdir" / "test.json"
    data = {"key": "value", "numbers": [1, 2, 3]}
    atomic_write_json(data, target)

    assert target.exists()
    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_validate_run_artifact(tmp_path: Path):
    valid_file = tmp_path / "valid.json"
    data = {
        "policy": "P0",
        "final_metrics": {"pr_auc": 0.72, "roc_auc": 0.85},
    }
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    assert validate_run_artifact(valid_file) is True

    # Empty file
    empty_file = tmp_path / "empty.json"
    empty_file.touch()
    assert validate_run_artifact(empty_file) is False

    # Incomplete schema
    bad_file = tmp_path / "bad.json"
    with open(bad_file, "w", encoding="utf-8") as f:
        json.dump({"foo": "bar"}, f)
    assert validate_run_artifact(bad_file) is False


def test_manifest_job_lifecycle_and_resumption(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    artifacts_dir = tmp_path / "runs"
    zip_path = tmp_path / "archive.zip"

    manifest = ExperimentManifest(
        manifest_path=manifest_file,
        artifacts_dir=artifacts_dir,
        zip_archive_path=zip_path,
        experiment_name="TestExp",
        dataset_name="TestDS",
    )

    policy = "P1"
    seed = 42

    assert manifest.is_job_completed(policy, seed) is False

    # Start job
    manifest.record_start(policy, seed, dataset_fingerprint="sha256:abc")
    assert manifest.is_job_completed(policy, seed) is False

    # Complete job
    dummy_res = DummyRunResult()
    art_path = manifest.record_completion(
        policy=policy,
        seed=seed,
        run_result=dummy_res,
        wall_clock_duration_s=12.5,
        dataset_fingerprint="sha256:abc",
        code_version="v1.0.0",
    )

    assert Path(art_path).exists()
    assert manifest.is_job_completed(policy, seed) is True
    assert zip_path.exists()

    # Re-instantiate manifest (simulating notebook restart)
    new_manifest = ExperimentManifest(
        manifest_path=manifest_file,
        artifacts_dir=artifacts_dir,
        zip_archive_path=zip_path,
        experiment_name="TestExp",
        dataset_name="TestDS",
    )
    # Must immediately recognize completed job and skip
    assert new_manifest.is_job_completed(policy, seed) is True

    # Another job (P2, seed 42) is NOT completed
    assert new_manifest.is_job_completed("P2", seed) is False


def test_manifest_recovery_from_disk(tmp_path: Path):
    """If the manifest file itself is deleted or lost, existing valid run artifacts are recovered."""
    manifest_file = tmp_path / "manifest.json"
    artifacts_dir = tmp_path / "runs"

    manifest = ExperimentManifest(
        manifest_path=manifest_file,
        artifacts_dir=artifacts_dir,
    )

    dummy_res = DummyRunResult()
    manifest.record_start("P0", 101)
    manifest.record_completion("P0", 101, dummy_res, wall_clock_duration_s=5.0)
    assert manifest.is_job_completed("P0", 101) is True

    # Delete manifest file
    manifest_file.unlink()
    assert not manifest_file.exists()

    # Create fresh manifest instance
    fresh_manifest = ExperimentManifest(
        manifest_path=manifest_file,
        artifacts_dir=artifacts_dir,
    )
    # The artifact exists on disk and is valid, so is_job_completed recovers it
    assert fresh_manifest.is_job_completed("P0", 101) is True
    expected_jid = fresh_manifest.make_job_id("P0", 101)
    assert expected_jid in fresh_manifest.jobs
    assert fresh_manifest.jobs[expected_jid].status == "COMPLETED"

