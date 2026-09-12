"""
Experiment Job Manifest and Atomic Checkpoint Manager for Objective 1.

Provides robust resumability for long-running streaming experiments:
1. Each (policy, seed) execution is treated as an independent job.
2. Checkpoints are written atomically (write to temp file, then rename).
3. Manifest tracks job lifecycle: PENDING -> RUNNING -> COMPLETED | FAILED.
4. If interrupted, completed jobs are skipped upon re-execution.
5. Live mirroring to a zip archive allows downloading partial progress anytime.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    experiment: str
    dataset: str
    policy: str
    seed: int
    status: str  # "PENDING", "RUNNING", "COMPLETED", "FAILED"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    wall_clock_duration_s: Optional[float] = None
    dataset_fingerprint: Optional[str] = None
    code_version: Optional[str] = None
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    secondary_metrics: Dict[str, Any] = field(default_factory=dict)
    adaptation_count: int = 0
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_tx_per_sec: Optional[float] = None
    artifact_path: Optional[str] = None
    error_message: Optional[str] = None
    window_size: Optional[int] = None
    no_swap: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def atomic_write_json(data: Any, target_path: Path, indent: int = 2) -> None:
    """Write JSON data to a target path atomically using a temporary file."""
    target_path = Path(target_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target_path.parent

    # Create temporary file in the same directory to ensure same filesystem for atomic rename
    fd, temp_file_path = tempfile.mkstemp(
        prefix=f".{target_path.stem}_",
        suffix=".tmp",
        dir=temp_dir,
    )
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        # Atomic rename (on Windows os.replace replaces existing file)
        os.replace(temp_file_path, target_path)
    except Exception:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass
        raise


def validate_run_artifact(artifact_path: Path) -> bool:
    """Verify that a run artifact exists and is a valid non-empty JSON file."""
    if not artifact_path.exists() or artifact_path.stat().st_size == 0:
        return False
    try:
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            isinstance(data, dict)
            and "policy" in data
            and "final_metrics" in data
            and "pr_auc" in data.get("final_metrics", {})
        )
    except Exception:
        return False


class ExperimentManifest:
    """Manages execution state, resumption, and atomic artifact tracking."""

    def __init__(
        self,
        manifest_path: Path,
        artifacts_dir: Path,
        zip_archive_path: Optional[Path] = None,
        experiment_name: str = "Objective1_E1_E2",
        dataset_name: str = "IEEE-CIS",
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.zip_archive_path = Path(zip_archive_path).resolve() if zip_archive_path else None
        self.experiment_name = experiment_name
        self.dataset_name = dataset_name

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, JobRecord] = {}
        self.load()

    @staticmethod
    def _get_artifact_filename(
        policy: str,
        seed: int,
        window_size: Optional[int] = None,
        no_swap: bool = False,
    ) -> str:
        parts = [f"run_{policy}"]
        if window_size is not None and window_size != 5000:
            parts.append(f"W{window_size}")
        if no_swap:
            parts.append("noswap")
        parts.append(f"seed{seed}.json")
        return "_".join(parts)

    def make_job_id(
        self,
        policy: str,
        seed: int,
        window_size: Optional[int] = None,
        no_swap: bool = False,
    ) -> str:
        parts = [self.dataset_name, self.experiment_name, policy]
        if window_size is not None and window_size != 5000:
            parts.append(f"W{window_size}")
        if no_swap:
            parts.append("noswap")
        parts.append(f"seed{seed}")
        return "_".join(parts)

    def load(self) -> None:
        """Load manifest from disk if it exists."""
        if not self.manifest_path.exists():
            return
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_jobs = data.get("jobs", {})
            for jid, r in raw_jobs.items():
                self.jobs[jid] = JobRecord(**r)
        except Exception:
            # If manifest is corrupted, do not crash; start fresh or recover from artifacts
            pass

    def save(self) -> None:
        """Save manifest to disk atomically."""
        data = {
            "experiment": self.experiment_name,
            "dataset": self.dataset_name,
            "last_updated": _utc_now_str(),
            "total_jobs": len(self.jobs),
            "completed_jobs": sum(1 for j in self.jobs.values() if j.status == "COMPLETED"),
            "jobs": {jid: j.as_dict() for jid, j in self.jobs.items()},
        }
        atomic_write_json(data, self.manifest_path, indent=2)
        if self.zip_archive_path:
            self.mirror_to_zip()

    def mirror_to_zip(self) -> None:
        """Mirror current manifest and all completed run artifacts into a zip archive."""
        if not self.zip_archive_path:
            return
        self.zip_archive_path.parent.mkdir(parents=True, exist_ok=True)
        temp_zip = self.zip_archive_path.with_suffix(".tmp.zip")
        try:
            with zipfile.ZipFile(temp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                if self.manifest_path.exists():
                    zf.write(self.manifest_path, arcname=self.manifest_path.name)
                for job in self.jobs.values():
                    if job.status == "COMPLETED" and job.artifact_path:
                        p = Path(job.artifact_path)
                        if p.exists():
                            zf.write(p, arcname=f"runs/{p.name}")
            os.replace(temp_zip, self.zip_archive_path)
        except Exception:
            if temp_zip.exists():
                try:
                    os.remove(temp_zip)
                except OSError:
                    pass

    def is_job_completed(
        self,
        policy: str,
        seed: int,
        window_size: Optional[int] = None,
        no_swap: bool = False,
    ) -> bool:
        """Check if job is marked COMPLETED and artifact exists and is valid."""
        job_id = self.make_job_id(policy, seed, window_size=window_size, no_swap=no_swap)
        job = self.jobs.get(job_id)
        if job and job.status == "COMPLETED" and job.artifact_path:
            p = Path(job.artifact_path)
            if validate_run_artifact(p):
                return True

        # Check expected file on disk
        expected_fn = self._get_artifact_filename(policy, seed, window_size=window_size, no_swap=no_swap)
        expected_file = self.artifacts_dir / expected_fn
        if validate_run_artifact(expected_file):
            self._recover_job_from_artifact(job_id, policy, seed, expected_file, window_size=window_size, no_swap=no_swap)
            return True

        # Legacy alias check for historical W=5000 or un-annotated baseline
        if (window_size is None or window_size == 5000) and not no_swap:
            legacy_id = f"{self.dataset_name}_{self.experiment_name}_{policy}_seed{seed}"
            legacy_job = self.jobs.get(legacy_id)
            if legacy_job and legacy_job.status == "COMPLETED" and legacy_job.artifact_path:
                p = Path(legacy_job.artifact_path)
                if validate_run_artifact(p):
                    return True
            legacy_file = self.artifacts_dir / f"run_{policy}_seed{seed}.json"
            if validate_run_artifact(legacy_file):
                self._recover_job_from_artifact(legacy_id, policy, seed, legacy_file, window_size=5000, no_swap=False)
                return True

        return False

    def _recover_job_from_artifact(
        self,
        job_id: str,
        policy: str,
        seed: int,
        file_path: Path,
        window_size: Optional[int] = None,
        no_swap: bool = False,
    ) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            fm = d.get("final_metrics", {})
            lp = d.get("latency_percentiles", {})
            rec_w = d.get("window_size", window_size)
            rec_ns = d.get("no_swap", no_swap)
            self.jobs[job_id] = JobRecord(
                job_id=job_id,
                experiment=self.experiment_name,
                dataset=self.dataset_name,
                policy=policy,
                seed=seed,
                status="COMPLETED",
                start_time=d.get("start_time"),
                end_time=d.get("end_time"),
                wall_clock_duration_s=d.get("wall_clock_duration_s"),
                dataset_fingerprint=d.get("dataset_fingerprint"),
                code_version=d.get("code_version"),
                pr_auc=fm.get("pr_auc"),
                roc_auc=fm.get("roc_auc"),
                secondary_metrics=fm,
                adaptation_count=len(d.get("adaptation_log", [])),
                latency_p50_ms=lp.get("p50_ms"),
                latency_p95_ms=lp.get("p95_ms"),
                latency_p99_ms=lp.get("p99_ms"),
                throughput_tx_per_sec=d.get("throughput_tx_per_sec"),
                artifact_path=str(file_path),
                window_size=rec_w,
                no_swap=rec_ns,
            )
            self.save()
        except Exception:
            pass

    def record_start(
        self,
        policy: str,
        seed: int,
        dataset_fingerprint: Optional[str] = None,
        window_size: Optional[int] = None,
        no_swap: bool = False,
    ) -> str:
        """Mark job as RUNNING."""
        job_id = self.make_job_id(policy, seed, window_size=window_size, no_swap=no_swap)
        self.jobs[job_id] = JobRecord(
            job_id=job_id,
            experiment=self.experiment_name,
            dataset=self.dataset_name,
            policy=policy,
            seed=seed,
            status="RUNNING",
            start_time=_utc_now_str(),
            dataset_fingerprint=dataset_fingerprint,
            window_size=window_size,
            no_swap=no_swap,
        )
        self.save()
        return job_id

    def record_completion(
        self,
        policy: str,
        seed: int,
        run_result: Any,
        wall_clock_duration_s: float,
        dataset_fingerprint: Optional[str] = None,
        code_version: Optional[str] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        trajectories: Optional[Dict[str, Any]] = None,
        window_size: Optional[int] = None,
        no_swap: bool = False,
    ) -> Path:
        """Atomically persist run artifact and update manifest."""
        job_id = self.make_job_id(policy, seed, window_size=window_size, no_swap=no_swap)
        start_time = self.jobs[job_id].start_time if job_id in self.jobs else _utc_now_str()
        end_time = _utc_now_str()

        # Extract metrics
        final_metrics = (
            run_result.final_metrics.as_dict()
            if hasattr(run_result.final_metrics, "as_dict")
            else dict(run_result.final_metrics)
        )
        raw_lat = (
            run_result.latency_percentiles.as_dict()
            if hasattr(run_result.latency_percentiles, "as_dict")
            else dict(run_result.latency_percentiles)
        )
        latency_pcts = dict(raw_lat)
        # Convert seconds to milliseconds if p50_ms is not already set
        p50_val = raw_lat.get("p50_ms", raw_lat.get("p50", 0.0) * (1.0 if "p50_ms" in raw_lat else 1000.0))
        p95_val = raw_lat.get("p95_ms", raw_lat.get("p95", 0.0) * (1.0 if "p95_ms" in raw_lat else 1000.0))
        p99_val = raw_lat.get("p99_ms", raw_lat.get("p99", 0.0) * (1.0 if "p99_ms" in raw_lat else 1000.0))
        latency_pcts["p50_ms"] = round(p50_val, 4)
        latency_pcts["p95_ms"] = round(p95_val, 4)
        latency_pcts["p99_ms"] = round(p99_val, 4)

        n_stream = run_result.n_stream
        throughput = n_stream / wall_clock_duration_s if wall_clock_duration_s > 0 else 0.0

        artifact_data = {
            "job_id": job_id,
            "experiment": self.experiment_name,
            "dataset": self.dataset_name,
            "policy": policy,
            "seed": seed,
            "window_size": window_size,
            "no_swap": no_swap,
            "start_time": start_time,
            "end_time": end_time,
            "wall_clock_duration_s": round(wall_clock_duration_s, 3),
            "throughput_tx_per_sec": round(throughput, 2),
            "dataset_fingerprint": dataset_fingerprint,
            "code_version": code_version,
            "config_snapshot": config_snapshot or {},
            "n_warmup": run_result.n_warmup,
            "n_stream": n_stream,
            "final_metrics": final_metrics,
            "latency_percentiles": latency_pcts,
            "adaptation_count": len(run_result.adaptation_log),
            "adaptation_log": run_result.adaptation_log,
            "drift_event_count": len(run_result.drift_event_log),
            "drift_event_log": run_result.drift_event_log,
            "trajectories": trajectories or {},
        }

        artifact_fn = self._get_artifact_filename(policy, seed, window_size=window_size, no_swap=no_swap)
        artifact_path = self.artifacts_dir / artifact_fn
        atomic_write_json(artifact_data, artifact_path, indent=2)

        # Validate before marking complete
        if not validate_run_artifact(artifact_path):
            raise RuntimeError(f"Written artifact at {artifact_path} failed validation!")

        self.jobs[job_id] = JobRecord(
            job_id=job_id,
            experiment=self.experiment_name,
            dataset=self.dataset_name,
            policy=policy,
            seed=seed,
            status="COMPLETED",
            start_time=start_time,
            end_time=end_time,
            wall_clock_duration_s=round(wall_clock_duration_s, 3),
            dataset_fingerprint=dataset_fingerprint,
            code_version=code_version,
            pr_auc=final_metrics.get("pr_auc"),
            roc_auc=final_metrics.get("roc_auc"),
            secondary_metrics=final_metrics,
            adaptation_count=len(run_result.adaptation_log),
            latency_p50_ms=latency_pcts.get("p50_ms"),
            latency_p95_ms=latency_pcts.get("p95_ms"),
            latency_p99_ms=latency_pcts.get("p99_ms"),
            throughput_tx_per_sec=round(throughput, 2),
            artifact_path=str(artifact_path),
            window_size=window_size,
            no_swap=no_swap,
        )
        self.save()
        return artifact_path

    def record_failure(
        self,
        policy: str,
        seed: int,
        error_message: str,
        window_size: Optional[int] = None,
        no_swap: bool = False,
    ) -> None:
        """Record job failure."""
        job_id = self.make_job_id(policy, seed, window_size=window_size, no_swap=no_swap)
        start_time = self.jobs[job_id].start_time if job_id in self.jobs else _utc_now_str()
        self.jobs[job_id] = JobRecord(
            job_id=job_id,
            experiment=self.experiment_name,
            dataset=self.dataset_name,
            policy=policy,
            seed=seed,
            status="FAILED",
            start_time=start_time,
            end_time=_utc_now_str(),
            error_message=error_message,
            window_size=window_size,
            no_swap=no_swap,
        )
        self.save()

    def get_summary_dataframe(self) -> Any:
        """Return pandas DataFrame summarizing all jobs."""
        import pandas as pd
        rows = [j.as_dict() for j in self.jobs.values()]
        return pd.DataFrame(rows)
