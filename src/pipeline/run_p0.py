"""
Standalone Authoritative P0 Execution Runner.

Executes the P0 Incremental-Only Baseline Control under strict prequential
streaming semantics (predict-then-train, learn_one on all revealed labels,
zero retraining, zero model replacements, no retraining window W).

Reuses the canonical data loader, preprocessor, and prequential runner.
Saves artifacts atomically to outputs/checkpoints/objective1_runs/.

Usage:
    python -m src.pipeline.run_p0 --mode SMOKE
    python -m src.pipeline.run_p0 --mode FULL
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

from src.data.loader import load_ieee_cis
from src.data.preprocessing import StreamingPreprocessor
from src.pipeline.manifest import ExperimentManifest
from src.pipeline.runner import PrequentialRunner
from src.utils.seed import set_seed


def find_project_root(start_path: Path) -> Path:
    current = start_path.resolve()
    while current != current.parent:
        if (current / "src").is_dir() and (current / "pyproject.toml").is_file():
            return current
        current = current.parent
    return Path(".").resolve()


def resolve_ieee_cis_dir(project_root: Path) -> Path:
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob("train_transaction.csv"))
        if matches:
            return matches[0].parent

    for base in [project_root, Path.cwd(), Path("/kaggle/working")]:
        if base.exists():
            matches = list(base.rglob("train_transaction.csv"))
            if matches:
                return matches[0].parent

    raise FileNotFoundError("Could not locate 'train_transaction.csv'.")


def extract_downsampled_trajectories(
    records: List[Any], window_size: int = 5000, max_points: int = 150
) -> Dict[str, Any]:
    n_records = len(records)
    if n_records == 0:
        return {"indices": [], "rolling_pr_auc": []}

    step = max(1, n_records // max_points)
    indices = []
    pr_aucs = []

    y_true_all = [r.y_true for r in records]
    y_prob_all = [r.y_prob for r in records]

    for end_idx in range(window_size, n_records + 1, step):
        start_idx = end_idx - window_size
        y_w = y_true_all[start_idx:end_idx]
        p_w = y_prob_all[start_idx:end_idx]

        if sum(y_w) > 0 and len(y_w) - sum(y_w) > 0:
            prec, rec, _ = precision_recall_curve(y_w, p_w)
            score = float(auc(rec, prec))
        else:
            score = 0.0

        indices.append(end_idx)
        pr_aucs.append(round(score, 5))

    return {
        "indices": indices,
        "rolling_pr_auc": pr_aucs,
        "window_size": window_size,
    }


def run_p0(mode: str = "SMOKE", seeds: Optional[List[int]] = None, output_dir: Optional[Path] = None) -> None:
    project_root = find_project_root(Path.cwd())
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if seeds is None:
        seeds = [42, 101]

    # Mode parameters
    is_smoke = (mode.upper() == "SMOKE")
    smoke_row_limit = 1000 if is_smoke else None
    grace_period = 50 if is_smoke else 200
    delta = 0.002
    warmup_frac = 0.15
    rolling_window = 200 if is_smoke else 5000
    max_traj_pts = 50 if is_smoke else 200

    print(f"=== Objective 1 P0 Incremental Control Runner ===")
    print(f"Mode:             {mode.upper()}")
    print(f"Project Root:     {project_root}")
    print(f"Seeds:            {seeds}")

    # Output paths
    if output_dir is not None:
        checkpoint_dir = Path(output_dir).resolve()
        manifest_path = checkpoint_dir.parent / "objective1_manifest.json"
        zip_archive_path = checkpoint_dir.parent.parent / "objective1_artifacts_live.zip"
    else:
        kaggle_working = Path("/kaggle/working")
        outputs_base = kaggle_working if kaggle_working.exists() else project_root
        checkpoint_dir = outputs_base / "outputs" / "checkpoints" / "objective1_runs"
        manifest_path = outputs_base / "outputs" / "checkpoints" / "objective1_manifest.json"
        zip_archive_path = outputs_base / "outputs" / "objective1_artifacts_live.zip"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = ExperimentManifest(
        manifest_path=manifest_path,
        artifacts_dir=checkpoint_dir,
        zip_archive_path=zip_archive_path,
        experiment_name="Objective1_E1_E2",
        dataset_name="IEEE-CIS",
    )

    # Ingestion & Preprocessing
    data_dir = resolve_ieee_cis_dir(project_root)
    print(f"Data directory:   {data_dir}")
    df_raw = load_ieee_cis(data_dir=data_dir, split="train", join_identity=True)
    if smoke_row_limit is not None:
        df_raw = df_raw.iloc[:smoke_row_limit].copy()
    print(f"Loaded raw data:  {len(df_raw):,} transactions")

    # Chronological check
    assert df_raw["TransactionDT"].is_monotonic_increasing, "Dataset must be sorted by TransactionDT!"
    data_fingerprint = hashlib.sha256(
        f"{len(df_raw)}_{df_raw['TransactionID'].iloc[0]}_{df_raw['TransactionID'].iloc[-1]}".encode("utf-8")
    ).hexdigest()[:16]
    print(f"Fingerprint:      {data_fingerprint}")

    warmup_size = int(len(df_raw) * warmup_frac)
    stream_size = len(df_raw) - warmup_size
    print(f"Warmup size:      {warmup_size:,} | Stream size: {stream_size:,}")

    preprocessor = StreamingPreprocessor(dataset_name="ieee_cis", scale_features=False)
    print("Fitting preprocessor on warmup partition...")
    preprocessor.fit(df_raw.iloc[:warmup_size])
    X_all, y_all, seg_all = preprocessor.transform(df_raw)

    # Execution Loop over seeds
    for seed in seeds:
        job_id = manifest.make_job_id("P0", seed=seed, window_size=None, no_swap=False)
        print(f"\n--- Running Job {job_id} (Seed {seed}) ---")

        if manifest.is_job_completed("P0", seed=seed, window_size=None, no_swap=False):
            print(f"Job {job_id} already COMPLETED on disk -> SKIPPING.")
            continue

        manifest.record_start("P0", seed=seed, dataset_fingerprint=data_fingerprint, window_size=None, no_swap=False)
        set_seed(seed)

        config_snapshot = {
            "warmup_size": warmup_size,
            "stream_size": stream_size,
            "warmup_frac": warmup_frac,
            "window_size": None,
            "n_interval": 10000,
            "grace_period": grace_period,
            "delta": delta,
            "diagnostic_horizon": 500,
            "no_swap": False,
            "run_mode": mode.upper(),
        }

        t_start = time.perf_counter()
        runner = PrequentialRunner.from_config(
            policy_str="P0",
            detector_str="adwin",
            grace_period=grace_period,
            delta=delta,
            window_size=None,
            detector_kwargs={"delta": delta},
            segment_aware=False,
            seed=seed,
            p0_mode="incremental",
            diagnostic_horizon=500,
            no_swap=False,
        )

        run_result = runner.run(
            X=X_all,
            y=y_all,
            segment=seg_all,
            warmup_size=warmup_size,
        )
        wall_clock_s = time.perf_counter() - t_start

        trajectories = extract_downsampled_trajectories(
            run_result.records,
            window_size=rolling_window,
            max_points=max_traj_pts,
        )

        artifact_path = manifest.record_completion(
            policy="P0",
            seed=seed,
            run_result=run_result,
            wall_clock_duration_s=wall_clock_s,
            dataset_fingerprint=data_fingerprint,
            code_version="authoritative_o1_v3_p0_supplement",
            config_snapshot=config_snapshot,
            trajectories=trajectories,
            window_size=None,
            no_swap=False,
        )

        fm = run_result.final_metrics
        lp = run_result.latency_percentiles
        pr_val = fm.pr_auc if hasattr(fm, "pr_auc") else fm.get("pr_auc", 0.0)
        roc_val = fm.roc_auc if hasattr(fm, "roc_auc") else fm.get("roc_auc", 0.0)
        p95_val = lp.get("p95_ms", lp.get("p95", 0.0) * 1000)

        print(
            f"    [COMPLETED] Duration: {wall_clock_s:.1f}s | "
            f"PR-AUC: {pr_val:.5f} | ROC-AUC: {roc_val:.5f} | "
            f"Adaptations: {len(run_result.adaptation_log)} | "
            f"Latency p95: {p95_val:.2f}ms | Saved: {artifact_path.name}"
        )

    print("\nP0 execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Authoritative P0 Incremental Control")
    parser.add_argument(
        "--mode",
        choices=["SMOKE", "FULL"],
        default="SMOKE",
        help="Execution mode: SMOKE for fast local test (1000 rows), FULL for production (590k rows)",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101], help="Seeds to execute")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom checkpoint output directory")
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else None
    run_p0(mode=args.mode, seeds=args.seeds, output_dir=out_dir)
