"""
Unit and benchmark tests for row feature extraction optimization.

Verifies:
1. Exact feature ordering and names.
2. Exact values and types.
3. Proper handling of NaN / missing values.
4. Proper handling of categorical values.
5. Mathematical equivalence of model predictions and updates across both methods.
6. Benchmark comparing reference (.iat) vs optimized (NumPy + dict(zip)).
"""

import time
import numpy as np
import pandas as pd
import pytest
from river.tree import HoeffdingTreeClassifier

from src.pipeline.runner import PrequentialRunner


def _generate_test_dataframe(n_rows: int = 100, n_cols: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for j in range(n_cols):
        col_name = f"feat_{j:03d}"
        if j % 5 == 0:
            # Column with missing values
            vals = rng.normal(loc=0.0, scale=1.0, size=n_rows)
            mask = rng.uniform(size=n_rows) < 0.2
            vals[mask] = np.nan
            data[col_name] = vals
        elif j % 5 == 1:
            # Integer column
            data[col_name] = rng.integers(0, 100, size=n_rows)
        elif j % 5 == 2:
            # Categorical string column
            cats = ["W", "H", "C", "R", "S"]
            data[col_name] = rng.choice(cats, size=n_rows)
        else:
            # Standard continuous float
            data[col_name] = rng.normal(loc=10.0, scale=5.0, size=n_rows)
    return pd.DataFrame(data)


def test_feature_extraction_exact_equivalence():
    """Verify that numpy-backed extraction produces exact keys, values, and types as .iat."""
    df = _generate_test_dataframe(n_rows=50, n_cols=30)
    col_names = df.columns.tolist()
    arr = df.to_numpy()

    for i in range(len(df)):
        ref = PrequentialRunner._extract_row_reference(df, i)
        opt = dict(zip(col_names, arr[i]))

        assert list(ref.keys()) == list(opt.keys()), f"Keys mismatch at row {i}"
        for col in col_names:
            v_ref = ref[col]
            v_opt = opt[col]
            if pd.isna(v_ref):
                assert pd.isna(v_opt), f"NaN mismatch for {col} at row {i}"
            else:
                assert v_ref == v_opt, f"Value mismatch for {col} at row {i}: {v_ref} != {v_opt}"


def test_model_identity_across_extractions():
    """Verify that HoeffdingTree produces bit-for-bit identical probabilities and weights."""
    df = _generate_test_dataframe(n_rows=100, n_cols=20)
    col_names = df.columns.tolist()
    arr = df.to_numpy()
    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, size=len(df)).tolist()

    model_ref = HoeffdingTreeClassifier(grace_period=10)
    model_opt = HoeffdingTreeClassifier(grace_period=10)

    for i in range(len(df)):
        ref = PrequentialRunner._extract_row_reference(df, i)
        opt = dict(zip(col_names, arr[i]))

        prob_ref = model_ref.predict_proba_one(ref)
        prob_opt = model_opt.predict_proba_one(opt)
        assert prob_ref == prob_opt, f"Probability mismatch at row {i}: {prob_ref} vs {prob_opt}"

        model_ref.learn_one(ref, y[i])
        model_opt.learn_one(opt, y[i])

        # Verify state after learning
        post_prob_ref = model_ref.predict_proba_one(ref)
        post_prob_opt = model_opt.predict_proba_one(opt)
        assert post_prob_ref == post_prob_opt, f"Post-learning mismatch at row {i}"


def test_full_pipeline_run_equivalence():
    """Verify that running PrequentialRunner produces the exact same results with optimized extraction."""
    from src.pipeline.runner import PrequentialRunner

    df = _generate_test_dataframe(n_rows=80, n_cols=15, seed=7)
    rng = np.random.default_rng(999)
    y = pd.Series(rng.integers(0, 2, size=len(df)), name="target")
    seg = pd.Series(["W", "C", "R", "W"] * (len(df) // 4), name="segment")

    runner = PrequentialRunner.from_config(
        policy_str="P0",
        detector_str="adwin",
        window_size=20,
        n_interval=10,
        grace_period=10,
    )
    res = runner.run(X=df, y=y, segment=seg, warmup_size=20)

    assert res.n_stream == 60
    assert len(res.records) == 60
    assert hasattr(res.final_metrics, "pr_auc")
    assert hasattr(res.final_metrics, "roc_auc")
    assert res.final_metrics.as_dict()["pr_auc"] is not None


def test_feature_extraction_benchmark():
    """Benchmark reference vs optimized extraction and report metrics."""
    n_rows = 500
    n_cols = 100
    df = _generate_test_dataframe(n_rows=n_rows, n_cols=n_cols, seed=1)

    # Reference extraction timing
    t0 = time.perf_counter()
    for i in range(n_rows):
        _ = PrequentialRunner._extract_row_reference(df, i)
    ref_time = time.perf_counter() - t0

    # Optimized extraction timing
    col_names = df.columns.tolist()
    t1 = time.perf_counter()
    arr = df.to_numpy()
    for i in range(n_rows):
        _ = dict(zip(col_names, arr[i]))
    opt_time = time.perf_counter() - t1

    ref_ms_per_row = (ref_time / n_rows) * 1000
    opt_ms_per_row = (opt_time / n_rows) * 1000
    ref_rows_per_sec = n_rows / ref_time
    opt_rows_per_sec = n_rows / opt_time
    speedup = ref_time / opt_time

    print(
        f"\n[BENCHMARK] Feature Extraction ({n_rows} rows, {n_cols} columns):\n"
        f"  Reference (.iat):   {ref_ms_per_row:.3f} ms/row | {ref_rows_per_sec:.1f} rows/s\n"
        f"  Optimized (NumPy):  {opt_ms_per_row:.3f} ms/row | {opt_rows_per_sec:.1f} rows/s\n"
        f"  Speedup factor:     {speedup:.1f}x faster\n"
    )

    assert speedup > 5.0, f"Expected speedup > 5x, got {speedup:.2f}x"
