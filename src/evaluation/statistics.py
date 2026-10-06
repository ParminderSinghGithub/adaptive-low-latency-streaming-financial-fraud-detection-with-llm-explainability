"""
Inferential Statistical Analysis Infrastructure for Objective 2 (Experiment E3).

Implements the formal inferential statistical protocol (Source of Truth §13.1):
1. Pairwise policy comparisons across independent stochastic seed replications.
2. Effect size quantification: Paired Cohen's dz and Hedges' gz.
3. Multiplicity corrections:
   - Holm-Sidak step-down procedure for Family-Wise Error Rate (FWER) control.
   - Benjamini-Hochberg procedure for False Discovery Rate (FDR) control across multi-regime grids.
4. Policy x Drift Regime two-way ANOVA and rank concordance (Kendall's W)
   to evaluate Hypothesis H2 (interaction and rank stability).

Note on E3 Replication Assumptions:
- Inferential tests (paired t-test, Wilcoxon, Holm-Sidak, FDR) require independent
  stochastic replicates. In E3, this assumption is satisfied by the Gradual regime
  (10 stochastic transition sequences).
- For Sudden, Recurring, and Localized regimes, nominal seeds produce identical
  perturbations and deterministic metric outputs (s_D = 0). Downstream analysis
  must treat them as reproducibility/invariance audits, not independent samples.
- The two-way ANOVA requires independent within-cell replicate observations.
  When evaluating deterministic regimes, Kendall's W provides descriptive rank
  concordance rather than claiming an inferential multi-regime ANOVA F-test.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats


def cohens_dz(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute paired Cohen's dz effect size for repeated measures across seeds.

    d_z = mean(d) / std(d, ddof=1) where d = x - y.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) != len(y_arr):
        raise ValueError(f"Lengths must match: {len(x_arr)} vs {len(y_arr)}.")
    if len(x_arr) < 2:
        raise ValueError(f"Sample size must be >= 2, got {len(x_arr)}.")

    diff = x_arr - y_arr
    s_d = float(np.std(diff, ddof=1))
    if s_d == 0.0 or np.isnan(s_d):
        return 0.0
    return float(np.mean(diff) / s_d)


def hedges_gz(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute sample-size corrected paired Hedges' gz effect size.

    g_z = d_z * (1 - 3 / (4*(n - 1) - 1))
    """
    d_z = cohens_dz(x, y)
    n = len(x)
    if n <= 2:
        return d_z
    correction = 1.0 - (3.0 / (4.0 * (n - 1.0) - 1.0))
    return float(d_z * correction)


def paired_policy_comparison(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    policy_a: str = "A",
    policy_b: str = "B",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform comprehensive paired hypothesis testing between two policies across seeds.

    Tests difference d = scores_a - scores_b.
    Computes:
    - Shapiro-Wilk normality test on differences.
    - Paired Student's t-test (parametric).
    - Wilcoxon signed-rank test (non-parametric fallback).
    - Mean difference, standard error, and 95% confidence interval.
    - Cohen's dz and Hedges' gz effect sizes.
    """
    a_arr = np.asarray(scores_a, dtype=float)
    b_arr = np.asarray(scores_b, dtype=float)
    n = len(a_arr)
    if n != len(b_arr):
        raise ValueError(f"Arrays must have identical length: {n} vs {len(b_arr)}.")
    if n < 2:
        raise ValueError(f"At least 2 paired observations required, got {n}.")

    diff = a_arr - b_arr
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = float(std_diff / math.sqrt(n)) if n > 0 else 0.0

    # Degrees of freedom and t critical
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Normality test
    shapiro_stat, shapiro_p = (None, None)
    if n >= 3:
        try:
            sw = stats.shapiro(diff)
            shapiro_stat, shapiro_p = float(sw.statistic), float(sw.pvalue)
        except Exception:
            pass

    # Paired Student's t-test
    t_res = stats.ttest_rel(a_arr, b_arr)
    t_stat = float(t_res.statistic)
    t_p = float(t_res.pvalue)

    # Wilcoxon signed-rank test
    wilcoxon_stat, wilcoxon_p = (None, None)
    if not np.all(diff == 0):
        try:
            w_res = stats.wilcoxon(diff, zero_method="wilcox", correction=True)
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
        except Exception:
            pass

    # Effect sizes
    d_z = cohens_dz(a_arr, b_arr)
    g_z = hedges_gz(a_arr, b_arr)

    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "n_seeds": n,
        "mean_diff": round(mean_diff, 5),
        "std_diff": round(std_diff, 5),
        "se_diff": round(se_diff, 5),
        "ci_95": (round(ci_lower, 5), round(ci_upper, 5)),
        "t_statistic": round(t_stat, 4),
        "t_pvalue": round(t_p, 6),
        "shapiro_pvalue": round(shapiro_p, 4) if shapiro_p is not None else None,
        "wilcoxon_statistic": round(wilcoxon_stat, 4) if wilcoxon_stat is not None else None,
        "wilcoxon_pvalue": round(wilcoxon_p, 6) if wilcoxon_p is not None else None,
        "cohens_dz": round(d_z, 4),
        "hedges_gz": round(g_z, 4),
        "significant_at_alpha": bool(t_p < alpha),
    }


def holm_sidak_correction(
    p_values: Sequence[float], alpha: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """Apply the Holm-Sidak step-down procedure for Family-Wise Error Rate (FWER) control.

    Parameters
    ----------
    p_values:
        Collection of raw unadjusted p-values.
    alpha:
        Family-wise nominal significance level (default 0.05).

    Returns
    -------
    Tuple of (adjusted_p_values, rejected_hypotheses_boolean_list)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    # Track original positions
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p_sorted = [0.0] * m
    rejected_sorted = [False] * m

    running_max = 0.0
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        k = rank_idx + 1
        # Sidak threshold for step k
        sidak_alpha = 1.0 - (1.0 - alpha) ** (1.0 / (m - k + 1.0))

        # Adjusted p-value: 1 - (1 - p)^(m - k + 1)
        p_clamped = min(max(p_val, 0.0), 1.0)
        adj_p = 1.0 - (1.0 - p_clamped) ** (m - k + 1.0)
        adj_p = max(adj_p, running_max)  # enforce monotonicity
        adj_p = min(adj_p, 1.0)
        running_max = adj_p

        adj_p_sorted[rank_idx] = adj_p
        rejected_sorted[rank_idx] = bool(p_val <= sidak_alpha)

    # Reorder to match original p_values sequence
    adjusted_p = [0.0] * m
    rejected = [False] * m
    for rank_idx, (orig_idx, _) in enumerate(indexed_p):
        adjusted_p[orig_idx] = adj_p_sorted[rank_idx]
        rejected[orig_idx] = rejected_sorted[rank_idx]

    return adjusted_p, rejected


def benjamini_hochberg_fdr(
    p_values: Sequence[float], q: float = 0.05
) -> Tuple[List[float], List[bool]]:
    """Apply Benjamini-Hochberg procedure for False Discovery Rate (FDR) control.

    Parameters
    ----------
    p_values:
        Collection of raw unadjusted p-values.
    q:
        FDR threshold (default 0.05).

    Returns
    -------
    Tuple of (adjusted_p_values, rejected_hypotheses_boolean_list)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adj_p_sorted = [0.0] * m

    # Step down from largest p-value to smallest to enforce monotonicity
    running_min = 1.0
    for rev_rank, (orig_idx, p_val) in enumerate(reversed(indexed_p)):
        rank = m - rev_rank
        p_adj = (p_val * m) / float(rank)
        p_adj = min(p_adj, running_min)
        p_adj = min(p_adj, 1.0)
        running_min = p_adj
        adj_p_sorted[rank - 1] = p_adj

    adjusted_p = [0.0] * m
    rejected = [False] * m
    for rank_idx, (orig_idx, _) in enumerate(indexed_p):
        p_adj = adj_p_sorted[rank_idx]
        adjusted_p[orig_idx] = p_adj
        rejected[orig_idx] = bool(p_adj <= q)

    return adjusted_p, rejected


def policy_regime_interaction_test(
    df: pd.DataFrame,
    metric_col: str = "pr_auc",
    policy_col: str = "policy",
    regime_col: str = "regime",
    seed_col: str = "seed",
) -> Dict[str, Any]:
    """Test Policy x Drift Regime interaction effect (Hypothesis H2).

    Computes two-way ANOVA decomposition on multi-seed experiment results:
    - Main Effect of Policy: F, p-value, partial eta-squared.
    - Main Effect of Regime: F, p-value, partial eta-squared.
    - Interaction Effect Policy x Regime: F, p-value, partial eta-squared.
    - Kendall's W coefficient of concordance measuring rank stability across regimes.

    Methodological Note:
    - Inferential ANOVA F-tests strictly require independent within-cell replication.
      If applied to regimes with deterministic seeds (s_D = 0), within-cell variance
      is artificial, invalidating inferential p-values.
    - In such settings, Kendall's W serves as a descriptive measure of rank stability,
      and policy rankings should be evaluated descriptively across regimes.
    """
    required_cols = {metric_col, policy_col, regime_col, seed_col}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    clean_df = df.dropna(subset=[metric_col]).copy()
    policies = sorted(clean_df[policy_col].unique())
    regimes = sorted(clean_df[regime_col].unique())

    p_count = len(policies)
    r_count = len(regimes)
    if p_count < 2 or r_count < 2:
        raise ValueError(f"Need >=2 policies and >=2 regimes; got {p_count} policies, {r_count} regimes.")

    grand_mean = clean_df[metric_col].mean()
    ss_total = np.sum((clean_df[metric_col] - grand_mean) ** 2)
    n_total = len(clean_df)

    # 1. Main Effect: Policy
    policy_means = clean_df.groupby(policy_col)[metric_col].mean()
    policy_counts = clean_df.groupby(policy_col)[metric_col].count()
    ss_policy = np.sum(policy_counts * (policy_means - grand_mean) ** 2)
    df_policy = p_count - 1

    # 2. Main Effect: Regime
    regime_means = clean_df.groupby(regime_col)[metric_col].mean()
    regime_counts = clean_df.groupby(regime_col)[metric_col].count()
    ss_regime = np.sum(regime_counts * (regime_means - grand_mean) ** 2)
    df_regime = r_count - 1

    # 3. Cell Means & Interaction
    cell_means = clean_df.groupby([policy_col, regime_col])[metric_col].mean()
    cell_counts = clean_df.groupby([policy_col, regime_col])[metric_col].count()

    ss_cells = 0.0
    for (pol, reg), c_mean in cell_means.items():
        cnt = cell_counts[(pol, reg)]
        ss_cells += cnt * (c_mean - grand_mean) ** 2

    ss_interaction = ss_cells - ss_policy - ss_regime
    ss_interaction = max(0.0, float(ss_interaction))
    df_interaction = df_policy * df_regime

    # 4. Error / Residuals
    ss_error = max(1e-12, float(ss_total - ss_cells))
    df_error = n_total - (p_count * r_count)
    if df_error <= 0:
        df_error = 1

    ms_error = ss_error / df_error
    ms_policy = ss_policy / max(1, df_policy)
    ms_regime = ss_regime / max(1, df_regime)
    ms_interaction = ss_interaction / max(1, df_interaction)

    f_policy = ms_policy / ms_error
    p_policy = 1.0 - float(stats.f.cdf(f_policy, df_policy, df_error))
    eta_sq_policy = ss_policy / (ss_policy + ss_error)

    f_regime = ms_regime / ms_error
    p_regime = 1.0 - float(stats.f.cdf(f_regime, df_regime, df_error))
    eta_sq_regime = ss_regime / (ss_regime + ss_error)

    f_interaction = ms_interaction / ms_error
    p_interaction = 1.0 - float(stats.f.cdf(f_interaction, df_interaction, df_error))
    eta_sq_interaction = ss_interaction / (ss_interaction + ss_error)

    # 5. Kendall's W Coefficient of Concordance (Policy Rank Stability)
    # Pivot matrix: rows = regimes, columns = policies, values = mean metric
    pivot = clean_df.groupby([regime_col, policy_col])[metric_col].mean().unstack()
    # Rank policies within each regime (1 = lowest, k = highest)
    ranks = pivot.rank(axis=1, ascending=True)
    m_judges = len(ranks)  # regimes
    k_items = len(ranks.columns)  # policies
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = (m_judges * (k_items + 1)) / 2.0
    s_sq = np.sum((rank_sums - mean_rank_sum) ** 2)
    denom = (m_judges ** 2) * (k_items ** 3 - k_items) / 12.0
    kendall_w = float(s_sq / denom) if denom > 0 else 0.0

    return {
        "n_observations": n_total,
        "main_effect_policy": {
            "df": df_policy,
            "f_statistic": round(float(f_policy), 4),
            "p_value": round(float(p_policy), 6),
            "partial_eta_squared": round(float(eta_sq_policy), 4),
        },
        "main_effect_regime": {
            "df": df_regime,
            "f_statistic": round(float(f_regime), 4),
            "p_value": round(float(p_regime), 6),
            "partial_eta_squared": round(float(eta_sq_regime), 4),
        },
        "interaction_policy_x_regime": {
            "df": df_interaction,
            "f_statistic": round(float(f_interaction), 4),
            "p_value": round(float(p_interaction), 6),
            "partial_eta_squared": round(float(eta_sq_interaction), 4),
            "significant_interaction": bool(p_interaction < 0.05),
        },
        "rank_concordance": {
            "kendalls_w": round(kendall_w, 4),
            "interpretation": (
                "High rank stability across regimes"
                if kendall_w >= 0.7
                else "Moderate/low rank stability (policy rank shifts across regimes)"
            ),
        },
    }
