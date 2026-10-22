"""
Statistical Analysis Module for Public Housing Fairness Evaluation.
Implements hypothesis testing and correlation analytics using SciPy:
- Chi-Square Test of Independence
- One-Way ANOVA (Parametric)
- Levene's Test (Assumption check for Homogeneity of Variances)
- Shapiro-Wilk / D'Agostino Normality Checks
- Kruskal-Wallis H-Test (Non-parametric alternative)
- Pearson & Spearman Correlation Matrices with Significance Testing
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

ALPHA = 0.05

def run_chi_square_test(df: pd.DataFrame, col_x: str = "Income category", col_y: str = "Application status") -> Dict[str, Any]:
    """
    Perform Pearson's Chi-Square Test of Independence between two categorical variables.
    """
    contingency = pd.crosstab(df[col_x], df[col_y])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    # Calculate Cramer's V as effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
    
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns).round(2)
    
    is_significant = p_val < ALPHA
    
    interpretation = (
        f"With a Chi-Square statistic of {chi2:.2f} and p-value = {p_val:.4e} (df = {dof}), "
        f"we {'reject' if is_significant else 'fail to reject'} the null hypothesis at α = {ALPHA}. "
        f"There is a {'statistically significant' if is_significant else 'no statistically significant'} "
        f"association between {col_x} and {col_y} (Cramer's V = {cramers_v:.3f}). "
        "Caution: Statistical association does not demonstrate discriminatory bias; "
        "public housing statutory criteria deliberately prioritize lower income and higher vulnerability applicants."
    )
    
    return {
        "test_name": f"Chi-Square Test of Independence ({col_x} vs. {col_y})",
        "h0": f"Null Hypothesis (H0): {col_x} and {col_y} are independent (no relationship).",
        "h1": f"Alternative Hypothesis (H1): {col_x} and {col_y} are significantly associated.",
        "chi2_stat": round(float(chi2), 3),
        "p_value": float(p_val),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 3),
        "alpha": ALPHA,
        "is_significant": is_significant,
        "contingency_table": contingency,
        "expected_table": expected_df,
        "interpretation": interpretation
    }

def run_waiting_time_tests(df: pd.DataFrame, group_col: str = "Region", value_col: str = "Waiting time") -> Dict[str, Any]:
    """
    Evaluate waiting time differences across groups (e.g. Region or Income Category).
    Checks assumptions (Normality and Levene's Homogeneity of Variance),
    runs One-Way ANOVA, and performs Kruskal-Wallis H-test.
    """
    # Exclude zero or negative waiting times (e.g., rejected applicants)
    active_data = df[df[value_col] > 0]
    
    groups_dict = {}
    for group_name, group_df in active_data.groupby(group_col):
        vals = group_df[value_col].dropna().values
        if len(vals) >= 3:
            groups_dict[group_name] = vals
            
    group_names = list(groups_dict.keys())
    group_values = list(groups_dict.values())
    
    if len(group_values) < 2:
        return {"error": "Insufficient groups or data points to run tests."}
        
    # 1. Assumption Check: Levene's test for equality of variance
    levene_stat, levene_p = stats.levene(*group_values)
    equal_variance = levene_p >= ALPHA
    
    # 2. Assumption Check: D'Agostino-Pearson omnibus test or Shapiro for normality on residuals
    all_vals = active_data[value_col].dropna().values
    if len(all_vals) <= 5000:
        norm_stat, norm_p = stats.shapiro(all_vals[:500])  # Sample for Shapiro if large
    else:
        norm_stat, norm_p = stats.normaltest(all_vals)
    is_normal = norm_p >= ALPHA
    
    # 3. One-Way ANOVA (Parametric)
    f_stat, anova_p = stats.f_oneway(*group_values)
    anova_sig = anova_p < ALPHA
    
    # 4. Kruskal-Wallis H-Test (Non-Parametric)
    kw_stat, kw_p = stats.kruskal(*group_values)
    kw_sig = kw_p < ALPHA
    
    # Group level summary table
    group_stats = []
    for name, vals in groups_dict.items():
        group_stats.append({
            group_col: name,
            "Sample Count (N)": len(vals),
            "Mean (Months)": round(float(np.mean(vals)), 2),
            "Std Dev": round(float(np.std(vals, ddof=1)), 2),
            "Median (Months)": round(float(np.median(vals)), 2),
            "IQR": round(float(stats.iqr(vals)), 2)
        })
    group_summary_df = pd.DataFrame(group_stats).sort_values("Mean (Months)", ascending=False)
    
    # Overall interpretation
    recommended_test = "Kruskal-Wallis H-Test" if (not is_normal or not equal_variance) else "One-Way ANOVA"
    
    interpretation = (
        f"Assumption diagnostics: Levene's test p-value = {levene_p:.4e} "
        f"({'Equal variances assumed' if equal_variance else 'Unequal variances detected'}), "
        f"Normality check p-value = {norm_p:.4e} "
        f"({'Normal distribution supported' if is_normal else 'Non-normal distribution detected'}). "
        f"Because assumptions for standard ANOVA are {'met' if (is_normal and equal_variance) else 'violated (skewness / heteroscedasticity)'}, "
        f"the {recommended_test} serves as the primary rigorous benchmark. "
        f"Kruskal-Wallis statistic H = {kw_stat:.2f}, p-value = {kw_p:.4e}. "
        f"There is a {'statistically significant' if kw_sig else 'no statistically significant'} difference "
        f"in waiting durations across {group_col}s (α = {ALPHA}). Regional variations in housing inventory "
        "and turnover velocity substantially influence administrative wait cycles."
    )
    
    return {
        "group_col": group_col,
        "value_col": value_col,
        "levene_stat": round(float(levene_stat), 3),
        "levene_p": float(levene_p),
        "equal_variance": equal_variance,
        "is_normal": is_normal,
        "norm_stat": round(float(norm_stat), 3),
        "norm_p": float(norm_p),
        "f_stat": round(float(f_stat), 3),
        "anova_p": float(anova_p),
        "anova_sig": anova_sig,
        "kw_stat": round(float(kw_stat), 3),
        "kw_p": float(kw_p),
        "kw_sig": kw_sig,
        "recommended_test": recommended_test,
        "group_summary_df": group_summary_df,
        "interpretation": interpretation
    }

def run_correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman correlation matrices for numeric features.
    """
    numeric_cols = [
        "Annual income",
        "Household size",
        "Housing need score",
        "Waiting time",
        "Monthly rent",
        "Per_Capita_Income",
        "Rent_Burden_Ratio"
    ]
    # Filter to existing columns in df
    cols = [c for c in numeric_cols if c in df.columns]
    
    sub_df = df[cols].dropna()
    
    pearson_corr = sub_df.corr(method="pearson").round(3)
    spearman_corr = sub_df.corr(method="spearman").round(3)
    
    # Calculate p-values for Pearson correlations
    p_values = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for c1 in cols:
        for c2 in cols:
            if c1 == c2:
                p_values.loc[c1, c2] = 0.0
            else:
                _, p = stats.pearsonr(sub_df[c1], sub_df[c2])
                p_values.loc[c1, c2] = round(p, 4)
                
    return {
        "columns": cols,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "p_values": p_values
    }

def create_correlation_heatmap(corr_df: pd.DataFrame, title: str = "Feature Correlation Matrix") -> go.Figure:
    """
    Generate an interactive Plotly heatmap for correlation matrices.
    """
    fig = px.imshow(
        corr_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title=title
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig
