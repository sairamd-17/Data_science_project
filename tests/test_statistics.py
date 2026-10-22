"""
Unit Tests for Statistical Analysis Module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_generator import generate_synthetic_housing_data
from modules.data_cleaning import clean_dataset
from modules.statistics import (
    run_chi_square_test, run_waiting_time_tests, run_correlation_analysis
)

@pytest.fixture
def sample_clean_df():
    raw_df = generate_synthetic_housing_data(n_samples=600, random_state=42)
    clean_df, _ = clean_dataset(raw_df)
    return clean_df

def test_chi_square_test(sample_clean_df):
    """Test Pearson Chi-Square test calculation and schema."""
    res = run_chi_square_test(sample_clean_df, col_x="Income category", col_y="Application status")
    
    assert "chi2_stat" in res
    assert "p_value" in res
    assert "degrees_of_freedom" in res
    assert "cramers_v" in res
    assert "contingency_table" in res
    assert res["chi2_stat"] >= 0
    assert 0.0 <= res["p_value"] <= 1.0
    assert 0.0 <= res["cramers_v"] <= 1.0
    assert len(res["interpretation"]) > 20

def test_waiting_time_tests(sample_clean_df):
    """Test ANOVA, Kruskal-Wallis, and assumption diagnostics."""
    res = run_waiting_time_tests(sample_clean_df, group_col="Region", value_col="Waiting time")
    
    assert "levene_stat" in res
    assert "levene_p" in res
    assert "f_stat" in res
    assert "anova_p" in res
    assert "kw_stat" in res
    assert "kw_p" in res
    assert "group_summary_df" in res
    assert 0.0 <= res["anova_p"] <= 1.0
    assert 0.0 <= res["kw_p"] <= 1.0
    assert len(res["group_summary_df"]) > 0

def test_correlation_analysis(sample_clean_df):
    """Test Pearson and Spearman correlation matrices."""
    corr_res = run_correlation_analysis(sample_clean_df)
    
    assert "pearson_corr" in corr_res
    assert "spearman_corr" in corr_res
    assert "p_values" in corr_res
    
    p_mat = corr_res["pearson_corr"]
    assert p_mat.shape[0] == p_mat.shape[1]
    # Diagonal of correlation matrix must be 1.0
    for col in p_mat.columns:
        assert np.isclose(p_mat.loc[col, col], 1.0)
