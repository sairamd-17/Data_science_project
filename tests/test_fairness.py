"""
Unit Tests for Housing Fairness and Report Generation Modules.
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
from modules.fairness import (
    evaluate_group_equity, evaluate_vulnerable_households, evaluate_regional_supply_demand
)
from modules.report_generator import build_pdf_report

@pytest.fixture
def sample_clean_df():
    raw_df = generate_synthetic_housing_data(n_samples=500, random_state=42)
    clean_df, _ = clean_dataset(raw_df)
    return clean_df

def test_evaluate_group_equity(sample_clean_df):
    """Verify group equity evaluation output and disparity ratios."""
    eq_df = evaluate_group_equity(sample_clean_df, group_col="Income category")
    
    assert "Eligible_Approval_Rate_%" in eq_df.columns
    assert "Disparity_Ratio" in eq_df.columns
    assert "Potential_Disparity_Flag" in eq_df.columns
    assert (eq_df["Disparity_Ratio"] >= 0).all()
    assert (eq_df["Eligible_Approval_Rate_%"] <= 100).all()

def test_evaluate_vulnerable_households(sample_clean_df):
    """Verify vulnerable cohorts calculations."""
    v_df = evaluate_vulnerable_households(sample_clean_df)
    
    assert len(v_df) > 0
    assert "Vulnerable Cohort" in v_df.columns
    assert "Approval Rate (Eligible %)" in v_df.columns
    assert "Median Wait (Months)" in v_df.columns

def test_evaluate_regional_supply_demand(sample_clean_df):
    """Verify allocation-to-demand ratio computation."""
    sd_df = evaluate_regional_supply_demand(sample_clean_df)
    
    assert "Allocation_to_Demand_Ratio" in sd_df.columns
    assert "Eligible Demand (Households)" in sd_df.columns
    assert "Units Allocated (Supply)" in sd_df.columns
    assert (sd_df["Allocation_to_Demand_Ratio"] >= 0).all()

def test_pdf_report_generation(sample_clean_df):
    """Verify PDF report builder generates valid non-empty PDF bytes."""
    pdf_bytes = build_pdf_report(sample_clean_df)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 10000, "Generated PDF bytes should exceed 10KB"
    assert pdf_bytes.startswith(b"%PDF"), "Output must be a valid PDF file starting with %PDF"
