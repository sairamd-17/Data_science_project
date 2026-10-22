"""
Unit and Integration Tests for Data Cleaning and Preprocessing Pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_generator import generate_synthetic_housing_data
from modules.data_cleaning import (
    validate_columns, audit_dataset, clean_dataset, REQUIRED_COLUMNS
)

def test_data_generator():
    """Verify synthetic data generator produces valid rows and required schema."""
    df = generate_synthetic_housing_data(n_samples=1200, random_state=42)
    assert len(df) >= 1200, "Dataset must have at least 1,200 records"
    is_valid, missing = validate_columns(df)
    assert is_valid, f"Missing required columns in generated dataset: {missing}"

def test_validate_columns():
    """Verify schema validator detects missing columns correctly."""
    valid_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    is_valid, missing = validate_columns(valid_df)
    assert is_valid
    assert len(missing) == 0
    
    invalid_df = pd.DataFrame(columns=["Application ID", "Region"])
    is_valid_inv, missing_inv = validate_columns(invalid_df)
    assert not is_valid_inv
    assert "Annual income" in missing_inv

def test_cleaning_deduplication():
    """Verify duplicates are successfully removed."""
    df_raw = generate_synthetic_housing_data(n_samples=500, random_state=1)
    # Add duplicate rows
    df_with_dups = pd.concat([df_raw, df_raw.iloc[:15]], ignore_index=True)
    df_clean, audit = clean_dataset(df_with_dups)
    
    assert audit["duplicates_removed"] >= 15
    assert df_clean["Application ID"].duplicated().sum() == 0
    assert len(df_clean) == len(df_clean.drop_duplicates())

def test_cleaning_no_nans_and_anomalies():
    """Verify missing values are imputed and anomalies are corrected."""
    df_raw = generate_synthetic_housing_data(n_samples=500, random_state=10)
    df_clean, audit = clean_dataset(df_raw)
    
    # Check that required columns have 0 NaNs
    for col in REQUIRED_COLUMNS:
        assert df_clean[col].isnull().sum() == 0, f"Column {col} still has missing values after cleaning"
        
    # Check numeric bounds
    assert (df_clean["Annual income"] >= 0).all(), "Negative income found"
    assert (df_clean["Household size"] >= 1).all(), "Household size < 1 found"
    assert (df_clean["Waiting time"] >= 0).all(), "Negative waiting time found"
    assert (df_clean["Monthly rent"] >= 0).all(), "Negative monthly rent found"
    assert (df_clean["Housing need score"].between(10, 100)).all(), "Need score out of bounds [10, 100]"

def test_logical_consistency_and_derived_columns():
    """Verify logical business rules and derived features."""
    df_raw = generate_synthetic_housing_data(n_samples=500, random_state=20)
    df_clean, _ = clean_dataset(df_raw)
    
    # Ineligible rule
    ineligible = df_clean[df_clean["Eligibility status"] == "Ineligible"]
    assert (ineligible["Application status"] == "Rejected").all(), "Ineligible applicant was not marked as Rejected"
    assert (ineligible["Housing allocation"] == "Not Allocated").all(), "Ineligible applicant was marked as Allocated"
    
    # Derived columns existence
    for d_col in ["Per_Capita_Income", "Monthly_Income", "Rent_Burden_Ratio", "Need_Priority_Tier", "Waiting_Time_Category", "Is_Vulnerable"]:
        assert d_col in df_clean.columns, f"Derived column {d_col} is missing"
        
    # Derived column bounds
    assert (df_clean["Rent_Burden_Ratio"].between(0, 1.0)).all()
    assert (df_clean["Per_Capita_Income"] > 0).all()
