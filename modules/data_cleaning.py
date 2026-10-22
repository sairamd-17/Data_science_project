"""
Data Cleaning and Preprocessing Module for Public Housing Fairness Analysis.
Performs data auditing, anomaly detection, deduplication, imputation,
standardization, and feature engineering using Pandas.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

REQUIRED_COLUMNS = [
    "Application ID",
    "Region",
    "Household size",
    "Annual income",
    "Income category",
    "Gender",
    "Disability status",
    "Housing need score",
    "Eligibility status",
    "Application status",
    "Housing allocation",
    "Waiting time",
    "Monthly rent"
]

def validate_columns(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate that the uploaded/loaded DataFrame has the mandatory columns.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols

def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audit dataset health, detecting missing values, duplicates, and data anomalies.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["Application ID"].duplicated().sum()) if "Application ID" in df.columns else 0
    
    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pcts = ((df.isnull().sum() / total_rows) * 100).round(2).to_dict()
    
    # Value anomalies
    numeric_anomalies = {}
    if "Annual income" in df.columns:
        numeric_anomalies["negative_income"] = int((df["Annual income"] < 0).sum())
    if "Household size" in df.columns:
        numeric_anomalies["zero_or_negative_household"] = int((df["Household size"] <= 0).sum())
    if "Waiting time" in df.columns:
        numeric_anomalies["negative_waiting_time"] = int((df["Waiting time"] < 0).sum())
    if "Housing need score" in df.columns:
        numeric_anomalies["out_of_bound_need_score"] = int(((df["Housing need score"] < 0) | (df["Housing need score"] > 100)).sum())
    if "Monthly rent" in df.columns:
        numeric_anomalies["negative_rent"] = int((df["Monthly rent"] < 0).sum())
        
    # Memory usage
    mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    
    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "numeric_anomalies": numeric_anomalies,
        "memory_kb": mem_kb
    }

def clean_dataset(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive Pandas data cleaning pipeline.
    
    Steps:
    1. Deduplication (exact & ID-based)
    2. Categorical normalization (whitespace, casing, aliases)
    3. Missing value imputation with statistical proxies
    4. Numeric boundary validation & correction
    5. Logical constraint validation (e.g., Ineligible applicants)
    6. Feature engineering (derived metrics)
    """
    # Audit before cleaning
    before_audit = audit_dataset(df_raw)
    
    df = df_raw.copy()
    
    # ---------------------------------------------------------
    # STEP 1: Deduplication
    # ---------------------------------------------------------
    # First drop exact duplicate rows
    df = df.drop_duplicates()
    # If duplicate Application IDs exist with conflicting values, retain first occurrence
    if "Application ID" in df.columns:
        df = df.drop_duplicates(subset=["Application ID"], keep="first")
    df = df.reset_index(drop=True)
    
    # ---------------------------------------------------------
    # STEP 2: Standardize Categorical Data
    # ---------------------------------------------------------
    # Income category standardization
    if "Income category" in df.columns:
        def standardize_income_cat(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower().replace("-", " ")
            if "very" in s or "extremely" in s:
                return "Very Low Income"
            elif "low" in s:
                return "Low Income"
            elif "mod" in s:
                return "Moderate Income"
            elif "high" in s:
                return "High Income"
            return str(val).strip().title()
        df["Income category"] = df["Income category"].apply(standardize_income_cat)
        
    # Gender standardization
    if "Gender" in df.columns:
        def standardize_gender(val):
            if pd.isna(val):
                return "Unknown"
            s = str(val).strip().lower()
            if s in ["f", "female"]:
                return "Female"
            elif s in ["m", "male"]:
                return "Male"
            elif "non" in s or "binary" in s:
                return "Non-Binary"
            return str(val).strip().title()
        df["Gender"] = df["Gender"].apply(standardize_gender)
        
    # Disability status standardization
    if "Disability status" in df.columns:
        def standardize_disability(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip().lower()
            if s in ["y", "yes", "true", "1"]:
                return "Yes"
            elif s in ["n", "no", "false", "0"]:
                return "No"
            return "No"
        df["Disability status"] = df["Disability status"].apply(standardize_disability)
        # Impute missing disability status with mode ('No')
        df["Disability status"] = df["Disability status"].fillna("No")
        
    # Eligibility & Application status
    for col in ["Eligibility status", "Application status", "Housing allocation", "Region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            
    # ---------------------------------------------------------
    # STEP 3: Validate and Clean Numeric Values
    # ---------------------------------------------------------
    # Household size: must be positive integer >= 1
    if "Household size" in df.columns:
        df["Household size"] = pd.to_numeric(df["Household size"], errors="coerce")
        median_hh = int(df["Household size"][df["Household size"] > 0].median() or 3)
        df["Household size"] = df["Household size"].apply(lambda x: median_hh if pd.isna(x) or x <= 0 else int(x))
        
    # Annual income: must be non-negative
    if "Annual income" in df.columns:
        df["Annual income"] = pd.to_numeric(df["Annual income"], errors="coerce")
        # Treat negative as input error; take absolute value or impute
        df["Annual income"] = df["Annual income"].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
        
        # Impute missing annual income using median of income category or overall median
        overall_median_inc = df["Annual income"].median()
        if "Income category" in df.columns:
            income_by_cat = df.groupby("Income category")["Annual income"].transform("median")
            df["Annual income"] = df["Annual income"].fillna(income_by_cat).fillna(overall_median_inc)
        else:
            df["Annual income"] = df["Annual income"].fillna(overall_median_inc)
            
    # Align Income Category if missing or misaligned with annual income
    if "Income category" in df.columns and "Annual income" in df.columns:
        def infer_cat(row):
            if pd.notna(row["Income category"]) and row["Income category"] in ["Very Low Income", "Low Income", "Moderate Income", "High Income"]:
                return row["Income category"]
            inc = row["Annual income"]
            if inc <= 25000:
                return "Very Low Income"
            elif inc <= 45000:
                return "Low Income"
            elif inc <= 70000:
                return "Moderate Income"
            else:
                return "High Income"
        df["Income category"] = df.apply(infer_cat, axis=1)
        
    # Housing need score: bounded between 10 and 100
    if "Housing need score" in df.columns:
        df["Housing need score"] = pd.to_numeric(df["Housing need score"], errors="coerce")
        df["Housing need score"] = df["Housing need score"].fillna(df["Housing need score"].median())
        df["Housing need score"] = df["Housing need score"].clip(lower=10, upper=100).round().astype(int)
        
    # Waiting time: negative values replaced; rejected applicants have 0 waiting time
    if "Waiting time" in df.columns:
        df["Waiting time"] = pd.to_numeric(df["Waiting time"], errors="coerce")
        # Negative waiting time is invalid, replace with absolute or 0
        df["Waiting time"] = df["Waiting time"].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
        
        # If rejected, waiting time is 0
        if "Application status" in df.columns:
            df.loc[df["Application status"] == "Rejected", "Waiting time"] = 0.0
            
            # For approved/pending with missing waiting time, impute using regional median
            reg_wait_median = df[df["Waiting time"] > 0].groupby("Region")["Waiting time"].transform("median")
            df["Waiting time"] = df["Waiting time"].fillna(reg_wait_median).fillna(12.0)
            
        df["Waiting time"] = df["Waiting time"].round(1)

    # Monthly rent: rejected applicants pay 0 subsidized rent
    if "Monthly rent" in df.columns:
        df["Monthly rent"] = pd.to_numeric(df["Monthly rent"], errors="coerce")
        df["Monthly rent"] = df["Monthly rent"].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
        
        if "Application status" in df.columns:
            df.loc[df["Application status"] == "Rejected", "Monthly rent"] = 0.0
            # For approved/pending missing rent, estimate 30% of monthly income
            est_rent = (df["Annual income"] / 12.0) * 0.30 + (df["Household size"] * 35.0)
            df["Monthly rent"] = df["Monthly rent"].fillna(est_rent.round(0)).fillna(450.0)
            
        df["Monthly rent"] = df["Monthly rent"].round(0)

    # ---------------------------------------------------------
    # STEP 4: Logical Constraints & Sanity Corrections
    # ---------------------------------------------------------
    # Ineligible applicants cannot be Approved or Allocated
    if "Eligibility status" in df.columns and "Application status" in df.columns:
        ineligible_mask = df["Eligibility status"] == "Ineligible"
        df.loc[ineligible_mask, "Application status"] = "Rejected"
        if "Housing allocation" in df.columns:
            df.loc[ineligible_mask, "Housing allocation"] = "Not Allocated"
        if "Waiting time" in df.columns:
            df.loc[ineligible_mask, "Waiting time"] = 0.0
        if "Monthly rent" in df.columns:
            df.loc[ineligible_mask, "Monthly rent"] = 0.0
            
    # Rejected applicants cannot have Allocated housing
    if "Application status" in df.columns and "Housing allocation" in df.columns:
        df.loc[df["Application status"] == "Rejected", "Housing allocation"] = "Not Allocated"

    # ---------------------------------------------------------
    # STEP 5: Feature Engineering / Derived Columns
    # ---------------------------------------------------------
    # Per-Capita Income
    df["Per_Capita_Income"] = (df["Annual income"] / df["Household size"]).round(2)
    
    # Monthly Income
    df["Monthly_Income"] = (df["Annual income"] / 12.0).round(2)
    
    # Rent Burden Ratio (Monthly Rent * 12 / Annual Income)
    # Capped at 1.0 for realistic ratios
    df["Rent_Burden_Ratio"] = np.where(
        df["Annual income"] > 0,
        ((df["Monthly rent"] * 12.0) / df["Annual income"]).clip(0, 1.0).round(3),
        0.0
    )
    
    # Need Priority Tier
    def assign_priority_tier(score):
        if score >= 70:
            return "High Need (70+)"
        elif score >= 45:
            return "Medium Need (45-69)"
        else:
            return "Low Need (<45)"
    df["Need_Priority_Tier"] = df["Housing need score"].apply(assign_priority_tier)
    
    # Waiting Time Category
    def assign_waiting_cat(months, status):
        if status == "Rejected" or months == 0:
            return "None / Rejected"
        elif months < 6:
            return "Short (< 6 mos)"
        elif months <= 18:
            return "Moderate (6-18 mos)"
        elif months <= 36:
            return "Long (18-36 mos)"
        else:
            return "Extended (> 36 mos)"
            
    df["Waiting_Time_Category"] = df.apply(
        lambda r: assign_waiting_cat(r["Waiting time"], r["Application status"]), axis=1
    )
    
    # Vulnerability Flag
    # Household has disability OR large family (>=5) OR Very Low Income
    df["Is_Vulnerable"] = (
        (df["Disability status"] == "Yes") | 
        (df["Household size"] >= 5) | 
        (df["Income category"] == "Very Low Income")
    )
    
    # Audit after cleaning
    after_audit = audit_dataset(df)
    
    audit_summary = {
        "before": before_audit,
        "after": after_audit,
        "rows_removed": before_audit["total_rows"] - after_audit["total_rows"],
        "duplicates_removed": before_audit["duplicate_rows"],
        "missing_imputed": sum(before_audit["missing_counts"].values()) - sum(after_audit["missing_counts"].values()),
        "anomalies_corrected": sum(before_audit["numeric_anomalies"].values())
    }
    
    return df, audit_summary

def execute_cleaning_pipeline(input_csv: str = "data/public_housing_data.csv",
                              output_csv: str = "data/cleaned_housing_data.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load raw CSV, run cleaning pipeline, and save to cleaned CSV.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found at {input_csv}")
        
    df_raw = pd.read_csv(input_csv)
    df_cleaned, audit = clean_dataset(df_raw)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_cleaned.to_csv(output_csv, index=False)
    
    return df_cleaned, audit

if __name__ == "__main__":
    cleaned_df, audit = execute_cleaning_pipeline()
    print("=== DATA CLEANING COMPLETED ===")
    print(f"Original Rows: {audit['before']['total_rows']} -> Cleaned Rows: {audit['after']['total_rows']}")
    print(f"Duplicates Removed: {audit['duplicates_removed']}")
    print(f"Missing Values Imputed: {audit['missing_imputed']}")
    print(f"Anomalies Corrected: {audit['anomalies_corrected']}")
    print(f"Saved cleaned dataset to data/cleaned_housing_data.csv")
