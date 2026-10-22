"""
Data Generator Module for Public Housing Fairness Analysis.
Generates a realistic synthetic dataset of public housing applications
with intentional anomalies (duplicates, missing values, inconsistent casing, invalid numeric values)
to demonstrate robust data cleaning and manipulation.

DISCLAIMER: This dataset is entirely synthetic and generated for academic/demonstration
purposes only. It does not represent real government or individual records.
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_housing_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic public housing application dataset.
    
    Parameters:
        n_samples (int): Base number of application records.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Synthetic dataset with intentional real-world data flaws.
    """
    np.random.seed(random_state)
    
    # 1. Application IDs
    app_ids = [f"APP-{10000 + i}" for i in range(1, n_samples + 1)]
    
    # 2. Regions with realistic demographic distribution
    regions_list = ["North", "South", "East", "West", "Central"]
    region_probs = [0.22, 0.28, 0.18, 0.17, 0.15]
    regions = np.random.choice(regions_list, size=n_samples, p=region_probs)
    
    # 3. Household size (1 to 8, skewed towards 1-4)
    household_sizes = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n_samples, 
                                       p=[0.24, 0.26, 0.20, 0.15, 0.08, 0.04, 0.02, 0.01])
    
    # 4. Annual Income: Log-normal distribution reflective of low-to-moderate income applicants
    # Base income around $18,000 to $65,000, with scaling by household size
    base_incomes = np.random.lognormal(mean=10.1, sigma=0.45, size=n_samples)
    annual_incomes = np.round(base_incomes + (household_sizes * 2200), -2)
    # Clamp within realistic limits
    annual_incomes = np.clip(annual_incomes, 9500, 92000)
    
    # 5. Income Category (HUD/Local guideline based)
    # Very Low Income: <= 25,000
    # Low Income: 25,001 - 45,000
    # Moderate Income: 45,001 - 70,000
    # High Income: > 70,000
    def categorize_income(inc):
        if inc <= 25000:
            return "Very Low Income"
        elif inc <= 45000:
            return "Low Income"
        elif inc <= 70000:
            return "Moderate Income"
        else:
            return "High Income"
            
    income_categories = [categorize_income(inc) for inc in annual_incomes]
    
    # 6. Demographics: Gender and Disability Status
    genders = np.random.choice(["Female", "Male", "Non-Binary"], size=n_samples, p=[0.55, 0.41, 0.04])
    disability_status = np.random.choice(["Yes", "No"], size=n_samples, p=[0.24, 0.76])
    
    # 7. Housing Need Score (10 to 100)
    # Driven by low income, larger household size, disability status, plus random variance
    need_scores = []
    for inc, hh_size, dis in zip(annual_incomes, household_sizes, disability_status):
        score = 40.0
        # Income factor (lower income -> higher need)
        score += max(0, (50000 - inc) / 1000) * 0.8
        # Household size factor
        score += (hh_size - 1) * 3.5
        # Disability factor
        if dis == "Yes":
            score += 15.0
        # Random noise
        score += np.random.normal(0, 6)
        clamped_score = int(np.clip(np.round(score), 12, 99))
        need_scores.append(clamped_score)
        
    # 8. Eligibility Status
    # High income (> 70k) or very low need score has high ineligibility
    eligibility_status = []
    for inc, score in zip(annual_incomes, need_scores):
        if inc > 72000:
            eligibility_status.append(np.random.choice(["Ineligible", "Eligible"], p=[0.85, 0.15]))
        elif score < 25:
            eligibility_status.append(np.random.choice(["Ineligible", "Eligible"], p=[0.40, 0.60]))
        else:
            eligibility_status.append("Eligible")
            
    # 9. Application Status & Allocation
    # Eligible applicants with higher need have higher approval rates
    application_status = []
    housing_allocation = []
    waiting_times = []
    monthly_rents = []
    
    for reg, inc, hh, score, elig in zip(regions, annual_incomes, household_sizes, need_scores, eligibility_status):
        if elig == "Ineligible":
            status = "Rejected"
            alloc = "Not Allocated"
            w_time = 0.0
            rent = 0.0
        else:
            # Regional supply pressure variation
            reg_multiplier = {"Central": 0.80, "South": 0.95, "North": 1.05, "East": 1.10, "West": 1.00}[reg]
            p_approve = np.clip((score / 120.0) * reg_multiplier, 0.20, 0.90)
            p_pending = (1.0 - p_approve) * 0.45
            p_reject = 1.0 - p_approve - p_pending
            
            status = np.random.choice(["Approved", "Pending", "Rejected"], p=[p_approve, p_pending, p_reject])
            
            if status == "Approved":
                # Allocation: 75% of approved are allocated, 25% awaiting unit assignment
                alloc = np.random.choice(["Allocated", "Not Allocated"], p=[0.75, 0.25])
                # Waiting time: Central and South have longer waiting times
                base_w = {"Central": 26, "South": 20, "North": 14, "East": 12, "West": 16}[reg]
                w_time = float(np.round(np.random.gamma(shape=3.5, scale=base_w / 3.5), 1))
                w_time = np.clip(w_time, 2.0, 54.0)
                # Subsidized Monthly Rent: ~28-32% of monthly income, adjusted for bedroom size
                monthly_inc = inc / 12.0
                rent = float(np.round(np.clip(monthly_inc * 0.30 + (hh * 35), 250, 1650), 0))
            elif status == "Pending":
                alloc = "Not Allocated"
                base_w = {"Central": 16, "South": 12, "North": 8, "East": 7, "West": 10}[reg]
                w_time = float(np.round(np.random.gamma(shape=2.5, scale=base_w / 2.5), 1))
                w_time = np.clip(w_time, 1.0, 30.0)
                monthly_inc = inc / 12.0
                rent = float(np.round(np.clip(monthly_inc * 0.30 + (hh * 35), 250, 1650), 0))
            else:
                alloc = "Not Allocated"
                w_time = 0.0
                rent = 0.0
                
        application_status.append(status)
        housing_allocation.append(alloc)
        waiting_times.append(w_time)
        monthly_rents.append(rent)
        
    # Assemble Base DataFrame
    df = pd.DataFrame({
        "Application ID": app_ids,
        "Region": regions,
        "Household size": household_sizes,
        "Annual income": annual_incomes,
        "Income category": income_categories,
        "Gender": genders,
        "Disability status": disability_status,
        "Housing need score": need_scores,
        "Eligibility status": eligibility_status,
        "Application status": application_status,
        "Housing allocation": housing_allocation,
        "Waiting time": waiting_times,
        "Monthly rent": monthly_rents
    })
    
    # -------------------------------------------------------------
    # INTENTIONAL DATA QUALITY FLAWS FOR CLEANING DEMONSTRATION
    # -------------------------------------------------------------
    
    # 1. Inconsistent Category Casing & Formatting
    dirty_income_cats = df["Income category"].copy()
    inconsistent_indices = np.random.choice(df.index, size=65, replace=False)
    for idx in inconsistent_indices:
        cat = dirty_income_cats.loc[idx]
        if cat == "Very Low Income":
            dirty_income_cats.loc[idx] = np.random.choice(["very low income", "Very-Low Income", "VERY LOW INCOME", "Very Low Income "])
        elif cat == "Low Income":
            dirty_income_cats.loc[idx] = np.random.choice(["low income", "LOW INCOME", "Low-Income", "Low Income  "])
        elif cat == "Moderate Income":
            dirty_income_cats.loc[idx] = np.random.choice(["moderate income", "MODERATE INCOME", "Moderate Income "])
        else:
            dirty_income_cats.loc[idx] = np.random.choice(["high income", "HIGH INCOME"])
    df["Income category"] = dirty_income_cats
    
    # Inconsistent Gender & Disability formatting
    gender_dirty_indices = np.random.choice(df.index, size=35, replace=False)
    for idx in gender_dirty_indices:
        g = df.loc[idx, "Gender"]
        if g == "Female":
            df.loc[idx, "Gender"] = np.random.choice(["female", "FEMALE", "F", "Female "])
        elif g == "Male":
            df.loc[idx, "Gender"] = np.random.choice(["male", "MALE", "M", " Male"])
            
    dis_dirty_indices = np.random.choice(df.index, size=25, replace=False)
    for idx in dis_dirty_indices:
        d = df.loc[idx, "Disability status"]
        df.loc[idx, "Disability status"] = "Y" if d == "Yes" else "N"
        
    # 2. Inject Missing Values (NaNs)
    # Monthly rent missing (~4%)
    rent_nan_idx = np.random.choice(df.index, size=48, replace=False)
    df.loc[rent_nan_idx, "Monthly rent"] = np.nan
    
    # Waiting time missing (~3%)
    wait_nan_idx = np.random.choice(df.index, size=36, replace=False)
    df.loc[wait_nan_idx, "Waiting time"] = np.nan
    
    # Disability status missing (~2%)
    dis_nan_idx = np.random.choice(df.index, size=28, replace=False)
    df.loc[dis_nan_idx, "Disability status"] = np.nan
    
    # Annual income missing (~1.5%)
    inc_nan_idx = np.random.choice(df.index, size=22, replace=False)
    df.loc[inc_nan_idx, "Annual income"] = np.nan
    
    # 3. Invalid / Boundary Anomaly Entries
    # Negative income or waiting time
    anomaly_indices = np.random.choice(df.index, size=8, replace=False)
    df.loc[anomaly_indices[0], "Annual income"] = -4500.0
    df.loc[anomaly_indices[1], "Household size"] = 0
    df.loc[anomaly_indices[2], "Waiting time"] = -12.5
    df.loc[anomaly_indices[3], "Housing need score"] = 145  # out of bound > 100
    df.loc[anomaly_indices[4], "Monthly rent"] = -320.0
    
    # Ineligible applicant marked as Approved (logical inconsistency)
    inelig_approved_idx = df[(df["Eligibility status"] == "Ineligible") & (df["Application status"] == "Rejected")].index[:3]
    for idx in inelig_approved_idx:
        df.loc[idx, "Application status"] = "Approved"
        df.loc[idx, "Housing allocation"] = "Allocated"
        
    # 4. Duplicate Records (exact duplicates of existing rows)
    dup_sample = df.sample(n=35, random_state=123).copy()
    df_with_duplicates = pd.concat([df, dup_sample], ignore_index=True)
    
    # Shuffle slightly
    df_final = df_with_duplicates.sample(frac=1.0, random_state=99).reset_index(drop=True)
    
    return df_final

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "public_housing_data.csv")
    df = generate_synthetic_housing_data(n_samples=1500)
    df.to_csv(out_path, index=False)
    print(f"Generated synthetic housing dataset with {len(df)} rows.")
    print(f"Saved to: {out_path}")
