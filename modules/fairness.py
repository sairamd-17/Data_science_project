"""
Housing Fairness & Equity Analysis Module.
Evaluates distribution fairness, disparate impact, regional supply-to-demand ratios,
and outcomes for vulnerable demographic cohorts without relying on arbitrary scores.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

DISPARATE_IMPACT_THRESHOLD = 0.80  # Recognized 4/5ths (80%) rule benchmark

def evaluate_group_equity(df: pd.DataFrame, group_col: str = "Income category") -> pd.DataFrame:
    """
    Computes rigorous demographic and socio-economic equity metrics across groups.
    Accounts for eligibility and housing need score to avoid distorted raw comparisons.
    """
    records = []
    
    for group_name, gdf in df.groupby(group_col):
        total_apps = len(gdf)
        eligible_mask = gdf["Eligibility status"] == "Eligible" if "Eligibility status" in gdf.columns else pd.Series([True]*len(gdf))
        eligible_count = int(eligible_mask.sum())
        eligibility_rate = (eligible_count / total_apps * 100) if total_apps > 0 else 0.0
        
        approved_count = int((gdf["Application status"] == "Approved").sum())
        raw_approval_rate = (approved_count / total_apps * 100) if total_apps > 0 else 0.0
        eligible_approval_rate = (approved_count / eligible_count * 100) if eligible_count > 0 else 0.0
        
        allocated_count = int((gdf["Housing allocation"] == "Allocated").sum())
        allocation_rate_of_eligible = (allocated_count / eligible_count * 100) if eligible_count > 0 else 0.0
        
        active_wait = gdf[gdf["Waiting time"] > 0]["Waiting time"]
        median_wait = float(active_wait.median()) if len(active_wait) > 0 else 0.0
        mean_wait = float(active_wait.mean()) if len(active_wait) > 0 else 0.0
        mean_need = float(gdf["Housing need score"].mean()) if "Housing need score" in gdf.columns else 0.0
        
        records.append({
            group_col: group_name,
            "Total_Applicants": total_apps,
            "Eligible_Applicants": eligible_count,
            "Eligibility_Rate_%": round(eligibility_rate, 1),
            "Approved_Count": approved_count,
            "Raw_Approval_Rate_%": round(raw_approval_rate, 1),
            "Eligible_Approval_Rate_%": round(eligible_approval_rate, 1),
            "Allocated_Units": allocated_count,
            "Allocation_Rate_Eligible_%": round(allocation_rate_of_eligible, 1),
            "Mean_Need_Score": round(mean_need, 1),
            "Median_Wait_Months": round(median_wait, 1),
            "Mean_Wait_Months": round(mean_wait, 1)
        })
        
    equity_df = pd.DataFrame(records)
    
    # Calculate Disparate Impact Ratio relative to maximum eligible approval rate
    max_rate = equity_df["Eligible_Approval_Rate_%"].max()
    if max_rate > 0:
        equity_df["Disparity_Ratio"] = (equity_df["Eligible_Approval_Rate_%"] / max_rate).round(3)
        equity_df["Potential_Disparity_Flag"] = equity_df["Disparity_Ratio"].apply(
            lambda r: "Review Indicated (<0.80)" if r < DISPARATE_IMPACT_THRESHOLD else "Within Equity Standard (>=0.80)"
        )
    else:
        equity_df["Disparity_Ratio"] = 1.0
        equity_df["Potential_Disparity_Flag"] = "N/A"
        
    return equity_df

def evaluate_vulnerable_households(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates housing outcomes specifically for vulnerable applicant groups:
    - Individuals with reported disability
    - Extremely large households (5+ members)
    - High housing need tier (Need Score >= 70)
    - Very Low Income category
    """
    cohorts = [
        ("Disability Reported (Yes)", df["Disability status"] == "Yes"),
        ("No Disability Reported (No)", df["Disability status"] == "No"),
        ("Large Household (5+ Members)", df["Household size"] >= 5),
        ("Small/Standard Household (1-4)", df["Household size"] < 5),
        ("High Need Score (70+)", df["Housing need score"] >= 70),
        ("Medium/Low Need (<70)", df["Housing need score"] < 70),
        ("Very Low Income", df["Income category"] == "Very Low Income"),
        ("Moderate/High Income", df["Income category"].isin(["Moderate Income", "High Income"]))
    ]
    
    rows = []
    total_pop = len(df)
    
    for label, mask in cohorts:
        subset = df[mask]
        n_sub = len(subset)
        if n_sub == 0:
            continue
            
        elig_sub = subset[subset["Eligibility status"] == "Eligible"] if "Eligibility status" in subset.columns else subset
        n_elig = len(elig_sub)
        
        approved_n = int((subset["Application status"] == "Approved").sum())
        allocated_n = int((subset["Housing allocation"] == "Allocated").sum())
        
        elig_approval_pct = (approved_n / n_elig * 100) if n_elig > 0 else 0.0
        alloc_pct = (allocated_n / n_elig * 100) if n_elig > 0 else 0.0
        
        active_wait = subset[subset["Waiting time"] > 0]["Waiting time"]
        med_wait = round(float(active_wait.median()), 1) if len(active_wait) > 0 else 0.0
        mean_need = round(float(subset["Housing need score"].mean()), 1)
        
        rows.append({
            "Vulnerable Cohort": label,
            "Cohort Size (N)": n_sub,
            "% of Applicant Pool": round((n_sub / total_pop) * 100, 1),
            "Eligible Count": n_elig,
            "Approval Rate (Eligible %)": round(elig_approval_pct, 1),
            "Allocation Rate (%)": round(alloc_pct, 1),
            "Mean Need Score": mean_need,
            "Median Wait (Months)": med_wait
        })
        
    return pd.DataFrame(rows)

def evaluate_regional_supply_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates allocation relative to regional demand.
    Demand = Count of Eligible Applicants
    Supply = Count of Housing Units Allocated
    Allocation-to-Demand Ratio = Allocated Units / Eligible Applicants
    """
    records = []
    for reg, rdf in df.groupby("Region"):
        eligible_apps = int((rdf["Eligibility status"] == "Eligible").sum()) if "Eligibility status" in rdf.columns else len(rdf)
        allocated_units = int((rdf["Housing allocation"] == "Allocated").sum())
        alloc_to_demand = (allocated_units / eligible_apps) if eligible_apps > 0 else 0.0
        
        active_w = rdf[rdf["Waiting time"] > 0]["Waiting time"]
        med_w = float(active_w.median()) if len(active_w) > 0 else 0.0
        
        records.append({
            "Region": reg,
            "Eligible Demand (Households)": eligible_apps,
            "Units Allocated (Supply)": allocated_units,
            "Allocation_to_Demand_Ratio": round(alloc_to_demand, 3),
            "Allocation_to_Demand_%": round(alloc_to_demand * 100, 1),
            "Median Waiting Months": round(med_w, 1)
        })
        
    res_df = pd.DataFrame(records).sort_values("Allocation_to_Demand_Ratio", ascending=False)
    return res_df

# -------------------------------------------------------------
# PLOTLY CHARTS FOR FAIRNESS
# -------------------------------------------------------------

def create_disparity_ratio_chart(equity_df: pd.DataFrame, group_col: str) -> go.Figure:
    """
    Bar chart showing Disparate Impact Ratio across categories with standard 80% line.
    """
    colors = [
        "#EF4444" if r < DISPARATE_IMPACT_THRESHOLD else "#10B981" 
        for r in equity_df["Disparity_Ratio"]
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=equity_df[group_col],
        y=equity_df["Disparity_Ratio"],
        marker_color=colors,
        text=equity_df["Disparity_Ratio"].apply(lambda v: f"{v:.2f}"),
        textposition="outside",
        name="Disparity Ratio"
    ))
    
    # Add 0.80 benchmark threshold line
    fig.add_hline(
        y=DISPARATE_IMPACT_THRESHOLD,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text="80% Disparate Impact Benchmark (0.80)",
        annotation_position="top left"
    )
    
    fig.update_layout(
        template="plotly_white",
        title=f"Disparity Ratio by {group_col} (Benchmark = 1.00)",
        xaxis_title=group_col,
        yaxis_title="Ratio relative to Benchmark",
        yaxis=dict(range=[0, 1.25]),
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def create_supply_demand_chart(sd_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart displaying regional eligible demand vs unit allocation.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sd_df["Region"],
        y=sd_df["Eligible Demand (Households)"],
        name="Eligible Demand (Applicants)",
        marker_color="#94A3B8"
    ))
    fig.add_trace(go.Bar(
        x=sd_df["Region"],
        y=sd_df["Units Allocated (Supply)"],
        name="Units Allocated (Supply)",
        marker_color="#2563EB"
    ))
    
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Housing Unit Supply vs. Eligible Demand by Region",
        xaxis_title="Region",
        yaxis_title="Number of Households",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig
