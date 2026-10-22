"""
Analysis & Exploratory Data Analysis (EDA) Module.
Contains Pandas aggregation routines, KPI calculations, dynamic filtering,
and interactive Plotly visualizations for the Public Housing fairness evaluation.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple

# Modern aesthetic palette
CHART_THEME = "plotly_white"
COLOR_PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899"]
STATUS_COLOR_MAP = {
    "Approved": "#10B981",
    "Pending": "#F59E0B",
    "Rejected": "#EF4444"
}
ALLOC_COLOR_MAP = {
    "Allocated": "#2563EB",
    "Not Allocated": "#94A3B8"
}

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute core headline key performance indicators.
    """
    total_apps = len(df)
    if total_apps == 0:
        return {}
        
    status_counts = df["Application status"].value_counts().to_dict()
    approved_count = status_counts.get("Approved", 0)
    rejected_count = status_counts.get("Rejected", 0)
    pending_count = status_counts.get("Pending", 0)
    
    total_allocated = int((df["Housing allocation"] == "Allocated").sum()) if "Housing allocation" in df.columns else 0
    overall_approval_rate = round((approved_count / total_apps) * 100, 2)
    
    # Approval rate among eligible applicants
    eligible_mask = df["Eligibility status"] == "Eligible" if "Eligibility status" in df.columns else pd.Series([True]*len(df))
    eligible_count = int(eligible_mask.sum())
    eligible_approval_rate = round((approved_count / eligible_count * 100), 2) if eligible_count > 0 else 0.0
    
    avg_income = round(df["Annual income"].mean(), 2) if "Annual income" in df.columns else 0.0
    median_income = round(df["Annual income"].median(), 2) if "Annual income" in df.columns else 0.0
    
    # Waiting time among active applications (waiting > 0)
    active_wait = df[df["Waiting time"] > 0]["Waiting time"] if "Waiting time" in df.columns else pd.Series([])
    avg_waiting_time = round(active_wait.mean(), 1) if len(active_wait) > 0 else 0.0
    median_waiting_time = round(active_wait.median(), 1) if len(active_wait) > 0 else 0.0
    
    avg_need_score = round(df["Housing need score"].mean(), 1) if "Housing need score" in df.columns else 0.0
    
    return {
        "total_applications": total_apps,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "pending_count": pending_count,
        "total_allocated": total_allocated,
        "overall_approval_rate": overall_approval_rate,
        "eligible_applications": eligible_count,
        "eligible_approval_rate": eligible_approval_rate,
        "avg_income": avg_income,
        "median_income": median_income,
        "avg_waiting_time": avg_waiting_time,
        "median_waiting_time": median_waiting_time,
        "avg_need_score": avg_need_score
    }

def get_regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform GroupBy aggregation on Region to evaluate operational volume and outcomes.
    """
    summary = df.groupby("Region").agg(
        Total_Applications=("Application ID", "count"),
        Approved_Count=("Application status", lambda x: (x == "Approved").sum()),
        Rejected_Count=("Application status", lambda x: (x == "Rejected").sum()),
        Pending_Count=("Application status", lambda x: (x == "Pending").sum()),
        Allocated_Count=("Housing allocation", lambda x: (x == "Allocated").sum()),
        Mean_Income=("Annual income", "mean"),
        Median_Income=("Annual income", "median"),
        Mean_Need_Score=("Housing need score", "mean"),
        Mean_Waiting_Months=("Waiting time", lambda x: x[x > 0].mean()),
        Median_Waiting_Months=("Waiting time", lambda x: x[x > 0].median())
    ).reset_index()
    
    summary["Approval_Rate_%"] = ((summary["Approved_Count"] / summary["Total_Applications"]) * 100).round(2)
    summary["Allocation_Rate_%"] = ((summary["Allocated_Count"] / summary["Total_Applications"]) * 100).round(2)
    summary["Mean_Income"] = summary["Mean_Income"].round(0)
    summary["Median_Income"] = summary["Median_Income"].round(0)
    summary["Mean_Need_Score"] = summary["Mean_Need_Score"].round(1)
    summary["Mean_Waiting_Months"] = summary["Mean_Waiting_Months"].round(1)
    summary["Median_Waiting_Months"] = summary["Median_Waiting_Months"].round(1)
    
    return summary

def get_income_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform GroupBy aggregation on Income Category.
    """
    cat_order = ["Very Low Income", "Low Income", "Moderate Income", "High Income"]
    
    summary = df.groupby("Income category").agg(
        Total_Applications=("Application ID", "count"),
        Eligible_Count=("Eligibility status", lambda x: (x == "Eligible").sum()),
        Approved_Count=("Application status", lambda x: (x == "Approved").sum()),
        Allocated_Count=("Housing allocation", lambda x: (x == "Allocated").sum()),
        Mean_Need_Score=("Housing need score", "mean"),
        Mean_Waiting_Months=("Waiting time", lambda x: x[x > 0].mean()),
        Median_Waiting_Months=("Waiting time", lambda x: x[x > 0].median()),
        Mean_Household_Size=("Household size", "mean")
    ).reset_index()
    
    # Sort by standardized category order
    summary["order"] = summary["Income category"].apply(lambda c: cat_order.index(c) if c in cat_order else 99)
    summary = summary.sort_values("order").drop(columns=["order"])
    
    summary["Overall_Approval_Rate_%"] = ((summary["Approved_Count"] / summary["Total_Applications"]) * 100).round(2)
    summary["Eligible_Approval_Rate_%"] = np.where(
        summary["Eligible_Count"] > 0,
        ((summary["Approved_Count"] / summary["Eligible_Count"]) * 100).round(2),
        0.0
    )
    summary["Allocation_Rate_%"] = ((summary["Allocated_Count"] / summary["Total_Applications"]) * 100).round(2)
    summary["Mean_Need_Score"] = summary["Mean_Need_Score"].round(1)
    summary["Mean_Waiting_Months"] = summary["Mean_Waiting_Months"].round(1)
    summary["Median_Waiting_Months"] = summary["Median_Waiting_Months"].round(1)
    summary["Mean_Household_Size"] = summary["Mean_Household_Size"].round(1)
    
    return summary

# -------------------------------------------------------------
# PLOTLY INTERACTIVE VISUALIZATIONS
# -------------------------------------------------------------

def create_status_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of Application Status distribution.
    """
    counts = df["Application status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    
    fig = px.pie(
        counts,
        names="Status",
        values="Count",
        hole=0.45,
        color="Status",
        color_discrete_map=STATUS_COLOR_MAP,
        title="Application Status Distribution"
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#ffffff", width=2))
    )
    fig.update_layout(
        template=CHART_THEME,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def create_regional_applications_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing Application counts per Region, segmented by status.
    """
    reg_df = df.groupby(["Region", "Application status"]).size().reset_index(name="Count")
    
    fig = px.bar(
        reg_df,
        x="Region",
        y="Count",
        color="Application status",
        color_discrete_map=STATUS_COLOR_MAP,
        barmode="stack",
        title="Applications & Outcomes by Region",
        text="Count"
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        template=CHART_THEME,
        xaxis_title="Geographic Region",
        yaxis_title="Total Applications",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_approval_rate_by_income_chart(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing overall vs eligible approval rates across income categories.
    """
    inc_sum = get_income_category_summary(df)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=inc_sum["Income category"],
        y=inc_sum["Overall_Approval_Rate_%"],
        name="Overall Approval Rate (%)",
        marker_color="#3B82F6",
        text=inc_sum["Overall_Approval_Rate_%"].apply(lambda v: f"{v}%"),
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        x=inc_sum["Income category"],
        y=inc_sum["Eligible_Approval_Rate_%"],
        name="Eligible Approval Rate (%)",
        marker_color="#10B981",
        text=inc_sum["Eligible_Approval_Rate_%"].apply(lambda v: f"{v}%"),
        textposition="outside"
    ))
    
    fig.update_layout(
        template=CHART_THEME,
        barmode="group",
        title="Approval Rates by Income Category (Raw vs Eligible Base)",
        xaxis_title="Income Category",
        yaxis_title="Approval Rate (%)",
        yaxis=dict(range=[0, 105]),
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_allocation_by_region_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing housing units allocated vs unallocated by Region.
    """
    alloc_df = df.groupby(["Region", "Housing allocation"]).size().reset_index(name="Count")
    
    fig = px.bar(
        alloc_df,
        x="Region",
        y="Count",
        color="Housing allocation",
        color_discrete_map=ALLOC_COLOR_MAP,
        barmode="group",
        title="Housing Unit Allocation by Region",
        text="Count"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template=CHART_THEME,
        xaxis_title="Region",
        yaxis_title="Households",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_waiting_time_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Box plot showing Waiting Time distribution (in months) across Regions for active applications.
    """
    wait_df = df[df["Waiting time"] > 0]
    
    fig = px.box(
        wait_df,
        x="Region",
        y="Waiting time",
        color="Region",
        color_discrete_sequence=COLOR_PALETTE,
        points="outliers",
        title="Waiting Time Distribution by Region (Months)"
    )
    fig.update_layout(
        template=CHART_THEME,
        xaxis_title="Region",
        yaxis_title="Waiting Time (Months)",
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig

def create_income_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Histogram and KDE curve of Annual Income across application statuses.
    """
    fig = px.histogram(
        df,
        x="Annual income",
        color="Application status",
        color_discrete_map=STATUS_COLOR_MAP,
        nbins=40,
        marginal="box",
        barmode="overlay",
        opacity=0.65,
        title="Annual Income Distribution by Application Status"
    )
    fig.update_layout(
        template=CHART_THEME,
        xaxis_title="Annual Income ($)",
        yaxis_title="Frequency",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def create_need_score_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of Housing Need Score segmented by Application Status.
    """
    fig = px.histogram(
        df,
        x="Housing need score",
        color="Application status",
        color_discrete_map=STATUS_COLOR_MAP,
        nbins=30,
        barmode="overlay",
        marginal="violin",
        opacity=0.65,
        title="Housing Need Score Distribution by Outcome"
    )
    fig.update_layout(
        template=CHART_THEME,
        xaxis_title="Housing Need Score (10 - 100)",
        yaxis_title="Count of Applicants",
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig
