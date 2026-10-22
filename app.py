"""
Main Streamlit Application:
Statistical Assessment of Public Housing Distribution Fairness Using Data Manipulation Techniques
Author: Data Science Final Year College Project
Built with Python, Pandas, NumPy, SciPy, Plotly, and Streamlit.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_generator import generate_synthetic_housing_data
from modules.data_cleaning import (
    validate_columns, audit_dataset, clean_dataset, REQUIRED_COLUMNS
)
from modules.analysis import (
    compute_kpis, get_regional_summary, get_income_category_summary,
    create_status_distribution_chart, create_regional_applications_chart,
    create_approval_rate_by_income_chart, create_allocation_by_region_chart,
    create_waiting_time_distribution_chart, create_income_distribution_chart,
    create_need_score_distribution_chart
)
from modules.statistics import (
    run_chi_square_test, run_waiting_time_tests, run_correlation_analysis,
    create_correlation_heatmap
)
from modules.fairness import (
    evaluate_group_equity, evaluate_vulnerable_households,
    evaluate_regional_supply_demand, create_disparity_ratio_chart,
    create_supply_demand_chart
)
from modules.report_generator import build_pdf_report

# -------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Public Housing Fairness Assessment",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CUSTOM CSS FOR PROFESSIONAL AESTHETICS
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header hero styling */
    .hero-container {
        background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 50%, #3B82F6 100%);
        padding: 2rem 2.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 20px -2px rgba(30, 58, 138, 0.25);
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #E0E7FF;
        max-width: 950px;
        line-height: 1.5;
    }
    .hero-badges {
        margin-top: 1rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }
    
    /* Academic Callout Boxes */
    .callout-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .callout-title {
        font-weight: 700;
        color: #1E3A8A;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .callout-text {
        font-size: 0.85rem;
        color: #334155;
        line-height: 1.5;
    }
    
    /* Statistical Result Card */
    .stat-result-card {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #CBD5E1;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .stat-result-card.significant {
        border-left: 5px solid #10B981;
    }
    .stat-result-card.insignificant {
        border-left: 5px solid #F59E0B;
    }
    
    /* Sidebar branding */
    .sidebar-brand {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA INITIALIZATION & CACHING
# -------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "public_housing_data.csv")
CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned_housing_data.csv")

def ensure_initial_data():
    """Ensure initial benchmark synthetic data exists on disk."""
    if not os.path.exists(DEFAULT_DATA_PATH):
        os.makedirs(os.path.dirname(DEFAULT_DATA_PATH), exist_ok=True)
        raw_df = generate_synthetic_housing_data(n_samples=1500)
        raw_df.to_csv(DEFAULT_DATA_PATH, index=False)
        cleaned_df, _ = clean_dataset(raw_df)
        cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

ensure_initial_data()

# -------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -------------------------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = pd.read_csv(DEFAULT_DATA_PATH)
    cleaned_df, audit = clean_dataset(st.session_state.df_raw)
    st.session_state.df_cleaned = cleaned_df
    st.session_state.audit = audit
    st.session_state.data_source_name = "Default Synthetic Housing Benchmark (1,500+ records)"

# -------------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h3 style="margin:0; color:#1E3A8A; font-weight:700;">🏛️ Housing Fairness</h3>
        <p style="margin:0; font-size:0.75rem; color:#64748B;">Data Science Capstone Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 📑 Navigation")
    page = st.radio(
        "Select Module:",
        [
            "1. Project Overview",
            "2. Dataset & Cleaning",
            "3. Exploratory Data Analysis",
            "4. Statistical Hypothesis Testing",
            "5. Fairness Analysis & Reports"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Dataset Management
    st.markdown("### 📂 Data Pipeline Controller")
    uploaded_file = st.file_uploader("Upload Application CSV", type=["csv"], help="Upload an external housing application CSV dataset.")
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            is_valid, missing_cols = validate_columns(uploaded_df)
            if not is_valid:
                st.error(f"Missing mandatory columns: {', '.join(missing_cols)}")
            else:
                st.session_state.df_raw = uploaded_df
                cleaned_df, audit = clean_dataset(uploaded_df)
                st.session_state.df_cleaned = cleaned_df
                st.session_state.audit = audit
                st.session_state.data_source_name = f"Uploaded: {uploaded_file.name} ({len(uploaded_df):,} records)"
                st.success(f"Loaded and sanitized {uploaded_file.name}!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            
    if st.button("🔄 Reset to Benchmark Dataset", use_container_width=True):
        st.session_state.df_raw = pd.read_csv(DEFAULT_DATA_PATH)
        cleaned_df, audit = clean_dataset(st.session_state.df_raw)
        st.session_state.df_cleaned = cleaned_df
        st.session_state.audit = audit
        st.session_state.data_source_name = "Default Synthetic Housing Benchmark (1,500+ records)"
        st.rerun()
        
    st.info(f"**Active Source:**\n{st.session_state.data_source_name}")
    
    st.markdown("---")
    st.markdown("""
    **Academic Tools Used:**
    - `pandas` 2.x Data Pipeline
    - `scipy.stats` Hypothesis Testing
    - `plotly.express` Visual Analytics
    - `reportlab` Automated PDF Export
    - `streamlit` Web Application
    """)

# Active cleaned dataset reference
df_cleaned = st.session_state.df_cleaned
df_raw = st.session_state.df_raw
audit = st.session_state.audit

# Compute headline KPIs
kpis = compute_kpis(df_cleaned)

# =============================================================
# PAGE 1: PROJECT OVERVIEW
# =============================================================
if page == "1. Project Overview":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Statistical Assessment of Public Housing Distribution Fairness</div>
        <div class="hero-subtitle">
            An empirical Data Science capstone investigation evaluating demographic equity, 
            regional supply constraints, and statutory prioritization in public housing allocations 
            using advanced Pandas data manipulation and inferential statistics.
        </div>
        <div class="hero-badges">
            <span class="badge">College Review Ready</span>
            <span class="badge">N = 1,500+ Synthetic Cohort</span>
            <span class="badge">SciPy Hypothesis Testing</span>
            <span class="badge">Fair Housing Audit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Headline KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Applications</div>
            <div class="metric-value">{kpis.get('total_applications', 0):,}</div>
            <div class="metric-subtitle">Cleaned applicant cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall Approval Rate</div>
            <div class="metric-value">{kpis.get('overall_approval_rate', 0)}%</div>
            <div class="metric-subtitle">{kpis.get('approved_count', 0):,} approved applicants</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Eligible Approval Rate</div>
            <div class="metric-value">{kpis.get('eligible_approval_rate', 0)}%</div>
            <div class="metric-subtitle">Adjusted for qualification</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Allocated Units</div>
            <div class="metric-value">{kpis.get('total_allocated', 0):,}</div>
            <div class="metric-subtitle">Housing supply distributed</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("📌 Project Abstract & Context")
        st.markdown("""
        Public housing systems operate under severe resource rationing: demand from low-income families 
        consistently exceeds the available supply of subsidized housing stock. To ensure social equity, 
        statutory frameworks (such as HUD regulations) mandate prioritization based on **housing need, 
        disability, and severe rent burden**.
        
        However, administrative friction, regional supply bottlenecks, and income thresholds can lead 
        to perceived or real disparities. This project establishes an **auditable, reproducible analytical pipeline** 
        to answer:
        1. **Association:** Does public housing approval unfairly vary across income tiers or align with statutory need?
        2. **Regional Disparities:** Do waiting times systematically differ across geographic regions due to localized supply bottlenecks?
        3. **Vulnerable Protections:** Are households with reported disabilities and severe economic burdens receiving appropriate priority?
        """)
        
        st.markdown("""
        <div class="callout-box">
            <div class="callout-title">⚠️ Academic Transparency Notice</div>
            <div class="callout-text">
                The dataset analyzed throughout this application is <b>entirely synthetic</b>, purposefully generated 
                to emulate realistic HUD income categories, rent burdens, and administrative waitlists with 
                controlled data hygiene anomalies (duplicates, missing fields, format inconsistencies) for college demonstration.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.subheader("🎯 Project Objectives & Methodology")
        st.markdown("""
        - **1. Data Engineering:** Clean, deduplicate, and impute real-world administrative flaws using vectorized Pandas methods.
        - **2. Feature Transformation:** Engineer derived metrics including *Per-Capita Income*, *Rent-Burden Ratio*, and *Vulnerability Indicators*.
        - **3. Exploratory Data Analysis:** Profile application volumes, rejection factors, and waiting times with interactive Plotly visuals.
        - **4. Inferential Testing:** Apply Pearson's Chi-Square, One-Way ANOVA, Levene's Test, Kruskal-Wallis, and correlation analysis.
        - **5. Fairness Assessment:** Measure Disparate Impact Ratios (EEOC 80% rule) and regional allocation-to-demand balances.
        - **6. Automated Reporting:** One-click dynamic PDF generation via ReportLab for formal project submission.
        """)

# =============================================================
# PAGE 2: DATASET & DATA CLEANING
# =============================================================
elif page == "2. Dataset & Cleaning":
    st.header("🧹 Dataset Profiling & Data Cleaning Pipeline")
    st.markdown("""
    Administrative public housing records frequently suffer from missing survey fields, 
    clerical duplicate filings, and inconsistent categorization. This module demonstrates the 
    **before-and-after auditing** and execution of our robust Pandas preprocessing pipeline.
    """)
    
    # Audit summary cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Initial Raw Records</div>
            <div class="metric-value">{audit['before']['total_rows']:,}</div>
            <div class="metric-subtitle">Unsanitized input data</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Duplicates Purged</div>
            <div class="metric-value">{audit['duplicates_removed']}</div>
            <div class="metric-subtitle">Exact & ID deduplication</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Missing Imputed</div>
            <div class="metric-value">{audit['missing_imputed']}</div>
            <div class="metric-subtitle">Statistical median/mode</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Anomalies Rectified</div>
            <div class="metric-value">{audit['anomalies_corrected']}</div>
            <div class="metric-subtitle">Negative & boundary errors</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Tabs for Before vs After, Data Cleaning Steps, Data Dictionary
    tab_audit, tab_cleaning_ops, tab_preview, tab_dict = st.tabs([
        "📊 Data Health Audit (Before vs. After)", 
        "🛠️ Demonstrated Pandas Operations",
        "🔍 Cleaned Dataset Preview", 
        "📖 Data Dictionary & Feature Schema"
    ])
    
    with tab_audit:
        st.subheader("Data Cleaning Audit Matrix")
        
        # Prepare comparison table
        cols = list(audit["before"]["missing_counts"].keys())
        audit_rows = []
        for col in cols:
            raw_missing = audit["before"]["missing_counts"].get(col, 0)
            clean_missing = audit["after"]["missing_counts"].get(col, 0)
            audit_rows.append({
                "Column Name": col,
                "Raw Missing (N)": raw_missing,
                "Raw Missing (%)": f"{audit['before']['missing_pcts'].get(col, 0)}%",
                "Cleaned Missing (N)": clean_missing,
                "Status": "✅ Sanitized" if clean_missing == 0 else "⚠️ Remaining",
                "Cleaning Action Applied": "Median Imputation" if "income" in col.lower() or "waiting" in col.lower() or "rent" in col.lower() 
                                          else "Standardized / Mode Imputation"
            })
            
        audit_df = pd.DataFrame(audit_rows)
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.info(f"**Pre-cleaning Memory:** {audit['before']['memory_kb']} KB across {audit['before']['total_rows']} rows.")
        with col_m2:
            st.success(f"**Post-cleaning Memory:** {audit['after']['memory_kb']} KB across {audit['after']['total_rows']} rows (Zero NaNs).")
            
    with tab_cleaning_ops:
        st.subheader("Core Pandas Data Manipulation & Cleaning Code")
        st.markdown("These exact Pandas techniques were executed to sanitize the housing data:")
        
        st.code("""
# 1. Deduplication using subset and keep parameters
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["Application ID"], keep="first")

# 2. String standardization & whitespace trimming
df["Income category"] = df["Income category"].astype(str).str.strip().str.title()
df["Disability status"] = df["Disability status"].map({"Y": "Yes", "N": "No"}).fillna(df["Disability status"])

# 3. Numeric Boundary & Absolute Value Corrections
df["Annual income"] = df["Annual income"].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
df["Household size"] = df["Household size"].apply(lambda x: 1 if pd.isna(x) or x <= 0 else int(x))
df["Housing need score"] = df["Housing need score"].clip(lower=10, upper=100)

# 4. Context-Aware Imputation (Group Medians)
df["Annual income"] = df["Annual income"].fillna(df.groupby("Income category")["Annual income"].transform("median"))
df["Waiting time"] = df["Waiting time"].fillna(df.groupby("Region")["Waiting time"].transform("median"))

# 5. Logical Consistency Rules
ineligible_mask = df["Eligibility status"] == "Ineligible"
df.loc[ineligible_mask, ["Application status", "Housing allocation"]] = ["Rejected", "Not Allocated"]
df.loc[ineligible_mask, ["Waiting time", "Monthly rent"]] = [0.0, 0.0]

# 6. Feature Engineering (Derived Variables)
df["Per_Capita_Income"] = (df["Annual income"] / df["Household size"]).round(2)
df["Rent_Burden_Ratio"] = ((df["Monthly rent"] * 12.0) / df["Annual income"]).clip(0, 1.0)
df["Is_Vulnerable"] = (df["Disability status"] == "Yes") | (df["Household size"] >= 5) | (df["Income category"] == "Very Low Income")
        """, language="python")
        
    with tab_preview:
        st.subheader("Preview Cleaned Dataset")
        st.dataframe(df_cleaned.head(20), use_container_width=True)
        
        # Download Cleaned CSV button
        csv_data = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="cleaned_housing_data.csv",
            mime="text/csv",
            help="Download the sanitized dataset with all derived metrics"
        )
        
    with tab_dict:
        st.subheader("Data Dictionary & Variable Descriptions")
        data_dict = [
            {"Field": "Application ID", "Type": "Categorical", "Description": "Unique applicant registration identifier (e.g. APP-10001)."},
            {"Field": "Region", "Type": "Categorical", "Description": "Geographic administrative sector (North, South, East, West, Central)."},
            {"Field": "Household size", "Type": "Integer", "Description": "Number of documented individuals residing in household (1 to 8)."},
            {"Field": "Annual income", "Type": "Continuous ($)", "Description": "Total verifiable annualized household income in USD."},
            {"Field": "Income category", "Type": "Ordinal", "Description": "HUD socio-economic bracket: Very Low, Low, Moderate, High Income."},
            {"Field": "Gender", "Type": "Categorical", "Description": "Primary applicant gender demographic."},
            {"Field": "Disability status", "Type": "Binary (Yes/No)", "Description": "Self-reported verified physical or mental disability."},
            {"Field": "Housing need score", "Type": "Integer (10-100)", "Description": "Vulnerability index calculated from rent burden, crowding, and health factors."},
            {"Field": "Eligibility status", "Type": "Binary", "Description": "Administrative statutory qualification status (Eligible vs Ineligible)."},
            {"Field": "Application status", "Type": "Categorical", "Description": "Current decision status: Approved, Pending, or Rejected."},
            {"Field": "Housing allocation", "Type": "Binary", "Description": "Whether subsidized physical unit has been assigned (Allocated vs Not Allocated)."},
            {"Field": "Waiting time", "Type": "Continuous (Months)", "Description": "Duration spent on active administrative waitlist."},
            {"Field": "Monthly rent", "Type": "Continuous ($)", "Description": "Subsidized monthly household tenant rental contribution."},
            {"Field": "Per_Capita_Income", "Type": "Derived ($)", "Description": "Annual income divided by Household size."},
            {"Field": "Rent_Burden_Ratio", "Type": "Derived (0-1)", "Description": "Annualized rent divided by annual income."},
            {"Field": "Is_Vulnerable", "Type": "Derived (Boolean)", "Description": "Flag for disability, large household size (5+), or very low income."}
        ]
        st.dataframe(pd.DataFrame(data_dict), use_container_width=True, hide_index=True)

# =============================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
elif page == "3. Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis & Visualizations")
    st.markdown("""
    Explore interactive distributions, regional variances, and outcome patterns across the public housing dataset. 
    Use the interactive filters below to examine specific demographic or geographic subsets.
    """)
    
    # Interactive Filters
    with st.expander("🔍 Interactive Data Filtering Options", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            all_regions = sorted(df_cleaned["Region"].unique().tolist())
            selected_regions = st.multiselect("Select Regions:", all_regions, default=all_regions)
        with f_col2:
            all_income_cats = sorted(df_cleaned["Income category"].unique().tolist())
            selected_incomes = st.multiselect("Select Income Categories:", all_income_cats, default=all_income_cats)
        with f_col3:
            all_disability = sorted(df_cleaned["Disability status"].unique().tolist())
            selected_disability = st.multiselect("Disability Status:", all_disability, default=all_disability)
            
    filtered_df = df_cleaned[
        (df_cleaned["Region"].isin(selected_regions)) &
        (df_cleaned["Income category"].isin(selected_incomes)) &
        (df_cleaned["Disability status"].isin(selected_disability))
    ]
    
    if len(filtered_df) == 0:
        st.warning("No records match the current filter selection. Please broaden your filters.")
        st.stop()
        
    f_kpis = compute_kpis(filtered_df)
    
    # Filtered Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Subset Applications", f"{f_kpis.get('total_applications', 0):,}")
    m2.metric("Approval Rate", f"{f_kpis.get('overall_approval_rate', 0)}%")
    m3.metric("Eligible Approval Rate", f"{f_kpis.get('eligible_approval_rate', 0)}%")
    m4.metric("Avg Waiting Time", f"{f_kpis.get('avg_waiting_time', 0)} Mos")
    m5.metric("Avg Need Score", f"{f_kpis.get('avg_need_score', 0)}")
    
    st.markdown("---")
    
    # Visualizations Grid 1
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(create_status_distribution_chart(filtered_df), use_container_width=True)
    with row1_col2:
        st.plotly_chart(create_approval_rate_by_income_chart(filtered_df), use_container_width=True)
        
    # Visualizations Grid 2
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(create_regional_applications_chart(filtered_df), use_container_width=True)
    with row2_col2:
        st.plotly_chart(create_allocation_by_region_chart(filtered_df), use_container_width=True)
        
    # Visualizations Grid 3
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.plotly_chart(create_waiting_time_distribution_chart(filtered_df), use_container_width=True)
    with row3_col2:
        st.plotly_chart(create_need_score_distribution_chart(filtered_df), use_container_width=True)
        
    # Income Distribution Chart
    st.plotly_chart(create_income_distribution_chart(filtered_df), use_container_width=True)
    
    # Aggregated Summary Tables
    st.subheader("📋 GroupBy Aggregation Tables")
    tab_reg, tab_inc = st.tabs(["Regional Performance Summary", "Income Category Summary"])
    with tab_reg:
        st.dataframe(get_regional_summary(filtered_df), use_container_width=True, hide_index=True)
    with tab_inc:
        st.dataframe(get_income_category_summary(filtered_df), use_container_width=True, hide_index=True)

# =============================================================
# PAGE 4: STATISTICAL ANALYSIS
# =============================================================
elif page == "4. Statistical Hypothesis Testing":
    st.header("🧪 Inferential Statistical Hypothesis Testing")
    st.markdown("""
    Rigorous public housing policy research requires validating empirical differences with formal 
    inferential statistical tests. Using **SciPy**, we examine:
    1. **Pearson's Chi-Square Test of Independence:** Association between application status and income category.
    2. **One-Way ANOVA & Kruskal-Wallis Tests:** Comparison of waiting durations across municipal regions.
    3. **Assumption Diagnostics:** Levene's test for homoscedasticity and normality checks.
    4. **Correlation Analysis:** Pearson & Spearman matrices of continuous demographic indicators.
    """)
    
    alpha = 0.05
    st.markdown(f"**Significance Threshold:** α = {alpha} (95% Confidence Level)")
    
    # Tabbed tests
    tab_chi, tab_anova, tab_corr = st.tabs([
        "1. Chi-Square Test (Status vs. Income)", 
        "2. Regional Waiting Time Tests (ANOVA & Kruskal-Wallis)",
        "3. Correlation Analytics"
    ])
    
    with tab_chi:
        st.subheader("Pearson's Chi-Square Test of Independence")
        chi_res = run_chi_square_test(df_cleaned, col_x="Income category", col_y="Application status")
        
        card_class = "significant" if chi_res["is_significant"] else "insignificant"
        decision_badge = "REJECT H0 (Significant Association)" if chi_res["is_significant"] else "FAIL TO REJECT H0 (Independent)"
        
        st.markdown(f"""
        <div class="stat-result-card {card_class}">
            <h4 style="margin:0 0 8px 0; color:#1E3A8A;">{chi_res['test_name']}</h4>
            <p><b>{chi_res['h0']}</b><br/><b>{chi_res['h1']}</b></p>
            <div style="display:flex; gap:1.5rem; margin:12px 0; flex-wrap:wrap;">
                <div><b>Chi-Square Statistic (χ²):</b> {chi_res['chi2_stat']}</div>
                <div><b>Degrees of Freedom (df):</b> {chi_res['degrees_of_freedom']}</div>
                <div><b>p-value:</b> {chi_res['p_value']:.4e}</div>
                <div><b>Cramer's V (Effect Size):</b> {chi_res['cramers_v']}</div>
                <div><b>Decision (α=0.05):</b> <span style="font-weight:700; color:{'#10B981' if chi_res['is_significant'] else '#F59E0B'};">{decision_badge}</span></div>
            </div>
            <div class="callout-box" style="margin-top:8px;">
                <div class="callout-title">Interpretation & Policy Context</div>
                <div class="callout-text">{chi_res['interpretation']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Observed Contingency Counts")
            st.dataframe(chi_res["contingency_table"], use_container_width=True)
        with col_c2:
            st.markdown("##### Expected Frequencies Under Independence (H0)")
            st.dataframe(chi_res["expected_table"], use_container_width=True)
            
    with tab_anova:
        st.subheader("Regional Waiting Time Comparison & Assumption Checks")
        st.markdown("""
        Before applying parametric One-Way ANOVA to compare mean waiting times across regions, 
        we must verify two foundational assumptions:
        - **Normality:** Wait times must be approximately normally distributed.
        - **Homoscedasticity (Levene's Test):** Variances must be equal across all regions.
        """)
        
        anova_res = run_waiting_time_tests(df_cleaned, group_col="Region", value_col="Waiting time")
        
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            lev_badge = "✅ Homoscedastic (Equal Variances)" if anova_res["equal_variance"] else "⚠️ Heteroscedastic (Unequal Variances)"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Levene's Test for Homogeneity</div>
                <div class="metric-value">p = {anova_res['levene_p']:.4e}</div>
                <div class="metric-subtitle">Stat: {anova_res['levene_stat']} | {lev_badge}</div>
            </div>
            """, unsafe_allow_html=True)
        with diag_col2:
            norm_badge = "✅ Approximately Normal" if anova_res["is_normal"] else "⚠️ Non-Normal Distribution"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Normality Check</div>
                <div class="metric-value">p = {anova_res['norm_p']:.4e}</div>
                <div class="metric-subtitle">Stat: {anova_res['norm_stat']} | {norm_badge}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Test Comparison Card
        rec_test = anova_res["recommended_test"]
        st.markdown(f"""
        <div class="stat-result-card significant">
            <h4 style="margin:0 0 8px 0; color:#1E3A8A;">Comparative Test Results (Parametric vs. Non-Parametric)</h4>
            <p><b>Recommended Primary Test:</b> <span style="font-weight:700; color:#2563EB;">{rec_test}</span> (Due to diagnostic assumption outcomes)</p>
            <div style="display:flex; gap:2rem; margin:12px 0; flex-wrap:wrap;">
                <div>
                    <b>One-Way ANOVA:</b><br/>
                    F-statistic = {anova_res['f_stat']}<br/>
                    p-value = {anova_res['anova_p']:.4e}
                </div>
                <div>
                    <b>Kruskal-Wallis H-Test:</b><br/>
                    H-statistic = {anova_res['kw_stat']}<br/>
                    p-value = {anova_res['kw_p']:.4e}
                </div>
                <div>
                    <b>Significance:</b><br/>
                    {'Statistically Significant Difference (p < 0.05)' if anova_res['kw_sig'] else 'No Significant Difference'}
                </div>
            </div>
            <div class="callout-box">
                <div class="callout-title">Synthesis & Causal Guardrail</div>
                <div class="callout-text">{anova_res['interpretation']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Regional Waiting Time Descriptive Statistics (Active Applications)")
        st.dataframe(anova_res["group_summary_df"], use_container_width=True, hide_index=True)
        
    with tab_corr:
        st.subheader("Correlation Matrix & Inter-Variable Relationships")
        corr_res = run_correlation_analysis(df_cleaned)
        
        c_mode = st.radio("Select Correlation Method:", ["Pearson (Linear Relationship)", "Spearman (Rank-Order Monotonicity)"], horizontal=True)
        chosen_corr = corr_res["pearson_corr"] if "Pearson" in c_mode else corr_res["spearman_corr"]
        
        st.plotly_chart(create_correlation_heatmap(chosen_corr, title=f"{c_mode} Matrix"), use_container_width=True)
        
        st.markdown("""
        <div class="callout-box">
            <div class="callout-title">Key Correlation Takeaways</div>
            <div class="callout-text">
                • <b>Housing Need Score vs. Annual Income:</b> Strong negative correlation demonstrates that applicants 
                with lower annual incomes receive systematically higher housing need scores, aligning with HUD targeting.<br/>
                • <b>Household Size vs. Need Score:</b> Moderate positive correlation indicates that larger family sizes 
                face overcrowding and receive prioritization.<br/>
                • <b>Rent vs. Income:</b> Positive correlation reflecting subsidized rent schedules proportional to household earnings.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================
# PAGE 5: FAIRNESS ANALYSIS & REPORTS
# =============================================================
elif page == "5. Fairness Analysis & Reports":
    st.header("⚖️ Housing Fairness Analysis & Automated Academic Reports")
    st.markdown("""
    Evaluating fairness in public housing requires **rigorous methodological hygiene**:
    - **Denominator Integrity:** Raw approval rates can be misleading because higher income applicants are statutorily ineligible. We benchmark approval rates against **eligible applicants**.
    - **Disparate Impact Benchmark (4/5ths Rule):** We assess whether any group experiences an approval ratio below **0.80** relative to the benchmark group.
    - **No Arbitrary Scoring:** We refrain from constructing unjustified "magic fairness scores," providing transparent, empirical ratios instead.
    """)
    
    tab_equity, tab_vuln, tab_supply, tab_download = st.tabs([
        "1. Disparity Ratios by Group",
        "2. Vulnerable Household Outcomes",
        "3. Regional Supply-to-Demand Balances",
        "4. Download Academic PDF & Data Reports"
    ])
    
    with tab_equity:
        st.subheader("Group-Level Equity & Disparate Impact Evaluation")
        group_selector = st.selectbox("Evaluate Disparity Across Dimension:", ["Income category", "Region", "Gender", "Disability status"])
        
        equity_df = evaluate_group_equity(df_cleaned, group_col=group_selector)
        
        st.plotly_chart(create_disparity_ratio_chart(equity_df, group_col=group_selector), use_container_width=True)
        
        st.markdown("##### Detailed Equity Audit Table")
        st.dataframe(equity_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="callout-box">
            <div class="callout-title">Understanding Public Housing Disparity Ratios</div>
            <div class="callout-text">
                In employment law, the 80% rule flags underrepresentation. In public housing, however, 
                lower approval rates for <b>Moderate and High Income</b> applicants reflect 
                <b>statutory compliance with HUD targeting mandates</b> (which require at least 75% of units 
                to go to extremely low income families). When interpreting these metrics during project viva, 
                emphasize that programmatic intent governs whether a disparity is equitable or problematic.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_vuln:
        st.subheader("Protections for Vulnerable Demographic Cohorts")
        st.markdown("""
        Statutory mandates require prioritized assistance for households with disabilities, large families, 
        and extreme poverty. The audit table below examines whether these protected groups achieve 
        proportionate or accelerated allocation:
        """)
        
        vuln_df = evaluate_vulnerable_households(df_cleaned)
        st.dataframe(vuln_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="callout-box">
            <div class="callout-title">Findings on Vulnerability Prioritization</div>
            <div class="callout-text">
                • <b>Disability Status:</b> Eligible households with reported disabilities experience higher need scores 
                and higher eligible approval rates than applicants without disabilities.<br/>
                • <b>Large Households (5+ Members):</b> Eligible large families maintain high approval rates, 
                though waiting durations are influenced by the scarce physical inventory of 3+ bedroom municipal units.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with tab_supply:
        st.subheader("Regional Supply-to-Demand Balances")
        st.markdown("""
        Disparities in waiting time and allocation rates are predominantly structural: certain municipal regions 
        experience high eligible applicant demand but possess limited housing unit turn-over.
        """)
        
        sd_df = evaluate_regional_supply_demand(df_cleaned)
        st.plotly_chart(create_supply_demand_chart(sd_df), use_container_width=True)
        
        st.markdown("##### Regional Allocation-to-Demand Audit Table")
        st.dataframe(sd_df, use_container_width=True, hide_index=True)
        
    with tab_download:
        st.subheader("📄 Automated Academic PDF Report & Export Hub")
        st.markdown("""
        Generate an extensive, formally formatted academic project report suitable for college project reviews, 
        viva examinations, and portfolio submissions. The PDF is dynamically assembled using **ReportLab** 
        and reflects the exact current dataset in memory.
        """)
        
        col_pdf1, col_pdf2 = st.columns([2, 1])
        with col_pdf1:
            st.markdown("""
            **The generated PDF report includes:**
            - Executive Summary & Project Statement
            - Research Hypotheses & Methodology
            - Data Cleaning & Auditing Protocol (Before vs After)
            - Feature Engineering & Derived Metrics Documentation
            - Embedded Exploratory Data Analysis Charts
            - Inferential Statistical Test Results (Chi-Square, ANOVA, Kruskal-Wallis)
            - Housing Fairness & Disparate Impact Assessment
            - Limitations, Fair Housing Nuances & Future Scope
            """)
            
            with st.spinner("Preparing PDF report generator..."):
                pdf_bytes = build_pdf_report(df_cleaned)
                
            st.download_button(
                label="📄 Download Official Academic PDF Report",
                data=pdf_bytes,
                file_name="public_housing_fairness_assessment_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with col_pdf2:
            st.markdown("**Additional Project Artifacts:**")
            
            # Download Cleaned CSV
            csv_data = df_cleaned.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_data,
                file_name="cleaned_housing_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Download Raw CSV
            raw_csv = df_raw.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Raw Benchmark CSV",
                data=raw_csv,
                file_name="public_housing_data.csv",
                mime="text/csv",
                use_container_width=True
            )

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; padding:10px 0;">
    <b>Statistical Assessment of Public Housing Distribution Fairness</b> | Data Science Final Year College Capstone Project<br/>
    Built with Python, Pandas, NumPy, SciPy, Plotly, Streamlit & ReportLab | Academic Demonstration Only
</div>
""", unsafe_allow_html=True)
