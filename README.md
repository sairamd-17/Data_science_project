# Statistical Assessment of Public Housing Distribution Fairness Using Data Manipulation Techniques

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-150458.svg)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11+-8CAAE6.svg)](https://scipy.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/ReportLab-4.x-orange.svg)](https://www.reportlab.com/)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-brightgreen.svg)]()

> **Final Year College Capstone Project**  
> An empirical Data Science investigation into public housing distribution equity, regional allocation-to-demand balances, and statutory prioritization using Python, Pandas, SciPy inferential statistics, Plotly interactive graphics, and Streamlit.

---

## 📌 1. Project Overview & Problem Statement

Public housing programs provide subsidized housing units to low-income families, elderly individuals, and persons with disabilities. Due to severe physical housing shortages, municipal housing authorities must ration scarce housing units based on statutory need criteria (such as HUD regulations).

However, public scrutiny often questions whether administrative outcomes—such as application approval, waiting times, and unit allocations—are distributed fairly across socio-economic groups and geographic regions.

This project delivers an end-to-end Data Science pipeline that:
1. **Audits and sanitizes** administrative public housing records containing realistic clerical flaws.
2. **Applies advanced Pandas transformations** (vectorized calculations, conditional aggregation, multi-key grouping).
3. **Conducts inferential hypothesis testing** with SciPy (Pearson's Chi-Square, One-Way ANOVA, Levene's test, Kruskal-Wallis, and correlation analysis).
4. **Quantifies fairness** using Disparate Impact Ratios (the EEOC 80% rule) and regional allocation-to-demand metrics without relying on arbitrary scoring.
5. **Generates an academic PDF report** dynamically populated with actual figures, tables, and charts using ReportLab.

---

## 🏗️ 2. Repository Architecture

```
final_data_project/
├── app.py                     # Main 5-page interactive Streamlit dashboard
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation & viva preparation guide
├── data/
│   ├── public_housing_data.csv        # Raw synthetic housing benchmark (1,500+ records)
│   ├── cleaned_housing_data.csv       # Sanitized dataset with derived features
│   └── public_housing_fairness_report.pdf # Generated academic PDF report
├── modules/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic data generator with controlled real-world flaws
│   ├── data_cleaning.py       # Pandas data auditing, deduplication, and imputation pipeline
│   ├── analysis.py            # EDA aggregations, KPI calculations, and interactive Plotly charts
│   ├── statistics.py          # SciPy hypothesis testing (Chi-Square, ANOVA, Kruskal-Wallis, Corr)
│   ├── fairness.py            # Equity audit, disparate impact ratios, and supply-demand balance
│   └── report_generator.py   # ReportLab automated PDF report generator with static figures
└── tests/
    ├── __init__.py
    ├── test_cleaning.py       # Unit tests for data generation, validation, and cleaning
    ├── test_statistics.py     # Unit tests for Chi-Square, ANOVA, and correlations
    └── test_fairness.py       # Unit tests for equity metrics and PDF compilation
```

---

## 🧹 3. Data Cleaning & Manipulation Pipeline

Real-world administrative data exhibits missing values, duplicate filings, inconsistent categorical casing, and out-of-range numeric values. The cleaning pipeline implements:

### A. Deduplication
- Drops exact duplicate rows.
- Eliminates duplicate `Application ID` entries, keeping the first valid record:
  ```python
  df = df.drop_duplicates()
  df = df.drop_duplicates(subset=["Application ID"], keep="first")
  ```

### B. Categorical Normalization
- Resolves inconsistent casing, extra whitespace, and typos in `Income category`, `Gender`, and `Disability status` using string manipulation and dictionary mapping.

### C. Context-Aware Statistical Imputation
- Imputes missing continuous variables (`Annual income`, `Waiting time`, `Monthly rent`) using **group-level medians** (e.g. median income within the specific income bracket; median waiting time within the applicant's region) rather than naive global means.

### D. Constraint Validation
- Negative numbers in income, household size, waiting time, or rent are corrected using `abs()` or replaced with valid domain minimums.
- Enforces statutory constraints: Ineligible applicants cannot have an `Approved` or `Allocated` status.

### E. Feature Engineering & Derived Metrics
- **`Per_Capita_Income`**: $\frac{\text{Annual Income}}{\text{Household Size}}$
- **`Rent_Burden_Ratio`**: $\frac{\text{Monthly Rent} \times 12}{\text{Annual Income}}$ (clipped to $[0, 1.0]$)
- **`Need_Priority_Tier`**: Categorized into *High Need (70+)*, *Medium Need (45-69)*, and *Low Need (<45)*.
- **`Waiting_Time_Category`**: Grouped into *Short (<6m)*, *Moderate (6-18m)*, *Long (18-36m)*, and *Extended (>36m)*.
- **`Is_Vulnerable`**: Boolean flag indicating disability, large household size ($\ge 5$), or Very Low Income.

---

## 🧪 4. Inferential Statistical Testing Methodology

| Test | Variables Tested | Purpose | Decision Rule (α = 0.05) |
|---|---|---|---|
| **Pearson's Chi-Square ($\chi^2$)** | `Income category` vs. `Application status` | Test independence between income bracket and application decision | Reject $H_0$ if $p < 0.05$. Cramer's V measures association strength. |
| **Levene's Test** | `Waiting time` across `Region` | Test assumption of equal variances across regions | If $p < 0.05$, variances are unequal (violates standard ANOVA assumptions). |
| **Normality Test** | `Waiting time` | Check if waiting durations adhere to Gaussian distribution | If $p < 0.05$, non-normal distribution. |
| **One-Way ANOVA** | `Waiting time` across `Region` | Parametric comparison of regional mean waiting durations | Evaluates variance between groups vs within groups. |
| **Kruskal-Wallis ($H$)** | `Waiting time` across `Region` | Non-parametric test comparing median waiting times | Recommended primary test when ANOVA assumptions fail. |
| **Correlation Analysis** | Numerical features | Assess monotonic and linear relationships | Pearson ($r$) and Spearman ($\rho$) correlation matrices with p-values. |

---

## ⚖️ 5. Housing Fairness Assessment

The fairness module assesses administrative equity using regulatory standards:

1. **Eligible Denominator Principle**:
   - High-income applicants are statutorily ineligible. Evaluating raw approval rates distorts findings. We calculate:
     $$\text{Eligible Approval Rate} = \frac{\text{Approved Applicants}}{\text{Eligible Applicants}} \times 100$$
2. **Disparate Impact Ratio (EEOC 80% Rule)**:
   - Compares the eligible approval rate of each cohort against the highest-performing benchmark group:
     $$\text{Disparity Ratio} = \frac{\text{Group Eligible Approval Rate}}{\text{Benchmark Eligible Approval Rate}}$$
   - Ratios below **0.80** highlight potential disparities requiring policy audit.
3. **Regional Supply-to-Demand Ratio**:
   - Compares allocated units (supply) against eligible applicants (demand) across municipal sectors to explain structural waiting time differences.
4. **Vulnerable Household Safeguards**:
   - Assesses whether households with reported disabilities, large family sizes, and severe rent burdens receive statutory prioritization.

---

## 🚀 6. Installation & Execution Guide

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Unit & Integration Tests
```bash
pytest tests/ -v
```
All 12 unit tests should pass with 100% compliance.

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎓 7. Viva & Project Review Preparation

Here are key questions your evaluators may ask during your project review, along with recommended answers:

### Q1: Why did you choose Kruskal-Wallis alongside One-Way ANOVA?
> **Answer:** Standard One-Way ANOVA requires two key assumptions: normality of residuals and homogeneity of variance across groups (homoscedasticity). In our diagnostic tests, Levene's test yielded $p < 0.05$, indicating unequal variances across regions, and waiting times exhibited right-skewness. Therefore, the Kruskal-Wallis H-test is the scientifically sound non-parametric test because it relies on ranks rather than parametric assumptions.

### Q2: Why does the Moderate/High Income category have a low approval rate? Does this indicate discrimination?
> **Answer:** No. Public housing is a subsidized welfare safety net governed by statutory income limits (HUD regulations). By federal mandate, at least 75% of public housing admissions must be families whose incomes do not exceed 30% of Area Median Income (Extremely/Very Low Income). Therefore, lower approval for higher-income applicants reflects **statutory compliance with legislative intent**, not unlawful discrimination.

### Q3: Why is it important to use an "Eligible Approval Rate" rather than a raw approval rate?
> **Answer:** Using total applications as the denominator creates a distorted metric (known as the *base rate fallacy*). Because ineligible applicants are legally barred from receiving housing, including them in the denominator conflates programmatic ineligibility with administrative bias. Benchmarking outcomes against eligible demand provides an accurate assessment of equity.

### Q4: How does the automated PDF report work?
> **Answer:** The report is dynamically generated in memory using ReportLab and Matplotlib. It computes live summary statistics, hypothesis testing p-values, and equity audit tables from whichever dataset is currently loaded in the dashboard, builds the document flow with a two-pass `NumberedCanvas` (Page X of Y), and offers it as a direct download.

---

## 📄 License & Academic Attribution
Developed for college academic evaluation and portfolio demonstration. All datasets provided are synthetic and generated for educational and analytical purposes.
