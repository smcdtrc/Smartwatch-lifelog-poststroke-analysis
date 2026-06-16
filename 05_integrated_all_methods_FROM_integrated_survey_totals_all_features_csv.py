"""
Full-cohort association analysis for the smartwatch lifelog (post-stroke) study.

Pipeline:
  1. Load integrated survey + wearable data, build complete-case cohort (N=36).
  2. Build two pre-specified wearable domain composites (sign-aligned z-mean).
  3. Original analysis : survey x individual feature  -> Spearman + BH (32 tests).
  4. Revised analysis  : survey x domain composite    -> Spearman + 95% CI + BH (8 tests)
                         + age-adjusted partial Spearman.

Requires: pandas, numpy, scipy, scikit-learn, statsmodels, openpyxl
    pip install pandas numpy scipy scikit-learn statsmodels openpyxl
"""

import numpy as np
import pandas as pd
import numpy.linalg as la
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

XLSX = "integrated_SURVEY_TOTALS_ALL_features.xlsx"   # input file

# ----------------------------------------------------------------------
# Column definitions
# ----------------------------------------------------------------------
SURVEYS = {                       # survey total -> column
    "MoCA-K":   "survey__MoCA-K",
    "K-MMSE-2": "survey__K-MMSE-2",
    "SARC-F":   "survey__SARC-F",
    "EQ_5D-5L": "survey__EQ_5D-5L",
}
FEATURES = {                      # individual wearable feature -> column
    "stress_index": "hrv__stress_index", "RMSSD": "hrv__RMSSD", "SDNN": "hrv__SDNN",
    "pNN50": "hrv__pNN50", "HR": "hrv__HR",
    "enmo_mean_g": "imu__enmo_mean_g", "mad_g": "imu__mad_g", "Walking Speed": "imu__Walking Speed",
}
FEATURE_DOMAIN = {
    "stress_index": "HRV", "RMSSD": "HRV", "SDNN": "HRV", "pNN50": "HRV", "HR": "HRV",
    "enmo_mean_g": "Activity", "mad_g": "Activity", "Walking Speed": "Activity",
}
# Sign alignment within each domain (features negatively related to the construct are inverted).
# HRV construct = vagal tone: RMSSD/SDNN/pNN50 (+), HR/stress_index (-).
HRV_SIGNS = {"hrv__RMSSD": 1, "hrv__SDNN": 1, "hrv__pNN50": 1, "hrv__HR": -1, "hrv__stress_index": -1}
ACT_SIGNS = {"imu__enmo_mean_g": 1, "imu__mad_g": 1, "imu__Walking Speed": 1}
AGE = "survey__Age"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def fisher_ci(rho, n, alpha=0.05):
    """95% CI for a Spearman rho via Fisher z-transform."""
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - crit * se), np.tanh(z + crit * se)

def signed_zmean(data, signs):
    """Composite = mean of within-sample z-scores after sign alignment (== PC1 here)."""
    cols = list(signs)
    Z = StandardScaler().fit_transform(data[cols].values)
    Z = Z * np.array([signs[c] for c in cols])
    return Z.mean(axis=1)

def partial_spearman(x, y, cov):
    """Partial Spearman of x,y controlling cov, via correlation of rank residuals."""
    rx, ry, rc = stats.rankdata(x), stats.rankdata(y), stats.rankdata(cov)
    A = np.c_[np.ones_like(rc), rc]
    ex = rx - A @ la.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ la.lstsq(A, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), n - 3)
    return r, p

# ----------------------------------------------------------------------
# 1. Load + complete-case cohort
# ----------------------------------------------------------------------
df = pd.read_excel(XLSX, sheet_name="integrated")

analysis_cols = list(SURVEYS.values()) + list(FEATURES.values()) + [AGE]
d = df.dropna(subset=analysis_cols).copy()
N = len(d)
print(f"Analytic cohort: N = {N} (complete paired survey + wearable data)\n")

# ----------------------------------------------------------------------
# 2. Domain composites
# ----------------------------------------------------------------------
d["HRV"] = signed_zmean(d, HRV_SIGNS)
d["Activity"] = signed_zmean(d, ACT_SIGNS)
DOMAINS = {"HRV(autonomic)": "HRV", "Activity": "Activity"}

# ----------------------------------------------------------------------
# 3. Original analysis: survey x individual feature (BH over all such tests)
# ----------------------------------------------------------------------
rows = []
for s_name, s_col in SURVEYS.items():
    for f_name, f_col in FEATURES.items():
        rho, p = stats.spearmanr(d[s_col], d[f_col])
        lo, hi = fisher_ci(rho, N)
        ar, ap = partial_spearman(d[s_col].values, d[f_col].values, d[AGE].values)
        rows.append(dict(Survey=s_name, Feature=f_name, Domain=FEATURE_DOMAIN[f_name],
                         N=N, rho=rho, CI_low=lo, CI_high=hi, raw_p=p,
                         ageAdj_rho=ar, ageAdj_p=ap))
original = pd.DataFrame(rows)
original["BH_q"] = multipletests(original["raw_p"], method="fdr_bh")[1]
original = original.sort_values("raw_p").reset_index(drop=True)

print("=== ORIGINAL: survey x individual feature (BH family = %d tests) ===" % len(original))
print(original.round(4).to_string(index=False))
print("Survivors q<0.05:", int((original["BH_q"] < 0.05).sum()), "\n")

# ----------------------------------------------------------------------
# 4. Revised analysis: survey x domain composite (+ age-adjusted)
# ----------------------------------------------------------------------
rows = []
for s_name, s_col in SURVEYS.items():
    for dom_label, dom_col in DOMAINS.items():
        rho, p = stats.spearmanr(d[s_col], d[dom_col])
        lo, hi = fisher_ci(rho, N)
        ar, ap = partial_spearman(d[s_col].values, d[dom_col].values, d[AGE].values)
        rows.append(dict(Survey=s_name, Domain=dom_label, N=N,
                         rho=rho, CI_low=lo, CI_high=hi, raw_p=p,
                         ageAdj_rho=ar, ageAdj_p=ap))
revised = pd.DataFrame(rows)
revised["BH_q"] = multipletests(revised["raw_p"], method="fdr_bh")[1]
revised = revised.sort_values("raw_p").reset_index(drop=True)
revised["survives_BH"] = np.where(revised["BH_q"] < 0.05, "YES", "no")

print("=== REVISED: survey x domain composite (BH family = %d tests) ===" % len(revised))
print(revised.round(4).to_string(index=False))
print("Survivors q<0.05:", int((revised["BH_q"] < 0.05).sum()), "\n")

# ----------------------------------------------------------------------
# Optional: write tables to an Excel workbook
# ----------------------------------------------------------------------
with pd.ExcelWriter("analysis_output.xlsx") as xl:
    original.to_excel(xl, sheet_name="original_individual", index=False)
    revised.to_excel(xl, sheet_name="revised_composite", index=False)
    d[["Subject No.", AGE] + list(SURVEYS.values()) + ["HRV", "Activity"]].to_excel(
        xl, sheet_name="composite_scores", index=False)
    d[list(FEATURES.values())].corr(method="spearman").round(3).to_excel(
        xl, sheet_name="feature_collinearity")
print("Wrote analysis_output.xlsx")
