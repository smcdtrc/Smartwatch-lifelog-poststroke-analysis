# -*- coding: utf-8 -*-
"""
01_survey_revision_analysis.py
==============================

Survey-only revision analysis.

Input
-----
- servey.xlsx

Output
------
- survey_outputs/survey_participant_scores.xlsx
- survey_outputs/survey_revision_analysis_results.xlsx

What this script does
---------------------
1) Reads the survey Excel file.
2) Computes questionnaire scale scores.
3) Computes official PSQI-K 7 component scores and total score.
4) Creates participant-level survey dataset.
5) Exports reviewer-ready descriptive summaries.
6) Uses full-cohort exploratory correlations with FDR correction.
7) Keeps age-band and institution analyses as exploratory/supplementary.

Note
----
This script does NOT calculate HRV or IMU features.
Run the HRV and IMU scripts separately.
"""

import re
import itertools
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# User settings
# =============================================================================
SURVEY_FILE = "servey.xlsx"

OUT_DIR = Path("survey_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARTICIPANT_XLSX = OUT_DIR / "survey_participant_scores.xlsx"
OUT_ANALYSIS_XLSX = OUT_DIR / "survey_revision_analysis_results.xlsx"

ID_PAT = re.compile(r"^d\d{2}-\d{3}$", re.IGNORECASE)
TIME_PAT = re.compile(r"^\s*(\d{1,2})\s*:\s*(\d{2})\s*$")
NUM_PAT = re.compile(r"[-+]?\d*\.?\d+")

SITE_MAP = {"d01": "SMC", "d02": "KUAH", "d03": "HPH"}

SURVEY_SECTIONS = {
    "FSS": list(range(4, 13)),
    "MoCA-K": list(range(13, 23)),
    "K-MMSE-2": list(range(24, 43)),
    # PSQI-K is calculated separately using official component scoring.
    "낙상효능척도": list(range(69, 76)),
    "낙상경험": list(range(78, 90)),
    "GAD-7": list(range(90, 97)),
    "신체증상장애": list(range(97, 109)),
    "영양습관": list(range(109, 121)),
    "음주습관": list(range(121, 126)),
    "흡연습관": list(range(126, 131)),
    "SARC-F": list(range(131, 135)),
    "EQ_5D-5L": list(range(136, 142)),
}
SCORE_COLUMNS = list(SURVEY_SECTIONS.keys()) + ["PSQI-K"]


# =============================================================================
# Utility
# =============================================================================
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fdr_bh(pvals: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    mask = p.notna()
    if mask.sum() > 0:
        out.loc[mask] = multipletests(p.loc[mask].values, method="fdr_bh")[1]
    return out


def mean_ci_95(x: pd.Series) -> Tuple[float, float]:
    x = safe_numeric(x).dropna()
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return x.mean() - tcrit * se, x.mean() + tcrit * se


def spearman_ci_95(rho: float, n: int) -> Tuple[float, float]:
    if n < 4 or pd.isna(rho) or abs(rho) >= 1:
        return np.nan, np.nan
    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    lo, hi = z - 1.96 * se, z + 1.96 * se
    return np.tanh(lo), np.tanh(hi)


def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    if n <= k or pd.isna(H):
        return np.nan
    return max((H - k + 1) / (n - k), 0)


def eta_squared_anova(groups: List[pd.Series]) -> float:
    vals = [safe_numeric(g).dropna() for g in groups]
    all_values = pd.concat(vals, ignore_index=True)
    if len(all_values) == 0:
        return np.nan
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in vals if len(g) > 0)
    ss_total = ((all_values - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else np.nan


# =============================================================================
# PSQI scoring
# =============================================================================
def find_header_row(df: pd.DataFrame, max_scan: int = 50, probe: str = "Base_PSQI-K_") -> int:
    best_r, best_hits = 0, -1
    for r in range(min(max_scan, len(df))):
        hits = df.iloc[r].astype(str).str.contains(probe, na=False).sum()
        if hits > best_hits:
            best_r, best_hits = r, hits
    return best_r


def parse_hhmm_to_minutes(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("\n", " ")
    m = TIME_PAT.search(s)
    if m:
        h, mm = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 47 and 0 <= mm <= 59:
            return h * 60 + mm
    nums = NUM_PAT.findall(s)
    if len(nums) == 1:
        try:
            return float(nums[0])
        except Exception:
            return np.nan
    return np.nan


def parse_likert_1to4_to_0to3(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = NUM_PAT.search(s)
    if not m:
        return np.nan
    v = float(m.group())
    if 1 <= v <= 4:
        return v - 1.0
    return np.nan


def minutes_between_bed_and_wake(bed_min, wake_min):
    if pd.isna(bed_min) or pd.isna(wake_min):
        return np.nan
    bed_min = float(bed_min) % (24 * 60)
    wake_min = float(wake_min) % (24 * 60)
    dur = wake_min - bed_min
    if dur < 0:
        dur += 24 * 60
    if dur <= 0 or dur > 24 * 60:
        return np.nan
    return dur


def compute_psqi_components_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    header_row = find_header_row(raw_df)
    cols = raw_df.iloc[header_row].tolist()
    df = raw_df.copy()
    df.columns = cols

    id_col = None
    for c in df.columns:
        if df[c].astype(str).str.match(ID_PAT, na=False).any():
            id_col = c
            break
    if id_col is None:
        raise ValueError("dXX-XXX pattern ID column not found for PSQI scoring.")

    valid = df[id_col].astype(str).str.match(ID_PAT, na=False)
    df = df.loc[valid].copy().reset_index(drop=True)
    df.rename(columns={id_col: "Subject No."}, inplace=True)

    Q1 = df.get("Base_PSQI-K_1")
    Q2 = df.get("Base_PSQI-K_2")
    Q3 = df.get("Base_PSQI-K_3")
    Q4 = df.get("Base_PSQI-K_4")
    Q5 = {k: df.get(f"Base_PSQI-K_5-{k}") for k in list("abcdefghi")}
    Q6 = df.get("Base_PSQI-K_6")
    Q7 = df.get("Base_PSQI-K_7")
    Q8 = df.get("Base_PSQI-K_8")
    Q9 = df.get("Base_PSQI-K_9")

    idx = df.index
    bed_min = Q1.apply(parse_hhmm_to_minutes) if Q1 is not None else pd.Series(np.nan, index=idx)
    lat_min = Q2.apply(parse_hhmm_to_minutes) if Q2 is not None else pd.Series(np.nan, index=idx)
    wake_min = Q3.apply(parse_hhmm_to_minutes) if Q3 is not None else pd.Series(np.nan, index=idx)

    def parse_sleep_hours(x):
        m = parse_hhmm_to_minutes(x)
        if pd.isna(m):
            if pd.isna(x):
                return np.nan
            m2 = NUM_PAT.search(str(x))
            return float(m2.group()) if m2 else np.nan
        return m / 60.0

    sleep_hours = Q4.apply(parse_sleep_hours) if Q4 is not None else pd.Series(np.nan, index=idx)

    Q6_sc = Q6.apply(parse_likert_1to4_to_0to3) if Q6 is not None else pd.Series(np.nan, index=idx)
    Q7_sc = Q7.apply(parse_likert_1to4_to_0to3) if Q7 is not None else pd.Series(np.nan, index=idx)
    Q8_sc = Q8.apply(parse_likert_1to4_to_0to3) if Q8 is not None else pd.Series(np.nan, index=idx)
    Q9_sc = Q9.apply(parse_likert_1to4_to_0to3) if Q9 is not None else pd.Series(np.nan, index=idx)
    Q5_sc = {
        k: (Q5[k].apply(parse_likert_1to4_to_0to3) if Q5[k] is not None else pd.Series(np.nan, index=idx))
        for k in Q5
    }

    C1 = Q6_sc.clip(0, 3)

    def latency_minutes_to_score(m):
        if pd.isna(m):
            return np.nan
        m = float(m)
        if m <= 15:
            return 0
        if m <= 30:
            return 1
        if m <= 60:
            return 2
        return 3

    Q2_part = lat_min.apply(latency_minutes_to_score)
    C2_raw = Q2_part.add(Q5_sc["a"], fill_value=np.nan)

    def c2_collapse(x):
        if pd.isna(x):
            return np.nan
        x = int(round(x))
        if x <= 0:
            return 0
        if x <= 2:
            return 1
        if x <= 4:
            return 2
        return 3

    C2 = C2_raw.apply(c2_collapse)

    def sleep_hours_to_c3(h):
        if pd.isna(h):
            return np.nan
        h = float(h)
        if h > 7:
            return 0
        if h > 6:
            return 1
        if h > 5:
            return 2
        return 3

    C3 = sleep_hours.apply(sleep_hours_to_c3)

    time_in_bed_hours = pd.Series(
        [minutes_between_bed_and_wake(b, w) for b, w in zip(bed_min, wake_min)],
        index=idx,
        dtype="float",
    ) / 60.0
    efficiency = (sleep_hours / time_in_bed_hours) * 100.0

    def eff_to_c4(p):
        if pd.isna(p):
            return np.nan
        if p >= 85:
            return 0
        if p >= 75:
            return 1
        if p >= 65:
            return 2
        return 3

    C4 = efficiency.apply(eff_to_c4)

    disturb_sum = sum(Q5_sc[k] for k in list("bcdefghi"))

    def disturb_to_c5(s):
        if pd.isna(s):
            return np.nan
        if s <= 0:
            return 0
        if s <= 9:
            return 1
        if s <= 18:
            return 2
        return 3

    C5 = disturb_sum.apply(disturb_to_c5)
    C6 = Q7_sc.clip(0, 3)

    day_sum = Q8_sc.add(Q9_sc, fill_value=np.nan)

    def day_to_c7(s):
        if pd.isna(s):
            return np.nan
        s = int(round(s))
        if s <= 0:
            return 0
        if s <= 2:
            return 1
        if s <= 4:
            return 2
        return 3

    C7 = day_sum.apply(day_to_c7)

    comp = pd.DataFrame({
        "Subject No.": df["Subject No."],
        "PSQI_C1_subjective_quality": C1,
        "PSQI_C2_sleep_latency": C2,
        "PSQI_C3_sleep_duration": C3,
        "PSQI_C4_sleep_efficiency": C4,
        "PSQI_C5_sleep_disturbance": C5,
        "PSQI_C6_sleep_medication": C6,
        "PSQI_C7_daytime_dysfunction": C7,
    })
    psqi_cols = [c for c in comp.columns if c.startswith("PSQI_C")]
    comp["PSQI_TOTAL(0-21)"] = comp[psqi_cols].sum(axis=1, min_count=1)
    comp["PSQI_sleep_eff_%"] = efficiency
    comp["PSQI_sleep_hours"] = sleep_hours
    return comp


def load_and_score_survey(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_df = pd.read_excel(file_path, header=None, dtype=str)

    id_col_index = -1
    for col in raw_df.columns:
        if raw_df[col].astype(str).str.match(ID_PAT, na=False).any():
            id_col_index = col
            break
    if id_col_index == -1:
        raise ValueError("dXX-XXX pattern ID column not found in survey file.")

    valid_rows_mask = raw_df[id_col_index].astype(str).str.match(ID_PAT, na=False)
    end_col_index = min(id_col_index + 142, len(raw_df.columns))
    df = raw_df.loc[valid_rows_mask, range(id_col_index, end_col_index)].copy()
    df.columns = range(df.shape[1])
    df.rename(columns={0: "Subject No.", 1: "Sex", 2: "Age"}, inplace=True)

    for section_name, cols in SURVEY_SECTIONS.items():
        use_cols = []
        for c in cols:
            if c in df.columns:
                df[c] = safe_numeric(df[c])
                use_cols.append(c)
        df[section_name] = df[use_cols].sum(axis=1, min_count=1) if use_cols else np.nan

    psqi_comp = compute_psqi_components_from_raw(raw_df)
    df = df.merge(psqi_comp, on="Subject No.", how="left")
    df["PSQI-K"] = df["PSQI_TOTAL(0-21)"]

    df["institution"] = df["Subject No."].astype(str).str[:3].str.lower()
    df["site"] = df["institution"].map(SITE_MAP).fillna(df["institution"])
    df["Age"] = safe_numeric(df["Age"])
    df["Sex_num"] = safe_numeric(df["Sex"])
    df["sex_group"] = df["Sex_num"].replace({0: "Female", 1: "Male"})

    age_bins = [0, 49, 59, 69, 150]
    age_labels = ["<50", "50-59", "60-69", "70+"]
    df["age_group_exploratory"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=True)

    df["questionnaire_score_count"] = df[SCORE_COLUMNS].notna().sum(axis=1)

    comp_rows = []
    n_total = len(df)
    for sc in SCORE_COLUMNS:
        comp_rows.append({
            "score": sc,
            "N_available": int(df[sc].notna().sum()),
            "N_missing": int(df[sc].isna().sum()),
            "completion_pct": float(df[sc].notna().mean() * 100) if n_total > 0 else np.nan,
        })
    questionnaire_completion = pd.DataFrame(comp_rows)

    return df, psqi_comp, questionnaire_completion


# =============================================================================
# Analysis
# =============================================================================
def descriptive_table(df: pd.DataFrame, variables: List[str], by: Optional[str] = None) -> pd.DataFrame:
    rows = []
    if by is None:
        groups = [("Overall", df)]
    else:
        groups = [(str(g), sub) for g, sub in df.groupby(by, dropna=False, observed=False)]

    for group_name, sub in groups:
        for v in variables:
            if v not in sub.columns:
                continue
            x = safe_numeric(sub[v])
            lo, hi = mean_ci_95(x)
            rows.append({
                "grouping": by if by else "Overall",
                "level": group_name,
                "variable": v,
                "N": int(x.notna().sum()),
                "missing_N": int(x.isna().sum()),
                "mean": x.mean(),
                "SD": x.std(ddof=1),
                "mean_95CI_low": lo,
                "mean_95CI_high": hi,
                "median": x.median(),
                "IQR_Q1": x.quantile(0.25),
                "IQR_Q3": x.quantile(0.75),
                "min": x.min(),
                "max": x.max(),
            })
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for v in variables:
        if v not in df.columns:
            continue
        counts = df[v].value_counts(dropna=False)
        for level, cnt in counts.items():
            rows.append({
                "variable": v,
                "level": str(level),
                "N": int(cnt),
                "pct": float(cnt / n * 100) if n > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def exploratory_institution_tests(df: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        if v not in df.columns:
            continue
        sub = df[["site", v]].dropna()
        levels = [g for g in sorted(sub["site"].dropna().unique())]
        groups = [safe_numeric(sub.loc[sub["site"] == g, v]).dropna() for g in levels]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue

        normal_ok = all(len(g) >= 3 and stats.shapiro(g).pvalue >= 0.05 for g in groups)
        levene_p = stats.levene(*groups).pvalue if all(len(g) >= 2 for g in groups) else np.nan
        equal_var_ok = pd.notna(levene_p) and levene_p >= 0.05

        if normal_ok and equal_var_ok:
            stat, p = stats.f_oneway(*groups)
            test_name = "One-way ANOVA"
            df1 = len(groups) - 1
            df2 = sum(len(g) for g in groups) - len(groups)
            stat_name = "F"
            eff_name = "eta_squared"
            eff = eta_squared_anova(groups)
        else:
            stat, p = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
            df1 = len(groups) - 1
            df2 = np.nan
            stat_name = "H"
            eff_name = "epsilon_squared"
            eff = epsilon_squared_kruskal(stat, sum(len(g) for g in groups), len(groups))

        rows.append({
            "family": "exploratory_institution_comparison",
            "variable": v,
            "test": test_name,
            "statistic_name": stat_name,
            "statistic": stat,
            "df1": df1,
            "df2": df2,
            "raw_p": p,
            "effect_size_name": eff_name,
            "effect_size": eff,
            "N_total": int(sum(len(g) for g in groups)),
            "min_group_N": int(min(len(g) for g in groups)),
            "assumption_note": f"normal_ok={normal_ok}; Levene_p={levene_p}",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = fdr_bh(out["raw_p"])
    return out


def full_cohort_spearman(df: pd.DataFrame, x_vars: List[str], y_vars: List[str], family_name: str) -> pd.DataFrame:
    rows = []
    for x, y in itertools.product(x_vars, y_vars):
        if x == y or x not in df.columns or y not in df.columns:
            continue
        sub = df[[x, y]].copy()
        sub[x] = safe_numeric(sub[x])
        sub[y] = safe_numeric(sub[y])
        sub = sub.dropna()
        n = len(sub)
        if n < 4:
            rho, p, lo, hi = np.nan, np.nan, np.nan, np.nan
        else:
            rho, p = stats.spearmanr(sub[x], sub[y])
            lo, hi = spearman_ci_95(rho, n)
        rows.append({
            "family": family_name,
            "x": x,
            "y": y,
            "N": n,
            "test": "Spearman correlation",
            "rho": rho,
            "rho_95CI_low": lo,
            "rho_95CI_high": hi,
            "raw_p": p,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = fdr_bh(out["raw_p"])
    return out


def main():
    df, psqi_comp, questionnaire_completion = load_and_score_survey(SURVEY_FILE)

    categorical = categorical_summary(df, ["site", "institution", "sex_group", "age_group_exploratory"])
    descriptives = descriptive_table(df, ["Age", "questionnaire_score_count"] + SCORE_COLUMNS)
    by_institution = descriptive_table(df, SCORE_COLUMNS, by="site")
    by_age = descriptive_table(df, SCORE_COLUMNS, by="age_group_exploratory")
    institution_tests = exploratory_institution_tests(df, SCORE_COLUMNS)

    score_corr = full_cohort_spearman(
        df,
        x_vars=SCORE_COLUMNS,
        y_vars=SCORE_COLUMNS,
        family_name="exploratory_score_score_correlation",
    )
    if not score_corr.empty:
        score_corr["pair_key"] = score_corr.apply(lambda r: "--".join(sorted([r["x"], r["y"]])), axis=1)
        score_corr = score_corr.drop_duplicates("pair_key").drop(columns=["pair_key"])
        score_corr["p_fdr_bh"] = fdr_bh(score_corr["raw_p"])

    notes = pd.DataFrame([
        {"item": "analysis_frame", "note": "Survey-only analysis. HRV and IMU features are not calculated in this script."},
        {"item": "subgroup_analysis", "note": "Institution and age subgroup analyses should be treated as descriptive/exploratory."},
        {"item": "multiple_comparisons", "note": "Benjamini-Hochberg FDR-adjusted p-values are exported where applicable."},
        {"item": "reporting", "note": "Report N, statistic, df where applicable, effect size, CI, raw p, and FDR-adjusted p."},
    ])

    df.to_excel(OUT_PARTICIPANT_XLSX, index=False)

    with pd.ExcelWriter(OUT_ANALYSIS_XLSX, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="MANUSCRIPT_NOTES", index=False)
        df.to_excel(writer, sheet_name="participant_level_dataset", index=False)
        categorical.to_excel(writer, sheet_name="Table1_categorical", index=False)
        descriptives.to_excel(writer, sheet_name="Table1_descriptives", index=False)
        questionnaire_completion.to_excel(writer, sheet_name="questionnaire_completion", index=False)
        psqi_comp.to_excel(writer, sheet_name="PSQI_components", index=False)
        by_institution.to_excel(writer, sheet_name="descriptive_by_institution", index=False)
        by_age.to_excel(writer, sheet_name="SUPP_age_descriptives", index=False)
        institution_tests.to_excel(writer, sheet_name="exploratory_inst_tests", index=False)
        score_corr.to_excel(writer, sheet_name="full_cohort_score_corr", index=False)

    print("DONE")
    print(f"- Participant-level survey scores: {OUT_PARTICIPANT_XLSX}")
    print(f"- Survey analysis workbook:        {OUT_ANALYSIS_XLSX}")


if __name__ == "__main__":
    main()
