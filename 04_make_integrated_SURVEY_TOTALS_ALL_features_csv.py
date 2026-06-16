# -*- coding: utf-8 -*-
"""
04_make_integrated_SURVEY_TOTALS_ALL_features_csv.py
====================================================

Purpose
-------
Create one participant-level wide table that contains:

1) Subject/meta columns
2) Survey TOTAL / SCALE SCORE columns only
3) ALL IMU feature columns
4) ALL HRV feature columns

This script intentionally does NOT include survey item-level columns by default.
It is for making a clean integrated CSV for review/analysis.

Expected inputs
---------------
1) survey_outputs/survey_participant_scores.xlsx
2) imu_outputs/subject_imu_features.xlsx
3) hrv_outputs/subject_hrv_features.xlsx

Outputs
-------
integrated_survey_totals_all_features_outputs/
    integrated_SURVEY_TOTALS_ALL_features.csv
    integrated_SURVEY_TOTALS_ALL_features.xlsx
    selected_survey_total_columns.csv
    column_dictionary.csv
    missingness_summary.csv

Notes
-----
- Survey total columns are selected BEFORE merging, from the survey table only.
  This prevents IMU/HRV columns such as stress_index or total_steps from being
  accidentally treated as survey scores.
- IMU and HRV columns are all preserved with source prefixes: imu__*, hrv__*.
- Survey total columns are also prefixed as survey__* for clarity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# User settings
# =============================================================================
SURVEY_XLSX_CANDIDATES = [
    r"survey_outputs\survey_participant_scores.xlsx",
    r"survey_participant_scores.xlsx",
]
IMU_XLSX_CANDIDATES = [
    r"imu_outputs\subject_imu_features.xlsx",
    r"subject_imu_features.xlsx",
]
HRV_XLSX_CANDIDATES = [
    r"hrv_outputs\subject_hrv_features.xlsx",
    r"subject_hrv_features.xlsx",
]

OUT_DIR = Path("integrated_survey_totals_all_features_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "integrated_SURVEY_TOTALS_ALL_features.csv"
OUT_XLSX = OUT_DIR / "integrated_SURVEY_TOTALS_ALL_features.xlsx"
OUT_SELECTED_SURVEY = OUT_DIR / "selected_survey_total_columns.csv"
OUT_COLUMN_DICT = OUT_DIR / "column_dictionary.csv"
OUT_MISSINGNESS = OUT_DIR / "missingness_summary.csv"

# If True, include only columns listed in SURVEY_TOTAL_SPECS.
# If False, include SURVEY_TOTAL_SPECS plus additional auto-detected total/scale columns.
# For manuscript tables, True is safest. For broad inspection, False can help find other scale totals.
EXPLICIT_SURVEY_TOTALS_ONLY = False

# Preferred survey total/scale score columns.
# The key is the output name. The list contains possible source column aliases.
SURVEY_TOTAL_SPECS: Dict[str, List[str]] = {
    "MoCA-K": ["MoCA-K", "MoCA_K", "MoCA", "MOCA", "moca_total", "MoCA_total"],
    "K-MMSE-2": ["K-MMSE-2", "K_MMSE_2", "K-MMSE", "MMSE", "MMSE_total", "KMMSE"],
    "FSS": ["FSS", "FSS_total", "Fatigue Severity Scale"],
    "GAD-7": ["GAD-7", "GAD7", "GAD_7", "GAD-7_total", "GAD_TOTAL"],
    "SARC-F": ["SARC-F", "SARCF", "SARC_F", "SARC-F_total"],
    "EQ_5D-5L": ["EQ_5D-5L", "EQ-5D-5L", "EQ5D", "EQ_5D", "EQ_VAS", "EQ-VAS", "EQ5D_total"],
    # Prefer PSQI-K if present; otherwise use PSQI total alias.
    "PSQI-K": ["PSQI-K", "PSQI_K", "PSQI_TOTAL(0-21)", "PSQI_TOTAL", "PSQI total", "PSQI_total"],
}

# Metadata columns to keep from survey table if present.
SURVEY_META_COLS = [
    "Subject No.", "institution", "site", "Age", "Sex", "Sex_num",
]

# Auto-selection rules for additional survey total/scale columns.
# These are applied only to the SURVEY table, never to the merged dataframe.
AUTO_INCLUDE_REGEX = re.compile(
    r"(total|score|scale|index|sum|subscale|vas|eq[_\- ]?5d|moca|mmse|fss|gad|sarc|psqi)",
    flags=re.IGNORECASE,
)
AUTO_EXCLUDE_REGEX = re.compile(
    r"(^\d+$|^q\d+|_q\d+|item|question|문항|component|^c\d+|_c\d+|sleep_hours|sleep_eff|latency|disturbance|medication|dysfunction)",
    flags=re.IGNORECASE,
)
AUTO_EXCLUDE_EXACT = {
    "Subject No.", "institution", "site", "Age", "Sex", "Sex_num",
    "height", "weight", "BMI", "bmi",
}

# Prefixes for output columns.
SURVEY_PREFIX = "survey__"
IMU_PREFIX = "imu__"
HRV_PREFIX = "hrv__"

# =============================================================================
# Utility functions
# =============================================================================
def find_existing(candidates: Sequence[str]) -> Path:
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist: " + " | ".join(map(str, candidates)))


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def normalize_subject_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def norm_col(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")


def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    survey_path = find_existing(SURVEY_XLSX_CANDIDATES)
    imu_path = find_existing(IMU_XLSX_CANDIDATES)
    hrv_path = find_existing(HRV_XLSX_CANDIDATES)

    survey = clean_columns(pd.read_excel(survey_path))
    imu = clean_columns(pd.read_excel(imu_path))
    hrv = clean_columns(pd.read_excel(hrv_path))

    for name, df in [("survey", survey), ("imu", imu), ("hrv", hrv)]:
        if "Subject No." not in df.columns:
            raise ValueError(f"'Subject No.' column not found in {name} table.")
        df["Subject No."] = normalize_subject_id(df["Subject No."])

    print(f"[LOAD] survey: {survey_path} shape={survey.shape}")
    print(f"[LOAD] imu:    {imu_path} shape={imu.shape}")
    print(f"[LOAD] hrv:    {hrv_path} shape={hrv.shape}")
    return survey, imu, hrv, survey_path, imu_path, hrv_path


def find_alias_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    # exact first
    for a in aliases:
        if a in df.columns:
            return a
    # normalized match
    lookup = {norm_col(c): c for c in df.columns}
    for a in aliases:
        key = norm_col(a)
        if key in lookup:
            return lookup[key]
    return None


def auto_detect_survey_total_columns(survey: pd.DataFrame, already_selected: Sequence[str]) -> List[str]:
    already = set(already_selected)
    selected = []
    for col in survey.columns:
        c = str(col).strip()
        if c in already or c in AUTO_EXCLUDE_EXACT:
            continue
        if c == "Subject No.":
            continue
        if AUTO_EXCLUDE_REGEX.search(c):
            continue
        if not AUTO_INCLUDE_REGEX.search(c):
            continue
        x = safe_numeric(survey[c])
        # Total/scale score should be numeric and not almost empty.
        if x.notna().sum() >= 5 and x.nunique(dropna=True) >= 2:
            selected.append(c)
    return selected


def select_survey_total_columns(survey: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return survey subset with Subject No., metadata, and survey total/scale columns only."""
    output = pd.DataFrame({"Subject No.": survey["Subject No."]})
    rows = []

    # metadata
    for col in SURVEY_META_COLS:
        if col == "Subject No.":
            continue
        if col in survey.columns:
            out_col = f"{SURVEY_PREFIX}{col}"
            output[out_col] = survey[col]
            rows.append({
                "output_column": out_col,
                "source": "survey",
                "original_column": col,
                "role": "metadata",
                "selection_method": "metadata_list",
                "N_available": int(pd.Series(survey[col]).notna().sum()),
            })

    # explicit total score aliases
    explicit_originals = []
    for out_name, aliases in SURVEY_TOTAL_SPECS.items():
        src = find_alias_column(survey, aliases)
        if src is None:
            rows.append({
                "output_column": f"{SURVEY_PREFIX}{out_name}",
                "source": "survey",
                "original_column": "",
                "role": "survey_total_missing",
                "selection_method": "explicit_alias_not_found",
                "N_available": 0,
            })
            continue
        out_col = f"{SURVEY_PREFIX}{out_name}"
        output[out_col] = survey[src]
        explicit_originals.append(src)
        rows.append({
            "output_column": out_col,
            "source": "survey",
            "original_column": src,
            "role": "survey_total",
            "selection_method": "explicit_alias",
            "N_available": int(pd.Series(survey[src]).notna().sum()),
        })

    # optional auto-detected totals from survey only
    if not EXPLICIT_SURVEY_TOTALS_ONLY:
        auto_cols = auto_detect_survey_total_columns(survey, already_selected=explicit_originals + ["Subject No."] + SURVEY_META_COLS)
        existing_out_norm = {norm_col(c.replace(SURVEY_PREFIX, "")) for c in output.columns}
        for src in auto_cols:
            # Avoid adding a duplicate conceptual score if it normalizes to an existing output name.
            if norm_col(src) in existing_out_norm:
                continue
            out_col = f"{SURVEY_PREFIX}{src}"
            # If source column name somehow collides, add suffix.
            base = out_col
            k = 2
            while out_col in output.columns:
                out_col = f"{base}__dup{k}"
                k += 1
            output[out_col] = survey[src]
            rows.append({
                "output_column": out_col,
                "source": "survey",
                "original_column": src,
                "role": "survey_total_auto_detected",
                "selection_method": "auto_regex_total_scale",
                "N_available": int(pd.Series(survey[src]).notna().sum()),
            })

    selected_summary = pd.DataFrame(rows)
    return output, selected_summary


def prefix_non_key_columns(df: pd.DataFrame, prefix: str, source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep Subject No. as merge key; prefix every other column."""
    out = pd.DataFrame({"Subject No.": df["Subject No."]})
    rows = []
    for col in df.columns:
        if col == "Subject No.":
            continue
        out_col = f"{prefix}{col}"
        base = out_col
        k = 2
        while out_col in out.columns:
            out_col = f"{base}__dup{k}"
            k += 1
        out[out_col] = df[col]
        rows.append({
            "output_column": out_col,
            "source": source_name,
            "original_column": col,
            "role": "feature_or_metadata",
            "selection_method": "all_non_key_columns",
            "N_available": int(pd.Series(df[col]).notna().sum()),
        })
    return out, pd.DataFrame(rows)


def make_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_total = len(df)
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "N_total": n_total,
            "N_available": int(s.notna().sum()),
            "N_missing": int(s.isna().sum()),
            "availability_pct": float(s.notna().mean() * 100) if n_total else np.nan,
            "dtype": str(s.dtype),
        })
    return pd.DataFrame(rows)


def main() -> None:
    survey, imu, hrv, survey_path, imu_path, hrv_path = load_tables()

    survey_selected, survey_dict = select_survey_total_columns(survey)
    imu_prefixed, imu_dict = prefix_non_key_columns(imu, IMU_PREFIX, "imu")
    hrv_prefixed, hrv_dict = prefix_non_key_columns(hrv, HRV_PREFIX, "hrv")

    merged = survey_selected.merge(imu_prefixed, on="Subject No.", how="left")
    merged = merged.merge(hrv_prefixed, on="Subject No.", how="left")

    column_dict = pd.concat([survey_dict, imu_dict, hrv_dict], ignore_index=True)
    missingness = make_missingness_summary(merged)

    # Put Subject No. first; survey columns, then imu, then hrv.
    ordered_cols = ["Subject No."]
    ordered_cols += [c for c in merged.columns if c.startswith(SURVEY_PREFIX)]
    ordered_cols += [c for c in merged.columns if c.startswith(IMU_PREFIX)]
    ordered_cols += [c for c in merged.columns if c.startswith(HRV_PREFIX)]
    ordered_cols += [c for c in merged.columns if c not in ordered_cols]
    merged = merged[ordered_cols]

    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    column_dict.to_csv(OUT_COLUMN_DICT, index=False, encoding="utf-8-sig")
    missingness.to_csv(OUT_MISSINGNESS, index=False, encoding="utf-8-sig")
    survey_dict.to_csv(OUT_SELECTED_SURVEY, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="integrated", index=False)
        survey_dict.to_excel(writer, sheet_name="selected_survey_totals", index=False)
        column_dict.to_excel(writer, sheet_name="column_dictionary", index=False)
        missingness.to_excel(writer, sheet_name="missingness", index=False)

    survey_total_cols = [c for c in merged.columns if c.startswith(SURVEY_PREFIX) and c not in {f"{SURVEY_PREFIX}{x}" for x in ["Age", "Sex", "Sex_num", "institution", "site"]}]
    print("[DONE] Integrated CSV created.")
    print(f"[OUT] {OUT_CSV}")
    print(f"[OUT] {OUT_XLSX}")
    print(f"[ROWS] {len(merged)} participants")
    print(f"[COLS] total={len(merged.columns)}, survey_prefixed={sum(c.startswith(SURVEY_PREFIX) for c in merged.columns)}, imu_prefixed={sum(c.startswith(IMU_PREFIX) for c in merged.columns)}, hrv_prefixed={sum(c.startswith(HRV_PREFIX) for c in merged.columns)}")
    print("[SURVEY TOTALS SELECTED]")
    for c in survey_dict[survey_dict["role"].astype(str).str.contains("survey_total", na=False)]["output_column"].tolist():
        print(f"  - {c}")
    print("[NOTE] If auto-detected survey totals are still too many, set EXPLICIT_SURVEY_TOTALS_ONLY = True at the top of the script.")


if __name__ == "__main__":
    main()
