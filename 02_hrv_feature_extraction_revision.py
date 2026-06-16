# -*- coding: utf-8 -*-
"""
02_hrv_feature_extraction_revision.py
=====================================

HRV-only feature extraction for revision.

Input
-----
- Raw CSV folder containing patient files.
- Subject ID should appear in path or filename as dXX-XXX.
- PPG/HRV signal is assumed to be in M column, index 12.

Output
------
- hrv_outputs/per_file_hrv_features.csv
- hrv_outputs/subject_hrv_features.xlsx
- hrv_outputs/subject_hrv_features.csv

Features
--------
- HR
- median HR
- SDNN
- RMSSD
- pNN50
- cleaned SDNN/RMSSD/pNN50
- artifact_drop_rate
- stress_index = z(HR) - z(RMSSD)

This script does NOT calculate IMU or survey scores.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# User settings
# =============================================================================
ROOT_DIR = r"D:\dBrain_server\patient_data_csv_0313"
OUT_DIR = Path("hrv_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 25.0
SIGNAL_COL_IDX = 12  # M column, 0-based

LOW_HZ = 0.5
HIGH_HZ = 4.0
ORDER = 3

MIN_HR = 40
MAX_HR = 200

PROMINENCE_FRAC = 0.30
MIN_WIDTH_SEC = 0.06

IBI_MIN_MS = 300.0
IBI_MAX_MS = 2000.0
IBI_MEDIAN_TOL_FRAC = 0.20
DELTA_ABS_MAX_MS = 80.0
DELTA_REL_MAX_FRAC = 0.20

SUBJECT_RE = re.compile(r"(?i)(d\d{2}-\d{3})")
FILE_DT_PAT = re.compile(r"(\d{4}-\d{2}-\d{2})[_\s](\d{2})[-:](\d{2})[-:](\d{2})")


def extract_subject_from_path(path: str) -> Optional[str]:
    m = None
    for m in SUBJECT_RE.finditer(str(path)):
        pass
    return m.group(1).lower() if m else None


def subject_to_group(subj: Optional[str]) -> Optional[str]:
    if not subj:
        return None
    return subj.split("-")[0].lower()


def parse_datetime_from_path(path: str) -> pd.Timestamp:
    m = FILE_DT_PAT.search(str(path))
    if not m:
        return pd.NaT
    return pd.to_datetime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", errors="coerce")


def read_csv_column(filepath: str, col_idx: int) -> np.ndarray:
    try:
        df = pd.read_csv(filepath, header=None, usecols=[col_idx], engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(filepath, header=0, usecols=[col_idx], engine="python", on_bad_lines="skip")
    sig = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    sig = sig[np.isfinite(sig)]
    if sig.size == 0:
        raise ValueError(f"No valid numeric data in column index {col_idx}")
    return sig


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    if high >= nyq:
        raise ValueError(f"high cutoff {high} >= Nyquist {nyq}")
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def detect_ppg_peaks(sig: np.ndarray, fs: float) -> np.ndarray:
    min_distance = int(max(1, round(fs * 60.0 / MAX_HR)))
    p5, p95 = np.nanpercentile(sig, [5, 95])
    prom = max(1e-6, PROMINENCE_FRAC * (p95 - p5))
    width = max(1, int(round(MIN_WIDTH_SEC * fs)))
    peaks, _ = find_peaks(sig, distance=min_distance, prominence=prom, width=width)
    return peaks


def clean_ibi(ibi_ms: np.ndarray) -> Tuple[np.ndarray, float]:
    if ibi_ms.size < 5:
        return ibi_ms, 0.0
    med = np.median(ibi_ms)
    keep1 = np.abs(ibi_ms - med) <= (IBI_MEDIAN_TOL_FRAC * med)
    ibi1 = ibi_ms[keep1]
    if ibi1.size < 5:
        drop_rate = 1.0 - (ibi1.size / max(1, ibi_ms.size))
        return ibi1, float(drop_rate)

    d = np.diff(ibi1)
    thr = np.maximum(DELTA_ABS_MAX_MS, DELTA_REL_MAX_FRAC * ibi1[1:])
    keep2 = np.concatenate([[True], np.abs(d) <= thr])
    ibi_clean = ibi1[keep2]
    drop_rate = 1.0 - (ibi_clean.size / max(1, ibi_ms.size))
    return ibi_clean, float(drop_rate)


def compute_hrv_from_peaks(peaks: np.ndarray, fs: float) -> Dict[str, float]:
    if peaks is None or len(peaks) < 3:
        return dict(
            n_peaks=0 if peaks is None else int(len(peaks)),
            n_ibi=0,
            HR=np.nan,
            median_hr=np.nan,
            SDNN=np.nan,
            RMSSD=np.nan,
            pNN50=np.nan,
            SDNN_clean=np.nan,
            RMSSD_clean=np.nan,
            pNN50_clean=np.nan,
            artifact_drop_rate=np.nan,
        )

    ibi_ms = np.diff(peaks) / fs * 1000.0
    mask = (ibi_ms >= IBI_MIN_MS) & (ibi_ms <= IBI_MAX_MS) & np.isfinite(ibi_ms)
    ibi_ms = ibi_ms[mask]

    if ibi_ms.size < 2:
        return dict(
            n_peaks=int(len(peaks)),
            n_ibi=int(ibi_ms.size),
            HR=np.nan,
            median_hr=np.nan,
            SDNN=np.nan,
            RMSSD=np.nan,
            pNN50=np.nan,
            SDNN_clean=np.nan,
            RMSSD_clean=np.nan,
            pNN50_clean=np.nan,
            artifact_drop_rate=1.0,
        )

    HR = 60000.0 / np.mean(ibi_ms)
    median_hr = 60000.0 / np.median(ibi_ms)
    SDNN = np.std(ibi_ms, ddof=1)
    diff_ms = np.diff(ibi_ms)
    RMSSD = np.sqrt(np.mean(diff_ms ** 2)) if diff_ms.size > 0 else np.nan
    pNN50 = np.mean(np.abs(diff_ms) > 50.0) if diff_ms.size > 0 else np.nan

    ibi_clean, drop_rate = clean_ibi(ibi_ms)
    if ibi_clean.size >= 2:
        SDNN_clean = float(np.std(ibi_clean, ddof=1))
        d_c = np.diff(ibi_clean)
        RMSSD_clean = float(np.sqrt(np.mean(d_c ** 2))) if d_c.size > 0 else np.nan
        pNN50_clean = float(np.mean(np.abs(d_c) > 50.0)) if d_c.size > 0 else np.nan
    else:
        SDNN_clean = RMSSD_clean = pNN50_clean = np.nan

    return dict(
        n_peaks=int(len(peaks)),
        n_ibi=int(ibi_ms.size),
        HR=float(HR),
        median_hr=float(median_hr),
        SDNN=float(SDNN),
        RMSSD=float(RMSSD),
        pNN50=float(pNN50) if np.isfinite(pNN50) else np.nan,
        SDNN_clean=SDNN_clean,
        RMSSD_clean=RMSSD_clean,
        pNN50_clean=pNN50_clean,
        artifact_drop_rate=float(drop_rate),
    )


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root}")

    csv_files = [str(p) for p in root.rglob("*.csv")]
    print(f"[INFO] CSV files found: {len(csv_files)}")

    per_file_rows = []
    for fpath in csv_files:
        subj = extract_subject_from_path(fpath)
        if not subj:
            continue

        try:
            sig = read_csv_column(fpath, SIGNAL_COL_IDX)
            if sig.size < 10:
                raise ValueError("Too few valid samples")

            sig_bp = bandpass_filter(sig, FS, LOW_HZ, HIGH_HZ, ORDER)
            peaks = detect_ppg_peaks(sig_bp, FS)
            metrics = compute_hrv_from_peaks(peaks, FS)

            valid = 1 if np.isfinite(metrics.get("RMSSD_clean", np.nan)) or np.isfinite(metrics.get("RMSSD", np.nan)) else 0

            rec = {
                "Subject No.": subj,
                "institution": subject_to_group(subj),
                "file_path": fpath,
                "file_datetime": parse_datetime_from_path(fpath),
                "duration_sec": len(sig) / FS,
                "hrv_valid": valid,
                **metrics,
                "error": "",
            }
        except Exception as e:
            rec = {
                "Subject No.": subj,
                "institution": subject_to_group(subj),
                "file_path": fpath,
                "file_datetime": parse_datetime_from_path(fpath),
                "duration_sec": np.nan,
                "hrv_valid": 0,
                "HR": np.nan,
                "median_hr": np.nan,
                "SDNN": np.nan,
                "RMSSD": np.nan,
                "pNN50": np.nan,
                "SDNN_clean": np.nan,
                "RMSSD_clean": np.nan,
                "pNN50_clean": np.nan,
                "artifact_drop_rate": np.nan,
                "error": str(e),
            }
        per_file_rows.append(rec)

    per_file = pd.DataFrame(per_file_rows)
    per_file.to_csv(OUT_DIR / "per_file_hrv_features.csv", index=False, encoding="utf-8-sig")

    valid = per_file[per_file["hrv_valid"] == 1].copy()
    if valid.empty:
        print("[WARN] No valid HRV files.")
        return

    subject = valid.groupby("Subject No.").agg(
        institution=("institution", "first"),
        hrv_n_files_valid=("file_path", "count"),
        hrv_total_duration_sec=("duration_sec", "sum"),
        HR=("HR", "mean"),
        median_hr=("median_hr", "mean"),
        RMSSD=("RMSSD_clean", "mean"),
        SDNN=("SDNN_clean", "mean"),
        pNN50=("pNN50_clean", "mean"),
        artifact_drop_rate=("artifact_drop_rate", "mean"),
    ).reset_index()

    # Stress index: z(HR) - z(RMSSD)
    hr_std = subject["HR"].std(ddof=1)
    rmssd_std = subject["RMSSD"].std(ddof=1)
    hr_std = 1 if pd.isna(hr_std) or hr_std == 0 else hr_std
    rmssd_std = 1 if pd.isna(rmssd_std) or rmssd_std == 0 else rmssd_std
    subject["z_hr"] = (subject["HR"] - subject["HR"].mean()) / hr_std
    subject["z_rmssd"] = (subject["RMSSD"] - subject["RMSSD"].mean()) / rmssd_std
    subject["stress_index"] = subject["z_hr"] - subject["z_rmssd"]

    subject.to_csv(OUT_DIR / "subject_hrv_features.csv", index=False, encoding="utf-8-sig")
    subject.to_excel(OUT_DIR / "subject_hrv_features.xlsx", index=False)

    print("DONE")
    print(f"- Per-file HRV:     {OUT_DIR / 'per_file_hrv_features.csv'}")
    print(f"- Subject HRV CSV:  {OUT_DIR / 'subject_hrv_features.csv'}")
    print(f"- Subject HRV XLSX: {OUT_DIR / 'subject_hrv_features.xlsx'}")


if __name__ == "__main__":
    main()
