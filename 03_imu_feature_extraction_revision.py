# -*- coding: utf-8 -*-
"""
03_imu_feature_extraction_revision.py
=====================================

IMU-only feature extraction for revision.

Input
-----
- Raw CSV folder containing patient files.
- Subject ID should appear in path or filename as dXX-XXX.
- ACC columns are assumed to be 1,2,3 columns -> index [0,1,2].
- Gravity columns are assumed to be 9,10,11 columns -> index [8,9,10].

Output
------
- imu_outputs/per_file_imu_features.csv
- imu_outputs/subject_imu_features.xlsx
- imu_outputs/subject_imu_features.csv

Features
--------
- step_count
- Steps/min
- cadence_spm
- estimated Walking Speed
- estimated step length
- ENMO
- MAD
- active_minutes
- MVPA percentage

This script does NOT calculate HRV or survey scores.

Important
---------
Walking Speed is an acceleration-derived estimate. In the paper, describe it as
estimated walking speed or acceleration-derived walking speed, not lab-grade gait speed.
"""

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
OUT_DIR = Path("imu_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 100.0

IDX_ACC = [0, 1, 2]      # 1,2,3 columns
IDX_GRV = [8, 9, 10]     # 9,10,11 columns

G_CONST = 9.80665

BP_LOW = 0.7
BP_HIGH = 3.0
BP_ORDER = 4

APPLY_LP_FOR_ENMO = True
LP_CUTOFF_HZ = 20.0

HP_CUTOFF_HZ = 0.25
HP_ORDER = 2

MIN_STEP_S = 0.30
ACTIVE_MIN_STEP_TH = 20

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


def read_cols_fixed(fp: str, idx_list: List[int]) -> np.ndarray:
    try:
        df = pd.read_csv(fp, header=None, usecols=idx_list, engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(fp, header=0, usecols=idx_list, engine="python", on_bad_lines="skip")
    df = df.apply(pd.to_numeric, errors="coerce")
    arr = df.to_numpy(dtype="float64", copy=False)
    mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[mask]
    if arr.shape[0] == 0:
        raise ValueError(f"No valid numeric rows for columns {idx_list}")
    return arr


def butter_filter(x: np.ndarray, fs: float, cutoff, btype: str, order: int = 4):
    nyq = 0.5 * fs
    wn = [c / nyq for c in cutoff] if isinstance(cutoff, (list, tuple, np.ndarray)) else cutoff / nyq
    if isinstance(wn, list):
        if any(w >= 1 for w in wn):
            raise ValueError("Cutoff/Nyquist >= 1.0")
    else:
        if wn >= 1:
            raise ValueError("Cutoff/Nyquist >= 1.0")
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, x)


def lowpass_nd(X, fs, cutoff, order=4):
    if X.ndim == 1:
        return butter_filter(X, fs, cutoff, "low", order)
    Y = np.empty_like(X, dtype=float)
    for c in range(X.shape[1]):
        Y[:, c] = butter_filter(X[:, c], fs, cutoff, "low", order)
    return Y


def highpass_nd(X, fs, cutoff, order=2):
    if X.ndim == 1:
        return butter_filter(X, fs, cutoff, "high", order)
    Y = np.empty_like(X, dtype=float)
    for c in range(X.shape[1]):
        Y[:, c] = butter_filter(X[:, c], fs, cutoff, "high", order)
    return Y


def to_g_units(acc: np.ndarray, grav: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if grav is None:
        acc_mag = np.linalg.norm(acc, axis=1)
        med = np.nanmedian(acc_mag) if acc_mag.size else np.nan
        scale = G_CONST if (np.isfinite(med) and 7.0 <= med <= 12.0) else 1.0
        return acc / scale, None

    grav_mag = np.linalg.norm(grav, axis=1)
    med = np.nanmedian(grav_mag) if grav_mag.size else np.nan
    scale = G_CONST if (np.isfinite(med) and 7.0 <= med <= 12.0) else 1.0
    return acc / scale, grav / scale


def detect_steps_from_mag(mag_bp: np.ndarray, fs: float) -> np.ndarray:
    min_distance = int(round(fs * MIN_STEP_S))
    q25, q75 = np.nanpercentile(mag_bp, [25, 75])
    prom = max(1e-6, 0.2 * (q75 - q25))
    peaks, _ = find_peaks(mag_bp, distance=max(1, min_distance), prominence=prom)
    return peaks


def estimate_walking_speed(linacc_g: np.ndarray, cadence_spm: float, fs: float) -> Tuple[float, float]:
    if not np.isfinite(cadence_spm) or cadence_spm == 0 or linacc_g.shape[0] < fs:
        return np.nan, np.nan

    acc_mag_var = np.var(np.linalg.norm(linacc_g, axis=1))

    # Empirical approximation.
    # Keep this as an estimated speed, not validated gait-lab speed.
    K = 0.98
    step_length_m = K * (acc_mag_var ** 0.25)
    cadence_sps = cadence_spm / 60.0
    walking_speed_mps = step_length_m * cadence_sps

    return float(walking_speed_mps), float(step_length_m)


def compute_activity_metrics(linacc_g: np.ndarray, peaks_idx: np.ndarray) -> Dict[str, float]:
    N = linacc_g.shape[0]
    duration_sec = N / FS

    step_count = int(len(peaks_idx))
    cadence_spm = (step_count / duration_sec) * 60.0 if duration_sec > 0 else np.nan
    walking_speed_mps, step_length_m = estimate_walking_speed(linacc_g, cadence_spm, FS)

    linacc_proc = lowpass_nd(linacc_g, FS, LP_CUTOFF_HZ, order=4) if APPLY_LP_FOR_ENMO else linacc_g
    mag = np.linalg.norm(linacc_proc, axis=1)
    enmo = np.maximum(mag - 1.0, 0.0)
    enmo_mean = float(np.nanmean(enmo)) if enmo.size else np.nan
    mad = float(np.nanmean(np.abs(mag - np.nanmean(mag)))) if mag.size else np.nan

    total_minutes = int(np.ceil(duration_sec / 60.0))
    active_minutes = 0
    if total_minutes > 0 and len(peaks_idx) > 0:
        mins = (peaks_idx / FS) // 60
        step_per_min = np.bincount(mins.astype(int), minlength=total_minutes)
        active_minutes = int(np.sum(step_per_min >= ACTIVE_MIN_STEP_TH))

    return dict(
        duration_sec=float(duration_sec),
        step_count=step_count,
        cadence_spm=float(cadence_spm),
        walking_speed_mps=walking_speed_mps,
        estimated_step_length_m=step_length_m,
        enmo_mean_g=enmo_mean,
        mad_g=mad,
        active_minutes=active_minutes,
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
            acc = read_cols_fixed(fpath, IDX_ACC)
            try:
                grav = read_cols_fixed(fpath, IDX_GRV)
            except Exception:
                grav = None

            N = len(acc) if grav is None else min(len(acc), len(grav))
            if N < int(5 * FS):
                raise ValueError("Too few IMU samples")

            acc = acc[:N]
            if grav is not None:
                grav = grav[:N]

            acc_g, grav_g = to_g_units(acc, grav)
            if grav_g is not None:
                linacc_g = acc_g - grav_g
                mode = "subtract_grav"
            else:
                linacc_g = highpass_nd(acc_g, FS, HP_CUTOFF_HZ, order=HP_ORDER)
                mode = "highpass_acc"

            mag = np.linalg.norm(linacc_g, axis=1)
            mag_bp = butter_filter(mag, FS, [BP_LOW, BP_HIGH], "band", BP_ORDER)
            peaks = detect_steps_from_mag(mag_bp, FS)
            metrics = compute_activity_metrics(linacc_g, peaks)

            rec = {
                "Subject No.": subj,
                "institution": subject_to_group(subj),
                "file_path": fpath,
                "file_datetime": parse_datetime_from_path(fpath),
                "imu_valid": 1,
                "mode": mode,
                **metrics,
                "error": "",
            }
        except Exception as e:
            rec = {
                "Subject No.": subj,
                "institution": subject_to_group(subj),
                "file_path": fpath,
                "file_datetime": parse_datetime_from_path(fpath),
                "imu_valid": 0,
                "mode": "error",
                "duration_sec": np.nan,
                "step_count": np.nan,
                "cadence_spm": np.nan,
                "walking_speed_mps": np.nan,
                "estimated_step_length_m": np.nan,
                "enmo_mean_g": np.nan,
                "mad_g": np.nan,
                "active_minutes": np.nan,
                "error": str(e),
            }
        per_file_rows.append(rec)

    per_file = pd.DataFrame(per_file_rows)
    per_file.to_csv(OUT_DIR / "per_file_imu_features.csv", index=False, encoding="utf-8-sig")

    valid = per_file[per_file["imu_valid"] == 1].copy()
    if valid.empty:
        print("[WARN] No valid IMU files.")
        return

    valid["duration_x_cadence"] = valid["duration_sec"] * valid["cadence_spm"]
    valid["duration_x_speed"] = valid["duration_sec"] * valid["walking_speed_mps"]

    subject = valid.groupby("Subject No.").agg(
        institution=("institution", "first"),
        imu_n_files_valid=("file_path", "count"),
        imu_total_duration_sec=("duration_sec", "sum"),
        total_steps=("step_count", "sum"),
        active_minutes=("active_minutes", "sum"),
        cadence_num=("duration_x_cadence", "sum"),
        speed_num=("duration_x_speed", "sum"),
        enmo_mean_g=("enmo_mean_g", "mean"),
        mad_g=("mad_g", "mean"),
    ).reset_index()

    subject["cadence_spm"] = subject["cadence_num"] / subject["imu_total_duration_sec"]
    subject["walking_speed_mps"] = subject["speed_num"] / subject["imu_total_duration_sec"]
    subject["Steps/min"] = subject["total_steps"] / (subject["imu_total_duration_sec"] / 60.0)
    subject["MVPA"] = subject["active_minutes"] / (subject["imu_total_duration_sec"] / 60.0) * 100.0
    subject["Walking Speed"] = subject["walking_speed_mps"]

    subject = subject.drop(columns=["cadence_num", "speed_num"])

    subject.to_csv(OUT_DIR / "subject_imu_features.csv", index=False, encoding="utf-8-sig")
    subject.to_excel(OUT_DIR / "subject_imu_features.xlsx", index=False)

    print("DONE")
    print(f"- Per-file IMU:     {OUT_DIR / 'per_file_imu_features.csv'}")
    print(f"- Subject IMU CSV:  {OUT_DIR / 'subject_imu_features.csv'}")
    print(f"- Subject IMU XLSX: {OUT_DIR / 'subject_imu_features.xlsx'}")


if __name__ == "__main__":
    main()
