"""Read and preprocess the Who is Alyx? Empatica / Polar recordings.

Loader contract: return (ppg_segments, ecg_segments, subject_ids), where the
signals are aligned, normalized float32 arrays of shape (N, T).
This module has no dependency on the training models or their configuration.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from signal_preprocessing import butter_bandpass_ppg, fir_bandpass_ecg, segment_with_overlap


def default_data_dir() -> Path:
    data_root = Path(__file__).resolve().parents[1] / "data"
    unversioned = data_root / "who-is-alyx"
    return unversioned if unversioned.is_dir() else data_root / "who-is-alyx-2.1"


def read_alyx_signal(path: Path, *, native_sampling_rate: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Read timestamp -> [value] or Alyx index -> [value, receipt time] JSON.

    Indexed Alyx recordings use the native sample clock, anchored at the first
    sample's receipt time. Receipt times describe Bluetooth delivery batches,
    not individual acquisition times. Missing indices remain gaps; sample loss
    without missing indices cannot be recovered from this format.
    Numeric Unix times may use s/ms/us/ns.
    ISO timestamps are also accepted; naive ISO times must share a timezone.
    """
    with path.open(encoding="utf-8") as stream:
        samples = json.load(stream)["data"]
    if not isinstance(samples, dict) or len(samples) < 2:
        raise ValueError(f"{path}: expected at least two timestamped samples")
    try:
        timestamps = np.array([float(key) for key in samples], dtype=np.float64)
    except ValueError:
        timestamps = np.array([
            datetime.fromisoformat(key.replace("Z", "+00:00")).timestamp()
            for key in samples
        ])
    else:
        first_sample = next(iter(samples.values()))
        indexed = (isinstance(first_sample, list) and len(first_sample) >= 2
                   and isinstance(first_sample[1], str)
                   and np.all(timestamps == np.floor(timestamps))
                   and np.all((timestamps >= 0) & (timestamps < 1e11)))
        if indexed:
            if native_sampling_rate is None:
                native_sampling_rate = {"ECG": 130, "PPG": 64}.get(path.stem)
            if native_sampling_rate is None or not np.isfinite(native_sampling_rate) or native_sampling_rate <= 0:
                raise ValueError(f"{path}: indexed samples require a positive native_sampling_rate")
            first_index = int(np.argmin(timestamps))
            first_time = list(samples.values())[first_index][1]
            origin = datetime.fromisoformat(first_time.replace("Z", "+00:00")).timestamp()
            timestamps = origin + (timestamps - timestamps[first_index]) / native_sampling_rate
        else:
            magnitude = np.median(np.abs(timestamps))
            divisor = 1e9 if magnitude >= 1e17 else 1e6 if magnitude >= 1e14 else 1e3 if magnitude >= 1e11 else 1
            timestamps /= divisor
    values = np.array([
        value[0] if isinstance(value, list) else value for value in samples.values()
    ], dtype=np.float64)
    if not np.isfinite(timestamps).all() or not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite timestamps or samples")
    order = np.argsort(timestamps, kind="stable")
    timestamps, values = timestamps[order], values[order]
    timestamps, unique_indices = np.unique(timestamps, return_index=True)
    if len(timestamps) < 2:
        raise ValueError(f"{path}: fewer than two distinct timestamps")
    return timestamps, values[unique_indices]


def load_alyx_segments(data_dir: Optional[Path] = None, *, sampling_rate: int = 130,
                       window_seconds: float = 4.0, overlap_ratio: float = 0.2):

    """Align sessions on a common clock, filter continuous runs, then window.

    Windows never cross sessions or gaps exceeding three native sample periods.
    ECG is recorded at 130 Hz and Empatica PPG at 64 Hz.
    """
    data_dir = default_data_dir() if data_dir is None else Path(data_dir)
    window_size = int(round(window_seconds * sampling_rate))
    if sampling_rate <= 90 or window_size < 16 or not 0 <= overlap_ratio < 1:
        raise ValueError("Require sampling_rate > 90, window >= 16 samples and 0 <= overlap < 1")
    players = data_dir / "players"
    if not players.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {players}")
    ppg_parts, ecg_parts, subjects = [], [], []
    pairs_found = 0
    for ecg_path in sorted(players.glob("*/*/polar-h10/ECG.json")):
        session_dir = ecg_path.parent.parent
        ppg_path = session_dir / "empatica" / "PPG.json"
        if not ppg_path.is_file():
            continue
        pairs_found += 1
        ecg_time, ecg = read_alyx_signal(ecg_path, native_sampling_rate=130)
        ppg_time, ppg = read_alyx_signal(ppg_path, native_sampling_rate=64)
        start, end = max(ecg_time[0], ppg_time[0]), min(ecg_time[-1], ppg_time[-1])
        if end <= start:
            warnings.warn(f"No common signal interval: {session_dir}")
            continue
        # Work relative to the overlap start to preserve timestamp precision.
        ecg_time, ppg_time = ecg_time - start, ppg_time - start
        grid = np.arange(int(np.floor((end - start) * sampling_rate)) + 1) / sampling_rate
        valid = np.ones(len(grid), dtype=bool)
        for timestamps, native_fs in ((ecg_time, 130), (ppg_time, 64)):
            for gap in np.flatnonzero(np.diff(timestamps) > 3.0 / native_fs):
                left = np.searchsorted(grid, timestamps[gap], side="right")
                right = np.searchsorted(grid, timestamps[gap + 1], side="left")
                valid[left:right] = False
        aligned_ecg = np.interp(grid, ecg_time, ecg)
        aligned_ppg = np.interp(grid, ppg_time, ppg)
        edges = np.diff(np.r_[False, valid, False].astype(np.int8))
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            # filtfilt's 401-tap ECG filter requires more than 1203 samples.
            if right - left < max(window_size, 1204):
                continue
            clean_ecg = fir_bandpass_ecg(aligned_ecg[left:right], sampling_rate)
            clean_ppg = butter_bandpass_ppg(aligned_ppg[left:right], sampling_rate)
            ecg_windows = segment_with_overlap(clean_ecg, window_size, overlap_ratio)
            ppg_windows = segment_with_overlap(clean_ppg, window_size, overlap_ratio)
            usable = (np.ptp(ecg_windows, axis=1) > 1e-8) & (np.ptp(ppg_windows, axis=1) > 1e-8)
            ecg_windows, ppg_windows = ecg_windows[usable], ppg_windows[usable]
            if not len(ecg_windows):
                continue
            def normalize(windows):
                low = windows.min(axis=1, keepdims=True)
                return (2 * (windows - low) / np.ptp(windows, axis=1, keepdims=True) - 1).astype(np.float32)
            ecg_parts.append(normalize(ecg_windows))
            ppg_parts.append(normalize(ppg_windows))
            subjects.extend([int(session_dir.parent.name)] * len(ecg_windows))
    if not pairs_found:
        raise FileNotFoundError(
            f"No PPG.json / ECG.json pairs in {players}. If only *.dvc files exist, "
            f"retrieve the physiological files with DVC from {data_dir} before training."
        )
    if not ecg_parts:
        raise ValueError("No usable paired windows; check timestamps, gaps and recording lengths")
    return np.concatenate(ppg_parts), np.concatenate(ecg_parts), np.asarray(subjects)
