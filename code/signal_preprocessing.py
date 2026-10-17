"""Shared signal preprocessing utilities for dataset loaders."""

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline


def resample_signal_cubic(x: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if original_fs == target_fs:
        return x.copy()
    t_old = np.arange(len(x)) / original_fs
    duration = (len(x) - 1) / original_fs
    n_new = int(round(duration * target_fs)) + 1
    t_new = np.arange(n_new) / target_fs
    cs = CubicSpline(t_old, x)
    return cs(t_new)


def fir_bandpass_ecg(x: np.ndarray, fs: int, low: float = 0.5, high: float = 45.0, numtaps: int = 401):
    nyq = fs / 2.0
    taps = signal.firwin(numtaps, [low / nyq, high / nyq], pass_zero=False)
    return signal.filtfilt(taps, [1.0], x)


def butter_bandpass_ppg(x: np.ndarray, fs: int, low: float = 0.5, high: float = 8.0, order: int = 4):
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def median_denoise(x: np.ndarray, kernel_size: int = 5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return signal.medfilt(x, kernel_size=kernel_size)


def minmax_normalize_minus1_1(x: np.ndarray, eps: float = 1e-8):
    xmin = np.min(x)
    xmax = np.max(x)
    return 2.0 * (x - xmin) / (xmax - xmin + eps) - 1.0, xmin, xmax


def invert_minmax_minus1_1(x_norm: np.ndarray, xmin: float, xmax: float):
    return ((x_norm + 1.0) / 2.0) * (xmax - xmin) + xmin


def segment_with_overlap(x: np.ndarray, window_size: int, overlap_ratio: float = 0.2):
    step = max(1, int(window_size * (1.0 - overlap_ratio)))
    segments = []
    for start in range(0, len(x) - window_size + 1, step):
        segments.append(x[start:start + window_size])
    return np.stack(segments, axis=0) if segments else np.empty((0, window_size))
