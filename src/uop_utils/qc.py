"""Quality-control helpers."""

from __future__ import annotations

import numpy as np
from scipy import stats


def remove_spikes(data, time, window_size=12, threshold=3.0):
    """Remove spikes from a 1-D array using a moving robust-MAD threshold."""
    cleaned_data = np.copy(data)
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1

    if len(data) <= window_size:
        print('Warning: Data length is less than or equal to window size. Returning original data.')
        return data

    half_window = window_size // 2

    for i in range(len(data)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(data), i + half_window + 1)

        if i > start_idx and i < end_idx - 1:
            window_data = np.concatenate([data[start_idx:i], data[i + 1:end_idx]])
        elif i == start_idx and i < end_idx - 1:
            window_data = data[i + 1:end_idx]
        elif i > start_idx and i >= end_idx - 1:
            window_data = data[start_idx:i]
        else:
            continue

        window_data = window_data[~np.isnan(window_data)]
        if len(window_data) < 3:
            continue

        window_median = np.median(window_data)
        window_mad = stats.median_abs_deviation(window_data, nan_policy='omit')
        robust_std = 1.4826 * window_mad

        if not np.isnan(data[i]) and abs(data[i] - window_median) > threshold * robust_std:
            cleaned_data[i] = np.nan

    return cleaned_data
