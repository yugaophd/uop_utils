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


def apply_workhorse_qc_flags(ds, var, var_name):
    """Apply Workhorse QC flags to a variable before averaging."""
    return apply_qc_flags(ds, var, var_name, flag_name='Workhorse_flag')


def apply_nortek_qc_flags(ds, var, var_name):
    """Apply Nortek QC flags to a variable before averaging."""
    print(f"[apply_nortek_qc_flags] Using flag: 'Nortek_flag' for variable: {var_name}")
    return apply_qc_flags(ds, var, var_name, flag_name='Nortek_flag')


def apply_qc_flags(ds, var, var_name, flag_name='Workhorse_flag'):
    """Apply a binary QC flag variable to a data variable before averaging."""
    if flag_name not in ds:
        print(f'No {flag_name} found in dataset for {var_name}')
        return var

    qc_flag = ds[flag_name]
    print(f'    Applying {flag_name} to {var_name}...')

    flag_binary = (qc_flag > 0).astype(int)

    if qc_flag.dims == var.dims:
        var_qc = var.where(flag_binary == 0, np.nan)
    elif len(var.dims) == 1 and var.dims[0] in qc_flag.dims and 'bin_depth' in str(qc_flag.dims):
        print(f'    Using surface-level flags for surface variable {var_name}')
        if 'bin_depth' in qc_flag.dims:
            surface_dim = next(dim for dim in qc_flag.dims if 'bin_depth' in dim)
            surface_flag = flag_binary.isel({surface_dim: 0})
            var_qc = var.where(surface_flag == 0, np.nan)
        else:
            var_qc = var
    else:
        print(f'    Warning: Dimension mismatch between {flag_name} and {var_name}.')
        print(f'    Flag dims: {qc_flag.dims}, Variable dims: {var.dims}')
        print('    Skipping QC for this variable.')
        return var

    original_nan_count = np.sum(np.isnan(var.values))
    flagged_nan_count = np.sum(np.isnan(var_qc.values))
    newly_flagged = flagged_nan_count - original_nan_count
    total_points = var.size

    print(f'    Applied {flag_name} to {var_name}:')
    print(f'      Original NaN count: {original_nan_count} ({original_nan_count / total_points * 100:.1f}%)')
    print(f'      After flagging: {flagged_nan_count} ({flagged_nan_count / total_points * 100:.1f}%)')
    print(f'      Newly flagged: {newly_flagged} ({newly_flagged / total_points * 100:.1f}%)')

    return var_qc
