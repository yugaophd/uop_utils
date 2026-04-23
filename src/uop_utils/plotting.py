"""Plotting and LaTeX-report helpers."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from math import ceil


def create_custom_figure_latex(image_path, caption, label, placement='!htbp', width='0.96\\textwidth'):
    """Create LaTeX content for a figure using a fully qualified image path."""
    return (
        f"\\begin{{figure}}[{placement}]\n"
        f"    \\centering\n"
        f"    \\includegraphics[width={width}]{{{image_path}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        f"\\end{{figure}}\n\n"
    )


def create_figure_latex(base_image_dir, filename, group_name, freq='5-min', placement='!htbp', width='0.96\\textwidth'):
    """Create LaTeX content for a figure."""
    caption_text = filename.replace('.png', '').replace('_', ' ')
    print(caption_text)
    return create_custom_figure_latex(
        image_path=f"{base_image_dir}/{group_name}/{filename}",
        caption=f"Comparison of original and {freq} averaged data for {caption_text}.",
        label=f"fig:{filename.replace('.png', '')}",
        placement=placement,
        width=width,
    )


def create_two_figure_page_latex(base_image_dir, items, group_name, freq='5-min', placement='H', width='0.76\\textwidth'):
    """Create one LaTeX float containing up to two stacked figures."""
    lines = [f"\\begin{{figure}}[{placement}]\n", "    \centering\n"]
    for index, filename in enumerate(items):
        caption_text = filename.replace('.png', '').replace('_', ' ')
        print(caption_text)
        lines.append(f"    \\includegraphics[width={width}]{{{base_image_dir}/{group_name}/{filename}}}\n")
        lines.append(f"    \\caption{{Comparison of original and {freq} averaged data for {caption_text}.}}\n")
        lines.append(f"    \\label{{fig:{filename.replace('.png', '')}}}\n")
        if index != len(items) - 1:
            lines.append("    \\vspace{0.8em}\n")
    lines.append("\\end{figure}\n\n")
    return ''.join(lines)


def build_gap_aware_series(time_values, data_values, gap_factor=1.5):
    """Insert NaN/NaT separators so matplotlib breaks lines across large time gaps."""
    time_arr = np.asarray(time_values)
    value_arr = np.asarray(data_values)
    if time_arr.ndim != 1 or value_arr.ndim != 1 or len(time_arr) != len(value_arr):
        return time_arr, value_arr
    if len(time_arr) < 2:
        return time_arr, value_arr
    if np.issubdtype(time_arr.dtype, np.datetime64):
        numeric_time = time_arr.astype('datetime64[ns]').astype(np.int64)
        gap_marker = np.datetime64('NaT')
    else:
        try:
            numeric_time = time_arr.astype(float)
            gap_marker = np.nan
        except (TypeError, ValueError):
            return time_arr, value_arr
    diffs = np.diff(numeric_time)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return time_arr, value_arr
    nominal_step = np.median(positive_diffs)
    if nominal_step <= 0:
        return time_arr, value_arr
    gap_mask = diffs > nominal_step * gap_factor
    if not np.any(gap_mask):
        return time_arr, value_arr
    gap_times = []
    gap_values = []
    for index, (time_value, data_value) in enumerate(zip(time_arr, value_arr)):
        gap_times.append(time_value)
        gap_values.append(data_value)
        if index < len(gap_mask) and gap_mask[index]:
            gap_times.append(gap_marker)
            gap_values.append(np.nan)
    return np.asarray(gap_times), np.asarray(gap_values)


def plot_coare_input_multipanel(input_series, output_path):
    """Create one multipanel plot with all Wave Glider input time series."""
    if not input_series:
        print("No COARE input time series available for plotting.")
        return

    variables = ['u', 't', 'rh', 'P', 'ts', 'rain', 'Ss', 'lat', 'lon']
    ylabels = {
        'u': 'Wind speed (m s$^{-1}$)',
        't': 'Air temperature (degC)',
        'rh': 'Relative humidity (%)',
        'P': 'Pressure (hPa)',
        'ts': 'Sea temperature (degC)',
        'rain': 'Rain rate',
        'Ss': 'Salinity (PSU)',
        'lat': 'Latitude (deg)',
        'lon': 'Longitude (deg)',
    }
    titles = {
        'u': 'Wind speed',
        't': 'Air temperature',
        'rh': 'Relative humidity',
        'P': 'Pressure',
        'ts': 'Sea temperature',
        'rain': 'Rain rate',
        'Ss': 'Salinity',
        'lat': 'Latitude',
        'lon': 'Longitude',
    }

    ncols = 2
    nrows = ceil(len(variables) / ncols)
    panel_width = 6.0
    panel_height = 3.0
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=True,
    )
    axes = np.array(axes).reshape(-1)
    date_locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    date_formatter = mdates.ConciseDateFormatter(date_locator)

    for i, var in enumerate(variables):
        ax = axes[i]
        for entry in input_series:
            if var not in entry or len(entry[var]) == 0:
                continue

            time_values, data_values = build_gap_aware_series(entry['time'], entry[var])
            if np.any(np.isfinite(np.asarray(data_values, dtype=float))):
                ax.plot(
                    time_values,
                    data_values,
                    linewidth=0.9,
                    alpha=0.9,
                    label=entry['waveglider_name'],
                )

        ax.set_title(titles.get(var, var))
        ax.set_ylabel(ylabels.get(var, var))
        ax.grid(True)
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.tick_params(axis='x', labelrotation=30)
        if i >= len(variables) - ncols:
            ax.set_xlabel('Time')
        else:
            ax.tick_params(axis='x', labelbottom=False)

    for j in range(len(variables), len(axes)):
        axes[j].set_visible(False)

    legend_panel_var = 'lat'
    legend_idx = variables.index(legend_panel_var)
    legend_ax = axes[legend_idx]
    handles, labels = legend_ax.get_legend_handles_labels()
    if handles:
        nlegend_cols = 1 if len(labels) <= 4 else 2
        legend_ax.legend(
            handles,
            labels,
            loc='upper right',
            fontsize=7,
            frameon=True,
            ncol=nlegend_cols,
            borderaxespad=0.3,
            handlelength=1.4,
            labelspacing=0.25,
        )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved COARE input multipanel figure: {output_path}")
