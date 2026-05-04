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


def format_compact_date_labels(fig, axes):
    """Show the year on the first visible tick and month-day on later ticks."""
    axes = np.atleast_1d(axes).flatten()
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)

    for ax in axes:
        if ax.get_visible():
            ax.xaxis.set_major_locator(locator)

    fig.canvas.draw()

    for ax in axes:
        if not ax.get_visible():
            continue

        tick_positions = ax.get_xticks()
        x_min, x_max = ax.get_xlim()
        visible_ticks = [tick for tick in tick_positions if x_min <= tick <= x_max]
        if not visible_ticks:
            continue

        first_visible_tick = visible_ticks[0]
        labels = []
        for tick in tick_positions:
            if not (x_min <= tick <= x_max):
                labels.append('')
                continue
            tick_dt = mdates.num2date(tick)
            labels.append(tick_dt.strftime('%Y-%m-%d' if tick == first_visible_tick else '%m-%d'))

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(labels)


def plot_coare_input_multipanel(input_series, output_path, variables=None):
    """Create one multipanel plot with all Wave Glider input time series.
    
    Parameters
    ----------
    input_series : list of dict
        List of dictionaries containing time series data for each platform.
    output_path : str
        File path where the figure will be saved.
    variables : list of str, optional
        List of variable names to plot. If None, uses default list.
        Default: ['u', 't', 'rh', 'P', 'ts', 'rain', 'Ss', 'lat', 'lon']
    """
    if not input_series:
        print("No COARE input time series available for plotting.")
        return

    if variables is None:
        variables = ['u', 'wind_direction', 't', 'rh', 'P', 'ts', 'rain', 'Ss', 'lat', 'lon']
    
    all_ylabels = {
        'u': 'Wind speed (m s$^{-1}$)',
        'wind_direction': 'Wind direction (deg)',
        't': 'Air temperature (degC)',
        'rh': 'Relative humidity (%)',
        'P': 'Pressure (hPa)',
        'ts': 'Sea temperature (degC)',
        'rain': 'Rain rate',
        'Ss': 'Salinity (PSU)',
        'sw_dn': 'Shortwave radiation (W m$^{-2}$)',
        'lw_dn': 'Longwave radiation (W m$^{-2}$)',
        'lat': 'Latitude (deg)',
        'lon': 'Longitude (deg)',
    }
    all_titles = {
        'u': 'Wind speed',
        'wind_direction': 'Wind direction',
        't': 'Air temperature',
        'rh': 'Relative humidity',
        'P': 'Pressure',
        'ts': 'Sea temperature',
        'rain': 'Rain rate',
        'Ss': 'Salinity',
        'sw_dn': 'Shortwave radiation',
        'lw_dn': 'Longwave radiation',
        'lat': 'Latitude',
        'lon': 'Longitude',
    }
    
    ylabels = {k: all_ylabels.get(k, k) for k in variables}
    titles = {k: all_titles.get(k, k) for k in variables}

    # Compute global time range across all entries and all plotted variables
    global_t_min = None
    global_t_max = None
    for entry in input_series:
        for var in variables:
            if var not in entry or len(entry[var]) == 0:
                continue
            t = np.asarray(entry['time'])
            if np.issubdtype(t.dtype, np.datetime64):
                t_num = mdates.date2num(t.astype('datetime64[ms]').astype(object))
            else:
                try:
                    t_num = t.astype(float)
                except (TypeError, ValueError):
                    continue
            t_min = np.nanmin(t_num)
            t_max = np.nanmax(t_num)
            if global_t_min is None or t_min < global_t_min:
                global_t_min = t_min
            if global_t_max is None or t_max > global_t_max:
                global_t_max = t_max

    ncols = 2
    nrows = ceil(len(variables) / ncols)
    panel_width = 6.0
    panel_height = 3.0
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        sharex=False,
    )
    axes = np.array(axes).reshape(-1)

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
        date_locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        date_formatter = mdates.ConciseDateFormatter(date_locator)
        ax.xaxis.set_major_locator(date_locator)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', labelbottom=True, labelrotation=30)
        if global_t_min is not None and global_t_max is not None:
            ax.set_xlim(global_t_min, global_t_max)

    for j in range(len(variables), len(axes)):
        axes[j].set_visible(False)

    # Place legend on the first panel at the best location
    legend_ax = axes[0]
    handles, labels = legend_ax.get_legend_handles_labels()
    if handles:
        nlegend_cols = 1 if len(labels) <= 4 else 2
        legend_ax.legend(
            handles,
            labels,
            loc='best',
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
