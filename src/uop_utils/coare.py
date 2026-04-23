"""COARE-processing helpers shared across project scripts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr


def load_processing_config(config_path, section=None):
    """Load a JSON processing configuration, optionally returning one section."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if section is None:
        return config
    return config.get(section, {})


def add_scalar_input_variable(ds_output, name, value, attrs):
    """Add a scalar COARE input/provenance value as a dataset variable."""
    data = np.asarray(value)
    if data.dtype.kind in {"U", "S", "O"}:
        data = np.asarray(str(value), dtype=str)
    ds_output[name] = xr.DataArray(data, attrs=attrs)


def annotate_reference_height_outputs(ds_output, zrf_u, zrf_t, zrf_q):
    """Update reference-height output metadata to reflect the actual run settings."""
    variable_height_map = {
        "wind_speed_at_reference_height": (
            zrf_u,
            f"wind speed at {zrf_u:.1f} m",
            "COARE wind-speed reference height used to compute this profile-adjusted output",
        ),
        "neutral_wind_speed_at_reference_height": (
            zrf_u,
            f"neutral wind speed at {zrf_u:.1f} m",
            "COARE wind-speed reference height used to compute this neutral profile-adjusted output",
        ),
        "air_temperature_at_reference_height": (
            zrf_t,
            f"air temperature at {zrf_t:.1f} m",
            "COARE air-temperature reference height used to compute this profile-adjusted output",
        ),
        "neutral_air_temperature_at_reference_height": (
            zrf_t,
            f"neutral value of air temperature at {zrf_t:.1f} m",
            "COARE air-temperature reference height used to compute this neutral profile-adjusted output",
        ),
        "specific_humidity_at_reference_height": (
            zrf_q,
            f"air specific humidity at {zrf_q:.1f} m",
            "COARE humidity reference height used to compute this profile-adjusted output",
        ),
        "neutral_specific_humidity_at_reference_height": (
            zrf_q,
            f"neutral value of air specific humidity at {zrf_q:.1f} m",
            "COARE humidity reference height used to compute this neutral profile-adjusted output",
        ),
        "relative_humidity_at_reference_height": (
            zrf_q,
            f"air relative humidity at {zrf_q:.1f} m",
            "COARE humidity reference height used to compute this profile-adjusted output",
        ),
        "air_density_at_reference_height": (
            zrf_t,
            f"air density at {zrf_t:.1f} m",
            "COARE reference height associated with this profile-adjusted air-density output",
        ),
    }

    for var_name, (height, long_name, comment) in variable_height_map.items():
        if var_name not in ds_output:
            continue
        ds_output[var_name].attrs["long_name"] = long_name
        ds_output[var_name].attrs["reference_height_m"] = float(height)
        ds_output[var_name].attrs["comment"] = comment


def process_surface_current(ds, depth=4, max_gap_hours=2):
    """Interpolate surface current at the requested depth for COARE processing."""
    print(f"Processing surface current at {depth}m depth (20-min averages)")

    if "Workhorse_vel_east" not in ds or "Workhorse_vel_north" not in ds:
        print("Surface current data not available")
        return None, None, None

    vel_x = ds.Workhorse_vel_east
    vel_y = ds.Workhorse_vel_north

    if "Workhorse_bin_depth" not in ds.Workhorse_vel_east.coords:
        print("Depth information not available in current data")
        return None, None, None

    depth_data = ds.Workhorse_vel_east.Workhorse_bin_depth.data
    depth_idx = np.argmin(np.abs(depth_data - depth))
    actual_depth = depth_data[depth_idx]
    print(f"Using depth {actual_depth}m (index: {depth_idx})")

    u_vel = vel_x.isel(Workhorse_range=depth_idx)
    v_vel = vel_y.isel(Workhorse_range=depth_idx)
    valid_mask = ~np.isnan(u_vel) & ~np.isnan(v_vel)

    if np.sum(valid_mask) < 3:
        print(f"Too few valid surface current points: {np.sum(valid_mask)}")
        return None, None, None

    print(f"Interpolating gaps less than {max_gap_hours} hours for 20-min data...")
    time_vals = u_vel.time.values
    time_diffs = np.diff(time_vals)
    median_timestep = np.median(time_diffs).astype("timedelta64[s]").astype(float)
    points_per_gap = int(max_gap_hours * 3600 / median_timestep)

    df = pd.DataFrame({"u": u_vel.values, "v": v_vel.values}, index=pd.DatetimeIndex(time_vals))
    df_interpolated = df.interpolate(method="linear", limit=points_per_gap)

    u_interp_count = np.sum(np.isnan(df["u"]) & ~np.isnan(df_interpolated["u"]))
    v_interp_count = np.sum(np.isnan(df["v"]) & ~np.isnan(df_interpolated["v"]))
    print(f"Interpolated {u_interp_count} points in U velocity (20-min)")
    print(f"Interpolated {v_interp_count} points in V velocity (20-min)")

    return df_interpolated["u"].values, df_interpolated["v"].values, time_vals


def process_radiation(datasets, freq="20min"):
    """Average radiation data across platforms on a common time grid."""
    print(f"Processing and averaging radiation data across platforms ({freq} output)")

    common_start = pd.Timestamp(min(ds.time.min().item() for ds in datasets.values())).floor(freq)
    common_end = pd.Timestamp(max(ds.time.max().item() for ds in datasets.values())).ceil(freq)
    common_time = pd.date_range(start=common_start, end=common_end, freq=freq)

    aligned_datasets = {}
    sw_fluxes = []
    lw_fluxes = []

    for name, ds in datasets.items():
        ds_time = pd.to_datetime(ds["time"].values)
        data_vars = {}

        if "SMP21_shortwave_flux" in ds.variables:
            sw_series = pd.Series(ds["SMP21_shortwave_flux"].values, index=ds_time).resample(freq).mean()
            sw_reindexed = sw_series.reindex(common_time)
            sw_fluxes.append(xr.DataArray(sw_reindexed.values, coords={"time": common_time}, dims=["time"]))
            data_vars["SMP21_shortwave_flux"] = ("time", sw_reindexed.values)

        lw_var = None
        if "SGR4_longwave_flux" in ds.variables:
            lw_var = "SGR4_longwave_flux"
        elif "IR02_longwave_flux" in ds.variables:
            lw_var = "IR02_longwave_flux"

        if lw_var is not None:
            lw_series = pd.Series(ds[lw_var].values, index=ds_time).resample(freq).mean()
            lw_reindexed = lw_series.reindex(common_time)
            lw_fluxes.append(xr.DataArray(lw_reindexed.values, coords={"time": common_time}, dims=["time"]))
            data_vars[lw_var] = ("time", lw_reindexed.values)

        ds_light = xr.Dataset(data_vars=data_vars, coords={"time": common_time}, attrs=ds.attrs.copy())
        aligned_datasets[name] = ds_light

    sw_flux_mean = xr.concat(sw_fluxes, dim="dataset").mean(dim="dataset", skipna=True) if sw_fluxes else None
    lw_flux_mean = xr.concat(lw_fluxes, dim="dataset").mean(dim="dataset", skipna=True) if lw_fluxes else None

    sw_valid = int(np.sum(np.isfinite(sw_flux_mean.values))) if sw_flux_mean is not None else 0
    lw_valid = int(np.sum(np.isfinite(lw_flux_mean.values))) if lw_flux_mean is not None else 0
    print(f"Processed radiation data ({freq}): SW flux has {sw_valid} valid points")
    print(f"Processed radiation data ({freq}): LW flux has {lw_valid} valid points")

    return sw_flux_mean, lw_flux_mean, aligned_datasets


def resample_dataset(ds, freq="20min"):
    """Resample a time-series dataset while preserving key coordinate metadata."""
    print(f"Resampling dataset to {freq} averages...")
    ds_resampled = ds.resample(time=freq).mean(skipna=True)

    for coord_name in ["latitude", "longitude"]:
        if coord_name in ds.coords and coord_name not in ds_resampled.coords:
            coord_resampled = ds[coord_name].resample(time=freq).mean(skipna=True)
            ds_resampled = ds_resampled.assign_coords({coord_name: coord_resampled})
            print(f"Restored {coord_name} coordinate after resampling")

    ds_resampled.attrs = ds.attrs.copy()
    ds_resampled.attrs["temporal_resolution"] = "20 minutes"
    ds_resampled.attrs["resampling_note"] = (
        "Resampled from 5-minute to 20-minute averages before flux calculations"
    )

    for var_name in ds_resampled.data_vars:
        if var_name in ds.data_vars:
            ds_resampled[var_name].attrs = ds[var_name].attrs.copy()
            ds_resampled[var_name].attrs["processing_note"] = "20-minute average of 5-minute L3 data"

    print(f"Original dataset: {len(ds.time)} points")
    print(f"Resampled dataset: {len(ds_resampled.time)} points")
    return ds_resampled


def write_coare_scalar_latex_table(scalar_rows, output_path):
    """Write per-platform scalar COARE inputs as a LaTeX table."""
    if not scalar_rows:
        print("No scalar COARE inputs available for LaTeX table.")
        return

    header = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Scalar inputs to \\texttt{coare36vn\\_zrf\\_et} for each Wave Glider.}\n"
        "\\small\n"
        "\\begin{tabular}{lccccccccc}\n"
        "\\hline\n"
        "Wave Glider & $z_u$ & $z_t$ & $z_q$ & $z_i$ & $z_{rf,u}$ & $z_{rf,t}$ & $z_{rf,q}$ & coolskin & albedo \\\\\n"
        "\\hline\n"
    )

    lines = [header]
    for row in scalar_rows:
        lines.append(
            f"{row['waveglider_name']} & "
            f"{row['zu']:.2f} & {row['zt']:.2f} & {row['zq']:.2f} & "
            f"{row['zi']:.1f} & {row['zrf_u']:.1f} & {row['zrf_t']:.1f} & {row['zrf_q']:.1f} & "
            f"{str(row['coolskin'])} & {row['albedo']} \\\\\n"
        )

    lines.append("\\hline\n\\end{tabular}\n\\end{table}\n")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved COARE scalar-input LaTeX table: {output_path}")
