"""Geometric and directional-data helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr


def calculate_wind_stress_components(stress_magnitude, wind_direction_degrees):
    """Calculate eastward and northward wind-stress components from magnitude and direction."""
    theta = np.radians(wind_direction_degrees)
    tau_x = -1 * stress_magnitude * np.sin(theta)
    tau_y = -1 * stress_magnitude * np.cos(theta)
    return tau_x, tau_y


def calculate_angle_between_vectors(wind_u, wind_v, curr_u, curr_v):
    """Calculate signed angle between wind and current vectors in degrees."""
    wind_mag = np.sqrt(wind_u**2 + wind_v**2)
    curr_mag = np.sqrt(curr_u**2 + curr_v**2)
    dot_product = wind_u * curr_u + wind_v * curr_v
    angle_rad = np.arccos(np.clip(dot_product / (wind_mag * curr_mag), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    cross_product = wind_v * curr_u - wind_u * curr_v
    angle_deg = np.where(cross_product < 0, -angle_deg, angle_deg)
    return angle_deg


def compute_current_relative_wind(wind_speed, wind_direction_deg, current_east, current_north):
    """Return wind adjusted for surface current where current is valid."""
    wind_speed = np.asarray(wind_speed, dtype=float)
    wind_direction_deg = np.asarray(wind_direction_deg, dtype=float)
    current_east = np.asarray(current_east, dtype=float)
    current_north = np.asarray(current_north, dtype=float)

    wind_dir_rad = np.radians(wind_direction_deg)
    wind_east = -wind_speed * np.sin(wind_dir_rad)
    wind_north = -wind_speed * np.cos(wind_dir_rad)

    rel_east = wind_east.copy()
    rel_north = wind_north.copy()

    valid_current = np.isfinite(current_east) & np.isfinite(current_north)
    rel_east[valid_current] = wind_east[valid_current] - current_east[valid_current]
    rel_north[valid_current] = wind_north[valid_current] - current_north[valid_current]

    rel_speed = np.sqrt(rel_east ** 2 + rel_north ** 2)
    rel_direction = np.degrees(np.arctan2(-rel_east, -rel_north)) % 360
    return rel_speed, rel_direction, valid_current


def vector_to_met_direction(u_component, v_component):
    """Convert vector components to meteorological direction (degrees from which flow comes)."""
    return (np.degrees(np.arctan2(-u_component, -v_component)) + 360.0) % 360.0


def circular_mean_resample(da, time_dim, freq='5min'):
    """Resample directional data using circular averaging."""
    rad_data = np.deg2rad(da)
    complex_data = np.exp(1j * rad_data)
    resampled_complex = complex_data.resample({time_dim: freq}).mean(skipna=True)
    resampled_angles = np.rad2deg(np.angle(resampled_complex))
    resampled_angles = np.where(resampled_angles < 0, resampled_angles + 360, resampled_angles)
    result = xr.DataArray(
        resampled_angles,
        coords=resampled_complex.coords,
        dims=resampled_complex.dims,
        attrs=da.attrs,
    )
    return result


def is_directional_variable(var_name, var_data=None):
    """Check whether a variable likely contains directional data."""
    if var_data is not None and hasattr(var_data, 'attrs') and 'units' in var_data.attrs:
        units = var_data.attrs['units'].lower()
        if 'degree' in units:
            temperature_indicators = ['_c', 'c', 'celsius', 'centigrade']
            is_temperature = any(temp in units for temp in temperature_indicators)
            if not is_temperature:
                return True
    directional_keywords = [
        'direction', 'dir', 'heading', 'bearing', 'azimuth', 'course',
        'wind_dir', 'wave_dir', 'current_dir', 'flow_dir', 'pitch', 'roll'
    ]
    var_name_lower = var_name.lower()
    return any(keyword in var_name_lower for keyword in directional_keywords)
