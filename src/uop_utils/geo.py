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
