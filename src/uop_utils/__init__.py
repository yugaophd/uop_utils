"""Shared utilities for UOP data-processing workflows."""

from .geo import (
    calculate_angle_between_vectors,
    calculate_wind_stress_components,
    circular_mean_resample,
    is_directional_variable,
)
from .io import create_dir, update_encoding
from .metadata import get_git_governance_info
from .naming import extract_campaign_names
from .qc import remove_spikes

__all__ = [
    'calculate_angle_between_vectors',
    'calculate_wind_stress_components',
    'circular_mean_resample',
    'create_dir',
    'extract_campaign_names',
    'get_git_governance_info',
    'is_directional_variable',
    'remove_spikes',
    'update_encoding',
]

__version__ = '0.1.0'
