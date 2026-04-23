"""Shared utilities for UOP data-processing workflows."""

from .coare import (
    add_scalar_input_variable,
    annotate_reference_height_outputs,
    load_processing_config,
    process_radiation,
    process_surface_current,
    resample_dataset,
    write_coare_scalar_latex_table,
)
from .geo import (
    calculate_angle_between_vectors,
    calculate_wind_stress_components,
    circular_mean_resample,
    compute_current_relative_wind,
    is_directional_variable,
)
from .io import create_dir, load_config, update_encoding
from .metadata import (
    add_metadata_comments,
    fix_waveglider_issues,
    get_git_governance_info,
    get_uop_coare_details,
    get_uop_coare_version,
    validate_time_range,
    write_git_provenance,
)
from .naming import extract_campaign_names
from .plotting import (
    build_gap_aware_series,
    create_custom_figure_latex,
    create_figure_latex,
    create_two_figure_page_latex,
    plot_coare_input_multipanel,
)
from .qc import remove_spikes

__all__ = [
    'add_metadata_comments',
    'add_scalar_input_variable',
    'annotate_reference_height_outputs',
    'build_gap_aware_series',
    'calculate_angle_between_vectors',
    'calculate_wind_stress_components',
    'circular_mean_resample',
    'compute_current_relative_wind',
    'create_custom_figure_latex',
    'create_dir',
    'load_config',
    'create_figure_latex',
    'create_two_figure_page_latex',
    'extract_campaign_names',
    'plot_coare_input_multipanel',
    'fix_waveglider_issues',
    'get_git_governance_info',
    'get_uop_coare_details',
    'get_uop_coare_version',
    'is_directional_variable',
    'load_processing_config',
    'process_radiation',
    'process_surface_current',
    'remove_spikes',
    'resample_dataset',
    'update_encoding',
    'validate_time_range',
    'write_coare_scalar_latex_table',
    'write_git_provenance',
]

__version__ = '0.1.0'
