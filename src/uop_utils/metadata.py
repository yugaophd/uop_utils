"""Metadata and provenance helpers."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from importlib import metadata as importlib_metadata
from typing import Optional

import numpy as np
import xarray as xr


def get_uop_coare_details():
    """Return version and import details for uop-coare."""
    details = {
        'version': None,
        'dist_name': None,
        'dist_path': None,
        'module_path': None,
        'python_executable': os.sys.executable,
    }

    for dist_name in ('uop-coare', 'uop_coare'):
        try:
            dist = importlib_metadata.distribution(dist_name)
            details['version'] = dist.version
            details['dist_name'] = dist.metadata['Name'] or dist_name
            details['dist_path'] = str(dist.locate_file(''))
            break
        except importlib_metadata.PackageNotFoundError:
            continue

    try:
        import uop_coare
        details['module_path'] = getattr(uop_coare, '__file__', None)
        if not details['version']:
            details['version'] = getattr(uop_coare, '__version__', None)
    except Exception:
        pass

    return details


def get_uop_coare_version():
    """Return the installed uop-coare version string if available."""
    return get_uop_coare_details()['version']


def get_uop_utils_details():
    """Return version and import details for uop-utils. Mirrors get_uop_coare_details()."""
    details = {
        'version': None,
        'dist_name': None,
        'dist_path': None,
        'module_path': None,
        'python_executable': os.sys.executable,
    }
    for dist_name in ('uop-utils', 'uop_utils'):
        try:
            dist = importlib_metadata.distribution(dist_name)
            details['version'] = dist.version
            details['dist_name'] = dist.metadata['Name'] or dist_name
            details['dist_path'] = str(dist.locate_file(''))
            break
        except importlib_metadata.PackageNotFoundError:
            continue
    try:
        import uop_utils
        details['module_path'] = getattr(uop_utils, '__file__', None)
        if not details['version']:
            details['version'] = getattr(uop_utils, '__version__', None)
    except Exception:
        pass
    return details


def compute_practical_salinity_from_conductivity(
    conductivity,
    temperature,
    depth_m,
    latitude,
    conductivity_units: str = 'S m-1',
):
    """Compute practical salinity and pressure from conductivity using TEOS-10.

    Parameters
    ----------
    conductivity : array-like
        Conductivity values.
    temperature : array-like
        In-situ temperature in degrees C.
    depth_m : array-like
        Depth in meters (positive down).
    latitude : array-like
        Latitude in degrees north.
    conductivity_units : str, optional
        Conductivity units. Supported values are S m-1 (or equivalent) and mS/cm.

    Returns
    -------
    salinity_psu : np.ndarray
        Practical salinity (unitless, often reported as PSU).
    pressure_dbar : np.ndarray
        Sea pressure in dbar computed from depth and latitude.
    conductivity_mscm : np.ndarray
        Conductivity converted to mS/cm used for salinity calculation.
    """
    try:
        import gsw
    except ImportError as e:
        raise ImportError('gsw is required for conductivity-to-salinity conversion.') from e

    conductivity_arr = np.asarray(conductivity, dtype=float)
    temperature_arr = np.asarray(temperature, dtype=float)
    depth_arr = np.asarray(depth_m, dtype=float)
    latitude_arr = np.asarray(latitude, dtype=float)
    units_norm = str(conductivity_units or '').strip().lower()

    if units_norm in {'s m-1', 's/m', 'siemens/m', 'siemens per meter'}:
        conductivity_mscm = conductivity_arr * 10.0
    elif units_norm in {'ms cm-1', 'ms/cm'}:
        conductivity_mscm = conductivity_arr
    else:
        raise ValueError(
            f'Unsupported conductivity units: "{conductivity_units}". '
            'Expected S m-1 (or equivalent) or mS/cm.'
        )

    pressure_dbar = gsw.p_from_z(-np.abs(depth_arr), latitude_arr)
    salinity_psu = gsw.SP_from_C(conductivity_mscm, temperature_arr, pressure_dbar)

    return salinity_psu, pressure_dbar, conductivity_mscm


def get_git_provenance_info(repo_path: Optional[str] = None, script_path: Optional[str] = None, status_limit: int = 20) -> dict:
    """Collect git provenance metadata for processing records."""
    repo_path = os.path.abspath(repo_path or os.getcwd())
    info = {
        'git_metadata_available': 0,
        'git_repo_path_requested': repo_path,
    }
    script_abs = None
    if script_path is not None:
        script_abs = os.path.abspath(script_path)
        info['processing_script'] = os.path.basename(script_abs)
        info['processing_script_path'] = script_abs

    def _run_git(args, cwd):
        result = subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True)
        return result.stdout.strip()

    try:
        repo_root = _run_git(['rev-parse', '--show-toplevel'], repo_path)
        info['git_repo_root'] = repo_root
        info['git_branch'] = _run_git(['branch', '--show-current'], repo_root)
        info['git_commit'] = _run_git(['rev-parse', 'HEAD'], repo_root)
        info['git_commit_short'] = _run_git(['rev-parse', '--short', 'HEAD'], repo_root)
        try:
            info['git_remote_origin_url'] = _run_git(['config', '--get', 'remote.origin.url'], repo_root)
        except Exception:
            info['git_remote_origin_url'] = ''
        try:
            info['git_describe'] = _run_git(['describe', '--always', '--dirty', '--tags'], repo_root)
        except Exception:
            info['git_describe'] = info['git_commit_short']
        status_text = _run_git(['status', '--short'], repo_root)
        status_lines = [line for line in status_text.splitlines() if line.strip()]
        info['git_is_dirty'] = int(len(status_lines) > 0)
        info['git_status'] = '; '.join(status_lines[:status_limit])
        info['git_metadata_available'] = 1
        if script_abs is not None:
            try:
                info['processing_script_relative_path'] = os.path.relpath(script_abs, repo_root)
            except Exception:
                info['processing_script_relative_path'] = script_abs
    except Exception as e:
        info['git_error'] = str(e)
    return info


def write_git_provenance(git_attrs_or_path, output_path=None):
    """Write git and library provenance to a LaTeX file.

    Supports two call styles::

        # 1-arg (S-MODE style) — git info collected internally from CWD
        write_git_provenance('/path/to/git_provenance.tex')

        # 2-arg — reuse a pre-built dict from get_git_provenance_info()
        write_git_provenance(git_attrs, '/path/to/git_provenance.tex')

    In both cases the output contains three subsections: the processing-repo
    git provenance, uop-coare package details, and uop-utils package details.

    Parameters
    ----------
    git_attrs_or_path : dict or str
        When called with two arguments, a dict returned by
        ``get_git_provenance_info()``.  When called with one argument, the
        output file path (git info is collected internally from the CWD).
    output_path : str, optional
        Destination ``.tex`` file path (only used in the 2-arg form).
    """
    if output_path is None:
        # 1-arg call: git_attrs_or_path is the output path; collect git info now
        out_path = git_attrs_or_path
        git_attrs = get_git_provenance_info()
    else:
        out_path = output_path
        git_attrs = git_attrs_or_path

    def escape(s):
        return str(s).replace('_', '\\_').replace('%', '\\%').replace('&', '\\&').replace('#', '\\#')

    def make_table(title, rows):
        block = [
            f'\\subsection*{{{title}}}',
            '\\begin{tabular}{ll}',
            '\\hline',
            '\\textbf{Field} & \\textbf{Value} \\\\',
            '\\hline',
        ]
        for label, value in rows:
            block.append(f'{label} & \\texttt{{{escape(value)}}} \\\\')
        block.append('\\hline')
        block.append('\\end{tabular}')
        block.append('')
        return block

    # --- Processing repo git provenance ---
    git_field_labels = [
        ('git_branch',        'Branch'),
        ('git_commit_full',   'Commit (full)'),
        ('git_commit_short',  'Commit (short)'),
        ('git_is_dirty',      'Working tree dirty'),
        ('git_script_path',   'Script path'),
        ('git_tag',           'Tag'),
    ]
    git_rows = [(label, git_attrs[key]) for key, label in git_field_labels if git_attrs.get(key) is not None]
    lines = make_table('Git Provenance', git_rows)

    # --- uop-coare library ---
    uop_coare_d = get_uop_coare_details()
    coare_rows = [
        ('Package', uop_coare_d.get('dist_name') or 'uop-coare'),
        ('Version', uop_coare_d.get('version') or 'unknown'),
        ('Install path', uop_coare_d.get('dist_path') or 'unknown'),
        ('Module path', uop_coare_d.get('module_path') or 'unknown'),
    ]
    lines += make_table('uop-coare', coare_rows)

    # --- uop-utils library ---
    uop_utils_d = get_uop_utils_details()
    utils_rows = [
        ('Package', uop_utils_d.get('dist_name') or 'uop-utils'),
        ('Version', uop_utils_d.get('version') or 'unknown'),
        ('Install path', uop_utils_d.get('dist_path') or 'unknown'),
        ('Module path', uop_utils_d.get('module_path') or 'unknown'),
    ]
    lines += make_table('uop-utils', utils_rows)

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')

    print(f'Git provenance LaTeX written: {out_path}')


def validate_time_range(start_date, end_date, time_coverage_start, max_year_difference=1):
    """Validate and adjust a date range against a dataset coverage start time."""
    try:
        start_date_obj = start_date.astype('M8[ms]').astype(datetime)
        end_date_obj = end_date.astype('M8[ms]').astype(datetime)
        coverage_start_obj = datetime.strptime(time_coverage_start, '%Y-%m-%d %H:%M:%S')
        if start_date_obj > end_date_obj:
            raise ValueError('The start_date must be earlier than the end_date.')
        coverage_year = coverage_start_obj.year
        start_year_diff = abs(start_date_obj.year - coverage_year)
        end_year_diff = abs(end_date_obj.year - coverage_year)
        if start_year_diff > max_year_difference or end_year_diff > max_year_difference:
            raise ValueError(f'Time range ({start_date} to {end_date}) is too far removed from the coverage start {time_coverage_start}.')
        if not (start_date_obj <= coverage_start_obj <= end_date_obj):
            adjusted_start_date = coverage_start_obj.date()
            adjusted_end_date = coverage_start_obj.date() + timedelta(days=364)
            return adjusted_start_date.strftime('%Y-%m-%d'), adjusted_end_date.strftime('%Y-%m-%d')
        print('Time range is valid.')
        return start_date, end_date
    except Exception as e:
        raise ValueError(f'Error in validating or adjusting the time range: {e}')


def add_metadata_comments(ds_resampled):
    """Add explanatory comments to empty variables and coordinate gaps."""
    print('Adding metadata comments for clarity...')
    for var_name in ds_resampled.data_vars:
        if var_name.startswith('INST_'):
            continue
        is_empty = False
        try:
            if hasattr(ds_resampled[var_name], 'values') and ds_resampled[var_name].size > 0:
                if np.isnan(ds_resampled[var_name].values).all():
                    is_empty = True
            elif ds_resampled[var_name].size == 0:
                is_empty = True
        except Exception:
            pass
        if is_empty:
            ds_resampled[var_name].attrs['comment'] = 'This variable contains no data because the instrument was not working during this deployment or was temporarily disabled.'
            print(f'Added empty variable comment to: {var_name}')
    coord_vars = ['latitude', 'longitude', 'wave_latitude', 'wave_longitude']
    gap_comment = 'Gaps in the data may occur when instruments were temporarily turned off due to lack of solar power. This does not indicate separate deployments.'
    for coord in coord_vars:
        if coord in ds_resampled.coords:
            if 'comment' in ds_resampled[coord].attrs:
                ds_resampled[coord].attrs['comment'] += ' ' + gap_comment
            else:
                ds_resampled[coord].attrs['comment'] = gap_comment
            print(f'Added gap explanation comment to: {coord}')
    return ds_resampled


def fix_waveglider_issues(ds, waveglider_name):
    """Apply waveglider-specific metadata fixes."""
    print(f'Applying waveglider-specific fixes for {waveglider_name}...')
    if 'coordinates' in ds.attrs:
        del ds.attrs['coordinates']
        print("Removed invalid 'coordinates' global attribute")
    for var_name in ['Workhorse_vel_Z', 'Workhorse_degraded_vel_Z']:
        if var_name in ds and 'standard_name' in ds[var_name].attrs:
            if ds[var_name].attrs['standard_name'] == 'z_sea_water_velocity':
                del ds[var_name].attrs['standard_name']
                print(f'Removed invalid standard_name from {var_name}')
    for var_name in ds.variables:
        if var_name.startswith('Workhorse_degraded_corr_') and 'units' in ds[var_name].attrs:
            if ds[var_name].attrs['units'] == '%':
                ds[var_name].attrs['units'] = 'A.U.'
                print(f'Changed units from % to A.U. for {var_name}')
                ds[var_name].attrs['comment'] = 'Correlation values range from 0 to 255, representing signal quality.'
    cdom_std_name = 'concentration_of_colored_dissolved_organic_matter_in_sea_water_expressed_as_equivalent_mass_fraction_of_quinine_sulfate_dihydrate'
    for var_name in ds.variables:
        if 'CDOM' in var_name.upper() and 'standard_name' not in ds[var_name].attrs:
            ds[var_name].attrs['standard_name'] = cdom_std_name
            print(f'Added CF standard name to {var_name}')
    if waveglider_name == 'WHOI43':
        for coord in ['latitude', 'longitude']:
            if coord in ds and 'instrument' in ds[coord].attrs:
                if ds[coord].attrs['instrument'] == 'INST_WXT':
                    ds[coord].attrs['instrument'] = 'INST_SITEX'
                    print(f'Fixed {coord} instrument attribute: INST_WXT → INST_SITEX')
        for var_name in ds.variables:
            if var_name.startswith('rbr_') and 'instrument' in ds[var_name].attrs:
                if ds[var_name].attrs['instrument'] == 'INST_RBR_Concerto':
                    ds[var_name].attrs['instrument'] = 'INST_RBR_Concerto_WHOI'
                    print(f'Fixed instrument reference for {var_name}')
    return ds


def fix_L2_metadata(ds, campaign_name):
    """Normalize L2 metadata for ASTraL waveglider outputs."""
    attrs = ds.attrs.copy()

    smode_attrs_to_remove = [
        'DOI',
        'id',
        'metadata_link',
        'acknowledgement',
        'program',
        'publisher_name',
        'publisher_email',
        'publisher_url',
        'publisher_type',
        'publisher_institution',
        'contributor_name',
        'contributor_role',
    ]

    for attr in smode_attrs_to_remove:
        if attr in attrs:
            del attrs[attr]

    attrs['project'] = 'Air-Sea Transfer in Rapid Atmosphere-Land transitions (ASTraL)'
    attrs['title'] = f'ASTraL 2024 Waveglider Observations, {campaign_name}, Bay of Bengal 2024'

    if 'summary' in attrs:
        attrs['summary'] = (
            f'Wave Glider {campaign_name} was deployed as part of the ASTraL (Air-Sea Transfer '
            f'in Rapid Atmosphere-Land transitions) Field Campaign in the Bay of Bengal during 2024. '
            f'This wave glider is equipped with oceanographic and meteorological instruments including '
            f'ADCP, GPS/IMU, GPCTD, Vaisala WXT536 weather station, shortwave and longwave radiometers, '
            f'and temperature/humidity probes for air-sea interaction studies.'
        )

    attrs['institution'] = 'Woods Hole Oceanographic Institution (WHOI) and Scripps Institution of Oceanography (SIO)'
    attrs['creator_institution'] = 'WHOI and SIO'
    attrs['sea_name'] = 'Indian Ocean'
    attrs['license'] = 'Data is freely available under Creative Commons CC BY 4.0'
    attrs['processing_level'] = 'L2'
    attrs['creator_name'] = 'Tom Farrar and Yu Gao'
    attrs['creator_email'] = 'jfarrar@whoi.edu, yu.gao@whoi.edu'
    attrs['comment'] = f'ASTraL project L2 data for {campaign_name} wave glider processed and converted to NetCDF.'


def append_history(existing_history, new_message):
    """Append a message to a CF-style history string."""
    if existing_history:
        return existing_history + ' ' + new_message
    return new_message


def build_data_array(values, time_values, lat, lon, attrs):
    """Build a time-indexed xarray DataArray with lat/lon as time-dependent coordinates."""
    clean_attrs = {key: value for key, value in attrs.items() if value not in (None, '')}
    return xr.DataArray(
        values,
        dims=['time'],
        coords={
            'time': time_values,
            'lat': ('time', lat),
            'lon': ('time', lon),
        },
        attrs=clean_attrs,
    )

    return attrs
