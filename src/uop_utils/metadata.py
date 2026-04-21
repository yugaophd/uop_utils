"""Metadata and provenance helpers."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from importlib import metadata as importlib_metadata
from typing import Optional

import numpy as np


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


def get_git_governance_info(repo_path: Optional[str] = None, script_path: Optional[str] = None, status_limit: int = 20) -> dict:
    """Collect git provenance metadata for processing-governance records."""
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


def write_git_provenance(out_path):
    """Write a LaTeX snippet with the current git commit info to out_path."""
    try:
        commit_meta = subprocess.check_output(['git', 'log', '-1', '--format=%h  %ci'], text=True).strip()
        commit_subject = subprocess.check_output(['git', 'log', '-1', '--format=%s'], text=True).strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()
        remote_url = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], text=True).strip()
    except subprocess.CalledProcessError:
        commit_meta, commit_subject, branch, remote_url = 'unavailable', '', 'unavailable', 'unavailable'

    uop_coare_details = get_uop_coare_details()
    uop_coare_version = uop_coare_details['version']
    if not uop_coare_version:
        raise RuntimeError('Unable to determine required uop-coare version. Install/import uop_coare before running processing.')

    def escape(s):
        return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#')

    lines = [
        '\\subsection{Processing Provenance}\n\n',
        '\\begin{description}\n',
        f'  \\item[Git Repository] \\url{{{remote_url}}}\n',
        f'  \\item[Branch] {escape(branch)}\n',
        f'  \\item[Commit] \\texttt{{{escape(commit_meta)}}}\n',
        f'  \\item[Message] {escape(commit_subject)}\n',
        f'  \\item[uop-coare] \\texttt{{{escape(str(uop_coare_version))}}}\n',
        '\\end{description}\n',
    ]
    with open(out_path, 'w') as f:
        f.writelines(lines)
    print(f'Wrote git provenance to {out_path}')


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
