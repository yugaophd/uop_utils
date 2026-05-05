"""I/O and filesystem helpers."""

from __future__ import annotations

import json
import os

import xarray as xr


def create_dir(directory):
    """Create a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def clear_png_files(directory):
    """Remove all .png files in a directory."""
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)
            print(f'Removed old plot: {filepath}')


def update_encoding(ds):
    """Update xarray Dataset encoding for compressed NetCDF output."""
    encoding = {}
    encoding_keys = (
        'shuffle', 'zlib', 'szip_coding', 'endian', 'szip_pixels_per_block',
        'contiguous', 'fletcher32', 'blosc_shuffle', 'quantize_mode', 'complevel',
        '_FillValue', 'chunksizes', 'least_significant_digit', 'dtype', 'compression',
        'significant_digits', 'scale_factor', 'add_offset'
    )

    for data_var in ds.data_vars:
        encoding[data_var] = {
            key: value for key, value in ds[data_var].encoding.items() if key in encoding_keys
        }
        encoding[data_var].update(zlib=True, complevel=4)

    for coord in ds.coords:
        if coord not in encoding:
            encoding[coord] = {
                key: value for key, value in ds[coord].encoding.items() if key in encoding_keys
            }
            encoding[coord].update(zlib=True, complevel=4)

    return encoding


def copy_inst_variables_to_data_vars(data_vars, source_ds, source_name, time_coord=None):
    """Copy all INST_ variables from source_ds into an output data_vars dictionary.

    Variables that include a time dimension are reindexed to time_coord when provided.
    Name collisions are resolved by inserting source_name while preserving the INST_ prefix.
    """
    inst_var_names = [name for name in source_ds.data_vars if name.startswith('INST_')]

    for inst_name in inst_var_names:
        inst_da = source_ds[inst_name]

        if time_coord is not None and 'time' in inst_da.dims:
            inst_da = inst_da.reindex(time=time_coord)

        output_name = inst_name
        if output_name in data_vars:
            suffix = inst_name[5:] if inst_name.startswith('INST_') else inst_name
            output_name = f'INST_{source_name}_{suffix}'
            counter = 2
            base_name = output_name
            while output_name in data_vars:
                output_name = f'{base_name}_{counter}'
                counter += 1

        inst_attrs = inst_da.attrs.copy()
        inst_attrs['source_dataset'] = source_name
        inst_attrs['source_variable'] = inst_name

        data_vars[output_name] = xr.DataArray(
            inst_da.values,
            dims=inst_da.dims,
            coords={
                dim_name: inst_da.coords[dim_name]
                for dim_name in inst_da.dims
                if dim_name in inst_da.coords
            },
            attrs=inst_attrs,
        )


def load_config(config_path):
    """Load a JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload, output_path, indent=2):
    """Write a dictionary payload to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
