"""I/O and filesystem helpers."""

from __future__ import annotations

import os


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


import json


def load_config(config_path):
    """Load a JSON configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload, output_path, indent=2):
    """Write a dictionary payload to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)
