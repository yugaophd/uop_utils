"""I/O and filesystem helpers."""

from __future__ import annotations

import os


def create_dir(directory):
    """Create a directory if it does not already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


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
