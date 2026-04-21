# uop_utils

Shared utility functions for UOP processing workflows across wavegliders, buoys, ships, and related observing platforms.

## Install

```bash
cd ~/Python/uop_utils
pip install -e .
```

## Included modules

- `uop_utils.metadata` - git provenance and metadata helpers
- `uop_utils.io` - directory and NetCDF encoding helpers
- `uop_utils.qc` - simple QC and despiking utilities
- `uop_utils.geo` - vector and directional-data utilities
- `uop_utils.naming` - filename parsing helpers
- `uop_utils.plotting` - plotting and LaTeX-report helpers

## Example

```python
from uop_utils.metadata import get_git_governance_info
from uop_utils.io import update_encoding
```

## Notable helpers

- `compute_current_relative_wind` preserves true wind when surface current is missing and applies vector relative-wind adjustment only where current is valid.
