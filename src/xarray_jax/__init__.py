"""Public API for the xarray_jax package."""

from .xarray_jax import (
    Variable,
    DataArray,
    Dataset,
    assign_coords,
    get_jax_coords,
    assign_jax_coords,
    unwrap_data,
    jax_data,
    jax_vars,
    apply_ufunc,
    pmap,
    dims_change_on_unflatten,
)

__all__ = [
    # Constructors
    "Variable",
    "DataArray",
    "Dataset",
    # Coordinate handling
    "assign_coords",
    "get_jax_coords",
    "assign_jax_coords",
    # Wrapping/unwrapping (post JaxArrayWrapper removal)
    "unwrap_data",
    "jax_data",
    "jax_vars",
    # Function application
    "apply_ufunc",
    # Parallelization
    "pmap",
    # Tree utilities context manager
    "dims_change_on_unflatten",
] 