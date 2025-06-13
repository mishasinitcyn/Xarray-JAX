# Xarray-JAX

This Python library provides support for [Xarray](https://docs.xarray.dev/en/latest/index.html) in [JAX](https://docs.jax.dev/en/latest/index.html), making it effectively the first named tensor implementation in a deep learning framework. 

Building on the [xarray_jax.py](https://github.com/google-deepmind/graphcast/blob/main/graphcast/xarray_jax.py) utility in Google Deepmind's [Graphcast](https://github.com/google-deepmind/graphcast) project, this standalone library leverages the Python Array API standard to natively support JAX-backed Xarray data.