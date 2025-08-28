
import numpy  as np
import xarray as xr
from scipy.signal import hilbert

def xphasesync(sig1, sig2, dim="time", bins=50):
    """
    Calculate phase synchronization using phase difference distribution for xarray DataArrays.

    Parameters:
        sig1 (xarray.DataArray): First time series signal.
        sig2 (xarray.DataArray): Second time series signal.
        dim (str): Dimension along which to calculate phase synchronization.
        bins (int or array): Number of bins or bin edges for the histogram.

    Returns:
        xarray.Dataset: Dataset containing phase synchronization index (PSI) and phase difference histogram PDF.
    """
    # Validate inputs
    if not isinstance(sig1, xr.DataArray) or not isinstance(sig2, xr.DataArray):
        raise ValueError("Inputs must be xarray DataArrays.")
    if sig1.dims != sig2.dims:
        raise ValueError("Signals must have the same dimensions.")

    # Compute analytic signal (Hilbert transform)
    hil1 = xr.apply_ufunc(hilbert, sig1, kwargs={"axis": sig1.get_axis_num(dim)}, dask="parallelized")
    hil2 = xr.apply_ufunc(hilbert, sig2, kwargs={"axis": sig2.get_axis_num(dim)}, dask="parallelized")

    # Extract phases
    phase1 = xr.apply_ufunc(np.angle, hil1)
    phase2 = xr.apply_ufunc(np.angle, hil2)

    # Compute phase difference
    pdiff = xr.apply_ufunc(
        lambda p1, p2: np.mod(p1 - p2 + np.pi, 2 * np.pi) - np.pi, phase1, phase2
    )

    # Calculate PSI (phase synchronization index)
    psi = xr.apply_ufunc(
        lambda pd: np.abs(np.mean(np.exp(1j * pd), axis=-1)),
        pdiff,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Histogram computation along the specified axis
    def histogram_along_axis(data, bins, axis_dim):
        def compute_hist(data_slice, bins):
            hist, _ = np.histogram(data_slice, bins=bins, density=True)
            return hist

        # Apply histogram calculation along the specified dimension
        hist_values = xr.apply_ufunc(
            compute_hist,
            data,
            kwargs={"bins": bins},
            input_core_dims=[[axis_dim]],  # Specify the input dimension to reduce
            output_core_dims=[["bins"]],  # Replace axis with 'bins'
            vectorize=True,  # Enable vectorized operation
            dask="parallelized",  # Enable Dask for large datasets
            output_dtypes=[float],  # Histogram values are floats
        )
        return hist_values

    # Compute histogram of phase differences
    hist = histogram_along_axis(pdiff, bins=bins, axis_dim=dim)

    # Define histogram bin edges and centers
    if isinstance(bins, int):
        edges = np.linspace(-np.pi, np.pi, bins + 1)
    else:
        edges = np.asarray(bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Add bin centers as a coordinate
    hist = hist.assign_coords(bins=centers)

    # Combine results
    return xr.Dataset({"PSI": psi, "Hist": hist})