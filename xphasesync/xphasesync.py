
import numpy  as np
import xarray as xr
from scipy.signal import hilbert

def xphasesync(sig1, sig2, dim="time", bins=50, m=1, n=1):
    """
    Calculate phase synchronization index (PSI) and phase difference histogram for xarray DataArrays.

    Parameters:
        sig1 (xr.DataArray): First time series signal.
        sig2 (xr.DataArray): Second time series signal.
        dim (str): Dimension along which to calculate phase synchronization.
        bins (int or array): Number of bins or bin edges for the histogram.
        m, n (int): Multipliers for generalized phase difference: δϕ = m*ϕ1 - n*ϕ2.

    Returns:
        xr.Dataset: Dataset containing:
            - PSI: Phase synchronization index
            - Hist: Histogram PDF of phase differences
    """
    # Input validation
    if not isinstance(sig1, xr.DataArray) or not isinstance(sig2, xr.DataArray):
        raise TypeError("Both inputs must be xarray DataArrays.")
    if sig1.dims != sig2.dims:
        raise ValueError("Signals must have identical dimensions.")

    # Compute analytic signals (Hilbert transform)
    hil1 = xr.apply_ufunc(hilbert, sig1, kwargs={"axis": sig1.get_axis_num(dim)}, dask="parallelized")
    hil2 = xr.apply_ufunc(hilbert, sig2, kwargs={"axis": sig2.get_axis_num(dim)}, dask="parallelized")

    # Extract instantaneous phases
    phase1 = xr.apply_ufunc(np.angle, hil1)
    phase2 = xr.apply_ufunc(np.angle, hil2)

    # Compute generalized phase difference δϕ = m*ϕ1 - n*ϕ2 (wrapped to [-π, π])
    pdiff = xr.apply_ufunc(
        lambda p1, p2: np.mod(m * p1 - n * p2 + np.pi, 2 * np.pi) - np.pi,
        phase1, phase2
    )

    # Compute Phase Synchronization Index (PSI)
    psi = xr.apply_ufunc(
        lambda pd: np.abs(np.mean(np.exp(1j * pd), axis=-1)),
        pdiff,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Histogram of phase differences
    def histogram_along_axis(data, bins, axis_dim):
        def compute_hist(slice_data, bins):
            hist, _ = np.histogram(slice_data, bins=bins, density=True)
            return hist

        return xr.apply_ufunc(
            compute_hist,
            data,
            kwargs={"bins": bins},
            input_core_dims=[[axis_dim]],
            output_core_dims=[["bins"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    hist = histogram_along_axis(pdiff, bins=bins, axis_dim=dim)

    # Define histogram bin centers
    edges = np.linspace(-np.pi, np.pi, bins + 1) if isinstance(bins, int) else np.asarray(bins)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = hist.assign_coords(bins=centers)

    # Return results as a Dataset
    return xr.Dataset({"PSI": psi, "Hist": hist})
