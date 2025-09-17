[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17148051.svg)](https://doi.org/10.5281/zenodo.17148051)


# xphasesync

Phase synchronization analysis of two time series using Numpy and Xarray

> 📌 **Note:**  
> If you use `xphasesync` in your published work, please cite:
>
> Stuecker, M. F., Zhao, S., Timmermann, A., Ghosh, R., Semmler, T., Lee, S.-S., Moon, J.-Y., Jin, F.-F., Jung, T. (2025). *Global climate mode resonance due to rapidly intensifying El Niño–Southern Oscillation.*  **Nature Communications**.


## Installation

```bash
pip install git+https://github.com/senclimate/xphasesync.git
```

or

```bash
pip install xphasesync
```


## Quick Start

```python
import numpy as np
import xarray as xr
from xphasesync import xphasesync

# Create sample sine waves
time = np.arange(1000)
sig1 = np.sin(0.1*time)
sig2 = np.sin(0.1*time + 0.5)

da1 = xr.DataArray(sig1, dims=["time"])
da2 = xr.DataArray(sig2, dims=["time"])

ds = xphasesync(da1, da2, dim="time")
print(ds)
```

## Applications
- ENSO phase synchronization changes (Fig. 4 and Supplementary Fig. 8 in Stuecker et al. 2025), an detailed example is available in [examples/phase_sync_illustration.ipynb](examples/phase_sync_illustration.ipynb)


## References
- Pikovsky, A., Rosenblum, M. & Kurths, J. Phase synchronization in regular and chaotic systems. International Journal of Bifurcation and Chaos 10, 2291–2305 (2000). https://doi.org/10.1142/S0218127400001481
