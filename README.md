# xphasesync

Phase synchronization analysis of two time series using Numpy and Xarray


## Installation

```bash
pip install git+https://github.com/senclimate/xphasesync.git
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
- ENSO phase synchronization changes (Fig. 4 and Supplementary Fig. 12 in Stuecker et al. 2025), an detailed example is available in [examples/illustration_phase_sync.ipynb](examples/illustration_phase_sync.ipynb)

## Acknowledgement

If you use the xentropy code in your published work, please kindly cite:

Stuecker, M. F., Zhao, S., Timmermann, A., Ghosh, R., Semmler, T., Lee, S.-S., et al. (2025). Global climate mode resonance due to rapidly intensifying El Niño–Southern Oscillation. **Nature Communications**. in revision. 


## References
- Pikovsky, A., Rosenblum, M. & Kurths, J. Phase synchronization in regular and chaotic systems. International Journal of Bifurcation and Chaos 10, 2291–2305 (2000). https://doi.org/10.1142/S0218127400001481
