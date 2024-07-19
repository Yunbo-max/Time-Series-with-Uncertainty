from darts.datasets import AirPassengersDataset
from matplotlib import pyplot as plt
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.seasonal import STL, seasonal_decompose, seasonal_mean
import numpy as np
from pandas.core.nanops import nanmean as pd_nanmean

df_data = AirPassengersDataset().load().pd_dataframe()
data_series = df_data["#Passengers"]
data_np = data_series.values

## The results are obtained by first estimating the trend by applying a convolution filter to the data.
## The trend is then removed from the series and the average of this de-trended series for each period is the
## returned seasonal component.

####==========
additive_decompose = seasonal_decompose(
    data_series,
    model="additive",
    period=12,
    two_sided=True)


## 1. ESTIMATE TREND
two_sided = True
period = 12
filt = None
if filt is None:
    if period % 2 == 0:  # split weights at ends
        filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
    else:
        filt = np.repeat(1.0 / period, period)

nsides = int(two_sided) + 1
trend = convolution_filter(data_np, filt, nsides)
detrended = data_np - trend

plt.figure()
plt.plot(data_np,label="raw")
plt.plot(trend, label="trend")
plt.legend()

## 2. ESTIMATE SEASONAL
period_averages = np.array([pd_nanmean(detrended[i::period], axis=0) for i in range(period)])
period_averages -= np.mean(period_averages, axis=0)
nobs = len(detrended)
seasonal = np.tile(period_averages.T, nobs // period + 1).T[:nobs]

plt.figure()
plt.plot(detrended, label="detrended")
plt.plot(seasonal, label="seasonal")
plt.legend()

## 3. RESIDUAL
residuals = detrended - seasonal

fig, axes = plt.subplots(2,1,sharex=True)
ax_raw = axes[0]
ax_residual = axes[1]
ax_raw.plot(data_np, label="raw")
ax_raw.plot(trend+seasonal, label="trend+seasonal")
ax_residual.plot(residuals, label="residual")
for ax in axes:
    ax.legend()

####==========