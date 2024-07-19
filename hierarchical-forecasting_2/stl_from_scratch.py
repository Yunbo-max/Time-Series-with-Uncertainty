# importing libraries
import pandas as pd
import statsmodels
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from darts.datasets import AirPassengersDataset
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.seasonal import STL, seasonal_decompose, seasonal_mean
import numpy as np
from pandas.core.nanops import nanmean as pd_nanmean
from tqdm import tqdm

df_data = AirPassengersDataset().load().pd_dataframe()
data_series = df_data["#Passengers"]
data_np = data_series.values

# index
X = np.arange(len(data_np))
y = data_np

# frac = 0.05
frac = 0.05

# period = 30
period = 12

init_T = 0

## STL
n_iterations = 50

for i in tqdm(range(n_iterations)):
    if i ==0:
        T = init_T
    else:
        T = trend_component

    ### 1. SEASONAL COMPONENT
    ## C = Cyclic(Loess(Detrended))
    ## L = LowPass(C)
    ## S = C-L

    detrended = y-T
    n_obs = len(detrended)

    lowess_ = []
    for i in range(period):
        cyclic_ = detrended[i::period]
        lowess_.append(lowess(cyclic_,np.arange(len(cyclic_)),frac=frac,missing="none")[:,1])
    cyclic_subseries = np.concatenate(lowess_)

    lowpass_ = np.mean(cyclic_subseries, axis=0)

    # lowpass_ = lowess(cyclic_subseries, np.arange(len(cyclic_subseries)), frac=0.95, missing="none")[:, 1]
    # lowpass_ = pd.Series(cyclic_subseries).rolling(window=3).mean().values
    # lowpass_ = pd.Series(lowpass_).rolling(window=3).mean().values
    # lowpass_ = lowess(lowpass_, np.arange(len(lowpass_)), frac=frac,missing="none")[:,1]
    # lowpass_ = pd.Series(lowpass_).bfill().values
    # lowpass_ = pd.Series(lowpass_).ffill().values

    seasonal_component =cyclic_subseries-lowpass_


    ### 2. TREND
    deseasonalised = y-seasonal_component

    trend_component = lowess(deseasonalised, np.arange(len(deseasonalised)),frac=frac,missing="none")[:,1]

residual = y-seasonal_component-trend_component
mean_residual = np.nanmean(np.abs(residual))
print(f"MEAN ABS RESIDUAL:{mean_residual:.4f} ")

plt.figure()
plt.plot(seasonal_component, label="seasonal")
plt.plot(trend_component, label="trend")
plt.legend()

# plt.figure()
# # plt.plot(X,y, label="raw")
# plt.plot(X,detrended, label="detrended")
# plt.plot(X,seasonal_component,label="seasonal_component")
# plt.legend()
#
# plt.figure()
# plt.plot(X,y, label="raw")
# plt.plot(X,deseasonalised, label="deseasonalised")
# plt.plot(X,trend_component, label="trend_component")
# plt.legend()
# #
# plt.figure()
# plt.plot(X,y, label="raw")
# plt.plot(X,trend_component+seasonal_component, label="trend+seasonal")
# plt.legend()


################
# from statsmodels.tsa.seasonal import STL
#
# stl = STL(data_series, seasonal=13)
# res = stl.fit()
# fig = res.plot()
#
# plt.figure()
# plt.plot(res.seasonal+res.trend)
# plt.plot(data_series.index,y)









