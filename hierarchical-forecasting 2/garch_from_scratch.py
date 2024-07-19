import pandas as pd
import scipy
import statsmodels
from sklearn.model_selection import train_test_split
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
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

###===============================================

## 0. LOAD DATA
df_data = AirPassengersDataset().load().pd_dataframe()

data_series = df_data["#Passengers"]
raw_data = data_series.values

trend = lowess(raw_data,np.arange(len(raw_data)),frac=0.2,missing="none")[:,1]

detrended = raw_data-trend
input_data = detrended
def compute_nll(parameters, input_data, return_full=False):
    omega = parameters[0]  ## const
    alpha = parameters[1]  ## for raw_2
    beta = parameters[2]  ## for recurrence

    len_data = len(input_data)
    sigma_2 = np.zeros(len_data)  ## results

    for i in range(len_data):
        if i == 0:
            sigma_2[i] = np.var(input_data)
        else:
            sigma_2[i] = omega + alpha * input_data[i - 1] ** 2 + beta * sigma_2[i - 1]

    ## NLL
    nll = np.log(sigma_2) + input_data ** 2 / sigma_2

    if return_full:
        return nll, sigma_2
    else:
        nll_sum = np.sum(nll)
        print(f"NLL : {nll_sum :.4f}")

        return nll_sum

## 3. OPT
parameters = [0.5,0.5,0.5]
opt = scipy.optimize.minimize(compute_nll, parameters, args=(input_data, False),
                                     bounds = ((.001,1),(.001,1),(.001,1)))
parameters_opt = opt.x
nll, garch_sigma = compute_nll(parameters_opt, input_data, return_full=True)
rolling_std = pd.Series(input_data).rolling(window=40).std().values

## PLOTS
fig, axes = plt.subplots(3,1)
ax_nll = axes[1]
ax_sigma = axes[2]
ax_raw= axes[0]

ax_sigma.plot(np.sqrt(garch_sigma), label="GARCH")
ax_sigma.plot(rolling_std, label="Rolling-std")
ax_raw.plot(input_data, label="detrended")
ax_raw.plot(trend, label="trend")
ax_raw.plot(raw_data, label="raw")
ax_nll.plot(nll)

ax_sigma.set_ylabel("SIGMA")
ax_raw.set_ylabel("INPUT DATA")
ax_nll.set_ylabel("NLL")

ax_raw.legend()
ax_sigma.legend()

## ROLLING STD
plt.figure()
plt.scatter(np.sqrt(garch_sigma),rolling_std)
plt.xlabel("GARCH")
plt.ylabel("Rolling-std")











