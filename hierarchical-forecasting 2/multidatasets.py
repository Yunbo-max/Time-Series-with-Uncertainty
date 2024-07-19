import pandas as pd
import statsmodels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from darts.datasets import (
    AirPassengersDataset,
    IceCreamHeaterDataset,
    AusBeerDataset,
    USGasolineDataset,
    ElectricityConsumptionZurichDataset,
    MonthlyMilkDataset,
)
from matplotlib import pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.seasonal import STL, seasonal_decompose, seasonal_mean
import numpy as np
from pandas.core.nanops import nanmean as pd_nanmean
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

###===============================================

## 1. LOAD DATA
datasets = {
    "icecream": IceCreamHeaterDataset,
    "beer": AusBeerDataset,
    "gasoline": USGasolineDataset,
    "milk": MonthlyMilkDataset,
}
dataset_col = {
    "icecream": "ice cream",
    "beer": "Y",
    "gasoline": "Gasoline",
    "milk": "Pounds per cow",
}

dataset_name = "milk"
# dataset_name = "icecream"
# dataset_name = "gasoline"
# dataset_name = "beer"

# df_data = USGasolineDataset().load().pd_dataframe()
## Weekly U.S. Product Supplied of Finished Motor Gasoline between 1991-02-08 and 2021-04-30

# df_data = AusBeerDataset().load().pd_dataframe()
## Total quarterly beer production in Australia (in megalitres) from 1956:Q1 to 2008:Q3

# df_data = IceCreamHeaterDataset().load().pd_dataframe()
## Monthly sales of heaters and ice cream between January 2004 and June 2020.

df_data = datasets[dataset_name]().load().pd_dataframe()

# data_series = df_data["#Passengers"]
# data_series = np.log(df_data["#Passengers"])
data_series = np.log(df_data[dataset_col[dataset_name]])


# data_series = data_series - data_series.rolling(10).mean().fillna(0).values  ## detrend

data_np = data_series.values
n_obs = len(data_np)

## 2. TREND FEATURES  : MA & EWM
# trend_type = "sma"
trend_type = "ewm"

trend_spans = [5, 10, 20]
trend_features = []

if trend_type == "sma":
    for trend_span in trend_spans:
        trend_term = data_series.rolling(trend_span).mean().values
        trend_features.append(trend_term)
elif trend_type == "ewm":
    for trend_span in trend_spans:
        trend_term = data_series.ewm(span=trend_span).mean().values
        trend_features.append(trend_term)
trend_features = np.moveaxis(np.array(trend_features), 0, 1)

## 3. SEASONAL FEATURES : FOURIER SERIES
k_fourier_order = 3
# k_fourier_order = 5
fourier_periods = [3, 6, 12]
# fourier_periods = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# fourier_periods = [10]

# fourier_orders = np.linspace(1, k_fourier_order, 10)
# fourier_orders = np.linspace(1, k_fourier_order, 5)
fourier_orders = np.arange(1, k_fourier_order)

fourier_features = []

for fourier_period in fourier_periods:
    for order in fourier_orders:
        sin_term = np.sin(2 * np.pi * order / fourier_period * np.arange(n_obs))
        cos_term = np.cos(2 * np.pi * order / fourier_period * np.arange(n_obs))
        fourier_features.append(sin_term)
        fourier_features.append(cos_term)
fourier_features = np.moveaxis(np.array(fourier_features), 0, 1)

## 4. AUTOREGRESSIVE
ar_order = 0
# ar_order = 5
max_trunc_start = int(np.max(trend_spans + fourier_periods) + 1 + ar_order)
max_trunc_end = 1  ## forecast horizon

trend_seasonal_features = np.concatenate((trend_features, fourier_features), axis=1)
# trend_seasonal_features = trend_features
# trend_seasonal_features = fourier_features

ar_features = []
for ar_order_i in range(ar_order + 1):
    ar_lagged = pd.DataFrame(trend_seasonal_features).shift(ar_order_i)
    ar_features.append(ar_lagged.values)

ar_features = np.concatenate(np.array(ar_features), axis=1)

ar_features = ar_features[max_trunc_start:-max_trunc_end]  ## truncate nan values


## 5. BAYESIAN REGRESSION
Y_h1 = data_series.shift(-1).values[
    max_trunc_start:-max_trunc_end, np.newaxis
]  ## truncate nan values
X_all_AR = ar_features

X_train_AR, X_test_AR, Y_train, Y_test = train_test_split(
    X_all_AR, Y_h1, test_size=0.5, shuffle=False
)

# model_AR = LinearRegression()
model_AR = BayesianRidge()
model_AR = model_AR.fit(X_train_AR, Y_train)

# Y_pred_all = model_AR.predict(X_all_AR)
# Y_pred_all = model_AR.predict(X_all_AR)[:, 0]
Y_pred_all, Y_pred_std = model_AR.predict(X_all_AR, return_std=True)

residuals_all = (Y_h1[:, 0] - Y_pred_all).flatten()

print(f"MAE: {np.abs(residuals_all).mean()}")
print(f"R2 : {r2_score(Y_h1, Y_pred_all)}")

## 6. BAYESIAN REGRESSION (+ RESIDUALS)
X_all_ARMA = np.concatenate((ar_features, residuals_all[:, np.newaxis]), axis=1)

residual_order = 1
residual_features = []
for i in range(residual_order):
    residual_feature_i = pd.Series(residuals_all).shift(i + 1).values
    residual_features.append(residual_feature_i)
residual_features = np.moveaxis(np.array(residual_features), 0, 1)

X_all_ARMA = np.concatenate((ar_features, residual_features), axis=1)[residual_order:]
Y_h1_ARMA = Y_h1[residual_order:]

X_train_ARMA, X_test_ARMA, Y_train, Y_test = train_test_split(
    X_all_ARMA, Y_h1_ARMA, test_size=0.5, shuffle=False
)
model_AR = BayesianRidge()
model_AR = model_AR.fit(X_train_ARMA, Y_train)
Y_pred_all_ARMA, Y_pred_std_ARMA = model_AR.predict(X_all_ARMA, return_std=True)
residuals_all_ARMA = (Y_h1_ARMA[:, 0] - Y_pred_all_ARMA).flatten()

print(f"MAE (+residual): {np.abs(residuals_all_ARMA).mean()}")
print(f"R2  (+residual): {r2_score(Y_h1_ARMA, Y_pred_all_ARMA)}")


## 6. PLOT AR
time_series = df_data.index[max_trunc_start:-max_trunc_end]

plt.figure()
plt.plot(time_series, Y_h1)
plt.plot(time_series, Y_pred_all)
plt.fill_between(
    time_series,
    Y_pred_all + 2 * Y_pred_std,
    Y_pred_all - Y_pred_std * 2,
    alpha=0.2,
    color="tab:blue",
)
plt.title("Bayesian AR")

plt.figure()
plt.scatter(Y_pred_all, Y_h1)
plt.title("AR")

## 6. PLOT ARMA
plt.figure()
plt.plot(time_series[1:], Y_h1_ARMA)
plt.plot(time_series[1:], Y_pred_all_ARMA)
plt.fill_between(
    time_series[1:],
    Y_pred_all_ARMA + 2 * Y_pred_std_ARMA,
    Y_pred_all_ARMA - Y_pred_std_ARMA * 2,
    alpha=0.2,
    color="tab:blue",
)
plt.title("Bayesian ARMA")

plt.figure()
plt.scatter(Y_pred_all_ARMA, Y_h1_ARMA)
plt.title("ARMA")

########### EVALUATE WINKLER  SCORE #######
# test_args = np.arange(len(Y_pred_all) // 2, len(Y_pred_all))
test_args = np.arange(len(Y_pred_all_ARMA) // 2, len(Y_pred_all_ARMA))

upper_bounds = Y_pred_all_ARMA[test_args] + 2 * Y_pred_std_ARMA[test_args]
lower_bounds = Y_pred_all_ARMA[test_args] - 2 * Y_pred_std_ARMA[test_args]

# upper_bounds = Y_pred_all[test_args] + 2 * Y_pred_std[test_args]
# lower_bounds = Y_pred_all[test_args] - 2 * Y_pred_std[test_args]

winkler_scores = upper_bounds - lower_bounds
alpha = 0.05
penalty_factor = 2 / alpha

# for i, y_ in enumerate(Y_h1[test_args]):
for i, y_ in enumerate(Y_h1_ARMA[test_args]):
    lower_bound_i = lower_bounds[i]
    upper_bound_i = upper_bounds[i]

    if y_ < lower_bounds[i]:
        winkler_scores[i] = winkler_scores[i] + penalty_factor * (lower_bound_i - y_)
    elif y_ > upper_bounds[i]:
        winkler_scores[i] = winkler_scores[i] + penalty_factor * (y_ - upper_bound_i)

# mean_winkler_score = np.mean(np.exp(winkler_scores))
# mean_winkler_score = np.mean(winkler_scores)
mean_winkler_score = np.exp(np.mean(winkler_scores))

print(f"MEAN WINKLER SCORE: {mean_winkler_score : .4f}")

###########################################
