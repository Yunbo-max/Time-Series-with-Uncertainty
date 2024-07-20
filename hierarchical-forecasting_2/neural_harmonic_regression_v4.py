# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-05-14 17:37:10
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-07-20 07:38:00
## ADD TREND TERMS (MORE)

import pandas as pd
import statsmodels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, BayesianRidge
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
from sklearn.metrics import r2_score
from darts.datasets import (
    AirPassengersDataset,
    IceCreamHeaterDataset,
    AusBeerDataset,
    USGasolineDataset,
    ElectricityConsumptionZurichDataset,
    MonthlyMilkDataset,
)

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

# dataset_name = "milk"
# dataset_name = "icecream"
# dataset_name = "gasoline"
dataset_name = "beer"

# df_data = USGasolineDataset().load().pd_dataframe()
## Weekly U.S. Product Supplied of Finished Motor Gasoline between 1991-02-08 and 2021-04-30

# df_data = AusBeerDataset().load().pd_dataframe()
## Total quarterly beer production in Australia (in megalitres) from 1956:Q1 to 2008:Q3

# df_data = IceCreamHeaterDataset().load().pd_dataframe()
## Monthly sales of heaters and ice cream between January 2004 and June 2020.

df_data = datasets[dataset_name]().load().pd_dataframe()

data_series = np.log(df_data[dataset_col[dataset_name]])


###==========================
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
trunc_start = np.max(trend_spans) + 1
trend_terms = np.moveaxis(np.array(trend_features), 0, 1)[trunc_start:]

###==========================

# trend_term = data_series.rolling(10).mean().fillna(0).values
# data_series = data_series - trend_term  ## detrend
data_series = data_series.iloc[trunc_start:]
# trend_term = trend_term[trunc_start:]


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
k_fourier_order = 2
fourier_periods = [12]
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
ar_order = 1
# ar_order = 0
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

####### 6. NEURAL REGRESSION ########

import torch


class HarmonicLayer(torch.nn.Module):
    def __init__(self, hidden_nodes=50, max_T=100):
        super().__init__()
        self.layer_A = torch.nn.Linear(hidden_nodes, 1)
        self.layer_B = torch.nn.Linear(hidden_nodes, 1)
        self.layer_T = torch.nn.Linear(hidden_nodes, 1)
        self.max_T = max_T

    def forward(self, x):
        A = torch.nn.functional.softplus(self.layer_A(x))
        B = torch.nn.functional.softplus(self.layer_B(x))
        # T = torch.nn.functional.softplus(self.layer_T(x)) + 1
        T = (torch.nn.functional.sigmoid(self.layer_T(x)) * self.max_T) + 1

        return A, B, T


# class NeuralHarmonicModel(torch.nn.Module):

#     # def cuda(self, *args, **kwargs):
#     #     super().cuda(*args, **kwargs)
#     #     for layer in self.output_layers_season:
#     #         layer.cuda()
#     #     for layer in self.output_layers_trend:
#     #         layer.cuda()
#     #     return self

#     def __init__(
#         self,
#         input_size=3,
#         hidden_nodes=50,
#         n_layers=1,
#         k_order=5,
#         t_order=1,
#         n_trends=3,
#     ):
#         super().__init__()

#         self.input_size = input_size
#         self.hidden_nodes = hidden_nodes
#         self.n_layers = n_layers
#         self.k_order = k_order
#         self.t_order = t_order

#         self.input_layer_season = torch.nn.Linear(input_size, hidden_nodes)
#         self.input_layer_trend = torch.nn.Linear(n_trends, hidden_nodes)

#         self.hidden_layers_season = []
#         for i in range(self.n_layers):
#             self.hidden_layers_season += [
#                 torch.nn.Linear(self.hidden_nodes, self.hidden_nodes),
#                 # torch.nn.Tanh(),
#                 torch.nn.GELU(),
#                 # torch.nn.Softplus(),
#             ]
#         self.hidden_layers_season = torch.nn.Sequential(*self.hidden_layers_season)

#         # self.output_layers_season = [
#         #     torch.nn.ModuleList(
#         #         [HarmonicLayer(self.hidden_nodes) for k in range(k_order)]
#         #     )
#         #     for t in range(t_order)
#         # ]

#         self.hidden_layers_trend = []
#         for i in range(self.n_layers):
#             self.hidden_layers_trend += [
#                 torch.nn.Linear(self.hidden_nodes, self.hidden_nodes),
#                 torch.nn.GELU(),
#             ]
#         self.hidden_layers_trend = torch.nn.Sequential(*self.hidden_layers_trend)

#         self.activation_func = torch.nn.functional.tanh

#         self.T = torch.nn.Parameter(torch.Tensor([[10.0]]))

#         # self.Ts = torch.nn.Parameter(torch.Tensor(torch.randn(t_order)))
#         # self.Ts = torch.nn.Parameter(torch.Tensor([5.0, 5.0]))
#         # self.Ts = torch.nn.Parameter(torch.Tensor([2.0, 5.0]))
#         # self.Ts = torch.nn.Parameter(torch.randn(self.t_order))
#         step_period = 3.0
#         self.Ts = torch.nn.Parameter(
#             torch.from_numpy(
#                 np.linspace(step_period, step_period * self.t_order, self.t_order)
#             ).float()
#         )

#         self.constant = torch.nn.Parameter(torch.Tensor([[1.0]]))

#         self.ABs = torch.nn.Parameter(
#             torch.Tensor(
#                 [[[0.01, 0.01] for k in range(k_order)] for t_ in range(t_order)]
#             )
#         )

#         self.output_layer_trend = torch.nn.Linear(self.hidden_nodes, 1)
#         self.output_layer_season = torch.nn.Linear(self.hidden_nodes, 1)

#     def forward(self, x, t, trend=None):
#         # repr = self.activation_func(self.input_layer(x))
#         # repr = self.hidden_layers(repr)
#         # yhat = self.output_layer(repr)

#         # fourier_components = []
#         # return 0
#         # T_ = torch.nn.functional.softplus(self.T) + 1
#         # T_ = self.T
#         fourier_components = 0
#         for t_order in range(self.t_order):
#             T_ = (torch.nn.functional.softplus(self.Ts[t_order]) + 1).reshape(-1, 1)
#             # T_ = torch.nn.functional.relu(self.Ts[t_order]) + 1
#             # T_ = self.Ts[t_order]

#             for k in range(self.k_order):
#                 # A = torch.nn.functional.softplus(self.output_layers[k][0](repr))
#                 # B = torch.nn.functional.softplus(self.output_layers[k][1](repr))
#                 # T = torch.nn.functional.softplus(self.output_layers[k][2](repr)) + 1

#                 # A, B, T = self.output_layers[k](repr)
#                 # A, B, _ = self.output_layers[t_order][k](repr)

#                 A, B = self.ABs[t_order][k]
#                 A = A.reshape(-1, 1)
#                 B = B.reshape(-1, 1)

#                 # A, B = torch.nn.functional.softplus(self.ABs[k])

#                 # A, B, T = self.output_layers[t_order][k](repr)

#                 # if k == 0 and t_order == 0:
#                 #     fourier_components = A * torch.sin(
#                 #         2 * torch.pi * t * k / 12
#                 #     ) + B * torch.cos(2 * torch.pi * t * T)
#                 #
#                 # else:
#                 #     fourier_components += A * torch.sin(
#                 #         2 * torch.pi * t * k / 12
#                 #     ) + B * torch.cos(2 * torch.pi * t * k / 12)

#                 fourier_components += A * torch.sin(
#                     2 * torch.pi * t * k / T_
#                 ) + B * torch.cos(2 * torch.pi * t * k / T_)

#                 # if k == 0 and t_order == 0:
#                 #     fourier_components = A * torch.sin(
#                 #         2 * torch.pi * t * k / self.T
#                 #     ) + B * torch.cos(2 * torch.pi * t * k / self.T)
#                 #
#                 # else:
#                 #     fourier_components += A * torch.sin(
#                 #         2 * torch.pi * t * k / self.T
#                 #     ) + B * torch.cos(2 * torch.pi * t * k / self.T)

#         # return fourier_components + self.constant
#         # return fourier_components + yhat + self.constant

#         # allocation = torch.nn.functional.sigmoid(self.constant)
#         # return yhat * (1 - allocation) + (x - yhat) * allocation + fourier_components
#         # return fourier_components + allocation

#         # return fourier_components + (yhat - x)

#         # deseasoned = x - fourier_components
#         # deseasoned = fourier_components - x

#         repr_trend = self.activation_func(self.input_layer_trend(trend))
#         repr_trend = self.hidden_layers_trend(repr_trend)
#         yhat_trend = self.output_layer_trend(repr_trend)

#         # detrend = x - trend
#         # repr_season = self.activation_func(self.input_layer_season(detrend))
#         # repr_season = self.hidden_layers_season(repr_season)
#         # yhat_season = self.output_layer_season(repr_season)

#         # return yhat_trend + fourier_components

#         # return yhat_season + trend

#         # return yhat_season + trend * self.constant + yhat_trend + fourier_components
#         # return yhat_season + trend + yhat_trend + fourier_components

#         # residual = fourier_components
#         # repr_season = self.activation_func(self.input_layer_season(residual))
#         # repr_season = self.hidden_layers_season(repr_season)
#         # yhat_season = self.output_layer_season(repr_season)

#         residual = x - (yhat_trend + fourier_components)
#         repr_season = self.activation_func(self.input_layer_season(residual))
#         repr_season = self.hidden_layers_season(repr_season)
#         yhat_season = self.output_layer_season(repr_season)
#         return yhat_season + yhat_trend + fourier_components

#         # residual = fourier_components
#         # repr_season = self.activation_func(self.input_layer_season(residual))
#         # repr_season = self.hidden_layers_season(repr_season)
#         # yhat_season = self.output_layer_season(repr_season)
#         # return yhat_season + yhat_trend + fourier_components


# torch.manual_seed(99)
# np.random.seed(7977)

# X_all = data_np[:-1, np.newaxis]
# Y_all = data_np[1:, np.newaxis]
# trend_tensor = trend_terms[:-1]
# trend_tensor = torch.from_numpy(trend_tensor).float().cuda()

# X_tensor = torch.from_numpy(X_all).float().cuda()
# t_tensor = torch.from_numpy(np.arange(n_obs)[:-1, np.newaxis]).float().cuda()
# Y_tensor = torch.from_numpy(Y_all).float().cuda()

# # X_tensor = torch.from_numpy(data_np[:, np.newaxis]).float()
# # t_tensor = torch.from_numpy(np.arange(n_obs)[:, np.newaxis]).float()

# (
#     X_train_tensor,
#     X_test_tensor,
#     Y_train_tensor,
#     Y_test_tensor,
#     t_train_tensor,
#     t_test_tensor,
#     trend_train_tensor,
#     trend_test_tensor,
# ) = train_test_split(
#     X_tensor, Y_tensor, t_tensor, trend_tensor, test_size=0.5, shuffle=False
# )

# n_ensemble = 5
# # nn_model = NeuralHarmonicModel(
# #     input_size=1,
# #     hidden_nodes=50,
# #     n_layers=1,
# #     k_order=5,
# #     t_order=2,
# #     n_trends=len(trend_spans),
# # )
# # nn_model = nn_model.to("cuda")
# # nn_model = nn_model.cuda()
# # fourier_components = nn_model(x=X_tensor, t=t_tensor, trend=trend_tensor)

# n_epochs = 1000

# nn_models = [
#     NeuralHarmonicModel(
#         input_size=1,
#         hidden_nodes=50,
#         n_layers=1,
#         k_order=5,
#         t_order=5,
#         n_trends=len(trend_spans),
#     ).cuda()
#     for i in range(n_ensemble)
# ]

# # optim = torch.optim.Adam(nn_model.parameters(), lr=2e-2)
# # optim = torch.optim.Adam(nn_model.parameters(), lr=1e-2)

# optims = [torch.optim.Adam(nn_model.parameters(), lr=1e-2) for nn_model in nn_models]

# for nn_i in range(len(nn_models)):

#     for epoch in range(n_epochs):
#         y_pred_tensor = nn_models[nn_i](
#             x=X_train_tensor, t=t_train_tensor, trend=trend_train_tensor
#         )
#         # loss = ((fourier_components - Y_tensor) ** 2).mean()
#         # loss = ((y_pred_tensor - Y_train_tensor)**2).mean()
#         loss = (torch.abs(y_pred_tensor - Y_train_tensor)).mean()

#         ## BACKPROP
#         optims[nn_i].zero_grad()
#         loss.backward()
#         optims[nn_i].step()

#         if epoch % 50 == 0:
#             print(f"{nn_i} LOSS {epoch}: {loss.item() : .4f}")

# y_pred_tensor_all = torch.stack(
#     [nn_model(x=X_tensor, t=t_tensor, trend=trend_tensor) for nn_model in nn_models]
# )
# y_pred_mean = y_pred_tensor_all.mean(0).squeeze(-1).detach().cpu().numpy()
# y_pred_std = y_pred_tensor_all.std(0).squeeze(-1).detach().cpu().numpy()

# ###===PLOTS===
# time_series = df_data.index[trunc_start + 1 :]
# plt.figure()
# plt.plot(time_series, Y_all, label="True")
# plt.plot(time_series, y_pred_mean, label="y_pred")
# plt.fill_between(
#     time_series,
#     y_pred_mean + 2 * y_pred_std,
#     y_pred_mean - 2 * y_pred_std,
#     alpha=0.5,
#     color="tab:blue",
# )
# plt.title(f"BNN ({dataset_name})")
# plt.legend()

# print(f"MAE: {(np.abs(y_pred_mean - Y_all)).mean()}")
# print(f"R2 : {r2_score(Y_all, y_pred_mean)}")

# ########### EVALUATE WINKLER  SCORE #######
# test_args = np.arange(len(y_pred_mean) // 2, len(y_pred_mean))

# upper_bounds = y_pred_mean[test_args] + 2 * y_pred_std[test_args]
# lower_bounds = y_pred_mean[test_args] - 2 * y_pred_std[test_args]

# winkler_scores = upper_bounds - lower_bounds
# alpha = 0.05
# penalty_factor = 2 / alpha

# for i, y_ in enumerate(Y_all[test_args]):
#     lower_bound_i = lower_bounds[i]
#     upper_bound_i = upper_bounds[i]

#     if y_ < lower_bounds[i]:
#         winkler_scores[i] = winkler_scores[i] + penalty_factor * (lower_bound_i - y_)
#     elif y_ > upper_bounds[i]:
#         winkler_scores[i] = winkler_scores[i] + penalty_factor * (y_ - upper_bound_i)

# mean_winkler_score = np.exp(np.mean(winkler_scores))

# print(f"MEAN WINKLER SCORE: {mean_winkler_score : .4f}")

# # plt.figure()
# # plt.plot(y_pred_std)


import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class NeuralHarmonicModel(torch.nn.Module):
    def __init__(
        self,
        input_size=3,
        hidden_nodes=50,
        n_layers=1,
        k_order=5,
        t_order=1,
        n_trends=3,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        self.n_layers = n_layers
        self.k_order = k_order
        self.t_order = t_order

        self.input_layer_season = torch.nn.Linear(input_size, hidden_nodes)
        self.input_layer_trend = torch.nn.Linear(n_trends, hidden_nodes)

        self.hidden_layers_season = []
        for i in range(self.n_layers):
            self.hidden_layers_season += [
                torch.nn.Linear(self.hidden_nodes, self.hidden_nodes),
                torch.nn.GELU(),
            ]
        self.hidden_layers_season = torch.nn.Sequential(*self.hidden_layers_season)

        self.hidden_layers_trend = []
        for i in range(self.n_layers):
            self.hidden_layers_trend += [
                torch.nn.Linear(self.hidden_nodes, self.hidden_nodes),
                torch.nn.GELU(),
            ]
        self.hidden_layers_trend = torch.nn.Sequential(*self.hidden_layers_trend)

        self.activation_func = torch.nn.functional.tanh

        self.T = torch.nn.Parameter(torch.Tensor([[10.0]]))

        step_period = 3.0
        self.Ts = torch.nn.Parameter(
            torch.from_numpy(
                np.linspace(step_period, step_period * self.t_order, self.t_order)
            ).float()
        )

        self.constant = torch.nn.Parameter(torch.Tensor([[1.0]]))

        self.ABs = torch.nn.Parameter(
            torch.Tensor(
                [[[0.01, 0.01] for k in range(k_order)] for t_ in range(t_order)]
            )
        )

        self.output_layer_trend = torch.nn.Linear(self.hidden_nodes, 1)
        self.output_layer_season = torch.nn.Linear(self.hidden_nodes, 1)

    def forward(self, x, t, trend=None):
        fourier_components = 0
        for t_order in range(self.t_order):
            T_ = (torch.nn.functional.softplus(self.Ts[t_order]) + 1).reshape(-1, 1)
            for k in range(self.k_order):
                A, B = self.ABs[t_order][k]
                A = A.reshape(-1, 1)
                B = B.reshape(-1, 1)

                fourier_components += A * torch.sin(
                    2 * torch.pi * t * k / T_
                ) + B * torch.cos(2 * torch.pi * t * k / T_)

        repr_trend = self.activation_func(self.input_layer_trend(trend))
        repr_trend = self.hidden_layers_trend(repr_trend)
        yhat_trend = self.output_layer_trend(repr_trend)

        residual = x - (yhat_trend + fourier_components)
        repr_season = self.activation_func(self.input_layer_season(residual))
        repr_season = self.hidden_layers_season(repr_season)
        yhat_season = self.output_layer_season(repr_season)
        return yhat_season + yhat_trend + fourier_components

# Ensure MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(99)
np.random.seed(7977)

X_all = data_np[:-1, np.newaxis]
Y_all = data_np[1:, np.newaxis]
trend_tensor = trend_terms[:-1]
trend_tensor = torch.from_numpy(trend_tensor).float().to(device)

X_tensor = torch.from_numpy(X_all).float().to(device)
t_tensor = torch.from_numpy(np.arange(n_obs)[:-1, np.newaxis]).float().to(device)
Y_tensor = torch.from_numpy(Y_all).float().to(device)

(
    X_train_tensor,
    X_test_tensor,
    Y_train_tensor,
    Y_test_tensor,
    t_train_tensor,
    t_test_tensor,
    trend_train_tensor,
    trend_test_tensor,
) = train_test_split(
    X_tensor, Y_tensor, t_tensor, trend_tensor, test_size=0.5, shuffle=False
)

n_ensemble = 5

nn_models = [
    NeuralHarmonicModel(
        input_size=1,
        hidden_nodes=50,
        n_layers=1,
        k_order=5,
        t_order=5,
        n_trends=len(trend_spans),
    ).to(device)
    for i in range(n_ensemble)
]

optims = [torch.optim.Adam(nn_model.parameters(), lr=1e-2) for nn_model in nn_models]

n_epochs = 1000

for nn_i in range(len(nn_models)):
    for epoch in range(n_epochs):
        y_pred_tensor = nn_models[nn_i](
            x=X_train_tensor, t=t_train_tensor, trend=trend_train_tensor
        )
        loss = (torch.abs(y_pred_tensor - Y_train_tensor)).mean()

        optims[nn_i].zero_grad()
        loss.backward()
        optims[nn_i].step()

        if epoch % 50 == 0:
            print(f"{nn_i} LOSS {epoch}: {loss.item() : .4f}")

y_pred_tensor_all = torch.stack(
    [nn_model(x=X_tensor, t=t_tensor, trend=trend_tensor) for nn_model in nn_models]
)
y_pred_mean = y_pred_tensor_all.mean(0).squeeze(-1).detach().cpu().numpy()
y_pred_std = y_pred_tensor_all.std(0).squeeze(-1).detach().cpu().numpy()

###===PLOTS===
time_series = df_data.index[trunc_start + 1 :]
plt.figure()
plt.plot(time_series, Y_all, label="True")
plt.plot(time_series, y_pred_mean, label="y_pred")
plt.fill_between(
    time_series,
    y_pred_mean + 2 * y_pred_std,
    y_pred_mean - 2 * y_pred_std,
    alpha=0.5,
    color="tab:blue",
)
plt.title(f"BNN ({dataset_name})")
plt.legend()

print(f"MAE: {(np.abs(y_pred_mean - Y_all)).mean()}")
print(f"R2 : {r2_score(Y_all, y_pred_mean)}")

########### EVALUATE WINKLER  SCORE #######
test_args = np.arange(len(y_pred_mean) // 2, len(y_pred_mean))

upper_bounds = y_pred_mean[test_args] + 2 * y_pred_std[test_args]
lower_bounds = y_pred_mean[test_args] - 2 * y_pred_std[test_args]

winkler_scores = upper_bounds - lower_bounds
alpha = 0.05
penalty_factor = 2 / alpha

for i, y_ in enumerate(Y_all[test_args]):
    lower_bound_i = lower_bounds[i]
    upper_bound_i = upper_bounds[i]

    if y_ < lower_bounds[i]:
        winkler_scores[i] = winkler_scores[i] + penalty_factor * (lower_bound_i - y_)
    elif y_ > upper_bounds[i]:
        winkler_scores[i] = winkler_scores[i] + penalty_factor * (y_ - upper_bound_i)

mean_winkler_score = np.exp(np.mean(winkler_scores))

print(f"MEAN WINKLER SCORE: {mean_winkler_score : .4f}")

