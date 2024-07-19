## ADD TREND TERM

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

###===============================================

## 1. LOAD DATA
df_data = AirPassengersDataset().load().pd_dataframe()

# data_series = df_data["#Passengers"]
data_series = np.log(df_data["#Passengers"])

trend_term = data_series.rolling(10).mean().fillna(0).values
# data_series = data_series - trend_term  ## detrend
data_series = data_series.iloc[10:]
trend_term = trend_term[10:]

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


class NeuralHarmonicModel(torch.nn.Module):

    # def cuda(self, *args, **kwargs):
    #     super().cuda(*args, **kwargs)
    #     for layer in self.output_layers_season:
    #         layer.cuda()
    #     for layer in self.output_layers_trend:
    #         layer.cuda()
    #     return self

    def __init__(self, input_size=3, hidden_nodes=50, n_layers=1, k_order=5, t_order=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        self.n_layers = n_layers
        self.k_order = k_order
        self.t_order = t_order

        self.input_layer_season = torch.nn.Linear(input_size, hidden_nodes)
        self.input_layer_trend = torch.nn.Linear(input_size, hidden_nodes)

        self.hidden_layers_season = []
        for i in range(self.n_layers):
            self.hidden_layers_season += [
                torch.nn.Linear(self.hidden_nodes, self.hidden_nodes),
                torch.nn.Tanh(),
                # torch.nn.Softplus(),
            ]
        self.hidden_layers_season = torch.nn.Sequential(*self.hidden_layers_season)

        # self.output_layers_season = [
        #     torch.nn.ModuleList(
        #         [HarmonicLayer(self.hidden_nodes) for k in range(k_order)]
        #     )
        #     for t in range(t_order)
        # ]

        self.hidden_layers_trend = []
        for i in range(self.n_layers):
            self.hidden_layers_trend += [
                torch.nn.Linear(self.hidden_nodes, self.hidden_nodes),
                torch.nn.Tanh(),
                # torch.nn.Softplus(),
            ]
        self.hidden_layers_trend = torch.nn.Sequential(*self.hidden_layers_trend)

        # self.output_layers_trend = [
        #     torch.nn.ModuleList(
        #         [HarmonicLayer(self.hidden_nodes) for k in range(k_order)]
        #     )
        #     for t in range(t_order)
        # ]

        self.activation_func = torch.nn.functional.tanh

        self.T = torch.nn.Parameter(torch.Tensor([[10.0]]))

        # self.Ts = torch.nn.Parameter(torch.Tensor(torch.randn(t_order)))
        # self.Ts = torch.nn.Parameter(torch.Tensor([5.0, 5.0]))
        # self.Ts = torch.nn.Parameter(torch.Tensor([2.0, 5.0]))
        self.Ts = torch.nn.Parameter(torch.randn(self.t_order))

        self.constant = torch.nn.Parameter(torch.Tensor([[1.0]]))

        self.ABs = torch.nn.Parameter(
            torch.Tensor(
                [[[0.01, 0.01] for k in range(k_order)] for t_ in range(t_order)]
            )
        )

        self.output_layer_trend = torch.nn.Linear(self.hidden_nodes, 1)
        self.output_layer_season = torch.nn.Linear(self.hidden_nodes, 1)

    def forward(self, x, t, trend=None):
        # repr = self.activation_func(self.input_layer(x))
        # repr = self.hidden_layers(repr)
        # yhat = self.output_layer(repr)

        # fourier_components = []
        # return 0
        # T_ = torch.nn.functional.softplus(self.T) + 1
        # T_ = self.T

        for t_order in range(self.t_order):
            T_ = (torch.nn.functional.softplus(self.Ts[t_order]) + 1).reshape(-1, 1)
            # T_ = torch.nn.functional.relu(self.Ts[t_order]) + 1
            # T_ = self.Ts[t_order]

            for k in range(self.k_order):
                # A = torch.nn.functional.softplus(self.output_layers[k][0](repr))
                # B = torch.nn.functional.softplus(self.output_layers[k][1](repr))
                # T = torch.nn.functional.softplus(self.output_layers[k][2](repr)) + 1

                # A, B, T = self.output_layers[k](repr)
                # A, B, _ = self.output_layers[t_order][k](repr)

                A, B = self.ABs[t_order][k]
                # A = A.reshape(-1, 1)
                # B = B.reshape(-1, 1)

                # A, B = torch.nn.functional.softplus(self.ABs[k])

                # A, B, T = self.output_layers[t_order][k](repr)

                # if k == 0 and t_order == 0:
                #     fourier_components = A * torch.sin(
                #         2 * torch.pi * t * k / 12
                #     ) + B * torch.cos(2 * torch.pi * t * T)
                #
                # else:
                #     fourier_components += A * torch.sin(
                #         2 * torch.pi * t * k / 12
                #     ) + B * torch.cos(2 * torch.pi * t * k / 12)

                if k == 0 and t_order == 0:
                    fourier_components = A * torch.sin(
                        2 * torch.pi * t * k / T_
                    ) + B * torch.cos(2 * torch.pi * t * k / T_)

                else:
                    fourier_components += A * torch.sin(
                        2 * torch.pi * t * k / T_
                    ) + B * torch.cos(2 * torch.pi * t * k / T_)

                # if k == 0 and t_order == 0:
                #     fourier_components = A * torch.sin(
                #         2 * torch.pi * t * k / self.T
                #     ) + B * torch.cos(2 * torch.pi * t * k / self.T)
                #
                # else:
                #     fourier_components += A * torch.sin(
                #         2 * torch.pi * t * k / self.T
                #     ) + B * torch.cos(2 * torch.pi * t * k / self.T)

        # return fourier_components + self.constant
        # return fourier_components + yhat + self.constant

        # allocation = torch.nn.functional.sigmoid(self.constant)
        # return yhat * (1 - allocation) + (x - yhat) * allocation + fourier_components
        # return fourier_components + allocation

        # return fourier_components + (yhat - x)

        # deseasoned = x - fourier_components
        # deseasoned = fourier_components - x
        # repr_trend = self.activation_func(self.input_layer_trend(deseasoned))
        # repr_trend = self.hidden_layers_trend(repr_trend)
        # yhat_trend = self.output_layer_trend(repr_trend)
        #
        # detrend = x - trend
        # repr_season = self.activation_func(self.input_layer_season(detrend))
        # repr_season = self.hidden_layers_season(repr_season)
        # yhat_season = self.output_layer_season(repr_season)

        # return yhat_trend + fourier_components

        # return yhat_season + trend

        # return yhat_season + trend * self.constant + yhat_trend + fourier_components
        # return yhat_season + trend + yhat_trend + fourier_components

        residual = x - (trend + fourier_components)
        repr_season = self.activation_func(self.input_layer_season(residual))
        repr_season = self.hidden_layers_season(repr_season)
        yhat_season = self.output_layer_season(repr_season)

        return yhat_season + fourier_components + trend

        # return yhat

        # return yhat - x

        # return fourier_components + (yhat - x)

        # return (fourier_components - x) * self.constant

        # return (x - yhat) + fourier_components * self.constant
        # allocation = torch.nn.functional.sigmoid(self.constant)
        # return (x - yhat) * (1 - allocation) + fourier_components * allocation

        # return yhat * (1 - allocation) + (yhat - x) * allocation


torch.manual_seed(777)
np.random.seed(777)

X_all = data_np[:-1, np.newaxis]
Y_all = data_np[1:, np.newaxis]
trend_tensor = trend_term[:-1, np.newaxis]
trend_tensor = torch.from_numpy(trend_tensor).float().cuda()

X_tensor = torch.from_numpy(X_all).float().cuda()
t_tensor = torch.from_numpy(np.arange(n_obs)[:-1, np.newaxis]).float().cuda()
Y_tensor = torch.from_numpy(Y_all).float().cuda()

# X_tensor = torch.from_numpy(data_np[:, np.newaxis]).float()
# t_tensor = torch.from_numpy(np.arange(n_obs)[:, np.newaxis]).float()

nn_model = NeuralHarmonicModel(
    input_size=1, hidden_nodes=50, n_layers=1, k_order=5, t_order=5
)
# nn_model = nn_model.to("cuda")
nn_model = nn_model.cuda()
# fourier_components = nn_model(x=X_tensor, t=t_tensor)

n_epochs = 1000

# optim = torch.optim.Adam(nn_model.parameters(), lr=2e-2)
optim = torch.optim.Adam(nn_model.parameters(), lr=1e-2)

for epoch in range(n_epochs):
    fourier_components = nn_model(x=X_tensor, t=t_tensor, trend=trend_tensor)
    # loss = ((fourier_components - Y_tensor) ** 2).mean()
    loss = (torch.abs(fourier_components - Y_tensor)).mean()

    ## BACKPROP
    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"LOSS {epoch}: {loss.item() : .4f}")

plt.figure()
plt.plot(Y_all, label="TRUE")
plt.plot(fourier_components.detach().cpu().numpy().flatten(), label="PRED")
plt.legend()

print(f"MAE: {(torch.abs(fourier_components - Y_tensor)).mean().item()}")
print(f"R2 : {r2_score(Y_all, fourier_components.detach().cpu().numpy())}")

#
# hidden_layers = []
# for i in range(nn_model.n_layers):
#     hidden_layers += [
#         torch.nn.Linear(nn_model.hidden_nodes, nn_model.hidden_nodes),
#         torch.nn.Tanh(),
#     ]
# hidden_layers = torch.nn.Sequential(*hidden_layers)
#
# output_layers = [
#     torch.nn.ModuleList([torch.nn.Linear(nn_model.hidden_nodes, 1) for i in range(3)])
#     for k in range(nn_model.k_order)
# ]
#
# ######################################

plt.figure()
plt.plot(data_series)
