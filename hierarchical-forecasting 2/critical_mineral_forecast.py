import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

symbol_dict = {"PL=F": "Platinum", "PA=F": "Palladium", "HG=F": "Copper"}
symbol_list = list(symbol_dict.keys())
symbol_names = list(symbol_dict.values())


df_raw = yf.download(symbol_list, period="12mo").reset_index()


## price data
close_cols = [("Close", symbol) for symbol in symbol_list]
price_close = df_raw[close_cols].values
price_close = np.moveaxis(price_close, 0, 1)


## zscore normalise
def rolling_norm(price_series, window=10):
    # rolling = pd.Series(price_series).rolling(window=window)
    rolling = pd.Series(price_series).ewm(span=window)
    price_mean = pd.Series(rolling.mean(0)).ffill().values
    price_std = pd.Series(rolling.std(0)).ffill().values

    price_upper = price_mean + 2 * price_std
    price_lower = price_mean - 2 * price_std

    return (price_series - price_lower) / (price_upper - price_lower) * 2 - 1


price_norm = np.array(
    [rolling_norm(np.log(price_close_), window=10) for price_close_ in price_close]
)

asset_i = 1


#######  NEURAL REGRESSION ########
data_np = price_norm[asset_i]
trend_type = "ewm"
trend_spans = [5, 10, 20]
trend_features = []

if trend_type == "sma":
    for trend_span in trend_spans:
        trend_term = pd.Series(data_np).rolling(trend_span).mean().values
        trend_features.append(trend_term)
elif trend_type == "ewm":
    for trend_span in trend_spans:
        trend_term = pd.Series(data_np).ewm(span=trend_span).mean().values
        trend_features.append(trend_term)
trunc_start = np.max(trend_spans) + 1
trend_terms = np.moveaxis(np.array(trend_features), 0, 1)[trunc_start:]
data_np = data_np[trunc_start:]
time_series = df_raw[("Date", "")].values[trunc_start:]

data_np = pd.Series(data_np).ffill().values
n_obs = len(data_np)

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
                # torch.nn.Tanh(),
                torch.nn.GELU(),
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
                torch.nn.GELU(),
            ]
        self.hidden_layers_trend = torch.nn.Sequential(*self.hidden_layers_trend)

        self.activation_func = torch.nn.functional.tanh

        self.T = torch.nn.Parameter(torch.Tensor([[10.0]]))

        # self.Ts = torch.nn.Parameter(torch.Tensor(torch.randn(t_order)))
        # self.Ts = torch.nn.Parameter(torch.Tensor([5.0, 5.0]))
        # self.Ts = torch.nn.Parameter(torch.Tensor([2.0, 5.0]))
        # self.Ts = torch.nn.Parameter(torch.randn(self.t_order))
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
        # repr = self.activation_func(self.input_layer(x))
        # repr = self.hidden_layers(repr)
        # yhat = self.output_layer(repr)

        # fourier_components = []
        # return 0
        # T_ = torch.nn.functional.softplus(self.T) + 1
        # T_ = self.T
        fourier_components = 0
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
                A = A.reshape(-1, 1)
                B = B.reshape(-1, 1)

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

        repr_trend = self.activation_func(self.input_layer_trend(trend))
        repr_trend = self.hidden_layers_trend(repr_trend)
        yhat_trend = self.output_layer_trend(repr_trend)

        # detrend = x - trend
        # repr_season = self.activation_func(self.input_layer_season(detrend))
        # repr_season = self.hidden_layers_season(repr_season)
        # yhat_season = self.output_layer_season(repr_season)

        # return yhat_trend + fourier_components

        # return yhat_season + trend

        # return yhat_season + trend * self.constant + yhat_trend + fourier_components
        # return yhat_season + trend + yhat_trend + fourier_components

        # residual = fourier_components
        # repr_season = self.activation_func(self.input_layer_season(residual))
        # repr_season = self.hidden_layers_season(repr_season)
        # yhat_season = self.output_layer_season(repr_season)

        residual = x - (yhat_trend + fourier_components)
        repr_season = self.activation_func(self.input_layer_season(residual))
        repr_season = self.hidden_layers_season(repr_season)
        yhat_season = self.output_layer_season(repr_season)
        return yhat_season + yhat_trend + fourier_components

        # residual = fourier_components
        # repr_season = self.activation_func(self.input_layer_season(residual))
        # repr_season = self.hidden_layers_season(repr_season)
        # yhat_season = self.output_layer_season(repr_season)
        # return yhat_season + yhat_trend + fourier_components


torch.manual_seed(99)
np.random.seed(7977)

X_all = data_np[:-1, np.newaxis]
Y_all = data_np[1:, np.newaxis]
trend_tensor = trend_terms[:-1]
trend_tensor = torch.from_numpy(trend_tensor).float().cuda()

X_tensor = torch.from_numpy(X_all).float().cuda()
t_tensor = torch.from_numpy(np.arange(n_obs)[:-1, np.newaxis]).float().cuda()
Y_tensor = torch.from_numpy(Y_all).float().cuda()

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

n_epochs = 1000

nn_models = [
    NeuralHarmonicModel(
        input_size=1,
        hidden_nodes=50,
        n_layers=1,
        k_order=5,
        t_order=5,
        n_trends=len(trend_spans),
    ).cuda()
    for i in range(n_ensemble)
]


# optim = torch.optim.Adam(nn_model.parameters(), lr=2e-2)
# optim = torch.optim.Adam(nn_model.parameters(), lr=1e-2)

optims = [torch.optim.Adam(nn_model.parameters(), lr=1e-2) for nn_model in nn_models]

for nn_i in range(len(nn_models)):

    for epoch in range(n_epochs):
        y_pred_tensor = nn_models[nn_i](
            x=X_train_tensor, t=t_train_tensor, trend=trend_train_tensor
        )
        # loss = ((fourier_components - Y_tensor) ** 2).mean()
        # loss = ((y_pred_tensor - Y_train_tensor)**2).mean()
        loss = (torch.abs(y_pred_tensor - Y_train_tensor)).mean()

        ## BACKPROP
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
test_args = np.arange(len(y_pred_mean) // 2, len(y_pred_mean))

time_series_trunc = time_series[1:][test_args]


plt.figure(figsize=(13, 6))
plt.plot(
    time_series_trunc,
    np.clip(Y_all[test_args], -1, 1),
    label="True",
    color="tab:orange",
)
plt.plot(
    time_series_trunc,
    np.clip(y_pred_mean[test_args], -1, 1),
    label="y_pred",
    color="tab:blue",
)
plt.fill_between(
    time_series_trunc,
    np.clip(y_pred_mean[test_args] + 2 * y_pred_std[test_args], -1, 1),
    np.clip(y_pred_mean[test_args] - 2 * y_pred_std[test_args], -1, 1),
    alpha=0.5,
    color="tab:blue",
)
plt.title(f"BNN ({symbol_names[asset_i]})")
plt.legend()
plt.grid()
plt.tight_layout()
print(f"MAE: {(np.abs(y_pred_mean - Y_all)).mean()}")
print(f"R2 : {r2_score(Y_all, y_pred_mean)}")
