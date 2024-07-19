import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt


symbol_dict = {"PL=F": "Platinum", "PA=F": "Palladium", "HG=F": "Copper"}
symbol_list = list(symbol_dict.keys())
symbol_names = list(symbol_dict.values())


df_raw = yf.download(symbol_list, period="12mo").reset_index()


time_series = df_raw[("Date", "")]


##
close_cols = [("Close", symbol) for symbol in symbol_list]
price_close = df_raw[close_cols].values
price_close = np.moveaxis(price_close, 0, 1)


## normalise
lookback_window = 10


def rolling_zscore(price_series, window=10):
    # rolling = pd.Series(price_series).rolling(window=window)
    rolling = pd.Series(price_series).ewm(span=window)

    return ((price_series - rolling.mean()) / rolling.std()).values


price_zscores = np.array(
    [
        rolling_zscore(np.log(price_close_), window=lookback_window)
        for price_close_ in price_close
    ]
)


price_zscore_mean = price_zscores.mean(0)
price_zscore_std = price_zscores.std(0)


price_zscore_upper = price_zscore_mean + 2 * price_zscore_std
price_zscore_lower = price_zscore_mean - 2 * price_zscore_std


anomaly_threshold = 2
# anomaly_args = np.argwhere(
#     (price_zscore_upper >= anomaly_threshold)
#     | (price_zscore_lower <= -anomaly_threshold)
# )
anomaly_upper = np.argwhere((price_zscore_upper >= anomaly_threshold))[:, 0]
anomaly_lower = np.argwhere((price_zscore_lower <= -anomaly_threshold))[:, 0]


plt.figure(figsize=(20, 5))
for price_smean_, symbol_name in zip(price_zscores, symbol_names):
    plt.plot(time_series, price_smean_, label=symbol_name)


plt.plot(
    time_series,
    price_zscore_mean,
    color="black",
    linestyle="--",
    label="Standardised mean",
)
plt.fill_between(
    time_series,
    price_zscore_upper,
    price_zscore_lower,
    color="black",
    alpha=0.2,
    linestyle="--",
)


# for anomaly_arg in anomaly_args:
#     plt.axvline(time_series[anomaly_arg], linestyle="--", color="tab:red", alpha=0.3)
plt.scatter(
    time_series[anomaly_lower],
    price_zscore_lower[anomaly_lower],
    marker="x",
    color="tab:red",
)
plt.scatter(
    time_series[anomaly_upper],
    price_zscore_upper[anomaly_upper],
    marker="x",
    color="tab:red",
)


# plt.axhline(-2, linestyle="--", color="tab:red", label="Anomaly")
# plt.axhline(2, linestyle="--", color="tab:red")


# plt.ylabel("Normalised price")
plt.ylabel("Normalised supply vs demand")


plt.legend()
plt.grid()
