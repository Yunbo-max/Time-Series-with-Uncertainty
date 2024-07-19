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

df_data = AirPassengersDataset().load().pd_dataframe().reset_index()
data_series = df_data["#Passengers"]
data_np = data_series.values
df_data['Month'] = pd.to_datetime(df_data['Month'])
df_data['month_num'] = df_data['Month'].dt.month

### IDEA: FIT FOURIER SERIES

# Get fourier features
for order in range(1, 10):
    df_data[f'fourier_sin_order_{order}'] = np.sin(2 * np.pi * order * df_data['month_num'] / 12)
    df_data[f'fourier_cos_order_{order}'] = np.cos(2 * np.pi * order * df_data['month_num'] / 12)

plt.figure()
plt.plot(df_data[f'fourier_sin_order_{1}'])
plt.plot(df_data[f'fourier_sin_order_{2}'])
plt.plot(df_data[f'fourier_sin_order_{3}'])
plt.plot(df_data[f'fourier_sin_order_{4}'])
plt.plot(df_data[f'fourier_sin_order_{5}'])
plt.plot(df_data[f'fourier_sin_order_{6}'])
plt.plot(df_data[f'fourier_sin_order_{7}'])
