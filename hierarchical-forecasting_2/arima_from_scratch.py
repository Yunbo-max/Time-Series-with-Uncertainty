import pandas as pd
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
###===============================================

## 1. LOAD DATA
df_data = AirPassengersDataset().load().pd_dataframe()

data_series = df_data["#Passengers"]
data_np = data_series.values

### 2. SPECIFY ARIMA PARAMETERS
lag_orders = 12
ma_orders = 1
difference_orders = 0

### 3.0 I (CREATE DIFFERENCED SERIES)
diff_series = data_series.copy()
for i in range(difference_orders):
    shift_series = diff_series.shift(1)
    diff_series = diff_series-shift_series
diff_series = diff_series.iloc[difference_orders:]

### 3.1 AR (CREATE LAGGED SERIES)
data_lagged = diff_series.shift(1).values

data_lagged = []
for i in range(lag_orders):
    data_lagged.append(diff_series.shift(i + 1).values)
data_lagged = np.moveaxis(np.array(data_lagged), 0, 1) ## swap feature axis with samples

### 4. CREATE TRAINING AND TESTING DATA
X_all_AR = data_lagged[lag_orders:]
Y_all = diff_series[lag_orders:,np.newaxis]

X_train_AR, X_test_AR, Y_train,Y_test = train_test_split(X_all_AR,Y_all, test_size=0.5,shuffle=False)

### 5. FIT MODEL
model_AR = LinearRegression()
model_AR = model_AR.fit(X_train_AR,Y_train)

residuals_all = (Y_all - model_AR.predict(X_all_AR)).flatten()
residuals_all_lagged = []
for i in range(ma_orders):
    residuals_all_lagged.append(pd.Series(residuals_all).shift(i + 1).values)
residuals_all_lagged = np.moveaxis(np.array(residuals_all_lagged), 0, 1) ## swap feature axis with samples

X_all_ARMA = np.concatenate((X_all_AR,residuals_all_lagged),axis=1)

X_train_ARMA, X_test_ARMA = train_test_split(X_all_ARMA, test_size=0.5,shuffle=False)

model_ARMA =LinearRegression()
model_ARMA = model_ARMA.fit(X_train_ARMA[ma_orders:],Y_train[ma_orders:])


### 4. TEST & VISUALISE
Y_test_pred_AR = model_AR.predict(X_test_AR)
Y_test_pred_ARMA = model_ARMA.predict(X_test_ARMA)

mae_AR = np.abs(Y_test_pred_AR-Y_test).mean()
mae_ARMA = np.abs(Y_test_pred_ARMA-Y_test).mean()

print(f"MAE_AR  : {mae_AR :.4f}")
print(f"MAE_ARMA: {mae_ARMA :.4f}")

### PLOT FIGURE
plt.figure()
plt.plot(np.arange(len(Y_train)),Y_train)
plt.plot(np.arange(len(Y_test))+len(Y_train),Y_test)
plt.plot(np.arange(len(Y_test))+len(Y_train),Y_test_pred_AR)
plt.plot(np.arange(len(Y_test))+len(Y_train),Y_test_pred_ARMA)

### UNDIFFERENCING
undiff_pred = np.cumsum(model_ARMA.predict(X_all_ARMA[ma_orders:]).flatten())
undiff_raw = np.cumsum(Y_all[ma_orders:])

plt.figure()
plt.plot(undiff_raw)
plt.plot(undiff_pred)



