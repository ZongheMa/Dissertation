import geopandas as gpd
import sys
sys.setrecursionlimit(1000000)  # Set the recursion limit to a higher value
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from tqdm import tqdm
import subprocess
import sys
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# load the data
flows_lsoa = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/processed data/flows_lsoa.shp'
flows_lsoa = gpd.read_file(flows_lsoa, crs={'init': 'epsg:27700'})
inoutter = '/Users/zonghe/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Zonghe Ma/Raw data/London administrative boundaries/lp-consultation-oct-2009-inner-outer-london-shp/lp-consultation-oct-2009-inner-outer-london.shp'
inoutter = gpd.read_file(inoutter, crs={'init': 'epsg:27700'})
flows_lsoa = gpd.sjoin(flows_lsoa, inoutter, how='inner', op='within')
flows_lsoa.drop(columns=['index_right', 'Source', 'Area_Ha', 'Shape_Leng', 'Shape_Area'], inplace=True)
flows_lsoa.rename(columns={'Boundary': 'Inner_Outer'}, inplace=True)

print(flows_lsoa.head())
print(flows_lsoa.columns)
print(len(flows_lsoa['LSOA21CD'].unique()))
print(len(flows_lsoa['date'].unique()))
print(flows_lsoa.describe())




# organize the flows data
flows_londonsum = flows_lsoa.groupby('date').aggregate(
    {'bus': 'sum', 'car': 'sum', 'cycle': 'sum', 'walks': 'sum', 'stationary': 'sum'})
flows_londonsum.index = pd.to_datetime(flows_londonsum.index)

flows_inouttersum = flows_lsoa.groupby(['date', 'Inner_Outer']).aggregate(
    {'bus': 'sum', 'car': 'sum', 'cycle': 'sum', 'walks': 'sum', 'stationary': 'sum'}).reset_index()
flows_inner = flows_inouttersum[flows_inouttersum['Inner_Outer'] == 'Inner London'].drop(columns=['Inner_Outer'])
flows_outer = flows_inouttersum[flows_inouttersum['Inner_Outer'] == 'Outer London'].drop(columns=['Inner_Outer'])

flows_inner.set_index('date', inplace=True)
flows_outer.set_index('date', inplace=True)
flows_inner.index = pd.to_datetime(flows_londonsum.index)
flows_outer.index = pd.to_datetime(flows_londonsum.index)

print(flows_londonsum.columns)
print(flows_inner.columns)

# The loop below is the main part for the time series analysis, including the ADF test, differencing, decomposition, and model fitting and evaluation

# Define a new class for the dataframe, which can be used to identify the name of the input dataframe
class InputDataFrame(pd.DataFrame):
    def __init__(self, *args, name=None, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)

    def input_name(self):
        return self.name

# Set the general dataframe for the time series analysis as df_temporal
df_temporal = InputDataFrame(flows_londonsum.copy())

# creat a list of the types of data, for the purpose of modelling and plotting
types = ['car',
         # 'bus', 'cycle', 'walks', 'stationary'
         ]



# split the data into train and test
train = df_temporal.iloc[:int(0.8 * (len(df_temporal)))]  # 80% of the data as train
test = df_temporal.iloc[int(0.8 * (len(df_temporal))):]  # 20% of the data as test

i=0
print('{:-^60s}'.format('Time series analysis for {} of {}'.format(types[i], df_temporal.input_name())))

# check the stationarity of the time series using ADF test
globals()['result_0_{}'.format(i + 1)] = adfuller(df_temporal[types[i]],
                                                  autolag='BIC')  # Use the BIC method to automatically select the lag due to the large scale of dataset
# Use 'globals()' to create a variable in the global scope, so that it can be used in the next loop for visualization

# print the results of the ADF test
print('\nThe ADF test for original time series:')
print('p-value: %f' % globals()['result_0_{}'.format(i + 1)][1])
print('ADF Statistic: %f' % globals()['result_0_{}'.format(i + 1)][0])
print('Critical Values:')
for key, value in globals()['result_0_{}'.format(i + 1)][4].items():
    print('\t%s: %.3f' % (key, value))

# Difference the time series
globals()['diff_1_{}'.format(i + 1)] = df_temporal[types[i]].diff().dropna()

# check the stationarity of the differenced time series using ADF test
print('\nThe ADF test for differenced time series (Difference order 1):')
globals()['result_1_{}'.format(i + 1)] = adfuller(globals()['diff_1_{}'.format(i + 1)])
print('p-value: %f' % globals()['result_1_{}'.format(i + 1)][1])
print('ADF Statistic: %f' % globals()['result_1_{}'.format(i + 1)][0])
print('Critical Values:')
for key, value in globals()['result_1_{}'.format(i + 1)][4].items():
    print('\t%s: %.3f' % (key, value))

# # fit the ARIMA model # This code block is for the ARIMA model, but it was replaced by the SARIMA model
# # the order of the model is (p, d, q), where p is the order of the AR term, d is the order of differencing, and q is the order of the MA term
# model = ARIMA(df_temporal[types[i]], order=(1, 1, 1))
# globals()['model_fit_{}'.format(i + 1)] = model.fit()
# # output the summary of the model
# print('Summary of the ARIMA model for {}:'.format(types[i]))
# print(globals()['model_fit_{}'.format(i + 1)].summary())
# # calculate the residuals
# globals()['residuals_{}'.format(i + 1)] = pd.DataFrame(globals()['model_fit_{}'.format(i + 1)].resid)
#
# # forecast the time series in ARIMA model
# globals()['forecast_{}'.format(i + 1)] = globals()['model_fit_{}'.format(i + 1)].forecast(steps=365)
# print(globals()['forecast_{}'.format(i + 1)])

print('\n{:-^60s}'.format('SARIMA Model'))

Find the best parameters for the SARIMA model using the grid search method

# generate all different combinations of p, d, and q triplets
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))

# generate all different combinations of seasonal p, d, and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

# find the best parameters for the model
best_bic = np.inf
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df_temporal[types[i]],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            if results.bic < best_bic:
                best_bic = results.bic
                best_params = param
                best_params_seasonal = param_seasonal
        except:
            continue

# print the best parameters
print('Best SARIMA parameters:', best_params, best_params_seasonal)
globals()['best_bic_{}'.format(i + 1)] = best_bic
globals()['best_params_{}'.format(i + 1)] = best_params
globals()['best_params_seasonal_{}'.format(i + 1)] = best_params_seasonal

# fit the SARIMA model using the best parameters

# After comparing the forecast results of SARIMA between directly using the steps=365 and using the loop to predict the next single value based on the all previous values, the latter one is better
# So the loop as a kind of improvement is used for forecasting

train_data = list(train[types[i]])
forcast = []
for data in test[types[i]]:
    model = sm.tsa.statespace.SARIMAX(pd.DataFrame(train_data),
                                      order=best_params,
                                      seasonal_order=best_params_seasonal,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    globals()['model_fit_{}'.format(i + 1)] = model.fit()
    pre = globals()['model_fit_{}'.format(i + 1)].forecast()
    pre = pre.to_frame(name='value')
    pre = pre.reset_index()
    prediction = pre['value'][0]
    forcast.append(prediction)
    train_data.append(data)

globals()['forecast_{}'.format(i + 1)] = forcast  # the forecast results of SARIMA model

# output the SARIMA model summary
print('Summary of the SARIMA model for {}:'.format(types[i]))
print(globals()['model_fit_{}'.format(i + 1)].summary())

# forecast the time series in SARIMA model
index = pd.date_range(start='2023-01-01', periods=len(test), freq='D')
globals()['forecast_{}'.format(i + 1)] = pd.DataFrame(globals()['forecast_{}'.format(i + 1)], columns=['value'],
                                                      index=index)
print(globals()['forecast_{}'.format(i + 1)])

# calculate the residuals to check the acceptability of the model
globals()['residuals_{}'.format(i + 1)] = pd.DataFrame(globals()['model_fit_{}'.format(i + 1)].resid)

