import pandas as pd
import numpy as np

# Input data preparation

# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = pd.concat(cols, axis=1)
	# drop rows with NaN values
	# if dropnan:
	# 	agg.dropna(inplace=True)
	return agg.values
def prepare_data_in(Qusgs_DF, Qwwtp_series, precip = None, Qusgs_lags = 1, Qwwtp_lags = 1):

    Qwwtp_series.rename('msdflow', inplace = True)
    lag_cols = []
    if Qusgs_lags > 0:
        for gage_name, gagedata in Qusgs_DF.items():

            for lag in range(1, Qusgs_lags + 1):
                new_col = gagedata.shift(lag).rename(f"{gage_name}_lag{lag}")
                lag_cols.append(new_col)
        Qusgs_DF = pd.concat([Qusgs_DF] + lag_cols, axis = 1)

    if Qwwtp_lags > 0:
        full_Qwwtp = series_to_supervised(Qwwtp_series.to_list(), n_in = Qwwtp_lags)
        Qwwtp_col_names = ['Qwwtp_t-' + str(i) for i in range(Qwwtp_lags,-1,-1)]
        full_Qwwtp = pd.DataFrame(full_Qwwtp, columns = Qwwtp_col_names, index = Qwwtp_series.index)
    else:
        full_Qwwtp = Qwwtp_series.to_frame()

    if precip is not None:
        # precip.rename("precip", inplace = True)
        Full_DF = pd.concat([Qusgs_DF, precip, full_Qwwtp], axis = 1)
    else:
        Full_DF = Qusgs_DF.join(full_Qwwtp)

    col_names = Full_DF.columns


    return Full_DF.values[max(Qusgs_lags, Qwwtp_lags):,:], col_names

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]