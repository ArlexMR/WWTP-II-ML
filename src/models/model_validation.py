from src.data.prepare_input_data import prepare_data_in, train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.base import clone

def simple_split_validation(init_model,no_lag_DF, lag_DF, gauge_list, n_test_frac, Qusgs_lags, Qwwtp_lags):
  train_list = list()
  test_list  = list()
  names_list = lag_DF.columns
  model      = clone(init_model)
  n_lags = max(Qusgs_lags,Qwwtp_lags)
  for name in names_list:
    no_lag_DF_filt  = no_lag_DF[gauge_list[name]]
    data, col_names = prepare_data_in(no_lag_DF_filt, lag_DF[name], Qusgs_lags = Qusgs_lags, Qwwtp_lags = Qwwtp_lags)
    n_test          = round(data.shape[0]*n_test_frac)
    
    # Split train and test data
    train, test = train_test_split(data, n_test)

    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy   = test[:, :-1], test[:, -1]

    # fit model
    model.fit(trainX, trainy)

    yhat_train = model.predict(trainX)
    yhat_test  = model.predict(testX)

    # Create output DF's
    train_DF         = pd.DataFrame(trainX, index = lag_DF.index[n_lags:-n_test], columns = col_names[:-1])
    train_DF['y']    = trainy
    train_DF['yhat'] = yhat_train

    test_DF         = pd.DataFrame(testX, index = lag_DF.index[-n_test:], columns = col_names[:-1])
    test_DF['y']    = testy
    test_DF['yhat'] = yhat_test

    train_list.append(train_DF)
    test_list.append(test_DF)
    
  return names_list, train_list, test_list