#Variabe importance calculation Function
from src.data.prepare_input_data import prepare_data_in, train_test_split



def get_var_importances(model, msd_series, usgs_df, precipDF, fraction_test = 0.2, Qusgs_lags = 1, Qwwtp_lags = 1):
  
  # prepare input data
  # no_lag_data     =  usgs_df.join(precipDF)
  data, col_names = prepare_data_in(usgs_df, msd_series, precip = precipDF, Qusgs_lags = Qusgs_lags, Qwwtp_lags = Qwwtp_lags)
  n_test          = round(data.shape[0]*fraction_test)

  # Split train and test data
  train, test = train_test_split(data, n_test)

  # split into input and output columns
  trainX, trainy = train[:, :-1], train[:, -1]
  testX, testy   = test[:, :-1], test[:, -1]

  # fit model
  model.fit(trainX, trainy)

  # get output
  var_importances  = model.feature_importances_
  var_names       = col_names[:-1]

  return var_importances, var_names 