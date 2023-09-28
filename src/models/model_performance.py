from src.models.performance_metrics import calc_kge, calc_nse, calc_nse_ln
from src.data.prepare_input_data import prepare_data_in
from src.data.prepare_input_data import train_test_split

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def performance_gage_simple_split(model_in, dischargeDF, msd_series, discharge_gauges, Qusgs_lags, Qwwtp_lags, n_test_frac):
  mae_list         = list()
  r2_list          = list()
  NSE_list         = list()
  KGE_list         = list()
  lognse_list      = list()

#   model = clone(model_in)
  model = model_in
  for i in range(1,len(discharge_gauges)+1):
    discharge             = dischargeDF[discharge_gauges[0:i]]
    data, col_names       = prepare_data_in(discharge, msd_series, Qusgs_lags = Qusgs_lags, Qwwtp_lags = Qwwtp_lags)
    n_test                = round(data.shape[0]*n_test_frac)

    # Split train and test data
    train, test = train_test_split(data, n_test)

    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy   = test[:, :-1], test[:, -1]

    # fit model
    model.fit(trainX, trainy)

    yhat_test  = model.predict(testX)

    # performance metrics
    mae = mean_absolute_error(testy,yhat_test)
    r2 = r2_score(testy,yhat_test)
    nse = calc_nse(testy,yhat_test)
    kge = calc_kge(testy,yhat_test)
    lognse = calc_nse_ln(testy,yhat_test)

    mae_list.append(mae)
    r2_list.append(r2)
    NSE_list.append(nse)
    KGE_list.append(kge)
    lognse_list.append(lognse)

  metrics_DF = pd.DataFrame({'MAE':mae_list, 'r2':r2_list, 'NSE':NSE_list, 'KGE':KGE_list, 'Log_NSE':lognse_list}, index = range(1,len(discharge_gauges)+1))

  return metrics_DF