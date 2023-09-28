from src.data.prepare_input_data import prepare_data_in
import numpy as np




class wwtp_model:
  """ gages should be a DF with gage predictors  
  """
  def __init__(self, gages, Q_in, model):
    self.gages = gages
    self.Q_in  = Q_in
    self.model = model
    self.gage_names = gages.columns


  def prepare_train_data(self, n_lags = 1):
    
    self.lags_used = n_lags
    train_data, self.predictor_names = prepare_data_in(self.gages, self.Q_in, n_lags = n_lags)
    self.trainX, self.trainy = train_data[:, :-1], train_data[:, -1]

  def train_model(self):
    self.model.fit(self.trainX, self.trainy)
    

  def get_RII_reference_values(self,gage_percentile = 0.05, lagQ_percentiles = 0.5):
    lag_Q_ref     = self.Q_in.quantile(lagQ_percentiles)
    Xref          = self.gages.quantile(gage_percentile).to_list()
    Xref.append(lag_Q_ref)
    Xref          = np.array([Xref])
    yhat_ref      = self.model.predict(Xref) 

    return yhat_ref, lag_Q_ref 

  def calc_RII(self, gage_values, gage_reference_percentile = 0.05, lagQ_reference_percentile = 0.5):
    if self.lags_used != 1:
      raise Exception("Not implemented for Q lags != 1")

    yhat_ref, lagged_Q_ref  = self.get_RII_reference_values(gage_reference_percentile, lagQ_reference_percentile)
    gage_values_array       = np.asarray(gage_values).reshape(-1,len(self.gage_names))
    
    lag_Q_array       = np.ones(shape=(gage_values_array.shape[0],1))*lagged_Q_ref
    input_array       = np.concatenate([gage_values_array, lag_Q_array], axis = 1)
    yhat              = self.model.predict(input_array)
    return  (yhat - yhat_ref) / yhat_ref
