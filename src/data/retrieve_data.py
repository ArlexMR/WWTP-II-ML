import json
import pandas as pd
from dataretrieval import nwis 
from settings import param_file
import numpy as np

#-----------------------msd_flows----------------------------------------------------------------------------------------
def get_msdFlow():

  params     = param_file.params

  msd_path = params["paths"]["MSD_flow"]
  
  msd_flow = pd.read_csv(msd_path,index_col='Date',parse_dates=True)
  
  max_msd_flows           = {'MFWQTC(MGD)' : 5,
                        'DRGWQTC(MGD)': 0,
                        'CCWQTC(MGD)' : 2, 
                        'FFWQTP(MGD)' : 0,
                        'HCWQTP(MGD)' : 2
                        }

  removed_data = {}
  for msd_name, max_flow in max_msd_flows.items():
      removed_data[msd_name] = (msd_flow[msd_name] < max_flow).sum()
      msd_flow.loc[msd_flow[msd_name] < max_flow, msd_name] = np.nan 


  return (msd_flow)

#-----------------------Stream gauge discharge data-----------------------------------------------------------------------

def get_usgs_data():

  # Retrieve the data
  params        = param_file.params
  siteNumber    = params["USGS_data"]["siteNumber"]
  parameterCode = params["USGS_data"]["parameterCode"]
  startDate     = params["USGS_data"]["startDate"]
  endDate       = params["USGS_data"]["endDate"]
  
  (dailyStreamflow, info) = nwis.get_dv(sites=siteNumber,
                                        parameterCd=parameterCode,
                                        start=startDate,
                                        end = endDate
                                        )
  # Reshape the DataFrame
  dischargeDF = dailyStreamflow.reset_index().pivot(index='datetime',columns='site_no')[parameterCode + "_Mean"]
  # Remove Time Zone info from the Discharge DF
  dischargeDF.index = dischargeDF.reset_index()['datetime'].dt.tz_localize(None)
  return dischargeDF
# print('/n/nUSGS streamflow:')
# display(dischargeDF.head(2))

#-----------------------Precipitation data--------------------------------------------------------------------------------------
def get_mean_precip():
  params  = param_file.params
  path    = params["paths"]["precip"]
  
  # Read csv
  precip_csv = pd.read_csv(path, index_col='local_time',parse_dates=True)

  # prepare Data
  temporal_DF     = precip_csv[['name','precip_rate','precip_type', 'temp_c']].reset_index()  # filter relevant columns
  sta_names       = temporal_DF.name.unique()                                           # Station names

  precipDF    = pd.DataFrame()                                                  # Initialize DF for P
  typeDF      = pd.DataFrame()                                                  # Initialize DF for type of P
  tempDF      = pd.DataFrame()                                                  # Initialize DF for type of Temperature

  # process one station at the time
  for name in sta_names:
    #precipitation DataFrame
    precipDF[name]          =  temporal_DF.loc[temporal_DF.name == name].drop_duplicates(subset='local_time').set_index('local_time').sort_index()['precip_rate']
    # Type of precipitation DF
    typeDF[name + '_type']  =  temporal_DF.loc[temporal_DF.name == name].drop_duplicates(subset='local_time').set_index('local_time').sort_index()['precip_type']
    #temperature DF
    tempDF[name]            =  temporal_DF.loc[temporal_DF.name == name].drop_duplicates(subset='local_time').set_index('local_time').sort_index()['temp_c']
    

  # replace nan 
  typeDF.fillna('notype', inplace = True)
  # Get mean values
  precipDF = precipDF.mean(axis = 1).rename('Precip').to_frame()
  return precipDF


def get_matched_data():
  msd_flow     = get_msdFlow()
  msd_names    = msd_flow.columns
  
  usgs_flow    = get_usgs_data()*0.646 # convert to MGD
  usgs_names   = usgs_flow.columns

  # match time series
  full_Df = usgs_flow.join(msd_flow).dropna().resample('D').asfreq().interpolate('linear')

  usgs_flow_match = full_Df[usgs_names]
  msd_match       = full_Df[msd_names]
  return msd_match, usgs_flow_match
