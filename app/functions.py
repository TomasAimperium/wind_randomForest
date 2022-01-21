import pandas as pd
from scipy.signal import savgol_filter
import numpy as np

def table2lags(table, max_lag, min_lag=0, separator='_'):
    """ Given a dataframe, return a dataframe with different lags of all its columns """
    values=[]
    for i in range(min_lag, max_lag + 1):
        values.append(table.shift(i).copy())
        values[-1].columns = [c + separator + str(i) for c in table.columns]
    return pd.concat(values, axis=1)


def savgol(X):
    savgol_data = savgol_filter(X, 21, 1)
    return savgol_data




def filter_agg(X,station):


    data = X.reset_index(drop = True)
    
    data['Meteo Station 04 - Wind Speed(m/s)'] = data['Meteo Station 04 - Wind Speed(m/s)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 04 - Wind Direction(º)'] = data['Meteo Station 04 - Wind Direction(º)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 04 - Wind Direction Rad(rad)'] = data['Meteo Station 04 - Wind Direction Rad(rad)'].apply(lambda x : 0 if x<-10 else x)
    data['Meteo Station 04 - Atmospheric Pressure(mB)'] = data['Meteo Station 04 - Atmospheric Pressure(mB)'].apply(lambda x : 887.82 if x<500 else x)
    data['Meteo Station 04 - External Ambient Temperature(ºC)'] = data['Meteo Station 04 - External Ambient Temperature(ºC)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 04 - Humidity(%)'] = data['Meteo Station 04 - Humidity(%)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 10 - Wind Direction(º)'] = data['Meteo Station 10 - Wind Direction(º)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 10 - Wind Speed(m/s)'] = data['Meteo Station 10 - Wind Speed(m/s)'].apply(lambda x : 0 if x<-1000 else x)
    data['Meteo Station 10 - Wind Direction Rad(rad)'] = data['Meteo Station 10 - Wind Direction Rad(rad)'].apply(lambda x : 0 if x<-10 else x)
    data['Datetime'] =  pd.to_datetime(data['Datetime'], format='%Y-%m-%d %H:%M:%S')
    data_agg = data.resample('5Min', on='Datetime').mean()
    data_noNa = data_agg.dropna()

    target = 'Meteo Station '+ station +' - Wind Speed(m/s)'
    y = data_noNa.loc[:,[target]].values
     
    
    return pd.Series(y.reshape(len(y)))



def prepro(X,st):
    Y = savgol(filter_agg(X,st))
    return Y     