import pandas as pd
import numpy as np
from functions import savgol
from functions import table2lags
from functions import filter_agg
from functions import prepro
from pipeline import pipe
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import config

import path

from pathlib import Path
BASE_DIR = Path(__file__).resolve(strict=True).parent


def train():
	print("############")
	print("loading data")
	print(" ")
	file = Path(BASE_DIR).joinpath(config.data_path)

	all_data = pd.read_csv(file,sep=";",decimal=",")
	data = pd.DataFrame(prepro(all_data,config.station),columns = ["y"])
	df = table2lags(data,config.lags).dropna().reset_index(drop = True)
	X = df[df.columns[config.t:]].values
	X_ = pipe.fit_transform(X)
	

	joblib.dump(pipe, Path(BASE_DIR).joinpath(config.pipe_file))
	y = df.iloc[:,0].values


	print("############")
	print("data loaded")
	print(" ")

	X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.33, random_state=42,shuffle=False)

	p = config.params

	rfr = RandomForestRegressor(bootstrap = p["bootstrap"],
	                           max_depth = p["max_depth"],
	                           max_features = p["max_features"],
	                           min_samples_leaf = p["min_samples_leaf"],
	                           min_samples_split = p["min_samples_split"],
	                           n_estimators = p["n_estimators"],verbose = 0)

	print("############")
	print("training")
	rfr.fit(X_train,y_train)

	pred_train = rfr.predict(X_train)
	pred_test = rfr.predict(X_test)



	outputs = {'mape_train': mean_absolute_percentage_error(pred_train,y_train),
	 'mape_test': mean_absolute_percentage_error(pred_test,y_test),
	 'mse_train': mean_squared_error(pred_train,y_train),
	 'mse_test': mean_squared_error(pred_test,y_test),
	 'r_2score_train': r2_score(pred_train,y_train),
	 'r_2score_test': r2_score(pred_test,y_test),
	 'hyperparameters':p,
	 'train_size': len(y_train),
	 'test_size': len(y_test)}


	return outputs
	
	

def predict(inputs):

	pkl = joblib.load(Path(BASE_DIR).joinpath(config.pipe_file))


	if not Path(BASE_DIR).joinpath(config.pipe_file).exists():
		return False
	

	rfr = joblib.load(Path(BASE_DIR).joinpath(config.model_file))
	y = pd.DataFrame(inputs,columns = ["y"])
	X = pkl.transform(table2lags(y,config.lags).dropna().reset_index(drop = True).iloc[:,:config.t-1].values)
	prediction_list = pd.Series(rfr.predict(X))[:(config.t)].to_dict()

	return prediction_list
	#return "culo"










