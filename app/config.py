from pathlib import Path
h = 4
lags = 10
t = lags - h
station = "10"



data_path = "datasets/train.csv"
model_file = "trained/RandomForest.joblib"
pipe_file = "trained/pipeline.pkl"


params = {'bootstrap': True,
 'max_depth': 80,
 'max_features': 3,
 'min_samples_leaf': 3,
 'min_samples_split': 12,
 'n_estimators': 10}



