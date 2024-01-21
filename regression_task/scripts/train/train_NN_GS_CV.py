import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scikeras.wrappers import KerasRegressor
from scipy.stats import uniform
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2

# sys argv
Layer1 = int(sys.argv[1]) if len(sys.argv) > 1 else 16
Layer2 = int(sys.argv[2]) if len(sys.argv) > 2 else 16
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 10
learning_rate = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
dropout_rate = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0

warnings.filterwarnings("ignore")
np.random.seed(12345)

# Read clean datasets
train = pd.read_csv("dataset/train_clean.csv")
test = pd.read_csv("dataset/test_clean.csv")

# divide into x and y
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# use grid search and cross validation
# Define the parameter grid
param_grid = {
    'batch_size': [8, 16, 32],
    'epochs': [5, 10, 20],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.0, 0.2, 0.4],
}

# Create the Neural Network model
def create_model(meta):
    model = Sequential()
    model.add(Dense(Layer1, activation='relu', input_dim=train_x.shape[1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(Layer2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = KerasRegressor(build_fn=create_model,loss="mean_squared_error", verbose=0)
# Create the GridSearchCV object
gs_cv = GridSearchCV(model, param_grid, scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
                     cv=5, verbose=3, return_train_score=True, n_jobs=-1, refit="neg_mean_squared_error")

# Fit grid search
gs_cv.fit(train_x, train_y)
# print results_
results_df = pd.DataFrame(gs_cv.cv_results_)
# choose columns
results_df = results_df[["params","mean_train_neg_mean_squared_error","mean_test_neg_mean_squared_error",
                        "mean_train_neg_mean_absolute_error","mean_test_neg_mean_absolute_error",
                        "mean_train_r2","mean_test_r2",
                        "mean_fit_time","mean_score_time"]]

# set mlflow experiment
mlflow.set_experiment("neural_network_wine_lrate_ntree")
experiment = mlflow.get_experiment_by_name("neural_network_wine_lrate_ntree")

for run in results_df.values:
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # log parameters
        for param in run[0].keys():
            mlflow.log_param(param, run[0][param])
        # log train metrics
        mlflow.log_metric("mse_train", -run[1])
        mlflow.log_metric("r2_train", run[5])
        mlflow.log_metric("mae_train", -run[3])
        # log validation metrics
        mlflow.log_metric("mse_valid", -run[2])
        mlflow.log_metric("r2_valid", run[6])
        mlflow.log_metric("mae_valid", -run[4])
        # log time metrics
        mlflow.log_metric("fit_time", run[7])
        mlflow.log_metric("score_time", run[8])
        
        # calculate full train metrics
        model = Sequential()
        model.add(Dense(Layer1, activation='relu', input_dim=train_x.shape[1]))
        model.add(Dropout(dropout_rate))
        model.add(Dense(Layer2, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)

        # predict on test set
        predicted_qualities = model.predict(test_x)
        # calculate metrics
        # full train errors
        fitted_qualities = model.predict(train_x)
        (mse_train, mae_train, r2_train) = eval_metrics(train_y, fitted_qualities)
        # full test errors
        (mse_test, mae_test, r2_test) = eval_metrics(test_y, predicted_qualities)
        # log test metrics
        mlflow.log_metric("mse_test", mse_test)
        mlflow.log_metric("r2_test", r2_test)
        mlflow.log_metric("mae_test", mae_test)
        # log full train metrics
        mlflow.log_metric("mse_train_full", mse_train)
        mlflow.log_metric("r2_train_full", r2_train)
        mlflow.log_metric("mae_full", r2_train)
        # log model

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(gs_cv.best_estimator_, f"model_{'_'.join([k for k in run[0].keys()])}", registered_model_name="NeuralNetworkWineModel",
                                        signature=mlflow.models.infer_signature(train_x, train_y))
        else:
            mlflow.sklearn.log_model(gs_cv.best_estimator_,  f"model_{'_'.join([k for k in run[0].keys()])}",
                                        signature=mlflow.models.infer_signature(train_x, train_y))