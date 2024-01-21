import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from scipy.stats import uniform, randint
from urllib.parse import urlparse

import mlflow
import mlflow.xgboost

def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2

# sys argv
n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 2
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
n_estimators_left = int(sys.argv[3]) if len(sys.argv) > 3 else 100
n_estimators_right = int(sys.argv[4]) if len(sys.argv) > 4 else 500
max_depth = int(sys.argv[5]) if len(sys.argv) > 5 else 6
subsample_left = float(sys.argv[6]) if len(sys.argv) > 6 else 0
subsample_right = float(sys.argv[7]) if len(sys.argv) > 7 else 1
min_child_weight = int(sys.argv[8]) if len(sys.argv) > 8 else 10

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
distributions = {
    'learning_rate': uniform(loc=learning_rate*0.5, scale=learning_rate*1.5), 
    'n_estimators': randint(low=n_estimators_left, high=n_estimators_right),
    'max_depth': randint(low=2, high=max_depth),
    'subsample': uniform(loc=subsample_left, scale=subsample_right),
    'min_child_weight': randint(low=1, high=min_child_weight),
}

# Create the XGBRegressor model
xgb_model = xgb.XGBRegressor(random_state=12345)

# Create the RandomizedSearchCV object
gs_cv = RandomizedSearchCV(xgb_model, distributions, scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
                            n_iter=n_iter,cv=5, verbose=3, return_train_score=True, n_jobs=-1, refit="neg_mean_squared_error")

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
mlflow.set_experiment("xgboost_wine_lrate_ntree")
experiment = mlflow.get_experiment_by_name("xgboost_wine_lrate_ntree")

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
        # use all parameters to train
        # model = xgb.XGBRegressor(learning_rate=run[0]["learning_rate"], n_estimators=run[0]["n_estimators"], max_depth=run[0]["max_depth"],
        #                         subsample=run[0]["subsample"], min_child_weight=run[0]["min_child_weight"], random_state=12345)
        # model = xgb.XGBRegressor(learning_rate=run[0]["learning_rate"], n_estimators=run[0]["n_estimators"], random_state=12345)
        model = xgb.XGBRegressor()
        model.fit(train_x, train_y)

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
            mlflow.xgboost.log_model(gs_cv.best_estimator_, f"model_{'_'.join([k for k in run[0].keys()])}", registered_model_name="XGBoostWineModel",
                                        signature=mlflow.models.infer_signature(train_x, train_y))
        else:
            mlflow.xgboost.log_model(gs_cv.best_estimator_,  f"model_{'_'.join([k for k in run[0].keys()])}",
                                        signature=mlflow.models.infer_signature(train_x, train_y))