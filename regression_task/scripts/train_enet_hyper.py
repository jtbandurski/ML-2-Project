import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


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

# iterate over parameters
for alpha in [0.1, 0.5]:
  for l1_ratio in [0.5]:
    with mlflow.start_run(experiment_id=0):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
        lr.fit(train_x, train_y)

        # train errors
        fitted_qualities = lr.predict(train_x)
        (rmse_train, mae_train, r2_train) = eval_metrics(train_y, fitted_qualities)

        # test errors
        predicted_qualities = lr.predict(test_x)
        (rmse_test, mae_test, r2_test) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        
        print("Train Errors:")
        print("  RMSE: %s" % rmse_train)
        print("  MAE: %s" % mae_train)
        print("  R2: %s" % r2_train)

        print("Test Errors:")
        print("  RMSE: %s" % rmse_test)
        print("  MAE: %s" % mae_test)
        print("  R2: %s" % r2_test)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        # log train metrics
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("mae_train", mae_train)
        # log test metrics
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_test", r2_test)
        mlflow.log_metric("mae_test", mae_test)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, f"model_{alpha}_{l1_ratio}", registered_model_name="ElasticnetWineModel",
                                        signature=mlflow.models.infer_signature(train_x, train_y))
        else:
            mlflow.sklearn.log_model(lr, f"model_{alpha}_{l1_ratio}",
                                        signature=mlflow.models.infer_signature(train_x, train_y))