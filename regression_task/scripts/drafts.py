import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from scipy.stats import uniform
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


# Read clean datasets
train = pd.read_csv("dataset/train_clean.csv")
test = pd.read_csv("dataset/test_clean.csv")

# divide into x and y
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# use randomized search and cross validation

distributions = {
        'alpha': uniform(loc=0, scale=1),
        'l1_ratio': uniform(loc=0, scale=1)
    }

# Create the ElasticNet model
lr = ElasticNet(random_state=12345)

# Create the RandomizedSearchCV object
rs_cv = RandomizedSearchCV(lr, distributions, scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
                                cv=5, n_iter=2, random_state=12345,
                                verbose=3, return_train_score=True,n_jobs=-1,
                                refit="neg_mean_squared_error")

# fit randomized search
rs_cv.fit(train_x, train_y)
# print results_
results_df = pd.DataFrame(rs_cv.cv_results_)
# print dataframe
results_df = results_df[["params","mean_train_neg_mean_squared_error","mean_test_neg_mean_squared_error",
                        "mean_train_neg_mean_absolute_error","mean_test_neg_mean_absolute_error",
                        "mean_train_r2","mean_test_r2",
                        "mean_fit_time","mean_score_time"]]


print(results_df[["params","mean_train_neg_mean_squared_error","mean_test_neg_mean_squared_error",
                    "mean_train_neg_mean_absolute_error","mean_test_neg_mean_absolute_error",
                    "mean_train_r2","mean_test_r2",
                    "mean_fit_time","mean_score_time"]])
print(results_df.columns)
print(rs_cv.cv_results_)
print(results_df.values)