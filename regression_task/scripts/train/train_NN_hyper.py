import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from urllib.parse import urlparse
import mlflow
import mlflow.tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K 
from tensorflow import convert_to_tensor, random

# sys argv
Layer1 = int(sys.argv[1]) if len(sys.argv) > 1 else 16
Layer2 = int(sys.argv[2]) if len(sys.argv) > 2 else 16
batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 10
learning_rate = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
dropout_rate = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0

def eval_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mse, mae, r2

# Create the Neural Network model
def create_model(Layer1, Layer2, dropout_rate):
    model = Sequential()
    model.add(Dense(Layer1, activation='relu', input_dim=train_x.shape[1]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(Layer2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='relu'))
    random.set_seed(12345)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

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
param_grid = {
    'batch_size': [16],
    'learning_rate': [0.001, 0.0001,0.005],
    'dropout_rate': [0.0, 0.2],
    'epochs_num': [15,20,25,30],
}

# set mlflow experiment
mlflow.set_experiment("neural_network_wine")
experiment = mlflow.get_experiment_by_name("neural_network_wine")

for epochs in param_grid["epochs_num"]:
    for batch_size in param_grid['batch_size']:
        for learning_rate in param_grid['learning_rate']:
            for dropout_rate in param_grid['dropout_rate']:
                # Your code here
                with mlflow.start_run(experiment_id=experiment.experiment_id):
                    # Create model
                    model = create_model(Layer1, Layer2, dropout_rate)
                    K.set_value(model.optimizer.learning_rate, learning_rate)
                    # Fit model
                    random.set_seed(12345)
                    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=0)
                    # train errors
                    fitted_qualities = model.predict(train_x)
                    (mse_train, mae_train, r2_train) = eval_metrics(train_y, fitted_qualities)

                    # test errors
                    predicted_qualities = model.predict(test_x)
                    (mse_test, mae_test, r2_test) = eval_metrics(test_y, predicted_qualities)

                    # log parameters
                    mlflow.log_param("Layer1", Layer1)
                    mlflow.log_param("Layer2", Layer2)
                    mlflow.log_param("dropout_rate", dropout_rate)
                    mlflow.log_param("learning_rate", learning_rate)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("epochs", epochs)

                    # log train metrics
                    mlflow.log_metric("mse_train", mse_train)
                    mlflow.log_metric("r2_train", r2_train)
                    mlflow.log_metric("mae_train", mae_train)

                    # log test metrics
                    mlflow.log_metric("mse_test", mse_test)
                    mlflow.log_metric("r2_test", r2_test)
                    mlflow.log_metric("mae_test", mae_test)
