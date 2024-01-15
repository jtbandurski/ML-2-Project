import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Load data from r2.csv
data = pd.read_csv('../dataset/r2.csv')

# Perform train-test split 80/20
train_data, test_data = train_test_split(data, test_size=0.2, random_state=12345)
# Save train and test data to CSV files
train_data.to_csv('../dataset/train.csv', index=False)
test_data.to_csv('../dataset/test.csv', index=False)
