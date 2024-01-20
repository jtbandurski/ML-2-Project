import pandas as pd

# Load the data
train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
# Define the columns to drop
columns_to_drop = ['feat01', 'feat02', "feat03","feat05","feat06","feat08","feat09","feat10"]

# Drop the columns
train.drop(columns_to_drop, axis=1, inplace=True)
test.drop(columns_to_drop, axis=1, inplace=True)

# Save the cleaned data
train.to_csv('../dataset/train_clean.csv', index=False)
test.to_csv('../dataset/test_clean.csv', index=False)
