


from sklearn.model_selection import train_test_split, cross_val_score
import joblib

# Add the necessary imports for the starter code.

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

import pandas as pd
import numpy as np


# Add code to load in the data.

data = pd.read_csv('data/census.csv')
data.columns = data.columns.str.strip()

print("data.shape")
print(data.shape)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

print("train.shape")
print(train.shape)

print("test.shape")
print(test.shape)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Inference
# Load the saved model
print('Loading saved model ...')
filename = 'model/logistic_regression.joblib'
model = joblib.load(filename)

# Proces the test data with the process_data function.
print("Processing data")
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
    )

print("y_test.shape")
print(y_test.shape)

print("y_test")
print(y_test)
print(type(y_test))


# print("X_test.shape")
# print(X_test.shape)

# print("X_test")
# print(X_test)

# Use the loaded model to make predictions
preds = inference(model,X_test)

# Create a pandas Series object
s = pd.Series(y_test)

# Convert the Series object to a numpy array
arr = y_test.values
print(arr)

# Reshape the numpy array
reshaped_arr = arr.reshape(-1,1)
print(reshaped_arr)

binarized = lb.fit_transform(y_test)
# print(binarized)


print(binarized)

y_test = binarized


# print("y_test.shape")
# print(y_test.shape)

# print("y_test")
# print(y_test)

# Get model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(precision)
print(recall)
print(fbeta)
print('... Success ...')

