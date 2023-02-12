# Script to train machine learning model.

from sklearn.model_selection import train_test_split, cross_val_score
import joblib

# Add the necessary imports for the starter code.

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

import pandas as pd


# Add code to load in the data.

data = pd.read_csv('data/census.csv')
data.columns = data.columns.str.strip()


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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


X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )

print(X_train.shape)

# Train and save a model.
print('Training model ...')
model = train_model(X_train,y_train)

#Save model
print('Saving model ...')
filename = 'model/logistic_regression.joblib'
joblib.dump(model, filename)

