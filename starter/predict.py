


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

# Inference
# Load the saved model
print('Loading saved model ...')
filename = 'model/logistic_regression.joblib'
model = joblib.load(filename)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False
    )


# Use the loaded model to make predictions
preds = inference(model,X_test)

# Binarize the labels
binarized = encoder.fit_transform(y_test)

# Convert the binarized labels to a numpy array
y_test = np.array(binarized)

#Get model metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(precision)
print(recall)
print(fbeta)
print('... Success ...')

