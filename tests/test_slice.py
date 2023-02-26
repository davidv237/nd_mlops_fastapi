import pandas as pd
import numpy as np
import pytest
import joblib
import subprocess
from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from starter.ml.data import process_data


import os

# Reading environment variable named "ENVIRONMENT_VARIABLE_NAME"
setting = os.getenv("SETTING")

if setting == "development":
    print(f"Environment variable value: {setting}")
    @pytest.fixture
    def data():
        """ Simple function to generate some fake Pandas data."""
        #data = pd.read_csv('data/census.csv')
        data = pd.read_csv('./data/census.csv')

        data.columns = data.columns.str.strip()
        return data
else:
    print("Environment variable not set.")

    @pytest.fixture
    def data():
        data_path = os.path.join(os.environ["DVC_CACHE_DIR"], "my-cache-dir", "data.csv")
        # dvc_cache_dir = subprocess.check_output(["dvc", "cache", "dir", "--show"]).decode().strip()
        print(data_path)
        # List all the files in the cache directory
        files = os.listdir(data_path)

        # Print the list of files
        print("Files in DVC cache directory:")
        for file in files:
            print(file)



         # Test that the data file exists in the cache directory
        data_file_path = os.path.join(dvc_cache_dir, "data/census.csv")
        assert os.path.isfile(data_file_path), f"Data file {data_file_path} not found"

    # Add more tests here...
    # def data():
    #     """ Simple function to generate some fake Pandas data."""
    #     #data = pd.read_csv('data/census.csv')
    #     data = pd.read_csv('./data/census.csv')

    #     data.columns = data.columns.str.strip()
    #     return data


@pytest.fixture
def cat_features():
    """ Simple function to generate some fake Pandas data."""
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
    return cat_features


def test_data_shape(data):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    assert data.shape == (32561, 15) , "Data does not have the expected shape."


def test_process_data(data, cat_features):
    """ Tests the process data function"""
    X,y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
    )
    assert isinstance(X, np.ndarray) , "Data does not have the expected shape."
    assert X.size != 0, "Data cannot be empty."


def test_train_models(data, cat_features):
    """ Tests train models function"""

    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train,y_train)
    assert isinstance(model, LogisticRegression)
    assert model.C == 1.0


def test_make_predictions(data, cat_features):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""

    train, test = train_test_split(data, test_size=0.20)

    X_test, y_test, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=False
    )
    filename = 'model/logistic_regression.joblib'
    model = joblib.load(filename)
    preds = inference(model,X_test)

    assert isinstance(model, LogisticRegression)
    assert preds.size != 0

def test_model_performance(data, cat_features):
    """ Tests model performance"""
    pass
