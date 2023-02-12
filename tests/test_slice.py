import pandas as pd
import pytest


@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""

    data = pd.read_csv('data/census.csv')
    data.columns = data.columns.str.strip()
    return data


def test_data_shape(data):
    """ Tests if our data has all 6513 rows containing 107 features and 1 target variable each"""
    assert data.shape == (6513, 108), "Data does not have the expected shape."

