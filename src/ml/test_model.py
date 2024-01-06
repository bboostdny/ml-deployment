import pytest
import pandas as pd
import numpy as np
import pickle
from .model import train_model, compute_model_metrics, inference
from .data import process_data
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def data():
    """ Retrieve and process training dataset."""
    df = pd.read_csv('data/census.csv')
    df = df.drop("fnlgt", axis=1)
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

    X_train, y_train, _, _ = process_data(df, cat_features, label='salary', training=True)

    return X_train, y_train


@pytest.fixture
def trained_model():
    """ Load trained model"""
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)

    return model


def test_train_model(data):
    X_train, y_train = data
    model = train_model(X_train, y_train)
    assert isinstance(model, type(RandomForestClassifier()))


def test_inference(data, trained_model):
    X_train, _= data
    predictions = inference(trained_model, X_train)
    # Check if number of predictions is equal to count of input data
    assert X_train.shape[0] == predictions.shape[0]
    # Check if predictions are binary
    assert np.all(np.logical_or(predictions == 0, predictions == 1))

def test_compute_model_metrics(data, trained_model):
    X_train, y_train = data
    predictions = trained_model.predict(X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, predictions)
    metrics = [precision, recall, fbeta]
    # Check if all metrics are floats
    assert all([isinstance(x, float) for x in metrics])
    # Check if all metrics have value between 0 and 1
    assert all([0 <= x <= 1 for x in metrics])
