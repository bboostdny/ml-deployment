import sys
import os
import pickle
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import pandas as pd
from main import app
from ml.data import process_data
from fastapi.testclient import TestClient


client = TestClient(app)

with open('model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open(f'./model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)


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

def test_say_hello():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello World!"}


def test_run_inference_above_50k():
    test_data = {
        "age": 45,
        "workclass": "Private",
        "education": "Prof-school",
        "education-num": 15,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 8564,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    prediction = client.post("/inference", json=test_data)

    assert prediction.status_code == 200
    assert prediction.json() == {"prediction": ">50k"}


def test_run_inference_below_50k() -> None:
    test_data = {
        "age": 23,
        "workclass": "Private",
        "education": "9th",
        "education-num": 5,
        "marital-status": "Never-married",
        "occupation": "Farming-fishing",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 721,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    prediction = client.post("/inference", json=test_data)

    assert prediction.status_code == 200
    assert prediction.json() == {"prediction": "<50k"}
