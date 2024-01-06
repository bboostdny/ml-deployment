import requests

data = {
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

url = "https://udacity-ml-deployment-1d974bbfc827.herokuapp.com/inference"
response = requests.post(url, json=data)
print(f'response code: {response.status_code}')
print(f'Prediction: {response.json()}')