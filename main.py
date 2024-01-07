# Put the code for your API here.
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.ml.data import process_data
from src.ml.model import inference

app = FastAPI()

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

with open(f'./model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'./model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open(f'./model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)

class Data(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example":
                {
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
            }


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

# Use POST action to send data to the server
@app.post("/inference")
async def run_inference(data: Data):
    data = pd.DataFrame(data.dict(), index=[0])
    data.columns = data.columns.str.replace("_", "-")
    X, _, _, _ = process_data(data, cat_features, training=False, encoder=encoder, lb=lb)
    prediction = model.predict(X)

    if prediction == 0:
        pred = "<50k"
    elif prediction == 1:
        pred = ">50k"

    return {"prediction": pred}
