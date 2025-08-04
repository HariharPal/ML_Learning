from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler
from pydantic_schemas.features import Features

from pydantic_schemas.features import Features
app = FastAPI()

ridge_model  = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.get("/")
def test():
    return "Hello World"

@app.post("/fwi")
def fwiPrediction(features: Features):
    # Use the instance attributes of features
    new_data_scaled = standard_scaler.transform([[features.Temperature, features.RH, features.Ws, features.Rain, features.FFMC, features.DMC,  features.ISI, features.Classes, features.Region]])
    result = ridge_model.predict(new_data_scaled)
    return result[0]
