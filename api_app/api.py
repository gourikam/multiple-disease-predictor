#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:02:42 2025

@author: gourikamakhija
"""

import joblib
import numpy as np
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(title="Multiple Disease Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models (using relative paths for Heroku compatibility)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "diabetes_model.sav")
diabetes_model = joblib.load(MODEL_PATH)
heart_model = joblib.load(os.path.join(os.path.dirname(__file__), "saved_models", "heart_model.sav"))
parkinsons_model = joblib.load(os.path.join(os.path.dirname(__file__), "saved_models", "parkinsons_model.sav"))

# Create data models for inputs
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

class HeartDiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class ParkinsonsInput(BaseModel):
    fo: float
    fhi: float
    flo: float
    Jitter_percent: float
    Jitter_Abs: float
    RAP: float
    PPQ: float
    DDP: float
    Shimmer: float
    Shimmer_dB: float
    APQ3: float
    APQ5: float
    APQ: float
    DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    D2: float
    PPE: float

# Create API endpoints
@app.get("/")
def home():
    return {"message": "Welcome to the Disease Prediction API"}

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    # Convert input data to array
    input_data = np.array([
        data.Pregnancies, 
        data.Glucose, 
        data.BloodPressure, 
        data.SkinThickness, 
        data.Insulin, 
        data.BMI, 
        data.DiabetesPedigreeFunction, 
        data.Age
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = diabetes_model.predict(input_data)[0]
    
    # Return result
    if prediction == 1:
        return {"result": "The Person is Diabetic", "prediction": int(prediction)}
    else:
        return {"result": "The Person is Not Diabetic", "prediction": int(prediction)}

@app.post("/predict/heart")
def predict_heart_disease(data: HeartDiseaseInput):
    # Convert input data to array
    input_data = np.array([
        data.age, 
        data.sex, 
        data.cp, 
        data.trestbps, 
        data.chol, 
        data.fbs, 
        data.restecg, 
        data.thalach, 
        data.exang, 
        data.oldpeak, 
        data.slope, 
        data.ca, 
        data.thal
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = heart_model.predict(input_data)[0]
    
    # Return result
    if prediction == 1:
        return {"result": "The person is having heart disease", "prediction": int(prediction)}
    else:
        return {"result": "The person does not have any heart disease", "prediction": int(prediction)}

@app.post("/predict/parkinsons")
def predict_parkinsons(data: ParkinsonsInput):
    # Convert input data to array
    input_data = np.array([
        data.fo, data.fhi, data.flo, data.Jitter_percent, data.Jitter_Abs,
        data.RAP, data.PPQ, data.DDP, data.Shimmer, data.Shimmer_dB, 
        data.APQ3, data.APQ5, data.APQ, data.DDA, data.NHR, data.HNR, 
        data.RPDE, data.DFA, data.spread1, data.spread2, data.D2, data.PPE
    ]).reshape(1, -1)
    
    # Make prediction
    prediction = parkinsons_model.predict(input_data)[0]
    
    # Return result
    if prediction == 1:
        return {"result": "The person has Parkinson's disease", "prediction": int(prediction)}
    else:
        return {"result": "The person does not have Parkinson's disease", "prediction": int(prediction)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)