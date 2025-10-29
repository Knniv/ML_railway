from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Crear la app
app = FastAPI(title="API de Modelos IA - Seguros y Diabetes")

# Cargar los modelos entrenados
pipe_lin = joblib.load("pipe_lin.joblib")
pipe_log = joblib.load("pipe_log.joblib")

# Umbral óptimo del modelo de diabetes
UMBRAL_IDEAL = 0.45
COLS_DIA_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


# ----- MODELO DE ENTRADA PARA COSTOS DE SEGURO -----
class SeguroInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


# ----- MODELO DE ENTRADA PARA DIABETES -----
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    threshold: float = UMBRAL_IDEAL


@app.get("/")
def home():
    return {"message": "API en funcionamiento. Usa /predict/seguro o /predict/diabetes"}


# ----- ENDPOINT DE SEGURO -----
@app.post("/predict/seguro")
def predict_seguro(data: SeguroInput):
    df = pd.DataFrame([data.dict()])
    prediction = pipe_lin.predict(df)[0]
    return {"prediccion": float(prediction), "mensaje": f"Costo estimado del seguro: ${prediction:,.2f}"}


# ----- ENDPOINT DE DIABETES -----
@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    df = pd.DataFrame([data.dict()])
    for col in COLS_DIA_MISSING:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    prob = pipe_log.predict_proba(df)[:, 1][0]
    clase = "Diabetes" if prob >= data.threshold else "No diabetes"
    return {
        "probabilidad": round(float(prob), 4),
        "umbral_usado": data.threshold,
        "prediccion": clase
    }
