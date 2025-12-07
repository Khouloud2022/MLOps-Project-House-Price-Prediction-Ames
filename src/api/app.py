# src/api/app.py ← VERSION FINALE QUI MARCHE À 100%
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction - Khouloud Ouni")

# Chargement du modèle et préprocesseur
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("data/processed/preprocessor.pkl")

class HouseFeatures(BaseModel):
    Gr_Liv_Area: float = 1500
    Total_Bsmt_SF: float = 1000
    Year_Built: int = 2000
    Lot_Area: int = 8450
    Overall_Qual: int = 7
    Overall_Cond: int = 5
    Full_Bath: int = 2
    TotRms_AbvGrd: int = 7
    Garage_Cars: int = 2
    Garage_Area: int = 480
    First_Flr_SF: int = 1000
    Second_Flr_SF: int = 500
    Neighborhood: str = "NAmes"
    MS_Zoning: str = "RL"
    Sale_Condition: str = "Normal"
    Kitchen_Qual: str = "TA"
    Exter_Qual: str = "TA"
    Heating_QC: str = "Ex"
    Central_Air: str = "Y"
    Foundation: str = "PConc"

@app.get("/")
def home():
    return {"message": "API Prédiction Prix Maison - Khouloud Ouni", "docs": "/docs"}

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Toutes les colonnes nécessaires
        data = pd.DataFrame([{  
            'Gr Liv Area': features.Gr_Liv_Area,
            'Total Bsmt SF': features.Total_Bsmt_SF,
            'Year Built': features.Year_Built,
            'Lot Area': features.Lot_Area,
            'Overall Qual': features.Overall_Qual,
            'Overall Cond': features.Overall_Cond,
            'Full Bath': features.Full_Bath,
            'TotRms AbvGrd': features.TotRms_AbvGrd,
            'Garage Cars': features.Garage_Cars,
            'Garage Area': features.Garage_Area,
            '1st Flr SF': features.First_Flr_SF,
            '2nd Flr SF': features.Second_Flr_SF,
            'Neighborhood': features.Neighborhood,
            'MS Zoning': features.MS_Zoning,
            'Sale Condition': features.Sale_Condition,
            'Kitchen Qual': features.Kitchen_Qual,
            'Exter Qual': features.Exter_Qual,
            'Heating QC': features.Heating_QC,
            'Central Air': features.Central_Air,
            'Foundation': features.Foundation
        }])

        X = preprocessor.transform(data)
        prediction = model.predict(X)[0]

        return {"predicted_sale_price": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))