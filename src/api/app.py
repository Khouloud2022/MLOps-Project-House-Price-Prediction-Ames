# src/api/app.py  ← VERSION CORRIGÉE 100% FONCTIONNELLE
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="House Price Prediction API", version="1.0")

# Chargement du modèle et préprocesseur
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("data/processed/preprocessor.pkl")

class HouseFeatures(BaseModel):
    Gr_Liv_Area: float
    Total_Bsmt_SF: float
    Year_Built: int
    Lot_Area: int
    Overall_Qual: int
    Overall_Cond: int = 5
    Full_Bath: int = 2
    TotRms_AbvGrd: int = 7
    Garage_Cars: int = 2
    Garage_Area: int = 500
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
    return {"message": "API House Price Prediction - allez sur /docs pour tester"}

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Conversion en DataFrame avec TOUTES les colonnes attendues
        input_data = pd.DataFrame([{
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

        # Prétraitement + prédiction
        X_processed = preprocessor.transform(input_data)
        prediction = model.predict(X_processed)[0]

        return {"predicted_sale_price": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)