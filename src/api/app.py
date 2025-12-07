# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and preprocessor
model = joblib.load('models/best_model.pkl')
preprocessor = joblib.load('data/processed/preprocessor.pkl')

class HouseFeatures(BaseModel):
    Gr_Liv_Area: float
    Total_Bsmt_SF: float
    Year_Built: int
    Lot_Area: int
    Neighborhood: str
    MS_Zoning: str
    Sale_Condition: str

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        data = pd.DataFrame([features.dict()])
        X = preprocessor.transform(data)
        prediction = model.predict(X)[0]
        return {"predicted_sale_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)