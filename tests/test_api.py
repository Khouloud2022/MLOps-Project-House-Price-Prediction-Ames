# tests/test_api.py
import requests
import json

def test_api_endpoint():
    payload = {
        "Gr_Liv_Area": 1700, "Total_Bsmt_SF": 856, "Year_Built": 2003,
        "Lot_Area": 8450, "Overall_Qual": 7, "Overall_Cond": 5,
        "Full_Bath": 2, "TotRms_AbvGrd": 8, "Garage_Cars": 2,
        "Garage_Area": 548, "First_Flr_SF": 856, "Second_Flr_SF": 854,
        "Neighborhood": "CollgCr", "MS_Zoning": "RL", "Sale_Condition": "Normal",
        "Kitchen_Qual": "Gd", "Exter_Qual": "Gd", "Heating_QC": "Ex",
        "Central_Air": "Y", "Foundation": "PConc"
    }
    
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "predicted_sale_price" in result
    assert result["predicted_sale_price"] > 100000