# tests/test_models.py
import pytest
from src.models.predict_model import predict
import pandas as pd

def test_predict():
    sample_data = pd.DataFrame({
        'Gr Liv Area': [1500], 'Total Bsmt SF': [1000], 'Year Built': [2000], 'Lot Area': [8000],
        'Neighborhood': ['NAmes'], 'MS Zoning': ['RL'], 'Sale Condition': ['Normal']
    })
    preds = predict(sample_data)
    assert len(preds) == 1
    assert isinstance(preds[0], float)
    assert preds[0] > 0  # Reasonable house price