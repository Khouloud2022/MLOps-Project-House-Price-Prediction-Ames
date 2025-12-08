# tests/test_model.py
import joblib
import numpy as np
import os

def test_model_exists():
    assert os.path.exists("models/best_model.pkl")

def test_model_prediction():
    model = joblib.load("models/best_model.pkl")
    X_test = np.load("data/processed/X_test.npy")
    pred = model.predict(X_test[:5])
    assert len(pred) == 5
    assert all(pred > 0)  # prix positif