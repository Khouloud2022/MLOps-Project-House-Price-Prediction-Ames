# tests/test_features.py
import joblib
import numpy as np
import os

def test_preprocessor_exists():
    assert os.path.exists("data/processed/preprocessor.pkl")

def test_preprocessor_output():
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    X_train = np.load("data/processed/X_train.npy")
    assert X_train.shape[1] > 50  # après OneHotEncoder
    assert not np.isnan(X_train).any()  # pas de NaN