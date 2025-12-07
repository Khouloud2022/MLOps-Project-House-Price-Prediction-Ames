# src/models/predict_model.py
import pandas as pd
import joblib

def predict(input_data, model_path='models/best_model.pkl', preprocessor_path='data/processed/preprocessor.pkl'):
    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Preprocess input (assume input_data is a DataFrame with raw features)
    X = preprocessor.transform(input_data)
    
    # Predict
    predictions = model.predict(X)
    return predictions

# Example usage
if __name__ == "__main__":
    sample_data = pd.DataFrame({
        'Gr Liv Area': [1500], 'Total Bsmt SF': [1000], 'Year Built': [2000], 'Lot Area': [8000],
        'Neighborhood': ['NAmes'], 'MS Zoning': ['RL'], 'Sale Condition': ['Normal']
    })
    preds = predict(sample_data)
    print(f"Predicted SalePrice: {preds[0]}")