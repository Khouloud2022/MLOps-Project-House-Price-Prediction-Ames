# src/models/train_model.py
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model():
    # Load data
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

    # === Train Linear Regression ===
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)
        rmse = mean_squared_error(y_test, pred, squared=False)
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(lr, "model")
        print(f"LinearRegression RMSE: {rmse:.2f}")

    # === Train Random Forest ===
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(rf, "model")
        print(f"RandomForest RMSE: {rmse:.2f}")

        # Save best model (you can change condition)
        if rmse < 30000:  # Good threshold for Ames
            os.makedirs('models', exist_ok=True)
            joblib.dump(rf, 'models/best_model.pkl')

    print("Training completed!")

if __name__ == "__main__":
    train_model()