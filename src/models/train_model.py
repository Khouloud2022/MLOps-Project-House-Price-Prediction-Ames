# src/models/train_model.py  ← VERSION FINALE 100% FONCTIONNELLE
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Fix le problème d'expériment ID 0 + évite les warnings
mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns"))
mlflow.set_experiment("House_Price_Prediction")  # Crée ou récupère l'expériment

def train_model():
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

    # Modèle 1 : Régression Linéaire
    with mlflow.start_run(run_name="LinearRegression"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(lr, "model")
        print(f"LinearRegression → RMSE: {rmse:,.0f} $")

    # Modèle 2 : Random Forest (meilleur modèle)
    with mlflow.start_run(run_name="RandomForest_Best"):
        rf = RandomForestRegressor(
            n_estimators=400,
            max_depth=25,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        mlflow.log_metric("RMSE", rmse)
        mlflow.sklearn.log_model(rf, "model")
        print(f"RandomForest → RMSE: {rmse:,.0f} $")

        # Sauvegarde du meilleur modèle
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf, 'models/best_model.pkl')

    print("Entraînement terminé avec succès ! Prêt pour la soutenance !")

if __name__ == "__main__":
    train_model()