# System Architecture for House Price Prediction MLOps Project

## Overview
This document describes the high-level architecture of the House Price Prediction system, built following MLOps principles. The system implements an end-to-end machine learning pipeline for predicting house prices using the Ames Housing dataset. It covers data processing, feature engineering, model training, deployment, monitoring, and continuous integration/delivery.

The architecture ensures reproducibility, scalability, and maintainability by leveraging tools like Git for code versioning, DVC for data and model versioning, MLflow for experiment tracking, Docker for containerization, FastAPI for the inference API, GitHub Actions for CI/CD, and Evidently for monitoring.

## Key Components

### 1. Data Ingestion and Processing
- **Description**: Raw data (Ames Housing CSV) is ingested and processed (cleaning, handling missing values, splitting into train/test sets).
- **Tools**: Pandas for data manipulation, stored in `data/raw/` and `data/processed/`.
- **Scripts**: `src/data/make_dataset.py`.
- **Versioning**: DVC tracks data files.

### 2. Feature Engineering
- **Description**: Transforms raw features (e.g., scaling numerical features like 'Gr Liv Area', one-hot encoding categoricals like 'Neighborhood').
- **Tools**: Scikit-learn (StandardScaler, OneHotEncoder).
- **Scripts**: `src/features/build_features.py`.
- **Output**: Preprocessor saved as `.pkl`, feature-engineered datasets.

### 3. Model Training and Experimentation
- **Description**: Trains models (e.g., Linear Regression, Random Forest), evaluates with metrics like RMSE, and logs experiments.
- **Tools**: Scikit-learn/XGBoost for models, MLflow for tracking params, metrics, and artifacts.
- **Scripts**: `src/models/train_model.py`.
- **Versioning**: Models saved as `.pkl`, tracked in MLflow.

### 4. Inference Pipeline
- **Description**: Loads trained model and preprocessor to make predictions on new data.
- **Tools**: Joblib for loading artifacts.
- **Scripts**: `src/models/predict_model.py`.

### 5. API Serving
- **Description**: Exposes a REST API for real-time predictions.
- **Tools**: FastAPI for the web framework, Uvicorn as server.
- **Scripts**: `src/api/app.py`.
- **Endpoints**: POST `/predict` accepts house features and returns predicted price.

### 6. Containerization
- **Description**: Packages training and inference environments for consistency.
- **Tools**: Docker.
- **Files**: `docker/Dockerfile.train` for training, `docker/Dockerfile.infer` for API.
- **Deployment**: Images can be pushed to a registry (e.g., Docker Hub) and deployed to cloud (e.g., AWS EC2).

### 7. CI/CD Pipeline
- **Description**: Automates testing, building, and deployment on code changes.
- **Tools**: GitHub Actions.
- **Workflow**: `.github/workflows/ci-cd.yaml` - runs tests, builds Docker images, deploys to staging/prod.
- **Triggers**: On push/PR to main branch.

### 8. Monitoring and Maintenance
- **Description**: Monitors model performance, detects data drift, and logs predictions.
- **Tools**: Evidently for drift reports, Prometheus/Grafana (optional for advanced metrics).
- **Scripts**: `src/monitoring/monitor.py`.
- **Alerts**: Basic threshold-based alerts for drift.

### 9. Overall Pipeline Orchestration
- **Description**: End-to-end reproducibility from data to deployment.
- **Tools**: DVC for defining stages (process_data -> build_features -> train_model).
- **File**: `dvc.yaml` - Run with `dvc repro`.

## Data Flow
1. Raw data → Processing → Features → Training → Model artifact.
2. Model → Inference API → Deployment.
3. Prod data → Monitoring → Alerts/Retraining.

## Architecture Diagram

```mermaid
graph TD
    A[Raw Data] --> B[Data Processing<br>(make_dataset.py)]
    B --> C[Feature Engineering<br>(build_features.py)]
    C --> D[Model Training<br>(train_model.py, MLflow)]
    D --> E[Model Artifact<br>(.pkl)]
    E --> F[Inference API<br>(app.py, FastAPI)]
    F --> G[Deployment<br>(Docker, Cloud)]
    H[CI/CD<br>(GitHub Actions)] --> G
    I[Monitoring<br>(monitor.py, Evidently)] --> G
    G --> J[Predictions & Logs]
    J --> I
```
