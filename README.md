# House Price Prediction – Full MLOps Project

**Regression project** (Ames Housing Dataset) – 100% compliant with the cahier des charges

## Overview
End-to-end MLOps pipeline for predicting house prices in Ames, Iowa.  
All 8 weeks of the course are covered.

## Features Implemented
- Git + standard structure
- DVC for data & model versioning
- MLflow experiment tracking
- Reproducible pipeline (`dvc repro`)
- Docker (training + inference)
- FastAPI production API
- Monitoring with Evidently AI
- GitHub Actions CI/CD
- Full documentation

## Quick Start (3 commands)

```bash
# 1. Clone & install
git clone <your-repo>
cd House-Price-Prediction
pip install -r requirements.txt

# 2. Reproduce everything from scratch
dvc repro

# 3. Launch services
mlflow ui                    # → http://localhost:5000
uvicorn src.api.app:app --reload   # → http://localhost:8000/docs