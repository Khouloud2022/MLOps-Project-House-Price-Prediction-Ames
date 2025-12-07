# House Price Prediction MLOps Project

## Overview
End-to-end MLOps for predicting house prices using Ames dataset.

## Setup
1. Clone repo
2. pip install -r requirements.txt
3. dvc init && dvc add data/
4. Run pipeline: dvc repro

## Usage
- Train: python src/models/train_model.py
- API: uvicorn src.api.app:app
- Monitor: python src/monitoring/monitor.py

## Architecture
[Diagram here: Data -> Features -> Model -> API -> Monitoring]