# Ames House Price Prediction – MLOps Project

**Regression project – Ames Housing Dataset**  
**Khouloud Ouni & Eya Ben Khadhra – December 2025**  

---

### Project Overview
End-to-end MLOps pipeline for predicting house prices in Ames, Iowa (2,930 properties, 79 features).  
**Random Forest model → RMSE: $28,208** (excellent performance – top 10% level).

---

### Features Implemented 

| Requirement                      | Implementation                                                                | Status |
|-------------------------------------------------------|----------------------------------------------------------|--------|
| Git + standard structure                              | Clean repository, professional layout                    | Done   |
| DVC + EDA                                             | Data versioning + `notebooks/EDA.ipynb`                  | Done   |
| MLflow tracking + metrics                             | 2 tracked runs + RMSE logged                             | Done   |
| Reproducible pipeline + Docker + tests               | `dvc repro` works in one command + Dockerfiles            | Done   |
| FastAPI inference API + Dockerfile                    | Swagger UI (`/docs`) + live prediction                   | Done   |
| GitHub Actions + cloud-ready deployment              | `.github/workflows/ci-cd.yaml` ready for AWS              | Done   |


**Bonus beyond specifications**  
**Streamlit interface** connected to FastAPI  
Ultra-clean, production-ready code

---

### Project File Structure
```
project-house-price-mlops/
├── .dvc/                  # DVC config and cache (for data/model versioning)
├── .github/
│   └── workflows/         # CI/CD YAML files 
├── data/                  # Data files (versioned with DVC, not Git)
│   ├── raw/               # Original data (AmesHousing.csv)
│   ├── processed/         # Cleaned data
│   └── external/          # Any external datasets
├── notebooks/             # Exploratory notebooks
|   ├── analyze-src/       # Basic Functions used in the EDA: basic_data_inspection, missing_values_analysis,univariate_analysis, bivariate_analysis, multivariate_analysis
│   └── EDA.ipynb          
├── src/                   # Main code 
│   ├── __init__.py
│   ├── data/              # Data loading/processing
│   │   └── make_dataset.py
│   ├── features/          # Feature engineering
│   │   └── build_features.py
│   ├── models/            # Model training/evaluation
│   │   ├── train_model.py
│   │   └── predict_model.py
│   ├── api/               # Inference API
│   │   └── app.py         # FastAPI/Flask app
│   └── monitoring/        # Monitoring scripts (e.g., drift detection)
│       └── monitor.py
├── tests/                 # Unit/integration tests (used pytest)
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── docs/                  # Documentation
│   ├── architecture.md    # System architecture
│   ├── files_structure.md
│   └── api_docs.md        # API documentation (Swagger for FastAPI)
├── docker/                # Dockerfiles
│   ├── Dockerfile.train   # For training
│   └── Dockerfile.infer   # For inference/API
├── dvc.yaml               # DVC pipeline definitions
├── params.yaml            # MLflow/DVC parameters (hyperparameters)
├── requirements.txt       # Python dependencies 
├── README.md              # Comprehensive README 
├── .gitignore             # Ignore venv, __pycache__, etc.
```




---

### Quick Start – Setup & Run 

```powershell
# 1. Clone the project
git clone https://github.com/Khouloud2022/MLOps-Project-House-Price-Prediction-Ames
cd House-Price-Prediction

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize DVC and download data
dvc init
dvc remote add -d myremote [your-remote-storage]  
dvc pull    # if data is remote

# 5. Run the full reproducible pipeline (Weeks 1–4)
dvc repro

# 6. Launch services (4 terminals)
# Terminal 1 → MLflow tracking
mlflow ui --backend-store-uri mlruns

# Terminal 2 → Production API
uvicorn src.api.app:app --reload

# Terminal 3 → Beautiful user interface
streamlit run app_streamlit.py

# Terminal 4 → Generate monitoring report
python src/monitoring/monitor.py
```
---
