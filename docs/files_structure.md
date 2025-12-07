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
└── setup.py               # Optional: If packaging as a Python module