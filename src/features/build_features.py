# src/features/build_features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import os

def build_features(train_path='data/processed/train.csv',
                   test_path='data/processed/test.csv',
                   output_path='data/processed/'):

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # More robust feature selection (these are known strong predictors in Ames)
    numerical_cols = [
        'Gr Liv Area', 'Total Bsmt SF', 'Year Built', 'Lot Area',
        'Overall Qual', 'Overall Cond', 'Full Bath', 'TotRms AbvGrd',
        'Garage Cars', 'Garage Area', '1st Flr SF', '2nd Flr SF'
    ]
    categorical_cols = [
        'Neighborhood', 'MS Zoning', 'Sale Condition', 'Kitchen Qual',
        'Exter Qual', 'Heating QC', 'Central Air', 'Foundation'
    ]

    X_train = train[numerical_cols + categorical_cols]
    y_train = train['SalePrice']
    X_test = test[numerical_cols + categorical_cols]
    y_test = test['SalePrice']

    # FULL PREPROCESSING PIPELINE with IMPUTATION
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(output_path, 'preprocessor.pkl'))
    np.save(os.path.join(output_path, 'X_train.npy'), X_train_processed)
    np.save(os.path.join(output_path, 'X_test.npy'), X_test_processed)
    y_train.to_csv(os.path.join(output_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_path, 'y_test.csv'), index=False)

    print("Features built successfully (with imputation)!")

if __name__ == "__main__":
    build_features()