import pandas as pd
import os

def test_data_files_exist():
    assert os.path.exists("data/processed/train.csv")
    assert os.path.exists("data/processed/test.csv")

def test_train_test_split():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    assert len(train) > len(test)  # 80/20 split
    assert "SalePrice" in train.columns
    assert "SalePrice" in test.columns