import pandas as pd
from sklearn.model_selection import train_test_split
import os

def make_dataset(raw_path='data/raw/AmesHousing.csv', processed_path='data/processed/'):
    # Load raw data
    df = pd.read_csv(raw_path)
    
    # Basic cleaning (handle missing values, drop irrelevant columns)
    df = df.drop(columns=['PID', 'Order'])  # Irrelevant IDs
    df['Lot Frontage'] = df['Lot Frontage'].fillna(df['Lot Frontage'].median())  # Impute median
    df = df.dropna(subset=['SalePrice'])  # Ensure target is present
    
    # Split into train/test (80/20)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save processed files
    os.makedirs(processed_path, exist_ok=True)
    train.to_csv(os.path.join(processed_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(processed_path, 'test.csv'), index=False)
    
    print("Dataset processed and split.")

if __name__ == "__main__":
    make_dataset()