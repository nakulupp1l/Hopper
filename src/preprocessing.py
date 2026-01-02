import pandas as pd
import os
from src.logger import logger

def preprocess_data(input_path: str, output_path: str):
    logger.info("Starting Data Preprocessing...")
    
    # 1. Load the data
    df = pd.read_csv(input_path)
    
    # 2. Basic Cleaning: Strip spaces from strings
    df['shape'] = df['shape'].str.strip()
    df['flowability'] = df['flowability'].str.strip()
    
    # 3. Ordinal Encoding for Flowability
    # We map them from worst (0) to best (5)
    flow_map = {
        "Very Very Poor": 0,
        "Very Poor": 1,
        "Poor": 2,
        "Passable": 3,
        "Passsable": 3, # Handling that typo just in case
        "Fair": 4,
        "Good": 5,
        "Excellent": 6
    }
    df['flowability_score'] = df['flowability'].map(flow_map)
    
    # 4. One-Hot Encoding for Shape
    # This creates columns like shape_spherical, shape_angular, etc.
    df = pd.get_dummies(df, columns=['shape'], prefix='shape')
    
    # 5. Handle missing values (though we shouldn't have any in synthetic data)
    df = df.dropna()
    
    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Preprocessing complete. Cleaned data saved to {output_path}")
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data(
        "data/synthetic/synthetic_hopper_data.csv", 
        "data/processed/cleaned_hopper_data.csv"
    )