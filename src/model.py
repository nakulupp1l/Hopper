import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os
from src.logger import logger

def train_hopper_model(input_path: str):
    logger.info("Initializing Model Training with Smart Data...")
    
    # 1. Load data
    df = pd.read_csv(input_path)
    
    # 2. Features (X) - Only the Yellow inputs
    # We drop the actual target names and the original flowability/shape text columns
    X = df.drop(columns=['flowability', 'conical_half_angle', 'conical_outlet_dim', 
                         'plane_half_angle', 'plane_outlet_dim'])
    
    # 3. Targets (y) - The Green outputs
    y = df[['conical_half_angle', 'conical_outlet_dim', 'plane_half_angle', 'plane_outlet_dim']]
    
    # 4. Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Build Model
    # n_estimators=100 means we are building 100 decision trees
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    
    logger.info("Fitting Random Forest...")
    model.fit(X_train, y_train)
    
    # 6. Save Model and Columns
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/hopper_rf_model.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
    
    logger.info("Model training complete and saved.")
    print("Success! The Model is trained and stored in /models/hopper_rf_model.pkl")
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_hopper_model("data/processed/cleaned_hopper_data.csv")