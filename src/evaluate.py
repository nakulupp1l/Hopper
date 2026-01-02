import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from src.logger import logger

def evaluate_model():
    logger.info("Starting Model Evaluation...")
    
    # 1. Load Model and Data
    model = joblib.load('models/hopper_rf_model.pkl')
    features = joblib.load('models/feature_names.pkl')
    df = pd.read_csv('data/processed/cleaned_hopper_data.csv')
    
    # 2. Prepare Test Set (mimicking the split in model.py)
    # Note: In a real project, we'd save the test set separately, but this works for verification.
    X = df[features]
    y = df[['conical_half_angle', 'conical_outlet_dim', 'plane_half_angle', 'plane_outlet_dim']]
    
    # Predict on the whole dataset for a quick performance check
    predictions = model.predict(X)
    
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    
    print(f"\n--- Model Performance ---")
    print(f"R2 Score (Accuracy): {r2:.4f}") # 1.0 is perfect
    print(f"Mean Absolute Error: {mae:.2f} units")

    # 3. THE GOLDEN ROW TEST
    # Let's pick a row similar to your real data: Density 850, d50 74, Shape Irregular
    print("\n--- Golden Row Verification ---")
    
    # We must match the EXACT columns created by get_dummies in preprocessing
    # We create a dictionary of all zeros first
    golden_input = {col: 0 for col in features}
    
    # Fill in our test values (Yellow Inputs)
    golden_input['bulk_density'] = 852.9
    golden_input['d50'] = 74.0
    # Set the specific shape column to 1 (One-Hot Encoding)
    if 'shape_Irregular' in golden_input:
        golden_input['shape_Irregular'] = 1
    
    golden_df = pd.DataFrame([golden_input])
    prediction = model.predict(golden_df)[0]
    
    print(f"Input: Density=852.9, d50=74, Shape=Irregular")
    print(f"Predicted Conical Angle: {prediction[0]:.2f}° (Target: ~23°)")
    print(f"Predicted Conical Outlet: {prediction[1]:.2f} mm (Target: ~250mm)")
    
    logger.info("Evaluation complete.")

if __name__ == "__main__":
    evaluate_model()