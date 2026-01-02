import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
from src.logger import logger

def test_on_real_data():
    logger.info("Starting Real-World Data Test with Robust Loading...")

    # 1. Load Model and Feature Names
    try:
        model = joblib.load('models/hopper_rf_model.pkl')
        feature_cols = joblib.load('models/feature_names.pkl')
    except FileNotFoundError:
        print("Error: Model files not found. Please run training first.")
        return
    
    # 2. Robust File Loading
    # We look for the CSV or the XLSX version
    possible_files = [
        'data/raw/Hopper data.xlsx - Sheet1.csv',
        'data/raw/Hopper_data.csv',
        'data/raw/Hopper_data.xlsx'
    ]
    
    df_real = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.xlsx'):
                    df_real = pd.read_excel(file_path, header=1)
                else:
                    # 'latin1' handles the special characters like (°) found in Excel CSVs
                    df_real = pd.read_csv(file_path, header=1, encoding='latin1')
                print(f"Successfully loaded test data from: {file_path}")
                break
            except Exception as e:
                continue

    if df_real is None:
        print("Error: Could not find the real-world data file in data/raw/ folder.")
        return

    # 3. Clean and Map Columns
    mapping = {
        'Bulk Density - ρb (kg/m3)': 'bulk_density',
        'd50 (µm)': 'd50',
        'Shape ': 'shape',
        'Half Angle (°)': 'actual_conical_angle',
        'Outlet Dimension\nNB': 'actual_conical_outlet',
        'Half Angle (°).1': 'actual_plane_angle',
        'Outlet Dimension\nNB.1': 'actual_plane_outlet'
    }
    df_real = df_real.rename(columns=mapping)
    
    # Critical Step: Clean the strings in the 'shape' column to match model training
    df_real['shape'] = df_real['shape'].astype(str).str.strip()
    
    # Drop rows missing crucial inputs
    df_real = df_real.dropna(subset=['bulk_density', 'd50', 'shape'])

    # 4. Preprocess for Prediction (One-Hot Encoding)
    X_processed = pd.get_dummies(df_real[['bulk_density', 'd50', 'shape']], columns=['shape'], prefix='shape')
    
    # Ensure all feature columns from training are present
    for col in feature_cols:
        if col not in X_processed.columns:
            X_processed[col] = 0
            
    # Match the order of training features
    X_processed = X_processed[feature_cols]

    # 5. Predict
    predictions = model.predict(X_processed)
    
    # 6. Calculate Metrics
    y_actual = df_real[['actual_conical_angle', 'actual_conical_outlet', 'actual_plane_angle', 'actual_plane_outlet']]
    
    print("\n" + "="*40)
    print("REAL-WORLD ACCURACY REPORT")
    print("="*40)
    
    target_names = ['Conical Angle', 'Conical Outlet', 'Plane Angle', 'Plane Outlet']
    for i, name in enumerate(target_names):
        # Filter out NaN targets from the real data for accurate scoring
        mask = ~y_actual.iloc[:, i].isna()
        actual = y_actual.iloc[:, i][mask]
        pred = predictions[mask, i]
        
        if len(actual) > 0:
            r2 = r2_score(actual, pred)
            mae = mean_absolute_error(actual, pred)
            print(f"\nTarget: {name}")
            print(f"  - R2 Score: {r2:.4f} (Reliability of design)")
            print(f"  - Avg Error (MAE): {mae:.2f} units")

    # 7. Final Plotting
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(target_names[:2]): # Plotting first two for clarity
        plt.subplot(1, 2, i+1)
        plt.scatter(y_actual.iloc[:, i], predictions[:, i], alpha=0.7, color='teal')
        plt.plot([y_actual.iloc[:, i].min(), y_actual.iloc[:, i].max()], 
                 [y_actual.iloc[:, i].min(), y_actual.iloc[:, i].max()], 'r--')
        plt.title(f'Actual vs Predicted: {name}')
        plt.xlabel('Measured')
        plt.ylabel('Model Prediction')
        
    plt.tight_layout()
    plt.savefig('reports/figures/final_performance_check.png')
    print("\nPerformance visualization saved to: reports/figures/final_performance_check.png")

if __name__ == "__main__":
    test_on_real_data()