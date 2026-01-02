import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
import os

def train_hybrid_model():
    # 1. Load Synthetic Data (The Volume)
    df_synth = pd.read_csv('data/processed/cleaned_hopper_data.csv')
    
    # 2. Load and Clean Real Data (The Accuracy)
    df_real_raw = pd.read_excel('data/raw/Hopper_data.xlsx', header=1)
    mapping = {
        'Bulk Density - ρb (kg/m3)': 'bulk_density', 'd50 (µm)': 'd50',
        'Shape ': 'shape', 'Half Angle (°)': 'conical_half_angle',
        'Outlet Dimension\nNB': 'conical_outlet_dim', 'Half Angle (°).1': 'plane_half_angle',
        'Outlet Dimension\nNB.1': 'plane_outlet_dim'
    }
    df_real = df_real_raw.rename(columns=mapping)
    df_real['shape'] = df_real['shape'].astype(str).str.strip()
    df_real = pd.get_dummies(df_real, columns=['shape'], prefix='shape')
    
    # 3. Combine them
    # We will "Boost" the real data by repeating it 10 times so the model pays more attention to it
    df_combined = pd.concat([df_synth, df_real, df_real, df_real], ignore_index=True).fillna(0)
    
    # 4. Features & Targets
    features = joblib.load('models/feature_names.pkl')
    X = df_combined[features]
    y = df_combined[['conical_half_angle', 'conical_outlet_dim', 'plane_half_angle', 'plane_outlet_dim']]
    
    # 5. Train
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42))
    model.fit(X, y)
    
    # 6. Save
    joblib.dump(model, 'models/hopper_rf_model.pkl')
    print("Hybrid Model Trained successfully!")

if __name__ == "__main__":
    train_hybrid_model()