import joblib
import pandas as pd
import numpy as np

def get_hopper_design():
    # 1. Load the "Brain" (Model) and the "Memory" (Feature Names)
    model = joblib.load('models/hopper_rf_model.pkl')
    features = joblib.load('models/feature_names.pkl')

    print("--- Hopper Project: Design Predictor ---")
    try:
        # Get User Inputs
        density = float(input("Enter Bulk Density (kg/m3): "))
        d50 = float(input("Enter Particle Size d50 (microns): "))
        print("Shapes: spherical, angular, Oval, Cystalline, lens shaped, Cuboid, Elomgated, Semirounded, Irregular")
        shape_input = input("Enter Material Shape: ").strip()

        # 2. Prepare the input data (matching One-Hot Encoding)
        input_data = {col: 0 for col in features}
        input_data['bulk_density'] = density
        input_data['d50'] = d50
        
        # Set the shape column to 1
        shape_col = f"shape_{shape_input}"
        if shape_col in input_data:
            input_data[shape_col] = 1
        else:
            print(f"Note: Shape '{shape_input}' not recognized, using default weights.")

        # 3. Predict
        df_input = pd.DataFrame([input_data])
        results = model.predict(df_input)[0]

        # 4. Display the Engineering Design
        print("\n" + "="*30)
        print("RECOMMENDED HOPPER DESIGN")
        print("="*30)
        print(f"Conical Half Angle:  {results[0]:.2f}°")
        print(f"Conical Outlet Size: {results[1]:.2f} mm")
        print(f"Plane Half Angle:    {results[2]:.2f}°")
        print(f"Plane Outlet Size:   {results[3]:.2f} mm")
        print("="*30)

    except ValueError:
        print("Please enter valid numbers for Density and d50.")

if __name__ == "__main__":
    get_hopper_design()