import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def verify_final_authenticity():
    # 1. Load the new "Reliable" Synthetic Data
    synth_path = "data/synthetic/synthetic_hopper_data.csv"
    df_synth = pd.read_csv(synth_path)
    
    # 2. Load the Master Lab Data (Hopper_data.xlsx)
    real_path = "data/raw/Hopper_data.xlsx"
    try:
        df_real = pd.read_excel(real_path, header=1)
    except:
        df_real = pd.read_csv(real_path, header=1, encoding='latin1')

    # 3. Align and Clean
    mapping = {
        'Bulk Density - ρb (kg/m3)': 'density',
        'd50 (µm)': 'd50',
        'Half Angle (°)': 'c_angle',
        'Outlet Dimension\nNB': 'c_outlet'
    }
    df_real = df_real.rename(columns=mapping)
    
    # Select only columns we want to compare
    real_compare = df_real[['density', 'c_angle', 'c_outlet']].dropna()
    synth_compare = df_synth[['bulk_density', 'conical_half_angle', 'conical_outlet_dim']]

    # 4. Statistical Summary
    print("\n" + "="*40)
    print("FINAL AUTHENTICITY REPORT")
    print("="*40)
    
    metrics = [
        ('Density', 'density', 'bulk_density'),
        ('Conical Angle', 'c_angle', 'conical_half_angle'),
        ('Conical Outlet', 'c_outlet', 'conical_outlet_dim')
    ]
    
    for label, real_col, synth_col in metrics:
        real_mean = real_compare[real_col].mean()
        synth_mean = synth_compare[synth_col].mean()
        diff = abs(real_mean - synth_mean) / real_mean * 100
        print(f"{label}:")
        print(f"  - Real Mean:  {real_mean:.2f}")
        print(f"  - Synth Mean: {synth_mean:.2f}")
        print(f"  - Deviation:  {diff:.2f}%")

    # 5. Visual Overlap (The "Digital Twin" Test)
    os.makedirs('reports/figures', exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    # Plot Angle Comparison
    plt.subplot(1, 2, 1)
    sns.kdeplot(real_compare['c_angle'], label='Real Lab Data', fill=True, color='blue', alpha=0.5)
    sns.kdeplot(synth_compare['conical_half_angle'], label='Synthetic Data', fill=True, color='orange', alpha=0.5)
    plt.title('Angle Distribution: Real vs Synthetic')
    plt.legend()

    # Plot Outlet Comparison
    plt.subplot(1, 2, 2)
    sns.kdeplot(real_compare['c_outlet'], label='Real Lab Data', fill=True, color='blue', alpha=0.5)
    sns.kdeplot(synth_compare['conical_outlet_dim'], label='Synthetic Data', fill=True, color='orange', alpha=0.5)
    plt.title('Outlet Distribution: Real vs Synthetic')
    plt.legend()

    plt.tight_layout()
    plt.savefig('reports/figures/final_authenticity_check.png')
    print("\nSuccess! Authenticity graph saved to: reports/figures/final_authenticity_check.png")

if __name__ == "__main__":
    verify_final_authenticity()