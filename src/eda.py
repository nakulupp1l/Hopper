import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.logger import logger

def perform_eda(input_path: str, report_dir: str):
    logger.info("Starting Exploratory Data Analysis...")
    
    # 1. Load the cleaned data
    df = pd.read_csv(input_path)
    
    # Create the reports folder if it doesn't exist
    os.makedirs(report_dir, exist_ok=True)
    
    # Set a professional style
    sns.set_theme(style="whitegrid")

    # 2. Plot Distribution of the Output Targets (The "Green" columns)
    plt.figure(figsize=(15, 10))
    
    targets = ['conical_half_angle', 'plane_half_angle', 'conical_outlet_dim', 'plane_outlet_dim']
    for i, col in enumerate(targets, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[col], kde=True, color='teal')
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "target_distributions.png"))
    logger.info("Saved target distributions plot.")

    # 3. Correlation Heatmap
    # This shows which inputs (Yellow) have the most influence on outputs (Green)
    plt.figure(figsize=(12, 10))
    # Filter for numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(report_dir, "correlation_heatmap.png"))
    logger.info("Saved correlation heatmap.")

    # 4. Input vs Output Analysis
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bulk_density', y='conical_outlet_dim', hue='flowability_score', palette='viridis')
    plt.title('Bulk Density vs Conical Outlet Dimension')
    plt.savefig(os.path.join(report_dir, "density_vs_outlet.png"))
    
    print(f"EDA complete! Plots are saved in: {report_dir}")
    logger.info("EDA finished successfully.")

if __name__ == "__main__":
    perform_eda(
        "data/processed/cleaned_hopper_data.csv", 
        "reports/figures"
    )