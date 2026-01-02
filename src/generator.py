import pandas as pd
import numpy as np
import os
from src.logger import logger
from src.schema import HopperDataSchema, SHAPE_TYPES, FLOWABILITY_TYPES
from pydantic import ValidationError
import typing

def generate_complex_synthetic_data(num_rows: int = 2500):
    logger.info("Starting Complex Logic Generation (Pattern Authenticity Mode)...")
    
    shapes = list(typing.get_args(SHAPE_TYPES))
    flowabilities = list(typing.get_args(FLOWABILITY_TYPES))
    
    synthetic_rows = []
    while len(synthetic_rows) < num_rows:
        # 1. Inputs (Yellow) - Anchored to Lab Distributions
        density = np.random.normal(762, 450)
        d50 = np.random.lognormal(mean=np.log(80), sigma=0.8) # Lognormal captures the 'long tail' of particles
        
        # Ensure physical bounds
        density = np.clip(density, 118, 3500)
        d50 = np.clip(d50, 2, 600)
        
        shape = np.random.choice(shapes)
        flowability = np.random.choice(flowabilities)

        # 2. COMPLEX RELATIONS (Randomized Logic)
        # We pick an 'Engineering School' for this specific row
        school = np.random.choice(['conservative', 'standard', 'optimized'], p=[0.2, 0.6, 0.2])
        
        # Base logic derived from Hopper_data.xlsx regression
        # Conical Angle Logic (Primary driver: Density + Shape interaction)
        angle_base = 25.0 + (0.001 * density)
        if shape in ['Oval', 'lens shaped', 'Cystalline']:
            angle_base += 12.0 # Complexity: Shape shifts the whole requirement
        elif shape == 'spherical':
            angle_base -= 4.0
            
        # Add non-linear interaction: d50 effect is stronger for lower densities
        angle_logic = angle_base + (d50 * 0.02 * (1500/density))
        
        # Conical Outlet Logic (Primary driver: Flowability + Angle Link)
        # Real-world data shows: Higher Angle often means smaller Outlet is allowed
        outlet_base = 600.0 - (0.25 * density) - (5.0 * angle_logic)
        
        # Flowability "Step Functions" (The most complex part of your data)
        flow_mod = 0
        if "Very" in flowability:
            flow_mod = np.random.uniform(150, 300) # Poor flow jumps the outlet size
        elif flowability in ["Excellent", "Good"]:
            flow_mod = np.random.uniform(-150, -250)

        outlet_logic = outlet_base + flow_mod

        # 3. Apply the "Engineering School" Multiplier
        if school == 'conservative':
            angle_logic *= 1.15
            outlet_logic *= 1.2
        elif school == 'optimized':
            angle_logic *= 0.85
            outlet_logic *= 0.9

        # 4. Plane Geometry Relationships
        p_angle = angle_logic + np.random.uniform(5, 10)
        p_outlet = outlet_logic * np.random.uniform(0.6, 0.8)

        # 5. Final Assembly with Gaussian Noise
        row_dict = {
            "bulk_density": density,
            "d50": d50,
            "shape": shape,
            "flowability": flowability,
            "conical_half_angle": np.clip(angle_logic + np.random.normal(0, 2), 5, 55),
            "conical_outlet_dim": np.clip(outlet_logic + np.random.normal(0, 15), 40, 950),
            "plane_half_angle": np.clip(p_angle + np.random.normal(0, 1), 10, 65),
            "plane_outlet_dim": np.clip(p_outlet + np.random.normal(0, 10), 40, 750)
        }

        try:
            valid_row = HopperDataSchema(**row_dict)
            synthetic_rows.append(valid_row.model_dump())
        except ValidationError:
            continue

    df = pd.DataFrame(synthetic_rows)
    df.to_csv("data/synthetic/synthetic_hopper_data.csv", index=False)
    print(f"Success! {num_rows} Complex Authentic rows generated.")

if __name__ == "__main__":
    generate_complex_synthetic_data(2500)