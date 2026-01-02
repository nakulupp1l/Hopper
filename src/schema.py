from pydantic import BaseModel, Field, validator
from typing import Literal

# Define our allowed categories based on your input
SHAPE_TYPES = Literal[
    "spherical", "angular", "Oval", 
    "Cystalline", "lens shaped", "Cuboid", "Elomgated", "Semirounded",
    "Irregular ", "Irregular and angular" # Including variants found in your CSV
]

FLOWABILITY_TYPES = Literal[
    "Very Very Poor", "Passable", "Very Poor", "Fair", "Poor","Excellent" # Matching CSV casing
]

class HopperDataSchema(BaseModel):
    # Inputs (Yellow)
    bulk_density: float = Field(..., gt=0, description="Bulk Density in kg/m3")
    d50: float = Field(..., gt=0, description="Particle size in micrometers")
    shape: SHAPE_TYPES
    
    # Outputs (Green)
    flowability: FLOWABILITY_TYPES
    conical_half_angle: float = Field(..., ge=0, le=90)
    conical_outlet_dim: float = Field(..., gt=0)
    plane_half_angle: float = Field(..., ge=0, le=90)
    plane_outlet_dim: float = Field(..., gt=0)

    @validator('plane_half_angle')
    def angle_logic(cls, v, values):
        # Example of physical rule: Plane angle is often larger than conical
        if 'conical_half_angle' in values and v < values['conical_half_angle']:
            # This is just a warning/check; engineering-wise plane angles 
            # are typically steeper or equal to conical for mass flow.
            pass
        return v