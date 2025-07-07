from pydantic import BaseModel, Field
from typing import Annotated

class SimulationInput(BaseModel):
    commodity: str  # e.g. "copper"
    n_days: int = 5     # forecast horizon
    percentil: Annotated[int, Field(gt=0, le=50)] = 5 # Percentil de la simulación entre ]0,50] 



class CalibrationRequest(BaseModel):
    commodity: str
    look_back: int = 10
    epochs: int = 20

class ForecastRequest(BaseModel):
    commodity: str
    n_days: int = 10

