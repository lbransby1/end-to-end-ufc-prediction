from pydantic import BaseModel
from typing import Dict, Any

# What the frontend will send to the backend
class FightPredictionRequest(BaseModel):
    fighter_red: str
    fighter_blue: str
    # If your frontend passes pre-computed features instead of just names, 
    # you can change this to accept a dictionary of features:
    # features: Dict[str, float]

# What the backend will return to the frontend
class PredictionResponse(BaseModel):
    winner: str
    confidence: float
    inference_results: Dict[str, Any]