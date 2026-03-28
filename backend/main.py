import pickle
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from schemas import FightPredictionRequest, PredictionResponse

# Dictionary to hold our loaded models and data
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    #print("Loading ML models and data into memory...")
    try:
        # 1. Load the Models
        with open("models/RandomForest_Opt_model.pkl", "rb") as f:
            app_state["rf_model"] = pickle.load(f)
        with open("models/feature_columns.pkl", "rb") as f:
            app_state["feature_columns"] = pickle.load(f)
            
        # 2. Load the Data ONCE into memory so it's lightning fast
        app_state["fighters_df"] = pd.read_csv("processed_data/fighter_averages.csv")
        # Ensure the index is the fighter name for super fast lookups
        if 'Name' in app_state["fighters_df"].columns:
            app_state["fighters_df"].set_index('Name', inplace=True)
            
        #print("Models and Data loaded successfully!")
    except Exception as e:
        print(f"Startup Error: {e}")
    
    yield 
    app_state.clear()

app = FastAPI(title="UFC Fight Predictor API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "API is live! Go to http://localhost:8000/docs to test it."}
@app.post("/predict", response_model=PredictionResponse)
def predict_fight(request: FightPredictionRequest):
    df = app_state["fighters_df"]
    model = app_state["rf_model"]
    cols = app_state["feature_columns"]

    # 1. Get Stats
    red_stats = df.loc[request.fighter_red].to_dict()
    blue_stats = df.loc[request.fighter_blue].to_dict()

    # 2. CREATE TWO SCENARIOS (The "Anti-Bias" Move)
    # Scenario A: Red is Fighter 1, Blue is Fighter 2
    # Scenario B: Red is Fighter 2, Blue is Fighter 1
    def get_features(f1_stats, f2_stats):
        feats = {}
        # List of columns we KNOW the model needs
        # (Add or remove based on your feature_columns.pkl)
        categorical_cols = ["Stance"] 
        
        def clean_val(v):
            """Force value to be a simple type or return 0 if it's a dict/list"""
            if isinstance(v, (dict, list)):
                return 0 
            return v

        for k, v in f1_stats.items():
            if k in categorical_cols or isinstance(v, (int, float, np.number)):
                if k not in ["Name", "DOB"]:
                    feats[f"red_{k}"] = clean_val(v)
        
        for k, v in f2_stats.items():
            if k in categorical_cols or isinstance(v, (int, float, np.number)):
                if k not in ["Name", "DOB"]:
                    feats[f"blue_{k}"] = clean_val(v)
        
        X = pd.DataFrame([feats])
        
        # Safe One-Hot Encoding
        existing_cats = [c for c in ["red_Stance", "blue_Stance"] if c in X.columns]
        if existing_cats:
            X = pd.get_dummies(X, columns=existing_cats)
        
        return X.reindex(columns=cols, fill_value=0)

    # Calculate Probabilities for both scenarios
    X_a = get_features(red_stats, blue_stats)
    X_b = get_features(blue_stats, red_stats) # Swapped names

    probs_a = model.predict_proba(X_a)[0] # [Blue_win, Red_win]
    probs_b = model.predict_proba(X_b)[0] # [Blue_win, Red_win]

    # 3. SWAP-AVERAGE
    # In Scenario B, the "Red" win prob is actually the "Blue" fighter's win prob.
    # We must flip probs_b to align them.
    avg_blue_prob = (probs_a[1] + probs_b[0]) / 2
    avg_red_prob = (probs_a[0] + probs_b[1]) / 2

    winner = request.fighter_red if avg_red_prob > avg_blue_prob else request.fighter_blue
    confidence = max(avg_red_prob, avg_blue_prob)

    return PredictionResponse(
        winner=winner,
        confidence=confidence,
        inference_results={
            "RandomForest": {
                "red_win_prob": float(avg_red_prob),
                "blue_win_prob": float(avg_blue_prob)
            }
        }
    )