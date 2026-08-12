from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = "/code" if __import__("os").path.exists("/code") else ".."

bundle = joblib.load(f"{BASE}/models/random_forest.pkl")
model, FEATURES, MEDIANS = bundle["model"], bundle["features"], bundle["medians"]

fighters = pd.read_csv(f"{BASE}/data/fighter_current_stats.csv")
lookup = {n: i for i, n in enumerate(fighters["Fighter_Name"])}

STATS = ["SLpM", "SApM", "Str_Acc", "Str_Def", "TD_Avg",
         "TD_Acc", "TD_Def", "Sub_Avg", "Wins", "Losses", "Reach"]


@app.get("/")
def root():
    return {"message": "UFC Fight Predictor API"}


@app.get("/fighters")
def get_fighters():
    df = fighters.fillna(0)
    return [
        {
            "id": row.Fighter_Name,
            "name": row.Fighter_Name,
            "record": f"{int(row.Wins)}-{int(row.Losses)}",
            "weightClass": "",
        }
        for row in df.itertuples()
    ]


@app.get("/predict")
def predict(fighter1: str, fighter2: str):
    if fighter1 not in lookup:
        return {"error": f"Fighter not found: {fighter1}"}
    if fighter2 not in lookup:
        return {"error": f"Fighter not found: {fighter2}"}
    if fighter1 == fighter2:
        return {"error": "Pick two different fighters"}

    f1 = fighters.iloc[lookup[fighter1]]
    f2 = fighters.iloc[lookup[fighter2]]

    row = {s.lower() + "_diff": f1[s] - f2[s] for s in STATS}

    # same column order as training, same fill values as training
    X = pd.DataFrame([row])[FEATURES].fillna(pd.Series(MEDIANS))

    prob = model.predict_proba(X)[0]
    p1 = float(prob[1])

    return {
        "fighter1": fighter1,
        "fighter2": fighter2,
        "fighter1_win_probability": round(p1, 3),
        "fighter2_win_probability": round(1 - p1, 3),
        "predicted_winner": fighter1 if p1 > 0.5 else fighter2,
    }