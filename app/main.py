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

model = joblib.load("/code/models/random_forest.pkl")
fighters = pd.read_csv("/code/data/ufc_fighters_final.csv",
    usecols=["Fighter_Name", "Reach", "SLpM", "Str_Acc", "SApM",
             "Str_Def", "TD_Avg", "TD_Acc", "TD_Def", "Sub_Avg",
             "Wins", "Losses", "Draws"])

fighters["Reach"] = fighters["Reach"].str.replace('"', '').astype(float)
pct_cols = ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]
for col in pct_cols:
    fighters[col] = fighters[col].str.replace("%", "").astype(float) / 100

@app.get("/")
def root():
    return {"message": "UFC Fight Predictor API"}

@app.get("/fighters")
def get_fighters():
    return fighters[["Fighter_Name", "Wins", "Losses", "Draws"]].fillna("").rename(
        columns={"Fighter_Name": "name"}
    ).assign(
        id=fighters["Fighter_Name"],
        record=fighters["Wins"].astype(int).astype(str) + "-" +
               fighters["Losses"].astype(int).astype(str) + "-" +
               fighters["Draws"].astype(int).astype(str),
        weightClass="",
    ).to_dict(orient="records")

@app.get("/predict")
def predict(fighter1: str, fighter2: str):
    f1 = fighters[fighters["Fighter_Name"] == fighter1]
    f2 = fighters[fighters["Fighter_Name"] == fighter2]

    if f1.empty:
        return {"error": f"Fighter not found: {fighter1}"}
    if f2.empty:
        return {"error": f"Fighter not found: {fighter2}"}

    f1 = f1.iloc[0]
    f2 = f2.iloc[0]

    features = {
        "reach_diff": f1["Reach"] - f2["Reach"],
        "slpm_diff": f1["SLpM"] - f2["SLpM"],
        "str_acc_diff": f1["Str_Acc"] - f2["Str_Acc"],
        "sapm_diff": f1["SApM"] - f2["SApM"],
        "str_def_diff": f1["Str_Def"] - f2["Str_Def"],
        "td_avg_diff": f1["TD_Avg"] - f2["TD_Avg"],
        "td_acc_diff": f1["TD_Acc"] - f2["TD_Acc"],
        "td_def_diff": f1["TD_Def"] - f2["TD_Def"],
        "sub_avg_diff": f1["Sub_Avg"] - f2["Sub_Avg"],
        "win_diff": f1["Wins"] - f2["Wins"],
        "loss_diff": f1["Losses"] - f2["Losses"],
    }

    X = pd.DataFrame([features]).fillna(0)
    prob = model.predict_proba(X)[0]

    return {
        "fighter1": fighter1,
        "fighter2": fighter2,
        "fighter1_win_probability": round(float(prob[1]), 3),
        "fighter2_win_probability": round(float(prob[0]), 3),
        "predicted_winner": fighter1 if prob[1] > 0.5 else fighter2
    }