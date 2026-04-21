from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "UFC Fight Predictor API"}

@app.get("/fighter/{name}")
def get_fighter(name: str):
    df = pd.read_csv("../data/ufc_fighters_final.csv")
    fighter = df[df["name"] == name].iloc[0]
    return fighter.to_dict()