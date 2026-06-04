import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# paths
BASE = "/code" if os.path.exists("/code") else ".."
DATA_PATH = f"{BASE}/data/ufc_fighters_final.csv"
FIGHTS_PATH = f"{BASE}/data/ufc_gold_dataset_final.csv"
MODEL_PATH = f"{BASE}/models/random_forest.pkl"

os.makedirs("models", exist_ok=True)

# load data
fights = pd.read_csv(FIGHTS_PATH)
fighters = pd.read_csv(DATA_PATH)

# clean fighters
fighters["Reach"] = fighters["Reach"].str.replace('"', '').astype(float)
pct_cols = ["Str_Acc", "Str_Def", "TD_Acc", "TD_Def"]
for col in pct_cols:
    fighters[col] = fighters[col].str.replace("%", "").astype(float) / 100

# merge fighter stats onto fights
df = fights.merge(fighters, left_on="Fighter_1", right_on="Fighter_Name", how="left")
df = df.rename(columns={col: "F1_" + col for col in fighters.columns if col != "Fighter_Name"})
df = df.merge(fighters, left_on="Fighter_2", right_on="Fighter_Name", how="left")
df = df.rename(columns={col: "F2_" + col for col in fighters.columns if col != "Fighter_Name"})

# target
df["target"] = (df["Winner"] == df["Fighter_1"]).astype(int)

# features
df["F1_Reach"] = pd.to_numeric(df["F1_Reach"], errors="coerce")
df["F2_Reach"] = pd.to_numeric(df["F2_Reach"], errors="coerce")

feature_cols = [
    "reach_diff", "slpm_diff", "str_acc_diff", "sapm_diff",
    "str_def_diff", "td_avg_diff", "td_acc_diff", "td_def_diff",
    "sub_avg_diff", "win_diff", "loss_diff"
]

df["reach_diff"] = df["F1_Reach"] - df["F2_Reach"]
df["slpm_diff"] = df["F1_SLpM"] - df["F2_SLpM"]
df["str_acc_diff"] = df["F1_Str_Acc"] - df["F2_Str_Acc"]
df["sapm_diff"] = df["F1_SApM"] - df["F2_SApM"]
df["str_def_diff"] = df["F1_Str_Def"] - df["F2_Str_Def"]
df["td_avg_diff"] = df["F1_TD_Avg"] - df["F2_TD_Avg"]
df["td_acc_diff"] = df["F1_TD_Acc"] - df["F2_TD_Acc"]
df["td_def_diff"] = df["F1_TD_Def"] - df["F2_TD_Def"]
df["sub_avg_diff"] = df["F1_Sub_Avg"] - df["F2_Sub_Avg"]
df["win_diff"] = df["F1_Wins"] - df["F2_Wins"]
df["loss_diff"] = df["F1_Losses"] - df["F2_Losses"]

X = df[feature_cols].fillna(df[feature_cols].median())
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")