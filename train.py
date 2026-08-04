import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# paths
BASE = "/code" if os.path.exists("/code") else "."
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

# drop draws / no contests (they aren't wins for either fighter)
df = df[df["Winner"].isin(df["Fighter_1"]) | df["Winner"].isin(df["Fighter_2"])]
df = df[(df["Winner"] == df["Fighter_1"]) | (df["Winner"] == df["Fighter_2"])]

df["F1_Reach"] = pd.to_numeric(df["F1_Reach"], errors="coerce")
df["F2_Reach"] = pd.to_numeric(df["F2_Reach"], errors="coerce")

stat_map = {
    "reach_diff":    ("F1_Reach",   "F2_Reach"),
    "slpm_diff":     ("F1_SLpM",    "F2_SLpM"),
    "str_acc_diff":  ("F1_Str_Acc", "F2_Str_Acc"),
    "sapm_diff":     ("F1_SApM",    "F2_SApM"),
    "str_def_diff":  ("F1_Str_Def", "F2_Str_Def"),
    "td_avg_diff":   ("F1_TD_Avg",  "F2_TD_Avg"),
    "td_acc_diff":   ("F1_TD_Acc",  "F2_TD_Acc"),
    "td_def_diff":   ("F1_TD_Def",  "F2_TD_Def"),
    "sub_avg_diff":  ("F1_Sub_Avg", "F2_Sub_Avg"),
    "win_diff":      ("F1_Wins",    "F2_Wins"),
    "loss_diff":     ("F1_Losses",  "F2_Losses"),
}
feature_cols = list(stat_map.keys())

for name, (a, b) in stat_map.items():
    df[name] = df[a] - df[b]

# original orientation
orig = df[feature_cols].copy()
orig["target"] = df["target"].values

# mirrored orientation: swap the two fighters, flip the label
mirror = (df[feature_cols] * -1).copy()
mirror["target"] = 1 - df["target"].values

balanced = pd.concat([orig, mirror], ignore_index=True)

X = balanced[feature_cols].fillna(balanced[feature_cols].median())
y = balanced["target"]

print(y.value_counts())   # should now be exactly 50/50

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")