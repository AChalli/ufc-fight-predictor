import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

BASE = "/code" if os.path.exists("/code") else "."
FIGHTERS_PATH = f"{BASE}/data/ufc_fighters_final.csv"
FIGHTS_PATH   = f"{BASE}/data/ufc_gold_dataset_final.csv"
MODEL_PATH    = f"{BASE}/models/random_forest.pkl"
SERVE_PATH    = f"{BASE}/data/fighter_current_stats.csv"
os.makedirs(f"{BASE}/models", exist_ok=True)

fights   = pd.read_csv(FIGHTS_PATH)
fighters = pd.read_csv(FIGHTERS_PATH)

fighters = fighters.drop_duplicates(subset="Fighter_Name", keep="first")
fights   = fights.drop_duplicates(subset="Fight_URL", keep="first")

fights["Event_Date"] = pd.to_datetime(fights["Event_Date"])
fights = fights.sort_values("Event_Date").reset_index(drop=True)

# keep only fights with a decisive winner (drops draws / no contests)
fights = fights[(fights["Winner"] == fights["Fighter_1"]) |
                (fights["Winner"] == fights["Fighter_2"])].reset_index(drop=True)

# static biometric — constant over time, so it never leaked
fighters["Reach"] = pd.to_numeric(
    fighters["Reach"].astype(str).str.replace('"', ''), errors="coerce")
reach_map = fighters.set_index("Fighter_Name")["Reach"]

# ---------- 1. one row per fighter per fight ----------
def side(df, me, opp):
    return pd.DataFrame({
        "fight_id":       df["Fight_URL"],
        "date":           df["Event_Date"],
        "fighter":        df[f"Fighter_{me}"],
        "won":            (df["Winner"] == df[f"Fighter_{me}"]).astype(int),
        "mins":           df["Total_Fight_Time_Sec"] / 60,
        "sig_landed":     df[f"F{me}_Sig_Landed"],
        "sig_att":        df[f"F{me}_Sig_Att"],
        "td_landed":      df[f"F{me}_TD_Landed"],
        "td_att":         df[f"F{me}_TD_Att"],
        "sub_att":        df[f"F{me}_Sub_Att"],
        "opp_sig_landed": df[f"F{opp}_Sig_Landed"],
        "opp_sig_att":    df[f"F{opp}_Sig_Att"],
        "opp_td_landed":  df[f"F{opp}_TD_Landed"],
        "opp_td_att":     df[f"F{opp}_TD_Att"],
    })

long = pd.concat([side(fights, 1, 2), side(fights, 2, 1)], ignore_index=True)
long = long.sort_values(["fighter", "date"]).reset_index(drop=True)

SUM_COLS = ["mins", "sig_landed", "sig_att", "td_landed", "td_att", "sub_att",
            "opp_sig_landed", "opp_sig_att", "opp_td_landed", "opp_td_att", "won"]

g = long.groupby("fighter")
for c in SUM_COLS:
    # cumsum minus current row = totals from PRIOR fights only
    long["prior_" + c] = g[c].cumsum() - long[c]
long["n_prior"] = g.cumcount()

# ---------- 2. derive rate stats from prior totals ----------
def rate(num, den):
    return (num / den).where(den > 0)

def derive(d, p):
    return pd.DataFrame({
        "SLpM":    rate(d[p+"sig_landed"], d[p+"mins"]),
        "SApM":    rate(d[p+"opp_sig_landed"], d[p+"mins"]),
        "Str_Acc": rate(d[p+"sig_landed"], d[p+"sig_att"]),
        "Str_Def": 1 - rate(d[p+"opp_sig_landed"], d[p+"opp_sig_att"]),
        "TD_Avg":  rate(d[p+"td_landed"], d[p+"mins"]) * 15,
        "TD_Acc":  rate(d[p+"td_landed"], d[p+"td_att"]),
        "TD_Def":  1 - rate(d[p+"opp_td_landed"], d[p+"opp_td_att"]),
        "Sub_Avg": rate(d[p+"sub_att"], d[p+"mins"]) * 15,
        "Wins":    d[p+"won"],
        "Losses":  d["n_prior"] - d[p+"won"],
    })

STATS = ["SLpM","SApM","Str_Acc","Str_Def","TD_Avg","TD_Acc","TD_Def","Sub_Avg","Wins","Losses","Reach"]

pre = pd.concat([long[["fight_id","fighter","n_prior"]], derive(long, "prior_")], axis=1)
pre["Reach"] = pre["fighter"].map(reach_map)

# ---------- 3. attach pre-fight stats back to each fight ----------
df = fights[["Fight_URL","Event_Date","Fighter_1","Fighter_2","Winner"]].copy()

for i, col in [(1, "Fighter_1"), (2, "Fighter_2")]:
    side_stats = pre.rename(columns={c: f"F{i}_{c}" for c in STATS + ["n_prior"]})
    df = df.merge(side_stats, left_on=["Fight_URL", col],
                  right_on=["fight_id", "fighter"], how="left") \
           .drop(columns=["fight_id", "fighter"])

# a debutant has no history — the model can't say anything about them
MIN_PRIOR = 2
df = df[(df["F1_n_prior"] >= MIN_PRIOR) & (df["F2_n_prior"] >= MIN_PRIOR)].reset_index(drop=True)

# ---------- 4. differentials + target ----------
df["target"] = (df["Winner"] == df["Fighter_1"]).astype(int)

feature_cols = []
for s in STATS:
    name = s.lower() + "_diff"
    df[name] = df[f"F1_{s}"] - df[f"F2_{s}"]
    feature_cols.append(name)

# ---------- 5. chronological split (train on past, test on future) ----------
cutoff = df["Event_Date"].quantile(0.8)
train_df = df[df["Event_Date"] <= cutoff]
test_df  = df[df["Event_Date"] >  cutoff]

def build_xy(sub):
    orig = sub[feature_cols].copy()
    orig["target"] = sub["target"].values
    mirror = (sub[feature_cols] * -1).copy()
    mirror["target"] = 1 - sub["target"].values
    both = pd.concat([orig, mirror], ignore_index=True)
    return both[feature_cols], both["target"]

X_train, y_train = build_xy(train_df)
X_test,  y_test  = build_xy(test_df)

# fill using TRAIN medians only — test medians would leak
med = X_train.median()
X_train = X_train.fillna(med)
X_test  = X_test.fillna(med)

print(f"train fights: {len(train_df)}   test fights: {len(test_df)}")
print(f"test cutoff:  {cutoff.date()}")
print(y_train.value_counts().to_dict())

model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
print(pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False))

joblib.dump({"model": model, "features": feature_cols, "medians": med.to_dict()}, MODEL_PATH)

# ---------- 6. current-state stats for serving ----------
totals = long.groupby("fighter")[SUM_COLS].sum()
totals["n_prior"] = long.groupby("fighter").size()
serve = derive(totals.add_prefix("prior_").rename(columns={"prior_n_prior": "n_prior"}), "prior_")
serve.insert(0, "Fighter_Name", totals.index)
serve["Reach"] = serve["Fighter_Name"].map(reach_map)
serve[["Fighter_Name"] + STATS].to_csv(SERVE_PATH, index=False)
print(f"Saved {len(serve)} fighters to {SERVE_PATH}")

# ---------- 7. sanity checks on the serve file ----------
missing_reach = serve["Reach"].isna().sum()
print(f"serve rows: {len(serve)}   missing reach: {missing_reach}")

for n in ["Jon Jones", "Islam Makhachev", "Merab Dvalishvili",
          "Sean O'Malley", "Ilia Topuria", "Alex Pereira"]:
    print(f"  {n}: {n in set(serve['Fighter_Name'])}")