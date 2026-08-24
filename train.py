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

WINDOW = 5          # recency window
MIN_PRIOR = 2

fights   = pd.read_csv(FIGHTS_PATH)
fighters = pd.read_csv(FIGHTERS_PATH)

fighters = fighters.drop_duplicates(subset="Fighter_Name", keep="first")
fights   = fights.drop_duplicates(subset="Fight_URL", keep="first")

fights["Event_Date"] = pd.to_datetime(fights["Event_Date"])
fights = fights.sort_values("Event_Date").reset_index(drop=True)
fights = fights[(fights["Winner"] == fights["Fighter_1"]) |
                (fights["Winner"] == fights["Fighter_2"])].reset_index(drop=True)

# static attributes
fighters["Reach"] = pd.to_numeric(
    fighters["Reach"].astype(str).str.replace('"', ''), errors="coerce")
reach_map = fighters.set_index("Fighter_Name")["Reach"]
dob_map   = fighters.set_index("Fighter_Name")["DOB"].pipe(pd.to_datetime, errors="coerce")

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

# career totals from prior fights only
for c in SUM_COLS:
    long["prior_" + c] = g[c].cumsum() - long[c]
long["n_prior"] = g.cumcount()

# NEW: last-WINDOW totals from prior fights only
for c in SUM_COLS:
    long["r_" + c] = g[c].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).sum())
long["r_n"] = g.cumcount().clip(upper=WINDOW)

# NEW: days since previous fight
long["layoff"] = g["date"].diff().dt.days

# ---------- 2. derive rate stats ----------
def rate(num, den):
    return (num / den).where(den > 0)

def derive(d, p, n_col):
    return pd.DataFrame({
        "SLpM":    rate(d[p+"sig_landed"], d[p+"mins"]),
        "SApM":    rate(d[p+"opp_sig_landed"], d[p+"mins"]),
        "Str_Acc": rate(d[p+"sig_landed"], d[p+"sig_att"]),
        "Str_Def": 1 - rate(d[p+"opp_sig_landed"], d[p+"opp_sig_att"]),
        "TD_Avg":  rate(d[p+"td_landed"], d[p+"mins"]) * 15,
        "TD_Acc":  rate(d[p+"td_landed"], d[p+"td_att"]),
        "TD_Def":  1 - rate(d[p+"opp_td_landed"], d[p+"opp_td_att"]),
        "Sub_Avg": rate(d[p+"sub_att"], d[p+"mins"]) * 15,
        "WinRate": rate(d[p+"won"], d[n_col]),
    })

CAREER = ["SLpM","SApM","Str_Acc","Str_Def","TD_Avg","TD_Acc","TD_Def","Sub_Avg","WinRate"]
RECENT = ["r" + s for s in CAREER]

pre = long[["fight_id","fighter","date","n_prior","layoff"]].copy()
pre = pd.concat([pre, derive(long, "prior_", "n_prior")], axis=1)

recent = derive(long, "r_", "r_n")
recent.columns = RECENT
pre = pd.concat([pre, recent], axis=1)

pre["Reach"]  = pre["fighter"].map(reach_map)
pre["Losses"] = long["n_prior"] - long["prior_won"]
pre["Wins"]   = long["prior_won"]

# NEW: age in years at fight date
pre["Age"] = (pre["date"] - pre["fighter"].map(dob_map)).dt.days / 365.25

ATTRS = CAREER + RECENT + ["Reach", "Wins", "Losses", "Age", "layoff"]

# ---------- 3. attach back to each fight ----------
df = fights[["Fight_URL","Event_Date","Fighter_1","Fighter_2","Winner"]].copy()

for i, col in [(1, "Fighter_1"), (2, "Fighter_2")]:
    s = pre.rename(columns={c: f"F{i}_{c}" for c in ATTRS + ["n_prior"]})
    df = df.merge(s[["fight_id","fighter"] + [f"F{i}_{c}" for c in ATTRS + ["n_prior"]]],
                  left_on=["Fight_URL", col], right_on=["fight_id","fighter"], how="left") \
           .drop(columns=["fight_id","fighter"])

df = df[(df["F1_n_prior"] >= MIN_PRIOR) & (df["F2_n_prior"] >= MIN_PRIOR)].reset_index(drop=True)

# ---------- 4. differentials + target ----------
df["target"] = (df["Winner"] == df["Fighter_1"]).astype(int)

feature_cols = []
for a in ATTRS:
    name = a.lower() + "_diff"
    df[name] = df[f"F1_{a}"] - df[f"F2_{a}"]
    feature_cols.append(name)

# ---------- 5. chronological split ----------
cutoff = df["Event_Date"].quantile(0.8)
train_df = df[df["Event_Date"] <= cutoff]
test_df  = df[df["Event_Date"] >  cutoff]

def build_xy(sub):
    orig = sub[feature_cols].copy();   orig["target"] = sub["target"].values
    mirr = (sub[feature_cols] * -1);   mirr["target"] = 1 - sub["target"].values
    both = pd.concat([orig, mirr], ignore_index=True)
    return both[feature_cols], both["target"]

X_train, y_train = build_xy(train_df)
X_test,  y_test  = build_xy(test_df)

med = X_train.median()
X_train, X_test = X_train.fillna(med), X_test.fillna(med)

print(f"features: {len(feature_cols)}   train: {len(train_df)}   test: {len(test_df)}")
print(f"cutoff: {cutoff.date()}")

model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                               random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

p = model.predict_proba(X_test)[:, 1]
print(f"\nAccuracy: {accuracy_score(y_test, (p > 0.5).astype(int)):.4f}")
print(f"prob range: {p.min():.3f} to {p.max():.3f}   above 0.70: {(p > 0.70).mean():.1%}")

print("\nCALIBRATION")
print(" bin        n   predicted   actual")
bins = np.arange(0, 1.01, 0.1)
idx = np.digitize(p, bins) - 1
for b in range(len(bins) - 1):
    m = idx == b
    if m.sum() >= 20:
        print(f"{bins[b]:.1f}-{bins[b+1]:.1f}  {m.sum():5d}   {p[m].mean():.3f}     {y_test.values[m].mean():.3f}")

print("\nTOP FEATURES")
print(pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12))

joblib.dump({"model": model, "features": feature_cols, "medians": med.to_dict()}, MODEL_PATH)

# ---------- 6. current-state stats for serving ----------
last = long.sort_values("date").groupby("fighter").tail(WINDOW)
career_tot = long.groupby("fighter")[SUM_COLS].sum()
career_tot["n"] = long.groupby("fighter").size()
recent_tot = last.groupby("fighter")[SUM_COLS].sum()
recent_tot["n"] = last.groupby("fighter").size()

serve   = derive(career_tot.add_prefix("prior_"), "prior_", "prior_n")
serve_r = derive(recent_tot.add_prefix("r_"),     "r_",     "r_n")
serve_r.columns = RECENT
serve = pd.concat([serve, serve_r], axis=1)

serve["Wins"]   = career_tot["won"]
serve["Losses"] = career_tot["n"] - career_tot["won"]
serve.insert(0, "Fighter_Name", career_tot.index)
serve["Reach"]     = serve["Fighter_Name"].map(reach_map)
serve["DOB"]       = serve["Fighter_Name"].map(dob_map)
serve["last_fight"] = long.groupby("fighter")["date"].max().values

serve.to_csv(SERVE_PATH, index=False)
print(f"\nSaved {len(serve)} fighters   missing reach: {serve['Reach'].isna().sum()}   missing DOB: {serve['DOB'].isna().sum()}")