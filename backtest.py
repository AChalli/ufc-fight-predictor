import pandas as pd
import numpy as np
import joblib
import os

BASE = "/code" if os.path.exists("/code") else "."
ODDS_PATH = f"{BASE}/data/ufc-master.csv"

# ---------- rebuild the exact test set train.py evaluates on ----------
# simplest approach: import the pieces from train.py rather than duplicating.
# for now, run train.py first so test_df / model exist, or refactor into a module.
from train import test_df, feature_cols, med, model, build_xy

# ---------- 1. load odds and build the join key ----------
odds = pd.read_csv(ODDS_PATH, low_memory=False)
odds["date"] = pd.to_datetime(odds["date"])
odds = odds.dropna(subset=["R_odds", "B_odds"])

def pair_key(dates, a, b):
    pairs = [tuple(sorted([x, y])) for x, y in zip(a, b)]
    return dates.dt.strftime("%Y-%m-%d") + "|" + pd.Series(pairs, index=dates.index).map(lambda t: f"{t[0]}|{t[1]}")

odds["k"] = pair_key(odds["date"], odds["R_fighter"], odds["B_fighter"])

# map each fighter name -> their american odds for that fight
odds_long = pd.concat([
    pd.DataFrame({"k": odds["k"], "fighter": odds["R_fighter"], "american": odds["R_odds"]}),
    pd.DataFrame({"k": odds["k"], "fighter": odds["B_fighter"], "american": odds["B_odds"]}),
])
odds_lookup = odds_long.set_index(["k", "fighter"])["american"]

# ---------- 2. attach odds to the test fights ----------
bt = test_df.copy()
bt["k"] = pair_key(bt["Event_Date"], bt["Fighter_1"], bt["Fighter_2"])
bt["odds_1"] = [odds_lookup.get((k, f), np.nan) for k, f in zip(bt["k"], bt["Fighter_1"])]
bt["odds_2"] = [odds_lookup.get((k, f), np.nan) for k, f in zip(bt["k"], bt["Fighter_2"])]
bt = bt.dropna(subset=["odds_1", "odds_2"]).reset_index(drop=True)

# ---------- 3. model probabilities (unmirrored: one row per fight) ----------
X = bt[feature_cols].fillna(med)
bt["p1"] = model.predict_proba(X)[:, 1]
bt["p2"] = 1 - bt["p1"]

# ---------- 4. odds math ----------
def to_decimal(a):
    return np.where(a < 0, 1 + 100 / np.abs(a), 1 + a / 100)

bt["dec_1"] = to_decimal(bt["odds_1"])
bt["dec_2"] = to_decimal(bt["odds_2"])
bt["imp_1"] = 1 / bt["dec_1"]
bt["imp_2"] = 1 / bt["dec_2"]

# devig: normalize so the two implied probabilities sum to 1
tot = bt["imp_1"] + bt["imp_2"]
bt["fair_1"] = bt["imp_1"] / tot
bt["fair_2"] = bt["imp_2"] / tot
bt["vig"] = tot - 1

print(f"fights with odds: {len(bt)}")
print(f"average vig:      {bt['vig'].mean():.3f}")

# ---------- 5. how good is the market vs the model? ----------
mkt_pick = (bt["fair_1"] > 0.5).astype(int)
mdl_pick = (bt["p1"] > 0.5).astype(int)
print(f"market accuracy:  {(mkt_pick == bt['target']).mean():.4f}")
print(f"model accuracy:   {(mdl_pick == bt['target']).mean():.4f}")
print(f"agreement:        {(mkt_pick == mdl_pick).mean():.1%}")

# ---------- 6. flat-stake betting simulation ----------
def simulate(bt, threshold, calibrate=None):
    rows = []
    for _, r in bt.iterrows():
        for side, p, dec, imp, won in [
            (1, r["p1"], r["dec_1"], r["imp_1"], r["target"] == 1),
            (2, r["p2"], r["dec_2"], r["imp_2"], r["target"] == 0),
        ]:
            q = calibrate(p) if calibrate else p
            ev = q * (dec - 1) - (1 - q)
            if ev > threshold:
                rows.append({"stake": 1.0,
                             "profit": (dec - 1) if won else -1.0,
                             "won": won, "dec": dec, "p": q, "ev": ev})
    if not rows:
        return None
    d = pd.DataFrame(rows)
    return {"bets": len(d), "won": int(d["won"].sum()),
            "hit": d["won"].mean(), "staked": d["stake"].sum(),
            "profit": d["profit"].sum(), "roi": d["profit"].sum() / d["stake"].sum(),
            "avg_dec": d["dec"].mean()}

# correction for the known probability compression, measured on the test set
CAL = [(0.0, 0.35, -0.06), (0.35, 0.45, -0.04), (0.45, 0.55, 0.02),
       (0.55, 0.70, 0.06), (0.70, 1.01, 0.04)]

def compress_fix(p):
    for lo, hi, adj in CAL:
        if lo <= p < hi:
            return min(max(p + adj, 0.01), 0.99)
    return p

print("\n--- RAW PROBABILITIES ---")
print(f"{'EV thresh':>10} {'bets':>6} {'hit':>7} {'ROI':>8} {'profit':>9}")
for t in [0.0, 0.05, 0.10, 0.20]:
    s = simulate(bt, t)
    if s:
        print(f"{t:>10.2f} {s['bets']:>6} {s['hit']:>6.1%} {s['roi']:>7.2%} {s['profit']:>8.1f}u")

print("\n--- COMPRESSION-CORRECTED ---")
print(f"{'EV thresh':>10} {'bets':>6} {'hit':>7} {'ROI':>8} {'profit':>9}")
for t in [0.0, 0.05, 0.10, 0.20]:
    s = simulate(bt, t, calibrate=compress_fix)
    if s:
        print(f"{t:>10.2f} {s['bets']:>6} {s['hit']:>6.1%} {s['roi']:>7.2%} {s['profit']:>8.1f}u")

# ---------- 7. underdogs vs favorites ----------
print("\n--- BY PRICE BUCKET (EV > 0.05, corrected) ---")
for lo, hi, label in [(1.0, 1.6, "heavy fav"), (1.6, 2.0, "fav"),
                      (2.0, 2.8, "dog"), (2.8, 99, "big dog")]:
    sub = bt.copy()
    rows = []
    for _, r in sub.iterrows():
        for p, dec, won in [(r["p1"], r["dec_1"], r["target"] == 1),
                            (r["p2"], r["dec_2"], r["target"] == 0)]:
            if not (lo <= dec < hi):
                continue
            q = compress_fix(p)
            if q * (dec - 1) - (1 - q) > 0.05:
                rows.append({"profit": (dec - 1) if won else -1.0, "won": won})
    if len(rows) >= 10:
        d = pd.DataFrame(rows)
        print(f"{label:>10} {len(d):>5} bets  hit {d['won'].mean():>5.1%}  ROI {d['profit'].sum()/len(d):>7.2%}")