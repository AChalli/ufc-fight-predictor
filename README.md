# UFC Fight Predictor

A machine learning system that predicts UFC fight outcomes from pre-fight
statistics, served as a REST API with a live web frontend. The project doubles
as an experiment: **can a fan's understanding of stylistic matchups be encoded
as features, and does it help a model predict fights the betting market has
already priced?**

**Live app:** [v0-yuncfc.vercel.app](https://v0-yuncfc.vercel.app)
**Frontend repo:** [ufc-fight-predictor-ui](https://github.com/AChalli/ufc-fight-predictor-ui)

---

## Results at a glance

| Metric | Value |
|---|---|
| Test accuracy (held-out, chronological) | 64.2% |
| Accuracy on fights with betting odds | 65.9% |
| Betting-market accuracy on the same fights | 70.3% |
| Backtested ROI vs closing odds | negative at every threshold |

The model is well-calibrated and predicts real signal, but **does not beat the
betting market** — a conclusion the project set out to test honestly rather than
assume.

---

## The core idea

Every naive feature in a fight predictor is a *difference*: fighter A's
takedown average minus fighter B's. But a grappler-vs-grappler fight and a
striker-vs-striker fight can produce identical differences while being
completely different fights. Difference-only features **destroy matchup
information** — exactly the information a knowledgeable fan uses to predict
upsets.

This project tests whether encoding that information — as style profiles and
interaction terms — measurably improves the model.

**It does.** The interaction feature `exp_str_success`
(`A_striking_accuracy × (1 − B_striking_defense)`, minus the reverse) became
the **second most important feature in the model**, behind only age. Adding
style and interaction features closed the accuracy gap to the market from 5.4
to 4.4 points.

---

## What predicts a fight

Top features after the full pipeline:

| Rank | Feature | Meaning |
|---|---|---|
| 1 | `age_diff` | age gap at fight date |
| 2 | `exp_str_success` | expected striking success (interaction term) |
| 3 | `winrate_diff` | pre-fight win rate gap |
| 4 | `sapm_diff` | strikes absorbed per minute |
| 5 | `rsapm_diff` | strikes absorbed, last 5 fights |

Findings that held across every version of the model:

- **Age is the strongest single predictor**, separating a fighter's résumé from
  their current ability.
- **Defense outranks offense** — strikes *absorbed* predicts better than strikes
  *landed*.
- **Matchup interaction terms rank near the top**, supporting the central
  hypothesis.
- **Reach ranks low** despite constant mention in broadcast commentary.

---

## Methodology, and the mistakes corrected along the way

This project's history is mostly a sequence of self-caught errors. They're
documented here because catching them was the point.

### Data leakage (the big one)

The first model reported **69.5% accuracy**. That was wrong. Features came from
career-average statistics scraped in 2026, used to predict fights as far back as
1993 — the model could see how each career turned out.

I only questioned the number because published projects on the same problem top
out at 62–64%, and I was beating them with a baseline model. The fix rebuilt
every feature from per-fight data as **pre-fight rolling statistics**
(cumulative sums excluding the current fight), switched to a **chronological
train/test split**, and restricted median imputation to the training set.
Accuracy fell to 61.2% — the first honest number. Confirmation the leak was
real: career win differential collapsed from the 2nd most important feature to
the 10th.

### Positional bias

The target was defined relative to "Fighter 1," who won 63% of the time in the
raw data — so a model could score 63% by always guessing Fighter 1. Fixed by
mirroring every fight (swap the fighters, flip the label), producing an exact
50/50 target.

### Probability compression

The Random Forest systematically pulled probabilities toward 50% — when it said
64%, the fighter actually won 72%. This is fine for classification but fatal for
betting, where expected value multiplies by the probability. Isotonic and Platt
calibration both cost accuracy at this dataset size. **Switching to XGBoost**
solved it directly — its log-loss objective produces calibrated probabilities
(top-bin error dropped from ~7.5 points to ~2).

### A betting "edge" that wasn't

An early backtest showed a +21% ROI on favorites. It was an artifact of two
mistakes: a calibration correction accidentally fit on the test set, and
post-hoc bucket selection across 16 comparisons. Refitting the correction on
training data only *reduced* ROI, proving the improvement had been leakage. The
apparent edge collapsed from a t-statistic of 2.5 to 1.2 — indistinguishable
from noise.

---

## Why it doesn't beat the market (and why that's expected)

The closing betting line aggregates bookmaker models, sharp money, and public
sentiment into the most accurate probability estimate available. Beating it
requires information the aggregate lacks — and the vig (≈4.5% here) is a floor
any strategy must clear first.

The model lands 4.4 points short of market accuracy with negative ROI at every
threshold. No price bucket shows a profit that survives multiple-comparison
scrutiny. That's consistent with everything known about efficient betting
markets, and it's the honest result: **domain-informed features made the model
measurably better, but not better than a market that already prices in most of
what they capture.**

---

## Architecture

```
train.py        ← ALL model work: feature engineering, training, evaluation
app/main.py     ← loads the model bundle, serves predictions
backtest.py     ← joins odds, computes EV and ROI against the market
data/           ← source CSVs + generated serving stats
models/         ← generated model bundle (gitignored)
Dockerfile      ← runs train.py at build time, so every deploy retrains
```

`train.py` owns the model entirely; `main.py` loads whatever bundle exists. The
model was swapped from Random Forest to XGBoost without touching the API or the
frontend. The bundle stores feature ordering and imputation medians alongside
the model, so serving reproduces training exactly — a mismatched column order
would silently produce wrong predictions rather than erroring.

### Endpoints

- `GET /fighters` — all fighters with UFC-only records
- `GET /predict?fighter1={name}&fighter2={name}` — win probabilities

---

## Stack

Python, XGBoost, scikit-learn, pandas, FastAPI. Containerized with Docker,
deployed on Render; frontend in Next.js/React on Vercel. Data from the
[UFC Dataset 1994–2026](https://www.kaggle.com/datasets/jossilva3110/ufc-dataset-1994-2026)
and betting odds from the
[Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset).

## Running locally

```bash
pip install -r requirements.txt
python train.py
cd app && uvicorn main:app --reload
```

## Known limitations

- Statistics are **UFC-only**; pre-UFC fights are invisible, so records start at
  zero on UFC debut and display smaller than fans expect.
- Fighters are joined across files by name; duplicate names collapse to the
  first occurrence.
- Reach is missing for ~24% of fighters and filled with the training median.
- **Open-stance experience** — a known stylistic factor — could not be tested
  because stance is missing for 19% of fighters.
- No weight-class feature, so division-specific base rates are not modeled.
