# UFC Fight Predictor API

A REST API that predicts UFC fight outcomes using a Random Forest classifier trained on 4,700+ historical fights, with features computed strictly from information available before each fight.

## Results

**61.2% accuracy** on a held-out chronological test set (fights after May 2023), against a 50% baseline.

An earlier version of this model reported 69.5%. That number was wrong, and finding out why was the most useful part of building this.

## The Leakage Problem

The original pipeline pulled fighter statistics from a roster file containing career averages as of the scrape date in 2026. Predicting a 2015 fight with a fighter's 2026 career stats means the model already knows how that fighter's career turned out. It was not predicting fights, it was reading the answer key.

Two symptoms made this visible:

- Accuracy sat well above the 62-64% range reported by comparable public projects
- Career win differential ranked as the second most important feature

The fix was to rebuild every feature from per-fight records. For each bout, a fighter's stats are the cumulative sum of all their *prior* fights only, computed via a grouped cumulative sum minus the current row. Three related corrections came with it:

- **Chronological split** instead of random. A random split lets the model train on 2025 fights to predict 2018 ones, which is a subtler version of the same problem.
- **Train-set medians** for missing-value imputation, so test-set statistics never influence training.
- **Debut filter** requiring 2+ prior fights, since a fighter with no history has no computable stats. Swept across 1, 2, and 3 priors; 2 performed best, though the differences fall within the noise floor for a test set this size.

After the fix, career win differential dropped from 2nd to 10th in feature importance. Most of its apparent predictive power had been leakage.

## What Predicts a Fight

| Rank | Feature | Importance |
|---|---|---|
| 1 | Strikes absorbed per minute | 0.122 |
| 2 | Striking defense | 0.116 |
| 3 | Strikes landed per minute | 0.111 |
| 4 | Takedowns per 15 min | 0.109 |
| ... | | |
| 9 | Reach | 0.058 |
| 10 | Win differential | 0.057 |
| 11 | Loss differential | 0.053 |

The top two are both defensive. Not getting hit predicts winning more strongly than hitting does.

Reach advantage, cited constantly in broadcast commentary, ranks 9th of 11. This held across both the leaky and corrected pipelines, which makes it the more robust of the two findings.

## Endpoints

### `GET /fighters`
All fighters with 1+ recorded UFC bouts, with UFC-only records.

### `GET /predict?fighter1={name}&fighter2={name}`
Win probabilities for a matchup.

```json
{
  "fighter1": "Jon Jones",
  "fighter2": "Stipe Miocic",
  "fighter1_win_probability": 0.71,
  "fighter2_win_probability": 0.29,
  "predicted_winner": "Jon Jones"
}
```

## Architecture

`train.py` owns all model work: feature engineering, training, evaluation, and writing both the model bundle and a serving-format stats file. `app/main.py` loads the bundle and serves it. The model can be swapped entirely without touching the API or the frontend.

The bundle stores the feature column order and training medians alongside the model, so serving reproduces training exactly. Mismatched column order would not raise an error in scikit-learn, it would silently produce wrong predictions.

## Stack

Python, FastAPI, scikit-learn, pandas. Containerized with Docker, deployed on Render. The model is retrained during the image build, so every deploy ships a freshly trained model.

## Running Locally

```bash
pip install -r requirements.txt
python train.py
cd app && uvicorn main:app --reload
```

## Known Limitations

- Records and statistics are UFC-only. Fights in other promotions are not in the dataset, so a fighter's pre-UFC experience is invisible to the model. Adding it as a static feature (career total minus UFC total) is a planned improvement.
- Fighters are matched across the two source files by name. Duplicate names are collapsed to the first occurrence, which assigns a small number of fighters the wrong biometrics.
- Reach is missing for roughly 24% of fighters in the serving set and is filled with the training median.

## Roadmap

- Weekly automated scraping and retraining
- XGBoost comparison against the current Random Forest
- Backtesting against historical closing odds, since accuracy against a 50% baseline says little about whether the model has edge over a market that already prices in most of this information

## Data

[UFC Dataset 1994-2026](https://www.kaggle.com/datasets/jossilva3110/ufc-dataset-1994-2026)
