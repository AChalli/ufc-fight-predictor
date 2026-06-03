# UFC Fight Predictor API

A REST API that predicts UFC fight outcomes using a Random Forest classifier trained on 8,500+ historical fights.

## How It Works

For any two fighters, the model computes differentials across 11 features derived from career statistics — striking volume, striking accuracy, strikes absorbed, takedown averages, takedown defense, submission attempts, reach, and win/loss record. These differentials are passed to a Random Forest classifier that returns win probabilities for each fighter.

**Model accuracy: 69.5%** on held-out test data (vs. 63% baseline of always picking Fighter 1).

**Key finding:** Striking volume differential (SLpM) is the strongest predictor of fight outcomes. Reach advantage, commonly cited in broadcast commentary, ranked last among all features.

## Endpoints

### `GET /fighters`
Returns a list of all 4,400+ fighters in the dataset with their name, record, and ID.

**Response:**
```json
[
  {
    "id": "Jon Jones",
    "name": "Jon Jones",
    "record": "27-1-0",
    "weightClass": ""
  }
]
```

### `GET /predict`
Returns win probabilities and a predicted winner for two fighters.

**Query Parameters:**
- `fighter1` — name of the first fighter
- `fighter2` — name of the second fighter

**Example:**
GET /predict?fighter1=Jon Jones&fighter2=Stipe Miocic
**Response:**
```json
{
  "fighter1": "Jon Jones",
  "fighter2": "Stipe Miocic",
  "fighter1_win_probability": 0.84,
  "fighter2_win_probability": 0.16,
  "predicted_winner": "Jon Jones"
}
```

## Tech Stack

- **Python** — FastAPI, scikit-learn, pandas, joblib
- **Model** — Random Forest Classifier (100 estimators)
- **Data** — UFC Stats (1993–2026), 8,500+ fights, 4,400+ fighters

## Running Locally

```bash
cd app
pip install fastapi uvicorn scikit-learn pandas joblib
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`.

## Data & Limitations

Fighter statistics are career averages at the time of data extraction, not historical snapshots per fight. This means the model uses current stats to predict past fights, which introduces a known data integrity limitation. Rolling averages calculated at fight time would improve accuracy and is a planned improvement.

## Dataset

[UFC Dataset 1994–2026](https://www.kaggle.com/datasets/jossilva3110/ufc-dataset-1994-2026) via Kaggle.
