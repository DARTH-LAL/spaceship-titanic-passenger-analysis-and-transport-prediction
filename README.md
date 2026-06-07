# Spaceship Titanic Data Analytics

A data analytics and machine learning project for the [Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic). The goal is to predict which passengers were transported to an alternate dimension using recovered ship records.

This repository currently uses a same-seed multi-config CatBoost blend with analytics-focused feature engineering for passenger groups, cabin structure, spending behavior, and family signals.

## Project Layout

- `train_and_submit.py`: training, feature engineering, cross-validation, and submission file generation
- `requirements.txt`: minimal Python dependencies for the current script
- `README.md`: project overview and run instructions

Competition data and generated submissions are intentionally excluded from version control.

## Environment

This project was verified with:

- Python `3.13`
- `numpy==2.1.3`
- `pandas==2.2.3`
- `catboost==1.2.10`

## Setup

Create a virtual environment and install dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Accept the Kaggle competition rules, then download the competition files yourself from the competition page:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Place those files in the project root before running the script.

## Run

Train the model and create a Kaggle submission:

```bash
python3.13 train_and_submit.py
```

If you want to skip cross-validation and write only the submission file:

```bash
python3.13 train_and_submit.py --skip-cv
```

The script writes predictions to:

```text
submission.csv
```

## Modeling Notes

The current pipeline includes:

- rule-based imputations for `CryoSleep`, `HomePlanet`, `Destination`, and `Age`
- engineered features such as `GroupSize`, `FamilySize`, `TotalSpend`, `LogTotalSpend`, `SpendPerAge`, and `LuxurySpend`
- categorical cabin-derived features such as `CabinDeck`, `CabinSide`, and `DeckSide`
- a two-model CatBoost blend built from closely related feature sets

## Publishing Note

This repository is set up to be GitHub-friendly:

- Kaggle competition data is gitignored
- generated `submission.csv` files are gitignored
- CatBoost training artifacts are gitignored

That makes it safe to publish the code and documentation while keeping competition files local.
