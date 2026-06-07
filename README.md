# Spaceship Titanic Passenger Analysis and Transport Prediction

This project is about analyzing Spaceship Titanic passenger data and building a model to predict which passengers were transported to an alternate dimension in the [Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic).

It is packaged as a data-analysis-focused portfolio project: the repository emphasizes data cleaning, feature engineering, evaluation, and a reproducible training workflow rather than only a final submission file.

## Portfolio Focus

- exploratory thinking around passenger behavior, cabin structure, and spending patterns
- practical data preparation for messy real-world competition data
- feature engineering for groups, families, cabin layout, and spending intensity
- gradient-boosted modeling with CatBoost for mixed numeric and categorical data
- repeatable training and submission generation from a single script

## Repository Layout

- `train_and_submit.py`: end-to-end pipeline for feature engineering, validation, training, and submission generation
- `requirements.txt`: project dependencies
- `README.md`: project overview, setup, and usage notes

Competition data, generated submissions, and CatBoost artifacts are intentionally excluded from version control.

## Environment

Verified with:

- Python `3.13`
- `numpy==2.1.3`
- `pandas==2.2.3`
- `catboost==1.2.10`

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

After joining the Kaggle competition and accepting its rules, download these files from the competition page and place them in the project root:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Those files are ignored by git so the repository stays safe to publish.

## Run

Train the model and generate `submission.csv`:

```bash
python3.13 train_and_submit.py
```

If you want to skip cross-validation and only write a submission:

```bash
python3.13 train_and_submit.py --skip-cv
```

## Modeling Approach

The current pipeline includes:

- rule-based imputations for missing passenger and spending fields
- engineered features for passenger groups, family structure, cabin location, and total spend behavior
- cabin-derived categorical features such as `CabinDeck`, `CabinSide`, and `DeckSide`
- a CatBoost-based blended classifier tailored to mixed-type competition data

## Public Repo Notes

- Kaggle competition CSV files are gitignored
- generated `submission.csv` files are gitignored
- CatBoost training artifacts are gitignored

That keeps the repository focused on analysis and modeling code while leaving competition assets local.
