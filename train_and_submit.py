#!/usr/bin/env python3
"""Train a same-seed multi-config CatBoost blend for Spaceship Titanic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "catboost is not installed. Install with: python -m pip install catboost"
    ) from exc


CV_SEED = 42
MODEL_SEEDS = [42]
BLEND_THRESHOLD = 0.500
ENABLE_PSEUDO_LABELING = False
PSEUDO_CONFIDENCE_THRESHOLD = 0.020
PSEUDO_SAMPLE_WEIGHT = 0.70
MIN_PSEUDO_ROWS = 20

BASE_CAT_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "CabinDeck",
    "CabinSide",
    "DeckSide",
    "AgeBin",
    "LastName",
]

BASE_NUM_FEATURES = [
    "Age",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "GroupSize",
    "FamilySize",
    "CabinNum",
    "TotalSpend",
    "LogTotalSpend",
    "SpendCount",
    "HasSpend",
    "AgeLogSpend",
    "CabinDeckNum",
    "IsAlone",
]

MODEL_SPECS = [
    {
        "name": "m0_base",
        "weight": 0.990,
        "cat_features": BASE_CAT_FEATURES,
        "num_features": BASE_NUM_FEATURES,
        "params": {
            "iterations": 2600,
            "depth": 7,
            "learning_rate": 0.038,
            "l2_leaf_reg": 6,
            "bagging_temperature": 0.7,
            "random_strength": 1.0,
        },
    },
    {
        "name": "m6_spendbin",
        "weight": 0.010,
        "cat_features": BASE_CAT_FEATURES + ["SpendBin"],
        "num_features": BASE_NUM_FEATURES + ["SpendPerAge", "LuxurySpend"],
        "params": {
            "iterations": 2600,
            "depth": 7,
            "learning_rate": 0.036,
            "l2_leaf_reg": 6,
            "bagging_temperature": 0.65,
            "random_strength": 1.0,
        },
    },
]


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic stratified train/validation indices without sklearn."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    unique_classes = np.unique(y)

    per_class_folds: dict[int, list[np.ndarray]] = {}
    for cls in unique_classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        per_class_folds[int(cls)] = list(np.array_split(cls_idx, n_splits))

    all_idx = np.arange(len(y))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_i in range(n_splits):
        val_parts = [per_class_folds[int(cls)][fold_i] for cls in unique_classes]
        val_idx = np.concatenate(val_parts)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_idx[train_mask]
        splits.append((train_idx, val_idx))
    return splits


def normalized_specs() -> list[dict]:
    total_weight = sum(float(spec["weight"]) for spec in MODEL_SPECS)
    if total_weight <= 0:
        raise ValueError("MODEL_SPECS weights must sum to a positive value.")

    specs: list[dict] = []
    for spec in MODEL_SPECS:
        out = dict(spec)
        out["weight"] = float(spec["weight"]) / total_weight
        specs.append(out)
    return specs


def build_features(df: pd.DataFrame, full_reference_df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering + rule-based imputations."""
    out = df.copy()

    pid_parts = out["PassengerId"].str.split("_", expand=True)
    out["GroupId"] = pid_parts[0]
    out["GroupNum"] = pd.to_numeric(pid_parts[1], errors="coerce")

    all_pid = full_reference_df["PassengerId"].str.split("_", expand=True)
    out["GroupSize"] = out["GroupId"].map(all_pid[0].value_counts(dropna=False))

    cabin_parts = out["Cabin"].fillna("Unknown/Unknown/Unknown").str.split("/", expand=True)
    out["CabinDeck"] = cabin_parts[0]
    out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    out["CabinSide"] = cabin_parts[2]
    out["DeckSide"] = out["CabinDeck"].astype(str) + "_" + out["CabinSide"].astype(str)
    out["CabinDeckNum"] = out["CabinDeck"].map(
        {deck: idx for idx, deck in enumerate(list("ABCDEFGT"))}
    )

    name_parts = out["Name"].fillna("Unknown Unknown").str.rsplit(" ", n=1, expand=True)
    out["LastName"] = name_parts[1]

    all_name_parts = (
        full_reference_df["Name"].fillna("Unknown Unknown").str.rsplit(" ", n=1, expand=True)
    )
    out["FamilySize"] = out["LastName"].map(all_name_parts[1].value_counts(dropna=False))
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")

    out.loc[out["VIP"].isna(), "VIP"] = False

    spend_sum_raw = out[spend_cols].sum(axis=1, skipna=True)
    out.loc[out["CryoSleep"].isna() & (spend_sum_raw > 0), "CryoSleep"] = False
    out.loc[out["CryoSleep"].isna() & (spend_sum_raw == 0), "CryoSleep"] = True

    cryo_true = out["CryoSleep"] == True
    for col in spend_cols:
        out.loc[cryo_true & out[col].isna(), col] = 0.0
        out[col] = out[col].fillna(0.0)

    grp_home = out.groupby("GroupId")["HomePlanet"].agg(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan
    )
    out["HomePlanet"] = out["HomePlanet"].fillna(out["GroupId"].map(grp_home))
    out["HomePlanet"] = out["HomePlanet"].fillna(
        out["CabinDeck"].map(
            {
                "A": "Europa",
                "B": "Europa",
                "C": "Europa",
                "T": "Europa",
                "G": "Earth",
            }
        )
    )
    out["HomePlanet"] = out["HomePlanet"].fillna("Earth")

    grp_dest = out.groupby("GroupId")["Destination"].agg(
        lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else np.nan
    )
    out["Destination"] = out["Destination"].fillna(out["GroupId"].map(grp_dest))
    out["Destination"] = out["Destination"].fillna("TRAPPIST-1e")

    age_median = out.groupby(["HomePlanet", "CryoSleep"])["Age"].transform("median")
    out["Age"] = out["Age"].fillna(age_median)
    out["Age"] = out["Age"].fillna(out["Age"].median())

    out["TotalSpend"] = out[spend_cols].sum(axis=1)
    out["LogTotalSpend"] = np.log1p(out["TotalSpend"])
    out["SpendCount"] = (out[spend_cols] > 0).sum(axis=1)
    out["HasSpend"] = (out["TotalSpend"] > 0).astype(int)
    out["AgeLogSpend"] = out["Age"] * out["LogTotalSpend"]
    out["SpendPerAge"] = out["TotalSpend"] / (out["Age"] + 1)
    out["LuxurySpend"] = out[["Spa", "VRDeck"]].sum(axis=1)

    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 25, 40, 60, 120],
        labels=["Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
    ).astype(str)

    out["SpendBin"] = pd.cut(
        out["TotalSpend"],
        bins=[-1, 0, 1, 200, 800, 2000, 100000],
        labels=["Zero", "Tiny", "Low", "Mid", "High", "Ultra"],
    ).astype(str)

    return out


def prepare_model_matrix(
    full_features: pd.DataFrame, cat_features: list[str], num_features: list[str]
) -> tuple[pd.DataFrame, list[int]]:
    cols = cat_features + num_features
    matrix = full_features[cols].copy()
    for col in cat_features:
        matrix[col] = matrix[col].astype(str)
    matrix = matrix.fillna(-1)
    cat_idx = [matrix.columns.get_loc(col) for col in cat_features]
    return matrix, cat_idx


def build_model(seed: int, params: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=seed,
        verbose=False,
        **params,
    )


def pseudo_refit_probs(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_idx: list[int],
) -> tuple[np.ndarray, int]:
    """Optionally refit with high-confidence pseudo labels and return test probs."""
    base_probs = model.predict_proba(X_test)[:, 1]
    if not ENABLE_PSEUDO_LABELING:
        return base_probs, 0

    pseudo_mask = (base_probs <= PSEUDO_CONFIDENCE_THRESHOLD) | (
        base_probs >= 1.0 - PSEUDO_CONFIDENCE_THRESHOLD
    )
    pseudo_rows = int(np.sum(pseudo_mask))
    if pseudo_rows < MIN_PSEUDO_ROWS:
        return base_probs, pseudo_rows

    pseudo_targets = (base_probs[pseudo_mask] >= 0.5).astype(int)
    X_aug = pd.concat([X_train, X_test.iloc[pseudo_mask]], axis=0)
    y_aug = np.concatenate([y_train, pseudo_targets])
    sample_weight = np.concatenate(
        [
            np.ones(len(y_train), dtype=float),
            np.full(pseudo_rows, PSEUDO_SAMPLE_WEIGHT, dtype=float),
        ]
    )

    pseudo_model = CatBoostClassifier(**model.get_params())
    pseudo_model.fit(
        X_aug,
        y_aug,
        cat_features=cat_idx,
        sample_weight=sample_weight,
    )
    return pseudo_model.predict_proba(X_test)[:, 1], pseudo_rows


def run_cv_accuracy(
    train_inputs: dict[str, tuple[pd.DataFrame, list[int]]], y: np.ndarray, specs: list[dict]
) -> tuple[float, float]:
    cv = stratified_kfold_indices(y=y, n_splits=5, seed=CV_SEED)
    ref_name = specs[0]["name"]
    n_rows = len(train_inputs[ref_name][0])

    oof = np.zeros(n_rows, dtype=float)
    fold_scores: list[float] = []

    for train_idx, val_idx in cv:
        y_tr, y_val = y[train_idx], y[val_idx]
        val_probs = np.zeros(len(val_idx), dtype=float)

        for spec in specs:
            X, cat_idx = train_inputs[spec["name"]]
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]

            seed_probs = np.zeros(len(val_idx), dtype=float)
            for seed in MODEL_SEEDS:
                model = build_model(seed, spec["params"])
                model.fit(
                    X_tr,
                    y_tr,
                    cat_features=cat_idx,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                    early_stopping_rounds=200,
                )
                seed_probs += model.predict_proba(X_val)[:, 1]

            seed_probs /= len(MODEL_SEEDS)
            val_probs += float(spec["weight"]) * seed_probs

        oof[val_idx] = val_probs
        fold_scores.append(accuracy_score_np(y_val, (val_probs >= BLEND_THRESHOLD).astype(int)))

    base = accuracy_score_np(y, (oof >= BLEND_THRESHOLD).astype(int))
    return float(np.mean(fold_scores)), float(base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="train.csv", type=Path)
    parser.add_argument("--test-path", default="test.csv", type=Path)
    parser.add_argument("--output-path", default="submission.csv", type=Path)
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip 5-fold CV and directly train on full training set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    full_ref = pd.concat([train_df.drop(columns=["Transported"]), test_df], ignore_index=True)

    train_features = build_features(train_df.drop(columns=["Transported"]), full_ref)
    test_features = build_features(test_df, full_ref)
    y_train = train_df["Transported"].astype(int).values

    specs = normalized_specs()
    train_inputs: dict[str, tuple[pd.DataFrame, list[int]]] = {}
    test_inputs: dict[str, tuple[pd.DataFrame, list[int]]] = {}

    for spec in specs:
        train_inputs[spec["name"]] = prepare_model_matrix(
            train_features, spec["cat_features"], spec["num_features"]
        )
        test_inputs[spec["name"]] = prepare_model_matrix(
            test_features, spec["cat_features"], spec["num_features"]
        )

    if not args.skip_cv:
        cv_mean, cv_oof = run_cv_accuracy(train_inputs, y_train, specs)
        print(f"Blend threshold: {BLEND_THRESHOLD:.3f}")
        print(f"Model seeds: {MODEL_SEEDS}")
        print(
            "Blend specs: "
            + ", ".join(f"{s['name']}={s['weight']:.3f}" for s in specs)
        )
        print(f"5-fold CV (mean fold): {cv_mean:.5f}")
        print(f"5-fold CV (OOF blend): {cv_oof:.5f}")

    test_probs = np.zeros(len(test_df), dtype=float)
    for spec in specs:
        X_train, cat_idx = train_inputs[spec["name"]]
        X_test, _ = test_inputs[spec["name"]]

        seed_probs = np.zeros(len(test_df), dtype=float)
        spec_pseudo_rows: list[int] = []
        for seed in MODEL_SEEDS:
            model = build_model(seed, spec["params"])
            model.fit(X_train, y_train, cat_features=cat_idx)

            probs, pseudo_rows = pseudo_refit_probs(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                cat_idx=cat_idx,
            )
            spec_pseudo_rows.append(pseudo_rows)
            seed_probs += probs

        seed_probs /= len(MODEL_SEEDS)
        test_probs += float(spec["weight"]) * seed_probs
        if ENABLE_PSEUDO_LABELING:
            print(
                f"{spec['name']} pseudo rows per seed: {spec_pseudo_rows} "
                f"(thr={PSEUDO_CONFIDENCE_THRESHOLD:.3f}, w={PSEUDO_SAMPLE_WEIGHT:.2f})"
            )

    test_pred = (test_probs >= BLEND_THRESHOLD).astype(bool)

    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_pred}
    )
    submission.to_csv(args.output_path, index=False)

    print(f"Blend threshold: {BLEND_THRESHOLD:.3f}")
    print("Blend specs: " + ", ".join(f"{s['name']}={s['weight']:.3f}" for s in specs))
    print(f"Saved submission to {args.output_path}")
    print(submission.head(10))


if __name__ == "__main__":
    main()
