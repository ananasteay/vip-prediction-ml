from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from src.data import DEFAULT_PATH, load_users, split_features_and_labels
from src.evaluate import select_threshold
from src.features import CATEGORICAL_FEATURES, build_features, fit_feature_params

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "vip_model.pkl"
TARGET_COLUMN = "віп/не віп"
RANDOM_STATE = 42

CANDIDATE_PARAMS = [
    {"depth": 4, "iterations": 300, "learning_rate": 0.05, "l2_leaf_reg": 5},
    {"depth": 3, "iterations": 300, "learning_rate": 0.05, "l2_leaf_reg": 10},
    {"depth": 4, "iterations": 500, "learning_rate": 0.03, "l2_leaf_reg": 3},
]


def _tie_safe_boundary(sorted_values: np.ndarray, boundary: int) -> int:
    """Push a split boundary forward past rows sharing the exact same value
    as the row at the boundary, so a temporal split never separates
    same-timestamp rows across train/calibration/holdout.
    """
    while boundary < len(sorted_values) and sorted_values[boundary] == sorted_values[boundary - 1]:
        boundary += 1
    return boundary


def split_train_calib_holdout(
    features_raw: pd.DataFrame,
    y: np.ndarray,
    split_mode: str = "random",
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """60/20/20 split into train (fit model), calibration (model selection +
    threshold selection), and holdout (final unbiased report only).

    "temporal" orders rows by FTD date so calibration/holdout always follow
    train chronologically - closer to how the model will actually be used
    (scored on users who convert after the training window closes) than a
    random shuffle. Rows tied on the exact same FTD timestamp are kept on
    the same side of each boundary. `split_mode` must be "random" or
    "temporal" - anything else raises rather than silently falling back to
    a random split.
    """
    if len(y) == 0:
        raise ValueError("no rows to split")
    if len(features_raw) != len(y):
        raise ValueError(f"features_raw and y must have the same length, got {len(features_raw)} and {len(y)}")

    if split_mode == "temporal":
        ftd_at = pd.to_datetime(features_raw.iloc[:, 1], errors="coerce")
        if ftd_at.isna().any():
            n_bad = int(ftd_at.isna().sum())
            raise ValueError(
                f"{n_bad} row(s) have a missing or unparseable FTD date (column B) - "
                "temporal splitting requires a valid date for every row."
            )
        order = np.argsort(ftd_at.to_numpy())
        features_sorted = features_raw.iloc[order].reset_index(drop=True)
        y_sorted = y[order]
        ftd_sorted = ftd_at.to_numpy()[order]

        n_train = _tie_safe_boundary(ftd_sorted, int(len(y_sorted) * 0.6))
        n_calib = _tie_safe_boundary(ftd_sorted, max(int(len(y_sorted) * 0.8), n_train))

        X_train, X_calib, X_holdout = (
            features_sorted.iloc[:n_train],
            features_sorted.iloc[n_train:n_calib],
            features_sorted.iloc[n_calib:],
        )
        y_train, y_calib, y_holdout = y_sorted[:n_train], y_sorted[n_train:n_calib], y_sorted[n_calib:]
    elif split_mode == "random":
        X_train, X_rest, y_train, y_rest = train_test_split(
            features_raw, y, test_size=0.4, stratify=y, random_state=random_state
        )
        X_calib, X_holdout, y_calib, y_holdout = train_test_split(
            X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=random_state
        )
    else:
        raise ValueError(f"split_mode must be 'random' or 'temporal', got {split_mode!r}")

    if y_train.sum() == 0:
        raise ValueError("training split has no positive (VIP) examples")
    if y_calib.sum() == 0:
        raise ValueError("calibration split has no positive (VIP) examples")

    return X_train, X_calib, X_holdout, y_train, y_calib, y_holdout


def _select_best_model(
    X_train: pd.DataFrame, y_train: np.ndarray, X_calib: pd.DataFrame, y_calib: np.ndarray
) -> tuple[CatBoostClassifier, float]:
    """Fit each candidate config, keep the one with the best calibration PR-AUC."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    best_model, best_score = None, -1.0
    for params in CANDIDATE_PARAMS:
        model = CatBoostClassifier(
            **params,
            cat_features=CATEGORICAL_FEATURES,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        model.fit(X_train, y_train)
        score = average_precision_score(y_calib, model.predict_proba(X_calib)[:, 1])
        if score > best_score:
            best_model, best_score = model, score
    return best_model, best_score


def train_model(
    data_path: Path = DEFAULT_PATH,
    include_ftd_amount: bool = True,
    split_mode: str = "random",
    model_path: Optional[Path] = MODEL_PATH,
) -> dict:
    """Train on columns A-K, select among a few candidate configs and
    calibrate a decision threshold using calibration data only, and
    optionally persist the artifact. Returns the artifact plus the
    calibration and holdout splits for evaluation.
    """
    users = load_users(data_path)
    features_raw, labels_raw = split_features_and_labels(users)
    y = labels_raw[TARGET_COLUMN].to_numpy()

    X_train_raw, X_calib_raw, X_holdout_raw, y_train, y_calib, y_holdout = split_train_calib_holdout(
        features_raw, y, split_mode=split_mode
    )

    params = fit_feature_params(X_train_raw)
    X_train = build_features(X_train_raw, params, include_ftd_amount)
    X_calib = build_features(X_calib_raw, params, include_ftd_amount)
    X_holdout = build_features(X_holdout_raw, params, include_ftd_amount)

    model, calib_pr_auc = _select_best_model(X_train, y_train, X_calib, y_calib)

    calib_proba = model.predict_proba(X_calib)[:, 1]
    threshold = select_threshold(y_calib, calib_proba)

    artifact = {
        "model": model,
        "feature_params": params,
        "threshold": threshold,
        "include_ftd_amount": include_ftd_amount,
        "feature_columns": list(X_train.columns),
        "calib_pr_auc": calib_pr_auc,
    }

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(artifact, f)

    return {
        "artifact": artifact,
        "calib": (X_calib, y_calib),
        "holdout": (X_holdout, y_holdout),
    }


if __name__ == "__main__":
    # Production config: temporal split (matches real deployment timing).
    # include_ftd_amount=False is comparable to True on calibration (see
    # README) - kept because the assignment values not relying on raw
    # FTD-day spend, not because it scores clearly higher.
    result = train_model(include_ftd_amount=False, split_mode="temporal")
    artifact = result["artifact"]
    print(f"Saved model to {MODEL_PATH}")
    print(f"Calibration PR-AUC: {artifact['calib_pr_auc']:.4f}")
    print(f"Selected threshold (from calibration data): {artifact['threshold']:.4f}")
