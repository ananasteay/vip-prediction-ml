from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.data import load_users, split_features_and_labels
from src.features import build_features
from src.train import MODEL_PATH

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "predictions.csv"


REQUIRED_ARTIFACT_KEYS = {"model", "feature_params", "threshold", "include_ftd_amount", "feature_columns"}


def load_artifact(model_path: Path = MODEL_PATH) -> dict:
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    missing = REQUIRED_ARTIFACT_KEYS - set(artifact.keys()) if isinstance(artifact, dict) else REQUIRED_ARTIFACT_KEYS
    if missing:
        raise ValueError(f"model artifact at {model_path} is invalid - missing keys: {sorted(missing)}")
    return artifact


def predict(input_path: Path, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    """Return a DataFrame with uid, vip_probability, predicted_vip for every
    row in the input workbook.
    """
    artifact = load_artifact(model_path)
    users = load_users(input_path)
    features_raw, _ = split_features_and_labels(users)

    X = build_features(features_raw, artifact["feature_params"], artifact["include_ftd_amount"])
    X = X[artifact["feature_columns"]]
    proba = artifact["model"].predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            # column A (uid) by position - the header text isn't guaranteed
            # to be literally "uid" on another workbook with the same layout.
            "uid": features_raw.iloc[:, 0].to_numpy(),
            "vip_probability": proba,
            "predicted_vip": (proba >= artifact["threshold"]).astype(int),
        }
    )


def predict_vip_uids(input_path: Path, model_path: Path = MODEL_PATH) -> pd.Series:
    predictions = predict(input_path, model_path)
    return predictions.loc[predictions["predicted_vip"] == 1, "uid"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Excel workbook with columns A-K")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="output CSV path")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="trained model artifact path")
    args = parser.parse_args()

    predictions = predict(args.input, args.model)
    vip_uids = predictions.loc[predictions["predicted_vip"] == 1, ["uid"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vip_uids.to_csv(args.output, index=False)

    print(f"Scored {len(predictions)} users, predicted {len(vip_uids)} VIP -> {args.output}")


if __name__ == "__main__":
    main()
