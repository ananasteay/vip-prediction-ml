from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "test_dataset.xlsx"
N_FEATURE_COLUMNS = 11  # columns A-K


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def load_users(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    return _strip_columns(df)


def load_deposits(path: Path = DEFAULT_PATH) -> pd.DataFrame | None:
    sheet_names = pd.ExcelFile(path).sheet_names
    if len(sheet_names) < 2:
        return None
    df = pd.read_excel(path, sheet_name=1)
    return _strip_columns(df)


def split_features_and_labels(users: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = users.iloc[:, :N_FEATURE_COLUMNS]
    labels = users.iloc[:, N_FEATURE_COLUMNS:]
    return features, labels