"""Synthetic A-K data builders shared across tests."""
import pandas as pd

from src.features import FEATURE_COLUMNS


def make_raw_features(n: int = 10, **overrides) -> pd.DataFrame:
    base_time = pd.Timestamp("2025-01-01")
    data = {
        "uid": [f"user-{i}" for i in range(n)],
        "ftd_at": [base_time + pd.Timedelta(hours=i) for i in range(n)],
        "registered_at": [base_time - pd.Timedelta(days=5) for _ in range(n)],
        "avg_bet_size": [1.0 + i for i in range(n)],
        "spins_count": [10.0 + i for i in range(n)],
        "email_domain": ["gmail.com"] * n,
        "app_installed_at": [base_time - pd.Timedelta(days=1) for _ in range(n)],
        "failed_deposits": [0] * n,
        "ftd_amount": [20.0 + i for i in range(n)],
        "tx_count": [1] * n,
        "avg_tx_check": [20.0 + i for i in range(n)],
    }
    df = pd.DataFrame(data)
    for col, values in overrides.items():
        df[col] = values
    assert list(df.columns) == FEATURE_COLUMNS
    return df
