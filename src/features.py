from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "uid",
    "ftd_at",
    "registered_at",
    "avg_bet_size",
    "spins_count",
    "email_domain",
    "app_installed_at",
    "failed_deposits",
    "ftd_amount",
    "tx_count",
    "avg_tx_check",
]
DATE_COLUMNS = ["ftd_at", "registered_at", "app_installed_at"]
CATEGORICAL_FEATURES = ["email_domain"]
DEFAULT_TOP_N_DOMAINS = 20


@dataclass
class FeatureParams:
    top_domains: list[str]
    domain_categories: list[str]


def _canonicalize(features_raw: pd.DataFrame) -> pd.DataFrame:
    if features_raw.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"expected {len(FEATURE_COLUMNS)} columns (A-K), got {features_raw.shape[1]}")
    df = features_raw.copy()
    df.columns = FEATURE_COLUMNS
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _normalize_domain(domain: pd.Series) -> pd.Series:
    return domain.astype("string").str.strip().str.lower().fillna("missing")


def fit_feature_params(features_raw: pd.DataFrame, top_n_domains: int = DEFAULT_TOP_N_DOMAINS) -> FeatureParams:
    df = _canonicalize(features_raw)
    top_domains = _normalize_domain(df["email_domain"]).value_counts().head(top_n_domains).index.tolist()
    # Fixed category list so train/calib/holdout/inference all encode
    # email_domain to the same integer codes - pandas' astype("category")
    # otherwise derives categories per-call from whatever values are present,
    # which silently desyncs the codes the model sees across splits.
    domain_categories = sorted(set(top_domains) | {"other", "missing"})
    return FeatureParams(top_domains=top_domains, domain_categories=domain_categories)


def build_features(
    features_raw: pd.DataFrame,
    params: FeatureParams,
    include_ftd_amount: bool = True,
) -> pd.DataFrame:
    """Transform raw A-K columns into model-ready features.

    `uid` is dropped here; callers needing it for output should keep
    features_raw's first column (uid, position 0) aside before calling
    this - don't assume the header text is literally "uid".
    """
    df = _canonicalize(features_raw)
    out = pd.DataFrame(index=df.index)

    out["ftd_hour"] = df["ftd_at"].dt.hour
    out["ftd_dow"] = df["ftd_at"].dt.day_of_week
    out["ftd_is_weekend"] = out["ftd_dow"].isin([5, 6]).astype(int)

    # Registration after FTD is invalid; clip instead of dropping to tolerate
    # rare clock-skew errors rather than losing the row entirely.
    reg_gap_days = (df["ftd_at"] - df["registered_at"]).dt.total_seconds() / 86400
    out["reg_to_ftd_days"] = reg_gap_days.clip(lower=0)

    install_gap_days = (df["ftd_at"] - df["app_installed_at"]).dt.total_seconds() / 86400
    out["has_install_date"] = df["app_installed_at"].notna().astype(int)
    out["install_to_ftd_days"] = install_gap_days.clip(lower=0)

    out["avg_bet_size"] = df["avg_bet_size"]
    out["spins_count"] = df["spins_count"]
    out["bet_spins_missing"] = df["avg_bet_size"].isna().astype(int)

    out["failed_deposits"] = df["failed_deposits"]
    out["tx_count"] = df["tx_count"]
    out["avg_tx_check"] = df["avg_tx_check"]
    out["failed_deposit_ratio"] = _safe_divide(df["failed_deposits"], df["failed_deposits"] + df["tx_count"])
    out["spins_per_tx"] = _safe_divide(df["spins_count"], df["tx_count"])

    if include_ftd_amount:
        out["ftd_amount"] = df["ftd_amount"]
        out["deposit_per_spin"] = _safe_divide(df["ftd_amount"], df["spins_count"])
        out["bet_to_deposit_ratio"] = _safe_divide(df["avg_bet_size"], df["ftd_amount"])

    domain = _normalize_domain(df["email_domain"])
    domain = domain.where(domain.isin(params.top_domains), "other")
    out["email_domain"] = pd.Categorical(domain, categories=params.domain_categories)

    return out
