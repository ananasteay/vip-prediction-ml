from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score

MIN_RECALL = 0.8
MAX_COUNT_RATIO_DEVIATION = 0.1


def _validate_scores(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if y_true.shape[0] == 0:
        raise ValueError("y_true is empty")
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            f"y_true and y_proba/y_pred must have the same length, got "
            f"{y_true.shape[0]} and {y_proba.shape[0]}"
        )


def recall_at_ratio(y_true: np.ndarray, y_proba: np.ndarray, ratio: float) -> tuple[float, int]:
    """Diagnostic only: recall (and the rank-based predicted count it
    corresponds to) if `ratio` x the actual VIP count were predicted by
    rank (top-k). Never used to pick a threshold - for reporting only.

    This is rank-based, not `proba >= threshold`-based, so unlike
    `select_threshold` it does not need to account for ties at a cutoff.
    """
    _validate_scores(y_true, y_proba)
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0, got {ratio}")
    actual_positives = int(y_true.sum())
    if actual_positives == 0:
        return float("nan"), 0
    k = min(max(1, round(ratio * actual_positives)), len(y_true))
    top_k = np.argsort(-y_proba)[:k]
    return float(y_true[top_k].sum() / actual_positives), k


def min_count_for_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = MIN_RECALL) -> int:
    """Diagnostic only: smallest predicted count (by rank) that reaches
    min_recall on this data. Never used to pick a threshold.

    Raises ValueError if the inputs are empty/mismatched, there are no
    positive examples (recall is undefined), or min_recall is outside
    [0, 1] (and therefore unreachable regardless of the data, since recall
    is non-decreasing in count and reaches 1.0 at count == len(y_true)).
    """
    _validate_scores(y_true, y_proba)
    actual_positives = int(y_true.sum())
    if actual_positives == 0:
        raise ValueError("y_true has no positive examples - recall is undefined")
    if not 0 <= min_recall <= 1:
        raise ValueError(f"min_recall must be in [0, 1], got {min_recall}")

    order = np.argsort(-y_proba)
    recalls = np.cumsum(y_true[order]) / actual_positives
    reaches_target = recalls >= min_recall
    if not reaches_target.any():
        # Shouldn't happen given the min_recall check above, but never trust
        # argmax on an all-False array - it silently returns index 0.
        raise ValueError(f"min_recall={min_recall} is unreachable on this data")
    return int(np.argmax(reaches_target)) + 1


def select_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = MIN_RECALL,
    max_count_ratio_deviation: float = MAX_COUNT_RATIO_DEVIATION,
) -> float:
    """Pick, on calibration data only, the strictest threshold that keeps
    recall >= min_recall and predicted count within max_count_ratio_deviation
    of the actual positive count. Falls back to the strictest threshold that
    still reaches min_recall (ignoring the count window) if no threshold
    satisfies both - the count constraint is typically the binding one given
    how rare VIPs are.

    Inference applies `proba >= threshold`, so a tie at the chosen cutoff
    pulls in every row with that exact score, not just one. Recall/ratio
    here are computed from that realized `>=` count, not from raw rank, so
    a tie never makes this report a tighter count than what `predict.py`
    would actually produce.
    """
    _validate_scores(y_true, y_proba)
    actual_positives = int(y_true.sum())
    if actual_positives == 0:
        raise ValueError("y_true has no positive examples - recall is undefined")
    if not 0 <= min_recall <= 1:
        raise ValueError(f"min_recall must be in [0, 1], got {min_recall}")
    if max_count_ratio_deviation < 0:
        raise ValueError(f"max_count_ratio_deviation must be >= 0, got {max_count_ratio_deviation}")

    order = np.argsort(-y_proba)
    sorted_proba = y_proba[order]
    sorted_labels = y_true[order]
    sorted_cum = np.cumsum(sorted_labels)

    # realized_count[i] = how many rows "proba >= sorted_proba[i]" actually
    # selects - equal to i+1 when there are no ties, larger when there are.
    ascending = np.sort(y_proba)
    realized_count = len(y_true) - np.searchsorted(ascending, sorted_proba, side="left")
    realized_tp = sorted_cum[realized_count - 1]

    recalls = realized_tp / actual_positives
    ratios = realized_count / actual_positives

    meets_recall = recalls >= min_recall
    meets_both = meets_recall & (np.abs(ratios - 1) <= max_count_ratio_deviation)

    if meets_both.any():
        idx = np.where(meets_both)[0][0]
    elif meets_recall.any():
        idx = np.where(meets_recall)[0][0]
    else:
        idx = len(y_true) - 1  # unreachable given the min_recall check above; kept as a safety net

    return float(sorted_proba[idx])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    _validate_scores(y_true, y_pred)
    actual_vip = int(y_true.sum())
    predicted_vip = int(y_pred.sum())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    ratio = predicted_vip / actual_vip if actual_vip else float("nan")
    return {
        "recall": recall,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "actual_vip_count": actual_vip,
        "predicted_vip_count": predicted_vip,
        "predicted_to_actual_ratio": ratio,
        "meets_recall_requirement": recall >= MIN_RECALL,
        "meets_count_requirement": bool(actual_vip) and abs(ratio - 1) <= MAX_COUNT_RATIO_DEVIATION,
    }


def _report(name: str, metrics: dict, pr_auc: float, y_true: np.ndarray, y_proba: np.ndarray) -> None:
    recall_flag = "OK" if metrics["meets_recall_requirement"] else "NOT MET"
    count_flag = "OK" if metrics["meets_count_requirement"] else "NOT MET"
    r90, k90 = recall_at_ratio(y_true, y_proba, 0.9)
    r100, k100 = recall_at_ratio(y_true, y_proba, 1.0)
    r110, k110 = recall_at_ratio(y_true, y_proba, 1.1)
    min_count = min_count_for_recall(y_true, y_proba)
    print(f"\n[{name}]")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(
        f"  recall @ 0.9x actual VIP (k={k90})={r90:.3f}   @1.0x (k={k100})={r100:.3f}   "
        f"@1.1x (k={k110})={r110:.3f}   (diagnostic only, never used to pick a threshold)"
    )
    print(f"  smallest predicted count reaching {MIN_RECALL:.0%} recall: {min_count} "
          f"({min_count / int(y_true.sum()):.1f}x actual VIP count)")
    print(f"  recall:    {metrics['recall']:.3f}  (target >= {MIN_RECALL})  -> {recall_flag}")
    print(f"  precision: {metrics['precision']:.3f}")
    print(f"  confusion matrix: {metrics['confusion_matrix']}")
    print(
        f"  actual VIP: {metrics['actual_vip_count']}   predicted VIP: {metrics['predicted_vip_count']}"
        f"   ratio: {metrics['predicted_to_actual_ratio']:.3f}  -> {count_flag}"
    )


if __name__ == "__main__":
    from src.train import train_model

    for split_mode in ["random", "temporal"]:
        for include_ftd_amount, label in [(True, "with FTD-day deposit amount"), (False, "without FTD-day deposit amount")]:
            result = train_model(include_ftd_amount=include_ftd_amount, split_mode=split_mode, model_path=None)
            threshold = result["artifact"]["threshold"]
            model = result["artifact"]["model"]
            X_holdout, y_holdout = result["holdout"]

            proba = model.predict_proba(X_holdout)[:, 1]
            y_pred = (proba >= threshold).astype(int)
            metrics = evaluate_predictions(y_holdout, y_pred)
            pr_auc = average_precision_score(y_holdout, proba)
            _report(f"Holdout [{split_mode}] - {label} (threshold={threshold:.4f})", metrics, pr_auc, y_holdout, proba)
