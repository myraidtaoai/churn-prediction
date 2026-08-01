"""Regression tests for the publication-safe aggregate model snapshot."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "public_data"
SOURCE_DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def read_rows(filename: str) -> list[dict[str, str]]:
    with (PUBLIC_DATA / filename).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def test_candidate_selection_matches_highest_validation_pr_auc() -> None:
    candidates = read_rows("model_comparison_metrics.csv")
    assert {row["algorithm"] for row in candidates} == {
        "balanced_random_forest",
        "xgboost",
        "lightgbm",
        "extra_trees",
    }
    selected = [row for row in candidates if row["selected"].lower() == "true"]
    assert len(selected) == 1
    assert float(selected[0]["validation_pr_auc"]) == max(
        float(row["validation_pr_auc"]) for row in candidates
    )


def test_champion_metrics_are_valid_probabilistic_scores() -> None:
    metrics = read_rows("model_validation_metrics.csv")
    assert len(metrics) == 1
    for name in {
        "validation_pr_auc",
        "test_roc_auc",
        "test_pr_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_balanced_accuracy",
        "classification_threshold",
    }:
        assert 0 <= float(metrics[0][name]) <= 1, name


def test_risk_portfolio_is_complete_and_monotonic() -> None:
    risk_rows = read_rows("risk_segment_summary.csv")
    by_segment = {row["risk_segment"]: row for row in risk_rows}
    assert set(by_segment) == {"High", "Medium", "Low"}
    with SOURCE_DATA.open(newline="", encoding="utf-8-sig") as source:
        source_count = sum(1 for _ in csv.DictReader(source))
    assert sum(int(row["customers"]) for row in risk_rows) == source_count
    assert (
        float(by_segment["High"]["average_churn_probability"])
        > float(by_segment["Medium"]["average_churn_probability"])
        > float(by_segment["Low"]["average_churn_probability"])
    )


def test_shap_export_is_sorted_and_matches_champion() -> None:
    shap_rows = read_rows("shap_feature_importance.csv")
    champion = read_rows("model_validation_metrics.csv")[0]["selected_algorithm"]
    importance = [float(row["mean_abs_shap"]) for row in shap_rows]
    assert importance == sorted(importance, reverse=True)
    assert all(value >= 0 for value in importance)
    assert {row["selected_algorithm"] for row in shap_rows} == {champion}
