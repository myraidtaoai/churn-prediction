"""Rules for promoting a candidate model to Champion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetrics:
    pr_auc: float
    recall: float


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reason: str


def evaluate_promotion(
    candidate: ModelMetrics,
    champion: ModelMetrics | None,
    minimum_first_model_pr_auc: float = 0.60,
    required_pr_auc_improvement: float = 0.01,
    maximum_recall_degradation: float = 0.02,
) -> PromotionDecision:
    """Determine whether Candidate should replace Champion."""

    if champion is None:
        approved = candidate.pr_auc >= minimum_first_model_pr_auc
        reason = (
            "First model meets the minimum PR-AUC requirement."
            if approved
            else "First model does not meet the minimum PR-AUC requirement."
        )
        return PromotionDecision(approved, reason)

    pr_auc_improvement = candidate.pr_auc - champion.pr_auc
    recall_degradation = champion.recall - candidate.recall

    if pr_auc_improvement < required_pr_auc_improvement:
        return PromotionDecision(
            False,
            "Candidate does not provide the required PR-AUC improvement.",
        )

    if recall_degradation > maximum_recall_degradation:
        return PromotionDecision(
            False,
            "Candidate recall degradation exceeds the allowed threshold.",
        )

    return PromotionDecision(
        True,
        "Candidate satisfies the promotion requirements.",
    )