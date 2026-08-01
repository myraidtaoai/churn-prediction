"""Rules for promoting a candidate model to Champion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetrics:
    pr_auc: float
    recall: float
    positive_rate: float | None = None


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reason: str


def evaluate_promotion(
    candidate: ModelMetrics,
    champion: ModelMetrics | None,
    minimum_first_model_pr_auc_lift: float = 2.0,
    minimum_first_model_recall: float = 0.60,
    required_pr_auc_improvement: float = 0.01,
    maximum_recall_degradation: float = 0.02,
    allow_equivalent_contract_upgrade: bool = False,
) -> PromotionDecision:
    """Determine whether Candidate should replace Champion."""

    if champion is None:
        if candidate.positive_rate is None or candidate.positive_rate <= 0:
            return PromotionDecision(
                False,
                "Missing a valid positive-rate baseline.",
            )

        pr_auc_lift = candidate.pr_auc / candidate.positive_rate
        approved = (
            pr_auc_lift >= minimum_first_model_pr_auc_lift
            and candidate.recall >= minimum_first_model_recall
        )
        reason = (
            "First model exceeds the no-skill PR-AUC baseline and recall requirement."
            if approved
            else "First model does not meet bootstrap quality requirements."
        )
        return PromotionDecision(approved, reason)

    required_improvement = (
        0.0 if allow_equivalent_contract_upgrade else required_pr_auc_improvement
    )
    pr_auc_improvement = candidate.pr_auc - champion.pr_auc
    recall_degradation = champion.recall - candidate.recall

    if pr_auc_improvement < required_improvement - 1e-12:
        return PromotionDecision(
            False,
            "Candidate does not provide the required PR-AUC improvement.",
        )

    if recall_degradation > maximum_recall_degradation:
        return PromotionDecision(
            False,
            "Candidate recall degradation exceeds the allowed threshold.",
        )

    reason = (
        "Candidate preserves model quality and upgrades the inference contract."
        if allow_equivalent_contract_upgrade
        else "Candidate satisfies the promotion requirements."
    )
    return PromotionDecision(True, reason)


def require_deployable_champion(
    decision: PromotionDecision,
    champion_version: str | None,
) -> None:
    """Stop the pipeline only when rejection leaves no model to score with."""
    if not decision.approved and champion_version is None:
        raise RuntimeError(
            "Candidate was rejected and no existing Champion is available. "
            "Batch scoring has been stopped."
        )
