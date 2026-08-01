from promotion_policy import ModelMetrics, evaluate_promotion


def test_first_model_is_approved_when_quality_is_sufficient():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.70, recall=0.75),
        champion=None,
    )

    assert decision.approved is True


def test_first_model_is_rejected_when_quality_is_insufficient():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.55, recall=0.75),
        champion=None,
    )

    assert decision.approved is False


def test_better_candidate_is_approved():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.73, recall=0.78),
        champion=ModelMetrics(pr_auc=0.70, recall=0.79),
    )

    assert decision.approved is True


def test_candidate_without_required_improvement_is_rejected():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.705, recall=0.80),
        champion=ModelMetrics(pr_auc=0.70, recall=0.79),
    )

    assert decision.approved is False


def test_candidate_with_excessive_recall_degradation_is_rejected():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.75, recall=0.70),
        champion=ModelMetrics(pr_auc=0.70, recall=0.80),
    )

    assert decision.approved is False


def test_equivalent_candidate_is_approved_for_contract_upgrade():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.70, recall=0.79),
        champion=ModelMetrics(pr_auc=0.70, recall=0.79),
        allow_equivalent_contract_upgrade=True,
    )

    assert decision.approved is True
    assert "upgrades the inference contract" in decision.reason


def test_worse_candidate_is_rejected_during_contract_upgrade():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.69, recall=0.80),
        champion=ModelMetrics(pr_auc=0.70, recall=0.79),
        allow_equivalent_contract_upgrade=True,
    )

    assert decision.approved is False


def test_recall_guardrail_applies_during_contract_upgrade():
    decision = evaluate_promotion(
        candidate=ModelMetrics(pr_auc=0.70, recall=0.76),
        champion=ModelMetrics(pr_auc=0.70, recall=0.79),
        allow_equivalent_contract_upgrade=True,
    )

    assert decision.approved is False
