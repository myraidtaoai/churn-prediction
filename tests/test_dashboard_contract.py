import re
from pathlib import Path


DASHBOARD_PATH = (
    Path(__file__).parents[1]
    / "dashboards"
    / "churn_exploration_and_model_results.lvdash.json"
)


def test_dashboard_uses_current_prediction_view():
    dashboard = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "current_customer_churn_scores" in dashboard
    assert re.search(
        r"(?<!current_)customer_churn_scores\b",
        dashboard,
    ) is None