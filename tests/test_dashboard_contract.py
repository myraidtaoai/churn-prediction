import json
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


def test_high_risk_table_has_all_query_fields_selected():
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    widgets = [
        item["widget"]
        for page in dashboard["pages"]
        for item in page["layout"]
        if item.get("widget", {}).get("name") == "high_risk_table_print"
    ]

    assert len(widgets) == 1
    widget = widgets[0]
    query_fields = {
        field["name"]
        for field in widget["queries"][0]["query"]["fields"]
    }
    selected_columns = {
        column["fieldName"]
        for column in widget["spec"]["encodings"]["columns"]
        if column.get("visible", True)
    }

    assert selected_columns == query_fields
