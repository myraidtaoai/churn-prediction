"""Contract tests for the Pipeline Observability dashboard."""

import json
from pathlib import Path

DASHBOARD_PATH = (
    Path(__file__).parents[1] / "dashboards" / "pipeline_observability.lvdash.json"
)


def test_dashboard_json_is_valid():
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    assert "datasets" in dashboard
    assert "pages" in dashboard
    assert len(dashboard["datasets"]) > 0
    assert len(dashboard["pages"]) > 0


def test_dashboard_queries_pipeline_runs():
    text = DASHBOARD_PATH.read_text(encoding="utf-8")
    assert "pipeline_runs" in text


def test_dashboard_queries_data_quality_metrics():
    text = DASHBOARD_PATH.read_text(encoding="utf-8")
    assert "data_quality_metrics" in text


def test_dashboard_queries_promotion_history():
    text = DASHBOARD_PATH.read_text(encoding="utf-8")
    assert "model_promotion_history" in text


def test_all_datasets_referenced_by_widgets():
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    dataset_names = {ds["name"] for ds in dashboard["datasets"]}
    referenced = set()
    for page in dashboard["pages"]:
        for item in page["layout"]:
            widget = item.get("widget", {})
            for query in widget.get("queries", []):
                ds_name = query.get("query", {}).get("datasetName")
                if ds_name:
                    referenced.add(ds_name)
    unreferenced = dataset_names - referenced
    assert not unreferenced, f"Datasets defined but never used: {unreferenced}"
