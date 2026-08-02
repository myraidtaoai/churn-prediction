"""Contract tests for the Auto Loader event ingestion (Phase 1.2).

These tests verify the code and bundle structure without requiring Spark
or a Databricks workspace.
"""

from __future__ import annotations

from pathlib import Path

import yaml

SRC = (
    Path(__file__).parents[1]
    / "src"
    / "churn_pipeline"
    / "ingestion"
    / "ingest_events.py"
)
WORKFLOW_PATH = Path(__file__).parents[1] / "resources" / "churn_workflow.yml"


def _load_jobs():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return workflow["resources"]["jobs"]


# ── Code contract ─────────────────────────────────────────────────────


def test_ingest_events_uses_auto_loader():
    src = SRC.read_text()
    assert "cloudFiles" in src
    assert "readStream" in src


def test_ingest_events_uses_available_now_trigger():
    """availableNow gives batch semantics with streaming guarantees."""
    src = SRC.read_text()
    assert "availableNow" in src


def test_ingest_events_has_checkpoint():
    """Checkpoint is the source of truth for consumed files."""
    src = SRC.read_text()
    assert "--checkpoint-path" in src
    assert "checkpointLocation" in src


def test_ingest_events_uses_rescue_mode():
    """Unexpected fields go to _rescued_data, not a crash."""
    src = SRC.read_text()
    assert "rescue" in src.lower()
    assert "_rescued_data" in src


def test_ingest_events_deduplicates_on_event_id():
    src = SRC.read_text()
    assert "dropDuplicates" in src
    assert "event_id" in src


def test_ingest_events_separates_event_and_ingestion_timestamps():
    """Bronze must carry both event_timestamp and ingestion_timestamp."""
    src = SRC.read_text()
    assert "event_ts" in src or "event_timestamp" in src
    assert "ingestion_timestamp" in src


def test_ingest_events_writes_to_events_bronze():
    """Events go to telco_events_bronze, not the CSV seed telco_bronze."""
    src = SRC.read_text()
    assert "telco_events_bronze" in src


# ── Bundle contract ───────────────────────────────────────────────────


def test_standalone_seed_bronze_job_exists():
    jobs = _load_jobs()
    assert "churn_seed_bronze" in jobs
    job = jobs["churn_seed_bronze"]
    assert job["max_concurrent_runs"] == 1
    assert "schedule" not in job  # manual, not scheduled
    task = job["tasks"][0]
    assert task["task_key"] == "ingest_bronze"
    assert "ingest.py" in task["spark_python_task"]["python_file"]


def test_data_pipeline_starts_with_event_ingest():
    jobs = _load_jobs()
    data_tasks = jobs["churn_data_pipeline"]["tasks"]
    task_keys = [t["task_key"] for t in data_tasks]

    # ingest_events is the first task with no dependency
    assert task_keys[0] == "ingest_events"
    assert "depends_on" not in data_tasks[0]

    # transform depends on ingest_events
    transform_idx = task_keys.index("transform_silver_and_gold")
    assert data_tasks[transform_idx]["depends_on"] == [{"task_key": "ingest_events"}]


def test_event_ingest_checkpoint_is_in_volume():
    """Checkpoint must be in the managed Volume, not DBFS or workspace."""
    jobs = _load_jobs()
    for job_key in ("churn_data_pipeline",):
        job = jobs[job_key]
        for task in job["tasks"]:
            if task["task_key"] != "ingest_events":
                continue
            params = task["spark_python_task"]["parameters"]
            # Find the --checkpoint-path value
            cp_idx = params.index("--checkpoint-path")
            cp_value = params[cp_idx + 1]
            assert "/Volumes/" in cp_value or "${resources.volumes" in cp_value


def test_event_schema_matches_generator():
    """The Bronze schema must declare all fields from the event contract.

    Now that ingest_events.py imports build_spark_schema() from contracts.py,
    the field names are no longer inline strings.  We verify the contract
    module itself contains every expected field instead.
    """
    import sys as _sys

    _contracts_dir = str(SRC.parent.parent)  # churn_pipeline/ root
    if _contracts_dir not in _sys.path:
        _sys.path.insert(0, _contracts_dir)
    from contracts import ALL_FIELD_NAMES

    expected_fields = {
        "schema_version",
        "event_id",
        "generation_id",
        "event_type",
        "event_timestamp",
        "event_date",
        "customer_id",
        "amount",
        "usage_gb",
        "payment_status",
        "support_topic",
        "plan_from",
        "plan_to",
        "complaint_severity",
        "cancellation_reason",
    }
    assert expected_fields == ALL_FIELD_NAMES, (
        f"Contract field mismatch.\n"
        f"  Missing from contract: {expected_fields - ALL_FIELD_NAMES}\n"
        f"  Extra in contract: {ALL_FIELD_NAMES - expected_fields}"
    )
