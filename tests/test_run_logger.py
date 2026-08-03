"""Unit tests for the pipeline_runs run logger."""

# run_logger lives under src/churn_pipeline/ops/ and uses churn_pipeline-level
# imports.  We put it on the path so the test runs locally without a cluster.
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "churn_pipeline"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ops.run_logger import log_run, track_run  # noqa: E402


@pytest.fixture()
def mock_spark():
    spark = MagicMock()
    spark.createDataFrame.return_value.write.format.return_value.mode.return_value.saveAsTable = MagicMock()
    return spark


def test_log_run_writes_one_row(mock_spark):
    started = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=42)
    log_run(
        spark=mock_spark,
        catalog="main",
        schema="churn_dev",
        task_name="transform",
        run_id="test-run-001",
        status="succeeded",
        started_at=started,
        finished_at=finished,
        output_summary={"rows": 100},
    )
    mock_spark.createDataFrame.assert_called_once()
    rows = mock_spark.createDataFrame.call_args[0][0]
    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "transform"
    assert row[1] == "test-run-001"
    assert row[2] == "succeeded"
    assert row[5] == 42.0  # duration_seconds


def test_track_run_success(mock_spark):
    with track_run(mock_spark, "main", "dev", "ingest") as ctx:
        ctx["output"] = {"ingested": 500}

    mock_spark.createDataFrame.assert_called_once()
    row = mock_spark.createDataFrame.call_args[0][0][0]
    assert row[0] == "ingest"
    assert row[2] == "succeeded"
    assert '"ingested": 500' in row[6]  # output_summary JSON


def test_track_run_failure(mock_spark):
    with pytest.raises(ValueError, match="boom"):
        with track_run(mock_spark, "main", "dev", "transform") as _ctx:
            raise ValueError("boom")

    row = mock_spark.createDataFrame.call_args[0][0][0]
    assert row[2] == "failed"
    assert row[7] == "boom"  # error_message
