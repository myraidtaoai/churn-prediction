"""Append-only run log for every pipeline task.

Each task calls ``log_run()`` at the end of its work.  The function writes
a single row to ``pipeline_runs`` — a Delta table that records what ran,
when, how long it took, and whether it succeeded.

The table is append-only and never mutated after write, making it safe for
concurrent tasks and auditable over time.

Schema::

    task_name       STRING    — logical task key (e.g. "ingest_events")
    run_id          STRING    — unique identifier for this execution
    status          STRING    — "succeeded" | "failed" | "skipped"
    started_at      TIMESTAMP — UTC wall-clock start
    finished_at     TIMESTAMP — UTC wall-clock end
    duration_seconds DOUBLE   — elapsed seconds
    output_summary  STRING    — JSON blob with task-specific metrics
    error_message   STRING    — populated only on failure
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator
from uuid import uuid4

from common import table as _table


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def log_run(
    spark: Any,  # pyspark.sql.SparkSession — deferred to avoid import at collection
    catalog: str,
    schema: str,
    task_name: str,
    run_id: str,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    output_summary: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    """Write a single run record to ``pipeline_runs``."""
    duration = (finished_at - started_at).total_seconds()
    row = [
        (
            task_name,
            run_id,
            status,
            started_at.isoformat(),
            finished_at.isoformat(),
            float(round(duration, 2)),
            json.dumps(output_summary or {}, sort_keys=True),
            error_message or "",
        )
    ]
    try:
        from pyspark.sql.types import (
            DoubleType,
            StringType,
            StructField,
            StructType,
        )

        spark_schema = StructType(
            [
                StructField("task_name", StringType(), False),
                StructField("run_id", StringType(), False),
                StructField("status", StringType(), False),
                StructField("started_at", StringType(), False),
                StructField("finished_at", StringType(), False),
                StructField("duration_seconds", DoubleType(), False),
                StructField("output_summary", StringType(), True),
                StructField("error_message", StringType(), True),
            ]
        )
    except ImportError:
        # Unit-test path — pyspark not installed locally.
        spark_schema = [
            "task_name",
            "run_id",
            "status",
            "started_at",
            "finished_at",
            "duration_seconds",
            "output_summary",
            "error_message",
        ]
    df = spark.createDataFrame(row, schema=spark_schema)
    target = _table(catalog, schema, "pipeline_runs")
    df.write.format("delta").mode("append").saveAsTable(target)


@contextmanager
def track_run(
    spark: Any,  # pyspark.sql.SparkSession
    catalog: str,
    schema: str,
    task_name: str,
    run_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager that times a task and logs the outcome.

    Usage::

        with track_run(spark, catalog, schema, "transform") as ctx:
            # ... do work ...
            ctx["output"] = {"rows": 42}

    On normal exit, logs status="succeeded".  On exception, logs
    status="failed" with the error message, then re-raises.
    """
    rid = run_id or uuid4().hex[:12]
    ctx: dict[str, Any] = {"run_id": rid, "output": {}}
    started = _now_utc()
    try:
        yield ctx
    except Exception as exc:
        log_run(
            spark=spark,
            catalog=catalog,
            schema=schema,
            task_name=task_name,
            run_id=rid,
            status="failed",
            started_at=started,
            finished_at=_now_utc(),
            error_message=str(exc)[:2000],
        )
        raise
    else:
        log_run(
            spark=spark,
            catalog=catalog,
            schema=schema,
            task_name=task_name,
            run_id=rid,
            status="succeeded",
            started_at=started,
            finished_at=_now_utc(),
            output_summary=ctx.get("output"),
        )
