"""Declarative data quality framework with three severity tiers.

Expectations are plain PySpark predicates — no external dependency.  Each
rule has a severity:

- **fail**: abort the run (e.g. zero rows ingested).
- **quarantine**: route the row to ``telco_quarantine`` with the violated
  rule name, continue with the surviving rows.
- **warn**: log to ``data_quality_metrics``, continue with all rows.

Usage::

    from quality import apply_quality_rules, EVENT_RULES, SILVER_RULES

    clean_df, metrics = apply_quality_rules(
        df=raw_df,
        rules=EVENT_RULES,
        spark=spark,
        catalog=catalog,
        schema=schema,
        run_id=run_id,
        stage="bronze_events",
    )
"""

from __future__ import annotations

import _path_helper  # noqa: F401 — adds churn_pipeline/ to sys.path

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Sequence

from contracts import EVENT_TYPES as _CONTRACT_EVENT_TYPES
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ── Rule definition ───────────────────────────────────────────────────


class Severity(Enum):
    FAIL = "fail"
    QUARANTINE = "quarantine"
    WARN = "warn"


@dataclass(frozen=True)
class QualityRule:
    """One expectation applied to a DataFrame."""

    name: str
    description: str
    severity: Severity
    # A PySpark Column expression that returns True for **valid** rows.
    # Rows where the predicate is False (or NULL) are violations.
    predicate_factory: object  # Callable[[], Column]

    def predicate(self):
        """Build the Column predicate (deferred so rules can be defined at
        import time before a SparkSession exists)."""
        return self.predicate_factory()


# ── Event Bronze rules ────────────────────────────────────────────────

EVENT_RULES: Sequence[QualityRule] = [
    QualityRule(
        name="event_id_not_null",
        description="event_id is required for dedup and lineage",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: (
            F.col("event_id").isNotNull() & (F.length(F.col("event_id")) > 0)
        ),
    ),
    QualityRule(
        name="customer_id_not_null",
        description="customer_id is required for attribution",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: (
            F.col("customer_id").isNotNull() & (F.length(F.col("customer_id")) > 0)
        ),
    ),
    QualityRule(
        name="event_timestamp_plausible",
        description="event_timestamp must parse and be within 2020-2030",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: (
            F.col("event_timestamp").isNotNull()
            & (
                F.to_timestamp(F.col("event_timestamp"), "yyyy-MM-dd'T'HH:mm:ss'Z'")
                >= F.lit("2020-01-01").cast("timestamp")
            )
            & (
                F.to_timestamp(F.col("event_timestamp"), "yyyy-MM-dd'T'HH:mm:ss'Z'")
                < F.lit("2030-01-01").cast("timestamp")
            )
        ),
    ),
    QualityRule(
        name="event_type_in_contract",
        description="event_type must be one of the contract-defined values",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: F.col("event_type").isin(
            *sorted(_CONTRACT_EVENT_TYPES)
        ),
    ),
    QualityRule(
        name="amount_non_negative",
        description="amount must be >= 0 when present",
        severity=Severity.WARN,
        predicate_factory=lambda: F.col("amount").isNull() | (F.col("amount") >= 0),
    ),
    QualityRule(
        name="usage_gb_non_negative",
        description="usage_gb must be >= 0 when present",
        severity=Severity.WARN,
        predicate_factory=lambda: F.col("usage_gb").isNull() | (F.col("usage_gb") >= 0),
    ),
    QualityRule(
        name="schema_version_present",
        description="schema_version is required for contract evolution",
        severity=Severity.WARN,
        predicate_factory=lambda: F.col("schema_version").isNotNull(),
    ),
]


# ── Silver rules ──────────────────────────────────────────────────────

SILVER_RULES: Sequence[QualityRule] = [
    QualityRule(
        name="customer_id_not_null",
        description="customer_id is the Silver primary key",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: (
            F.col("customer_id").isNotNull() & (F.length(F.col("customer_id")) > 0)
        ),
    ),
    QualityRule(
        name="churn_label_valid",
        description="churn_label must be 0 or 1",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: F.col("churn_label").isin(0, 1),
    ),
    QualityRule(
        name="senior_citizen_not_null",
        description="senior_citizen must be numeric",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: F.col("senior_citizen").isNotNull(),
    ),
    QualityRule(
        name="monthly_charges_not_null",
        description="monthly_charges must be numeric",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: F.col("monthly_charges").isNotNull(),
    ),
    QualityRule(
        name="tenure_valid",
        description="tenure must be non-negative",
        severity=Severity.QUARANTINE,
        predicate_factory=lambda: F.col("tenure").isNotNull() & (F.col("tenure") >= 0),
    ),
    QualityRule(
        name="total_charges_consistent",
        description="total_charges must be present for nonzero tenure",
        severity=Severity.WARN,
        predicate_factory=lambda: (
            (F.col("tenure") == 0) | F.col("total_charges").isNotNull()
        ),
    ),
]


# ── Engine ────────────────────────────────────────────────────────────


@dataclass
class QualityMetrics:
    """Per-rule pass/quarantine/fail counts for one run."""

    rule_name: str
    severity: str
    total_rows: int
    passing_rows: int
    violating_rows: int
    stage: str
    run_id: str
    evaluated_at: str


def apply_quality_rules(
    df: DataFrame,
    rules: Sequence[QualityRule],
    spark: SparkSession,
    catalog: str,
    schema: str,
    run_id: str,
    stage: str,
) -> tuple[DataFrame, list[QualityMetrics]]:
    """Apply rules, route violations, and return (clean_df, metrics).

    - **fail** rules: if any row violates, raise immediately.
    - **quarantine** rules: violating rows go to ``telco_quarantine``;
      clean rows continue.
    - **warn** rules: violations are counted and logged but all rows continue.

    Returns the DataFrame with quarantined rows removed, plus a list of
    per-rule metrics.
    """
    from common import table

    now = datetime.now(timezone.utc).isoformat()
    total_rows = df.count()
    metrics: list[QualityMetrics] = []
    quarantine_frames: list[DataFrame] = []

    # ── Evaluate each rule ────────────────────────────────────────────
    for rule in rules:
        predicate = rule.predicate()
        violating = df.filter(~predicate | predicate.isNull())
        violating_count = violating.count()
        passing_count = total_rows - violating_count

        metrics.append(
            QualityMetrics(
                rule_name=rule.name,
                severity=rule.severity.value,
                total_rows=total_rows,
                passing_rows=passing_count,
                violating_rows=violating_count,
                stage=stage,
                run_id=run_id,
                evaluated_at=now,
            )
        )

        if violating_count == 0:
            continue

        if rule.severity == Severity.FAIL:
            raise ValueError(
                f"FAIL rule '{rule.name}' violated by {violating_count} rows: "
                f"{rule.description}"
            )

        if rule.severity == Severity.QUARANTINE:
            # Tag quarantined rows with the violated rule.
            quarantine_batch = violating.select(
                F.to_json(F.struct(*[F.col(c) for c in df.columns])).alias(
                    "original_payload"
                ),
                F.lit(rule.name).alias("violated_rule"),
                F.lit(rule.description).alias("violation_description"),
                F.lit(run_id).alias("ingestion_run_id"),
                F.lit(now).cast("timestamp").alias("quarantined_at"),
            )
            quarantine_frames.append(quarantine_batch)

            # Remove violating rows from the working DataFrame.
            df = df.filter(predicate)
            total_rows = df.count()

        # Severity.WARN: just logged in metrics, nothing removed.

    # ── Write quarantined rows ────────────────────────────────────────
    if quarantine_frames:
        from functools import reduce

        quarantine_df = reduce(DataFrame.unionByName, quarantine_frames)
        quarantine_table = table(catalog, schema, "telco_quarantine")
        quarantine_df.write.format("delta").mode("append").saveAsTable(quarantine_table)

    # ── Write quality metrics ─────────────────────────────────────────
    metrics_rows = [
        (
            m.rule_name,
            m.severity,
            m.total_rows,
            m.passing_rows,
            m.violating_rows,
            m.stage,
            m.run_id,
            m.evaluated_at,
        )
        for m in metrics
    ]
    metrics_schema = [
        "rule_name",
        "severity",
        "total_rows",
        "passing_rows",
        "violating_rows",
        "stage",
        "run_id",
        "evaluated_at",
    ]
    metrics_df = spark.createDataFrame(metrics_rows, schema=metrics_schema)
    metrics_table = table(catalog, schema, "data_quality_metrics")
    metrics_df.write.format("delta").mode("append").saveAsTable(metrics_table)

    return df, metrics
