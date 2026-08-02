"""Versioned data contract shared by event producer and consumer.

This module is the single source of truth for the event schema.  Both
the generator (``ingestion/generate_events.py``) and the ingestion
job (``ingestion/ingest_events.py``) import from here so that
producer and consumer cannot drift independently.

Schema evolution policy:
  - Additive (new nullable field): bump ``CONTRACT_VERSION``, add to
    ``EVENT_FIELDS``.  Auto Loader rescue mode handles the transition;
    old files without the field are valid.
  - Breaking (rename, type change, remove required field): bump
    ``CONTRACT_VERSION``, update both producer and consumer in the same
    PR.  The ingestion job rejects events whose ``schema_version`` does
    not match a supported version.
  - The contract document (``docs/data-contract.md``) must be updated
    in every PR that touches this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Sequence, Tuple

# ── Contract version ──────────────────────────────────────────────────
# Increment on any schema change.  Old versions remain in
# SUPPORTED_VERSIONS until all historical data has been reprocessed.
CONTRACT_VERSION: int = 1
SUPPORTED_VERSIONS: FrozenSet[int] = frozenset({1})

# ── Event types ───────────────────────────────────────────────────────
EVENT_TYPES: FrozenSet[str] = frozenset(
    {
        "billing",
        "cancellation",
        "complaint",
        "payment",
        "plan_change",
        "support_call",
        "usage",
    }
)

# ── Field definitions ─────────────────────────────────────────────────


@dataclass(frozen=True)
class FieldDef:
    """One field in the event contract."""

    name: str
    spark_type: str  # PySpark type name: "string", "integer", "double"
    nullable: bool
    description: str


EVENT_FIELDS: Sequence[FieldDef] = (
    FieldDef("schema_version", "integer", False, "Contract version; currently 1"),
    FieldDef("event_id", "string", False, "Deterministic SHA-256 event identifier"),
    FieldDef("generation_id", "string", True, "SHA-256 batch configuration identifier"),
    FieldDef("event_type", "string", False, "One of the EVENT_TYPES enum values"),
    FieldDef("event_timestamp", "string", False, "UTC ISO-8601 event time"),
    FieldDef("event_date", "string", False, "UTC business date (YYYY-MM-DD)"),
    FieldDef("customer_id", "string", False, "Source customer identifier"),
    FieldDef("amount", "double", True, "Billing or payment amount"),
    FieldDef(
        "usage_gb", "double", True, "Internet usage; 0 for non-internet customers"
    ),
    FieldDef("payment_status", "string", True, "'succeeded' or 'failed'"),
    FieldDef("support_topic", "string", True, "Support-call category"),
    FieldDef("plan_from", "string", True, "Contract before a plan change"),
    FieldDef("plan_to", "string", True, "Contract after a plan change"),
    FieldDef("complaint_severity", "string", True, "'medium' or 'high'"),
    FieldDef("cancellation_reason", "string", True, "Cancellation category"),
)

# Convenience sets for validation.
REQUIRED_FIELDS: FrozenSet[str] = frozenset(
    f.name for f in EVENT_FIELDS if not f.nullable
)
ALL_FIELD_NAMES: FrozenSet[str] = frozenset(f.name for f in EVENT_FIELDS)
FIELD_NAME_TUPLE: Tuple[str, ...] = tuple(f.name for f in EVENT_FIELDS)

# ── Payment status enum ──────────────────────────────────────────────
PAYMENT_STATUSES: FrozenSet[str] = frozenset({"succeeded", "failed"})

# ── Complaint severities ──────────────────────────────────────────────
COMPLAINT_SEVERITIES: FrozenSet[str] = frozenset({"medium", "high"})

# ── Support topics ────────────────────────────────────────────────────
SUPPORT_TOPICS: FrozenSet[str] = frozenset(
    {"billing", "connectivity", "service_quality", "technical"}
)


def build_spark_schema():
    """Build a PySpark StructType from the contract field definitions.

    Deferred import so this module can be imported without pyspark
    (e.g. by the generator running in a lightweight environment or by
    unit tests).
    """
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    type_map = {
        "string": StringType(),
        "integer": IntegerType(),
        "double": DoubleType(),
    }
    return StructType(
        [StructField(f.name, type_map[f.spark_type], f.nullable) for f in EVENT_FIELDS]
    )


def validate_event(event: dict) -> list[str]:
    """Return a list of contract violations for a single event dict.

    Designed for use in the generator's self-check, not in the Spark
    pipeline (which uses the quality framework instead).
    """
    errors: list[str] = []

    # Required fields.
    for field_name in REQUIRED_FIELDS:
        if field_name not in event or event[field_name] is None:
            errors.append(f"missing required field: {field_name}")

    # Unknown fields.
    for key in event:
        if key not in ALL_FIELD_NAMES:
            errors.append(f"unknown field: {key}")

    # Enum checks.
    if event.get("event_type") and event["event_type"] not in EVENT_TYPES:
        errors.append(
            f"invalid event_type: {event['event_type']}; allowed: {sorted(EVENT_TYPES)}"
        )
    if event.get("payment_status") and event["payment_status"] not in PAYMENT_STATUSES:
        errors.append(f"invalid payment_status: {event['payment_status']}")
    if (
        event.get("complaint_severity")
        and event["complaint_severity"] not in COMPLAINT_SEVERITIES
    ):
        errors.append(f"invalid complaint_severity: {event['complaint_severity']}")
    if event.get("support_topic") and event["support_topic"] not in SUPPORT_TOPICS:
        errors.append(f"invalid support_topic: {event['support_topic']}")

    # Schema version.
    sv = event.get("schema_version")
    if sv is not None and sv not in SUPPORTED_VERSIONS:
        errors.append(
            f"unsupported schema_version: {sv}; supported: {sorted(SUPPORTED_VERSIONS)}"
        )

    return errors
