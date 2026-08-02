"""Tests for the versioned data contract module."""

from __future__ import annotations

import pytest

# ``pythonpath = ["src/churn_pipeline"]`` in pyproject.toml puts the
# pipeline modules on the path, so this is a plain top-level import.
from contracts import (
    ALL_FIELD_NAMES,
    COMPLAINT_SEVERITIES,
    CONTRACT_VERSION,
    EVENT_FIELDS,
    EVENT_TYPES,
    FIELD_NAME_TUPLE,
    PAYMENT_STATUSES,
    REQUIRED_FIELDS,
    SUPPORT_TOPICS,
    SUPPORTED_VERSIONS,
    FieldDef,
    validate_event,
)

# ── Structural invariants ────────────────────────────────────────────


def test_contract_version_in_supported():
    """Current version must be in the supported set."""
    assert CONTRACT_VERSION in SUPPORTED_VERSIONS


def test_field_count_matches_tuple():
    """EVENT_FIELDS and FIELD_NAME_TUPLE must stay in sync."""
    assert len(EVENT_FIELDS) == len(FIELD_NAME_TUPLE)
    assert tuple(f.name for f in EVENT_FIELDS) == FIELD_NAME_TUPLE


def test_no_duplicate_field_names():
    assert len(ALL_FIELD_NAMES) == len(EVENT_FIELDS)


def test_required_fields_subset():
    """Required fields must be a subset of all fields."""
    assert REQUIRED_FIELDS <= ALL_FIELD_NAMES


def test_event_types_nonempty():
    assert len(EVENT_TYPES) > 0


def test_all_field_defs_are_frozen():
    """FieldDefs must be immutable dataclasses."""
    for f in EVENT_FIELDS:
        assert isinstance(f, FieldDef)
        with pytest.raises(AttributeError):
            f.name = "hacked"


def test_spark_type_values_valid():
    """Every spark_type must be one we know how to map."""
    valid = {"string", "integer", "double"}
    for f in EVENT_FIELDS:
        assert f.spark_type in valid, f"{f.name} has unknown spark_type: {f.spark_type}"


# ── validate_event ───────────────────────────────────────────────────


def _minimal_valid_event() -> dict:
    return {
        "schema_version": CONTRACT_VERSION,
        "event_id": "abc123",
        "generation_id": None,
        "event_type": "usage",
        "event_timestamp": "2026-01-01T00:00:00Z",
        "event_date": "2026-01-01",
        "customer_id": "C001",
        "amount": None,
        "usage_gb": 5.0,
        "payment_status": None,
        "support_topic": None,
        "plan_from": None,
        "plan_to": None,
        "complaint_severity": None,
        "cancellation_reason": None,
    }


def test_valid_event_no_errors():
    assert validate_event(_minimal_valid_event()) == []


def test_missing_required_field():
    event = _minimal_valid_event()
    del event["event_id"]
    errors = validate_event(event)
    assert any("event_id" in e for e in errors)


def test_unknown_field_flagged():
    event = _minimal_valid_event()
    event["rogue_field"] = 42
    errors = validate_event(event)
    assert any("rogue_field" in e for e in errors)


def test_invalid_event_type():
    event = _minimal_valid_event()
    event["event_type"] = "teleportation"
    errors = validate_event(event)
    assert any("event_type" in e for e in errors)


def test_invalid_payment_status():
    event = _minimal_valid_event()
    event["payment_status"] = "maybe"
    errors = validate_event(event)
    assert any("payment_status" in e for e in errors)


def test_unsupported_schema_version():
    event = _minimal_valid_event()
    event["schema_version"] = 999
    errors = validate_event(event)
    assert any("schema_version" in e for e in errors)


# ── Enum consistency ─────────────────────────────────────────────────


def test_enum_sets_are_frozen():
    """All exported sets must be frozensets (immutable)."""
    for s in (EVENT_TYPES, PAYMENT_STATUSES, COMPLAINT_SEVERITIES, SUPPORT_TOPICS):
        assert isinstance(s, frozenset)


# ── PySpark schema builder ───────────────────────────────────────────


def _try_pyspark_available() -> bool:
    try:
        import pyspark  # noqa: F401

        return True
    except ImportError:
        return False


_needs_pyspark = pytest.mark.skipif(
    not _try_pyspark_available(), reason="pyspark not installed"
)


@_needs_pyspark
def test_build_spark_schema_field_count():
    from contracts import build_spark_schema

    schema = build_spark_schema()
    assert len(schema.fields) == len(EVENT_FIELDS)


@_needs_pyspark
def test_build_spark_schema_field_names_match():
    from contracts import build_spark_schema

    schema = build_spark_schema()
    schema_names = {f.name for f in schema.fields}
    assert schema_names == ALL_FIELD_NAMES
