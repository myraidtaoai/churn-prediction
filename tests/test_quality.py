"""Contract tests for the data quality framework (Phase 1.4).

These verify the rule definitions and code structure without Spark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "churn_pipeline"
TRANSFORMATION_DIR = SRC / "transformation"


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_needs_pyspark = pytest.mark.skipif(
    not _try_import("pyspark"), reason="pyspark not installed"
)
QUALITY_SRC = TRANSFORMATION_DIR / "quality.py"
TRANSFORM_SRC = TRANSFORMATION_DIR / "transform.py"


# ── Rule definitions ──────────────────────────────────────────────────


def test_quality_module_defines_three_severity_tiers():
    src = QUALITY_SRC.read_text()
    assert "FAIL" in src
    assert "QUARANTINE" in src
    assert "WARN" in src


def test_event_rules_cover_required_fields():
    src = QUALITY_SRC.read_text()
    assert "EVENT_RULES" in src
    # Must check the critical fields
    for field in ("event_id", "customer_id", "event_timestamp", "event_type"):
        assert field in src, f"EVENT_RULES should check {field}"


def test_silver_rules_cover_required_fields():
    src = QUALITY_SRC.read_text()
    assert "SILVER_RULES" in src
    for field in ("customer_id", "churn_label", "monthly_charges", "tenure"):
        assert field in src, f"SILVER_RULES should check {field}"


def test_quarantine_preserves_original_payload():
    """Quarantined rows must keep the original data for diagnosis."""
    src = QUALITY_SRC.read_text()
    assert "original_payload" in src
    assert "to_json" in src


def test_quarantine_records_violated_rule():
    src = QUALITY_SRC.read_text()
    assert "violated_rule" in src
    assert "violation_description" in src


def test_quarantine_records_run_id():
    src = QUALITY_SRC.read_text()
    assert "ingestion_run_id" in src


def test_quality_metrics_table_is_append_only():
    """Metrics must append (not overwrite) so quality is trendable."""
    src = QUALITY_SRC.read_text()
    assert "data_quality_metrics" in src
    assert '"append"' in src


def test_metrics_include_per_rule_counts():
    src = QUALITY_SRC.read_text()
    for col in ("rule_name", "total_rows", "passing_rows", "violating_rows"):
        assert col in src, f"Metrics must include {col}"


# ── Integration with transform ────────────────────────────────────────


def test_transform_uses_quality_framework():
    """transform.py must use apply_quality_rules, not raw ValueError."""
    src = TRANSFORM_SRC.read_text()
    assert "apply_quality_rules" in src
    assert "SILVER_RULES" in src


def test_transform_reports_quarantined_count():
    src = TRANSFORM_SRC.read_text()
    assert "quarantined_rows" in src


def test_transform_still_fails_on_all_rows_bad():
    """If quality leaves zero rows, the run must abort."""
    src = TRANSFORM_SRC.read_text()
    assert "All rows were quarantined" in src


# ── Rule importability (requires pyspark) ─────────────────────────────


@_needs_pyspark
def test_quality_rules_are_importable():
    """Rules should be importable without a running SparkSession."""
    import sys

    paths = [str(SRC), str(TRANSFORMATION_DIR)]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from quality import EVENT_RULES, SILVER_RULES, Severity

        assert len(EVENT_RULES) >= 5
        assert len(SILVER_RULES) >= 4
        assert Severity.FAIL.value == "fail"
        assert Severity.QUARANTINE.value == "quarantine"
        assert Severity.WARN.value == "warn"
        # Verify every rule has name, description, severity, predicate_factory
        for rule in list(EVENT_RULES) + list(SILVER_RULES):
            assert rule.name
            assert rule.description
            assert isinstance(rule.severity, Severity)
            assert callable(rule.predicate_factory)
    finally:
        for p in paths:
            if p in sys.path:
                sys.path.remove(p)


@_needs_pyspark
def test_event_rules_have_correct_severities():
    """Critical fields should quarantine, not just warn."""
    import sys

    paths = [str(SRC), str(TRANSFORMATION_DIR)]
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from quality import EVENT_RULES, Severity

        rules_by_name = {r.name: r for r in EVENT_RULES}
        # event_id and customer_id must quarantine, not warn
        assert rules_by_name["event_id_not_null"].severity == Severity.QUARANTINE
        assert rules_by_name["customer_id_not_null"].severity == Severity.QUARANTINE
        assert rules_by_name["event_type_in_contract"].severity == Severity.QUARANTINE
        # amount is a warn, not quarantine
        assert rules_by_name["amount_non_negative"].severity == Severity.WARN
    finally:
        for p in paths:
            if p in sys.path:
                sys.path.remove(p)
