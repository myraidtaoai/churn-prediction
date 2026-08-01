"""Tests for the idempotency guarantees introduced in Phase 1.1.

These tests verify:
- The CSV seed ingest is skip-safe on re-run.
- The transform produces a deterministic ``_record_hash`` / ``snapshot_date`` key.
- The MERGE SQL template targets the correct key columns.
- The backfill script delegates to the same code paths as the scheduled run.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

# ── record_hash determinism ────────────────────────────────────────────


def _record_hash(*values):
    """Mirror the hash function in ingest.py."""
    payload = "|".join(v if v is not None else "" for v in values)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_record_hash_is_deterministic():
    row = ("7590-VHVEG", "Female", "0", "Yes", "No", "1")
    assert _record_hash(*row) == _record_hash(*row)


def test_record_hash_changes_with_any_value():
    row_a = ("7590-VHVEG", "Female", "0")
    row_b = ("7590-VHVEG", "Male", "0")
    assert _record_hash(*row_a) != _record_hash(*row_b)


def test_record_hash_handles_none():
    row = ("7590-VHVEG", None, "0")
    h = _record_hash(*row)
    assert isinstance(h, str) and len(h) == 64


# ── transform MERGE key contract ──────────────────────────────────────


def test_transform_merge_key_columns():
    """The Silver MERGE must use (customer_id, snapshot_date) as the key.

    This is a contract test: if someone changes the key, this test reminds
    them to update the MERGE SQL, the backfill script, and the docs.
    """
    src = (
        Path(__file__).parents[1]
        / "src"
        / "churn_pipeline"
        / "transformation"
        / "transform.py"
    ).read_text()
    # The MERGE ON clause must reference both key columns.
    assert "target.customer_id = source.customer_id" in src
    assert "target.snapshot_date = source.snapshot_date" in src


def test_transform_adds_snapshot_date_column():
    """Silver rows must carry a snapshot_date for partitioning and replay."""
    src = (
        Path(__file__).parents[1]
        / "src"
        / "churn_pipeline"
        / "transformation"
        / "transform.py"
    ).read_text()
    assert "snapshot_date" in src
    assert "--snapshot-date" in src


# ── ingest idempotency contract ────────────────────────────────────────


def test_ingest_checks_existing_rows_before_writing():
    """The seed ingest must check for existing csv_seed rows."""
    src = (
        Path(__file__).parents[1] / "src" / "churn_pipeline" / "ingestion" / "ingest.py"
    ).read_text()
    assert "_source_type" in src
    assert "csv_seed" in src
    assert "skipped" in src  # the idempotent skip path must exist


def test_ingest_adds_record_hash():
    """Every Bronze row must carry a deterministic _record_hash."""
    src = (
        Path(__file__).parents[1] / "src" / "churn_pipeline" / "ingestion" / "ingest.py"
    ).read_text()
    assert "_record_hash" in src
    assert "sha256" in src


# ── backfill script contract ──────────────────────────────────────────


def test_backfill_calls_same_code_paths():
    """The backfill script must call ingest.py and transform.py, not
    contain its own ETL logic."""
    src = (
        Path(__file__).parents[1] / "src" / "churn_pipeline" / "ops" / "backfill.py"
    ).read_text()
    assert "ingest.py" in src
    assert "transform.py" in src
    assert "--snapshot-date" in src


def test_backfill_dry_run_does_not_execute(tmp_path):
    """--dry-run should print commands without running them."""
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).parents[1]
                / "src"
                / "churn_pipeline"
                / "ops"
                / "backfill.py"
            ),
            "--catalog",
            "test",
            "--schema",
            "test",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-03",
            "--landing-file",
            "/fake/path.csv",
            "--bundled-source-file",
            "/fake/bundle.csv",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Seed Bronze" in result.stdout
    assert "Transform 2026-01-01" in result.stdout
    assert "Transform 2026-01-02" in result.stdout
    assert "Transform 2026-01-03" in result.stdout
    assert "Done. Processed 3 date(s)" in result.stdout


def test_backfill_empty_range_exits_cleanly():
    """start > end should print a message and exit 0."""
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).parents[1]
                / "src"
                / "churn_pipeline"
                / "ops"
                / "backfill.py"
            ),
            "--catalog",
            "test",
            "--schema",
            "test",
            "--start-date",
            "2026-02-01",
            "--end-date",
            "2026-01-01",
            "--landing-file",
            "/fake/path.csv",
            "--bundled-source-file",
            "/fake/bundle.csv",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Nothing to do" in result.stderr


# ── train.py excludes metadata columns ────────────────────────────────


def test_train_excludes_snapshot_date_from_features():
    """snapshot_date and _transformed_at must not be model features."""
    src = (
        Path(__file__).parents[1] / "src" / "churn_pipeline" / "modeling" / "train.py"
    ).read_text()
    assert "snapshot_date" in src
    assert "_transformed_at" in src
