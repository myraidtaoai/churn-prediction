"""Tests for point-in-time feature snapshots and delayed labels.

The critical test: no feature in a snapshot for date D is derived from
an event with event_timestamp > D.  This is the leakage boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "churn_pipeline"
BUILD_FEATURES = SRC_DIR / "transformation" / "build_features.py"
GENERATE_LABELS = SRC_DIR / "transformation" / "generate_labels.py"


# ── Source-level tests (no Spark needed) ─────────────────────────────


class TestBuildFeaturesSource:
    """Static analysis of build_features.py to enforce leakage constraints."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = BUILD_FEATURES.read_text()

    def test_file_exists(self):
        assert BUILD_FEATURES.exists()

    def test_uses_event_ts_not_ingestion_timestamp_for_features(self):
        """Features must use event_ts (business time), never ingestion_timestamp."""
        # Ingestion_timestamp should only appear in column definitions or comments,
        # never in filter/window predicates.
        lines = self.src.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            if "ingestion_timestamp" in line and "filter" in line.lower():
                pytest.fail(
                    f"Line {i}: build_features.py uses ingestion_timestamp in a "
                    f"filter — features must use event_ts only."
                )

    def test_as_of_date_boundary_is_lte(self):
        """The main event filter must be <= as_of_date, not <."""
        # Look for the primary boundary filter.
        assert "<=" in self.src, (
            "build_features.py must filter events with event_ts <= as_of_date"
        )

    def test_output_table_is_gold_feature_snapshot(self):
        assert "gold_feature_snapshot" in self.src

    def test_snapshot_date_column_added(self):
        assert "snapshot_date" in self.src

    def test_idempotent_write(self):
        """Must use replaceWhere for partition-level idempotency."""
        assert "replaceWhere" in self.src

    def test_windowed_aggregates_exist(self):
        """Must have at least 7/30/90-day aggregation windows."""
        for window in ("7", "30", "90"):
            assert f"_{window}d" in self.src or f"days={window}" in self.src, (
                f"Missing {window}-day window in build_features.py"
            )

    def test_feature_categories_present(self):
        """Must aggregate payment, support, complaint, and usage features."""
        for category in ("payment", "support_call", "complaint", "usage"):
            assert category in self.src, (
                f"Missing {category} feature category in build_features.py"
            )


class TestGenerateLabelsSource:
    """Static analysis of generate_labels.py."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = GENERATE_LABELS.read_text()

    def test_file_exists(self):
        assert GENERATE_LABELS.exists()

    def test_maturity_guard_exists(self):
        """Must refuse to write labels before the horizon has elapsed."""
        assert "maturity_date" in self.src
        assert "skipped" in self.src

    def test_label_uses_cancellation_events(self):
        """Churn label must be derived from cancellation events."""
        assert "cancellation" in self.src

    def test_label_window_is_half_open(self):
        """Label window: (as_of_date, as_of_date + horizon].

        The label must not include events ON as_of_date (those are features).
        """
        # The start boundary must use > (strictly after as_of_date).
        assert ">" in self.src

    def test_default_horizon_30_days(self):
        assert "30" in self.src

    def test_training_view_created(self):
        assert "training_dataset" in self.src
        assert "CREATE OR REPLACE VIEW" in self.src

    def test_idempotent_write(self):
        assert "replaceWhere" in self.src

    def test_joins_to_gold_feature_snapshot(self):
        """The training view must join features to labels."""
        assert "gold_feature_snapshot" in self.src
        assert "gold_labels" in self.src


# ── Leakage boundary tests ───────────────────────────────────────────


class TestLeakageBoundary:
    """The critical invariant: features for date D use only events <= D,
    and labels use only events strictly after D."""

    # DOTALL so these survive the line wrapping that ``ruff format``
    # applies to long filter expressions.
    _FLAGS = re.DOTALL

    def test_feature_filter_excludes_future_events(self):
        """build_features.py must filter event_ts <= as_of_date."""
        src = BUILD_FEATURES.read_text()
        assert re.search(r'\.filter\(\s*F\.col\("event_ts"\)\s*<=', src, self._FLAGS), (
            "build_features.py must filter event_ts <= as_of_date"
        )

    def test_feature_filter_never_uses_gte_on_the_boundary(self):
        """The main boundary must not be >= — that would admit future events."""
        src = BUILD_FEATURES.read_text()
        assert not re.search(
            r'\.filter\(\s*F\.col\("event_ts"\)\s*>=\s*F\.lit\(str\(as_of_date\)',
            src,
            self._FLAGS,
        ), "build_features.py must not bound the main filter with >= as_of_date"

    def test_label_window_starts_after_as_of_date(self):
        """generate_labels.py label window must start strictly after as_of_date."""
        src = GENERATE_LABELS.read_text()
        assert re.search(
            r'\.filter\(\s*F\.col\("event_ts"\)\s*>\s*F\.lit\(label_start\)',
            src,
            self._FLAGS,
        ), "generate_labels.py must filter event_ts > as_of_date (strict)"

    def test_label_window_ends_at_maturity(self):
        """generate_labels.py label window must end at maturity_date."""
        src = GENERATE_LABELS.read_text()
        assert re.search(
            r'\.filter\(\s*F\.col\("event_ts"\)\s*<=\s*F\.lit\(label_end\)',
            src,
            self._FLAGS,
        ), "generate_labels.py must filter event_ts <= maturity_date"


# ── Bundle integration ───────────────────────────────────────────────


class TestBundleIntegration:
    """Feature and label jobs should be wirable into the bundle."""

    def test_build_features_has_catalog_schema_args(self):
        src = BUILD_FEATURES.read_text()
        assert "--catalog" in src
        assert "--schema" in src
        assert "--as-of-date" in src

    def test_generate_labels_has_catalog_schema_args(self):
        src = GENERATE_LABELS.read_text()
        assert "--catalog" in src
        assert "--schema" in src
        assert "--as-of-date" in src
        assert "--label-horizon-days" in src

    def test_generate_labels_imports_common(self):
        src = GENERATE_LABELS.read_text()
        assert "from common import" in src

    def test_build_features_imports_common(self):
        src = BUILD_FEATURES.read_text()
        assert "from common import" in src
