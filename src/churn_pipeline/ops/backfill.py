#!/usr/bin/env python3
"""Replay the data pipeline over a date range using the same code paths.

Usage (local, against a Databricks workspace):

    python src/churn_pipeline/ops/backfill.py \\
        --catalog main --schema churn_dev \\
        --start-date 2026-06-01 --end-date 2026-06-30 \\
        --landing-file /Volumes/main/churn_dev/landing/WA_Fn-UseC_-Telco-Customer-Churn.csv \\
        --bundled-source-file data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Or via ``databricks bundle run`` if you add this as a parameterised job task.

Design: this script calls the **same** ``ingest.py`` and ``transform.py`` that
the scheduled run uses.  There is no separate backfill code path — divergent
logic is the most common source of silent backfill bugs, and interviewers
probe for it.  The only difference is that this script loops snapshot dates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta


def _date_range(start: date, end: date):
    """Yield each date in [start, end] inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay ingest + transform for a date range."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive")
    parser.add_argument(
        "--landing-file",
        required=True,
        help="Path to the CSV in the managed landing Volume.",
    )
    parser.add_argument(
        "--bundled-source-file",
        required=True,
        help="Path to the bundle-packaged CSV (used only on the first seed run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        print(f"Nothing to do: start {start} is after end {end}.", file=sys.stderr)
        sys.exit(0)

    ingestion_dir = "src/churn_pipeline/ingestion"
    transformation_dir = "src/churn_pipeline/transformation"

    # ── Step 1: Seed Bronze (idempotent — skips if already loaded) ──────
    ingest_cmd = [
        sys.executable,
        f"{ingestion_dir}/ingest.py",
        "--catalog",
        args.catalog,
        "--schema",
        args.schema,
        "--landing-file",
        args.landing_file,
        "--bundled-source-file",
        args.bundled_source_file,
    ]
    print(f"[backfill] Seed Bronze: {' '.join(ingest_cmd)}")
    if not args.dry_run:
        subprocess.run(ingest_cmd, check=True)

    # ── Step 2: Transform for each date ─────────────────────────────────
    dates = list(_date_range(start, end))
    total = len(dates)
    for i, snapshot_date in enumerate(dates, 1):
        transform_cmd = [
            sys.executable,
            f"{transformation_dir}/transform.py",
            "--catalog",
            args.catalog,
            "--schema",
            args.schema,
            "--snapshot-date",
            snapshot_date.isoformat(),
        ]
        print(
            f"[backfill] ({i}/{total}) Transform {snapshot_date}: {' '.join(transform_cmd)}"
        )
        if not args.dry_run:
            subprocess.run(transform_cmd, check=True)

    print(f"[backfill] Done. Processed {total} date(s): {start} → {end}")


if __name__ == "__main__":
    main()
