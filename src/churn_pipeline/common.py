"""Shared helpers for the churn Databricks workflow."""

from __future__ import annotations

import argparse


def base_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Create the common command-line parser used by bundle tasks."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    return parser


def parse_catalog_schema() -> argparse.Namespace:
    parser = base_parser()
    return parser.parse_args()


def quoted(*parts: str) -> str:
    """Return a Unity Catalog identifier with each part safely quoted."""
    return ".".join(f"`{part.replace('`', '``')}`" for part in parts)


def table(catalog: str, schema: str, name: str) -> str:
    return quoted(catalog, schema, name)


def get_spark():
    """Return the Spark Connect session attached to the current Databricks job."""
    try:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.getOrCreate()
    except ValueError as exc:
        if "numpy.dtype size changed" not in str(exc):
            raise
        raise RuntimeError(
            "The task environment contains binary-incompatible NumPy extensions. "
            "Deploy the latest bundle and start a new workflow run; ingest and "
            "transform must use the dependency-free etl_v4 environment."
        ) from exc
