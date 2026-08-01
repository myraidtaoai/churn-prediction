"""Generate deterministic daily customer events into a Unity Catalog Volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .customer_state import CustomerState
except ImportError:  # Databricks executes this file directly as a task.
    from customer_state import CustomerState

SCHEMA_VERSION = 1
EVENT_TYPES = frozenset(
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
EVENT_FIELDS = (
    "schema_version",
    "event_id",
    "generation_id",
    "event_type",
    "event_timestamp",
    "event_date",
    "customer_id",
    "amount",
    "usage_gb",
    "payment_status",
    "support_topic",
    "plan_from",
    "plan_to",
    "complaint_severity",
    "cancellation_reason",
)


@dataclass(frozen=True)
class GenerationConfig:
    """Inputs that make a generated daily batch reproducible."""

    event_date: date
    seed: int = 20260801
    drift_level: float = 0.0

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0.0 <= self.drift_level <= 1.0:
            raise ValueError("drift_level must be between 0.0 and 1.0")

    @property
    def generation_id(self) -> str:
        value = (
            f"v{SCHEMA_VERSION}|{self.event_date.isoformat()}|"
            f"{self.seed}|{self.drift_level:.4f}"
        )
        return hashlib.sha256(value.encode()).hexdigest()


def _uniform(config: GenerationConfig, customer_id: str, purpose: str) -> float:
    value = f"{config.seed}|{config.event_date.isoformat()}|{customer_id}|{purpose}"
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _customer_uniform(
    config: GenerationConfig, customer_id: str, purpose: str
) -> float:
    """Return a stable customer property that does not change each day."""
    value = f"{config.seed}|{customer_id}|{purpose}"
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _event_timestamp(
    config: GenerationConfig, customer_id: str, event_type: str
) -> datetime:
    seconds = int(_uniform(config, customer_id, f"{event_type}:time") * 86400)
    start = datetime.combine(config.event_date, time.min, tzinfo=timezone.utc)
    return start + timedelta(seconds=min(seconds, 86399))


def _event(
    state: CustomerState,
    config: GenerationConfig,
    event_type: str,
    **details: Any,
) -> dict[str, Any]:
    timestamp = _event_timestamp(config, state.customer_id, event_type)
    identity = (
        f"v{SCHEMA_VERSION}|{config.generation_id}|{state.customer_id}|"
        f"{event_type}|{timestamp.isoformat()}"
    )
    event = {field: None for field in EVENT_FIELDS}
    event.update(
        {
            "schema_version": SCHEMA_VERSION,
            "event_id": hashlib.sha256(identity.encode()).hexdigest(),
            "generation_id": config.generation_id,
            "event_type": event_type,
            "event_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "event_date": config.event_date.isoformat(),
            "customer_id": state.customer_id,
            **details,
        }
    )
    return event


def generate_customer_events(
    state: CustomerState, config: GenerationConfig
) -> list[dict[str, Any]]:
    """Create a deterministic set of daily events for one customer."""
    drift = config.drift_level
    churn_risk = 0.015 * state.churn_signal
    events = [
        _event(
            state,
            config,
            "usage",
            usage_gb=(
                0.0
                if state.internet_service.lower() == "no"
                else round(
                    (2.0 + 48.0 * _uniform(config, state.customer_id, "usage:value"))
                    * (1.0 - 0.25 * drift),
                    3,
                )
            ),
        )
    ]

    billing_day = 1 + int(
        _customer_uniform(config, state.customer_id, "billing:day") * 28
    )
    if config.event_date.day == billing_day:
        events.append(
            _event(
                state,
                config,
                "billing",
                amount=round(state.monthly_charge * (1.0 + 0.08 * drift), 2),
            )
        )
        failure_probability = 0.02 + churn_risk + 0.18 * drift
        payment_status = (
            "failed"
            if _uniform(config, state.customer_id, "payment:status")
            < failure_probability
            else "succeeded"
        )
        events.append(
            _event(
                state,
                config,
                "payment",
                amount=round(state.monthly_charge * (1.0 + 0.08 * drift), 2),
                payment_status=payment_status,
            )
        )

    no_support = state.tech_support.lower() in {"no", "no internet service"}
    support_probability = 0.01 + 0.01 * no_support + churn_risk + 0.05 * drift
    if _uniform(config, state.customer_id, "support_call:emit") < support_probability:
        topics = ("billing", "connectivity", "service_quality", "technical")
        topic_index = int(
            _uniform(config, state.customer_id, "support_call:topic") * len(topics)
        )
        events.append(
            _event(
                state,
                config,
                "support_call",
                support_topic=topics[min(topic_index, len(topics) - 1)],
            )
        )

    complaint_probability = 0.005 + churn_risk + 0.04 * drift
    if _uniform(config, state.customer_id, "complaint:emit") < complaint_probability:
        severity_draw = _uniform(config, state.customer_id, "complaint:severity")
        severity = "high" if severity_draw < 0.2 + 0.4 * drift else "medium"
        events.append(
            _event(
                state,
                config,
                "complaint",
                complaint_severity=severity,
            )
        )

    plan_change_probability = 0.003 + 0.012 * drift
    if (
        _uniform(config, state.customer_id, "plan_change:emit")
        < plan_change_probability
    ):
        next_contract = {
            "Month-to-month": "One year",
            "One year": "Two year",
            "Two year": "Month-to-month",
        }.get(state.contract, "Month-to-month")
        events.append(
            _event(
                state,
                config,
                "plan_change",
                plan_from=state.contract,
                plan_to=next_contract,
            )
        )

    cancellation_probability = 0.0005 + 0.01 * state.churn_signal + 0.025 * drift
    if (
        _uniform(config, state.customer_id, "cancellation:emit")
        < cancellation_probability
    ):
        reasons = ("competitor", "price", "service", "unknown")
        reason_index = int(
            _uniform(config, state.customer_id, "cancellation:reason") * len(reasons)
        )
        events.append(
            _event(
                state,
                config,
                "cancellation",
                cancellation_reason=reasons[min(reason_index, len(reasons) - 1)],
            )
        )

    return sorted(
        events, key=lambda item: (item["event_timestamp"], item["event_type"])
    )


def generate_events(
    states: Iterable[CustomerState], config: GenerationConfig
) -> list[dict[str, Any]]:
    """Create a stable, globally ordered event batch."""
    events = [
        event
        for state in sorted(states, key=lambda item: item.customer_id)
        for event in generate_customer_events(state, config)
    ]
    event_ids = [event["event_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError("Generated duplicate event_id values")
    return sorted(
        events,
        key=lambda item: (
            item["event_timestamp"],
            item["customer_id"],
            item["event_type"],
        ),
    )


def event_file_path(output_root: Path, config: GenerationConfig) -> Path:
    """Return the immutable partitioned destination for one daily batch."""
    drift_basis_points = int(round(config.drift_level * 10_000))
    return (
        output_root
        / f"year={config.event_date.year:04d}"
        / f"month={config.event_date.month:02d}"
        / f"day={config.event_date.day:02d}"
        / f"events-seed={config.seed}-drift_bps={drift_basis_points}.jsonl"
    )


def serialize_events(events: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize a batch as deterministic newline-delimited JSON."""
    lines = [
        json.dumps(dict(event), sort_keys=True, separators=(",", ":"))
        for event in events
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode()


def write_event_file(
    output_root: Path, config: GenerationConfig, events: Iterable[Mapping[str, Any]]
) -> tuple[Path, str]:
    """Atomically create a batch, or prove an existing batch is identical."""
    destination = event_file_path(output_root, config)
    payload = serialize_events(events)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(
                "Refusing to overwrite a conflicting immutable event batch: "
                f"{destination}"
            )
        return destination, "unchanged"

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, "created"


def _quoted(*parts: str) -> str:
    return ".".join(f"`{part.replace('`', '``')}`" for part in parts)


def _parse_event_date(raw: str) -> date:
    if raw.strip():
        return date.fromisoformat(raw)
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--event-date", default="")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--drift-level", type=float, default=0.0)
    parser.add_argument("--max-customers", type=int, default=100_000)
    args = parser.parse_args()
    if args.max_customers <= 0:
        raise ValueError("max-customers must be positive")

    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()
    source = spark.table(_quoted(args.catalog, args.schema, "telco_silver"))
    required = {
        "customer_id",
        "tenure",
        "monthly_charges",
        "contract",
        "internet_service",
        "payment_method",
        "tech_support",
        "churn_label",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"telco_silver is missing required columns: {missing}")

    rows = source.select(*sorted(required)).limit(args.max_customers + 1).collect()
    if len(rows) > args.max_customers:
        raise ValueError(
            f"Customer snapshot exceeds max-customers={args.max_customers}; "
            "partition the generator before increasing this safety limit."
        )
    states = [CustomerState.from_mapping(row.asDict(recursive=True)) for row in rows]
    if not states:
        raise ValueError("telco_silver is empty; no customer events were generated")

    config = GenerationConfig(
        event_date=_parse_event_date(args.event_date),
        seed=args.seed,
        drift_level=args.drift_level,
    )
    events = generate_events(states, config)
    destination, status = write_event_file(Path(args.output_root), config, events)
    print(
        json.dumps(
            {
                "status": status,
                "destination": str(destination),
                "generation_id": config.generation_id,
                "customers": len(states),
                "events": len(events),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
