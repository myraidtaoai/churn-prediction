from datetime import date

import pytest

from data_generator.customer_state import CustomerState
from data_generator.generate_events import (
    EVENT_FIELDS,
    EVENT_TYPES,
    GenerationConfig,
    event_file_path,
    generate_events,
    write_event_file,
)


def state(customer_id: str, churn_label: int = 0) -> CustomerState:
    return CustomerState.from_mapping(
        {
            "customer_id": customer_id,
            "tenure": 12,
            "monthly_charges": 79.95,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
            "tech_support": "No",
            "churn_label": churn_label,
        }
    )


def test_customer_state_rejects_invalid_source_values() -> None:
    with pytest.raises(ValueError, match="customer_id"):
        state(" ")

    values = state("customer-1").__dict__ | {"monthly_charges": -1}
    with pytest.raises(ValueError, match="monthly_charges"):
        CustomerState.from_mapping(
            {
                "customer_id": values["customer_id"],
                "tenure": values["tenure_months"],
                "monthly_charges": values["monthly_charges"],
                "contract": values["contract"],
                "internet_service": values["internet_service"],
                "payment_method": values["payment_method"],
                "tech_support": values["tech_support"],
                "churn_label": values["churn_signal"],
            }
        )


def test_generation_is_deterministic_and_contract_complete() -> None:
    states = [state("customer-2", 1), state("customer-1")]
    config = GenerationConfig(date(2026, 8, 2), seed=42, drift_level=0.25)

    first = generate_events(states, config)
    second = generate_events(reversed(states), config)

    assert first == second
    assert len(first) >= len(states)
    assert len({event["event_id"] for event in first}) == len(first)
    assert all(tuple(event) == EVENT_FIELDS for event in first)
    assert {event["event_type"] for event in first}.issubset(EVENT_TYPES)
    assert all(event["generation_id"] == config.generation_id for event in first)


def test_seed_changes_event_identity() -> None:
    states = [state("customer-1")]
    first = generate_events(states, GenerationConfig(date(2026, 8, 2), seed=1))
    second = generate_events(states, GenerationConfig(date(2026, 8, 2), seed=2))

    assert {event["event_id"] for event in first} != {
        event["event_id"] for event in second
    }


def test_customer_has_exactly_one_billing_day_in_28_day_cycle() -> None:
    customer = state("customer-1")
    events = [
        event
        for day in range(1, 29)
        for event in generate_events(
            [customer], GenerationConfig(date(2026, 8, day), seed=42)
        )
    ]

    assert sum(event["event_type"] == "billing" for event in events) == 1
    assert sum(event["event_type"] == "payment" for event in events) == 1


def test_controlled_drift_increases_adverse_event_volume() -> None:
    states = [
        state(f"customer-{number:05d}", number % 4 == 0) for number in range(2000)
    ]
    baseline = generate_events(
        states, GenerationConfig(date(2026, 8, 2), seed=7, drift_level=0.0)
    )
    drifted = generate_events(
        states, GenerationConfig(date(2026, 8, 2), seed=7, drift_level=1.0)
    )
    adverse = {"cancellation", "complaint", "support_call"}

    assert sum(event["event_type"] in adverse for event in drifted) > sum(
        event["event_type"] in adverse for event in baseline
    )


def test_event_batch_is_partitioned_append_only_and_idempotent(tmp_path) -> None:
    config = GenerationConfig(date(2026, 8, 2), seed=42, drift_level=0.25)
    events = generate_events([state("customer-1")], config)

    destination, status = write_event_file(tmp_path, config, events)
    same_destination, repeated_status = write_event_file(tmp_path, config, events)

    assert destination == event_file_path(tmp_path, config)
    assert destination.parts[-4:-1] == (
        "year=2026",
        "month=08",
        "day=02",
    )
    assert status == "created"
    assert same_destination == destination
    assert repeated_status == "unchanged"

    conflicting = [{**event, "usage_gb": 999.0} for event in events]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_event_file(tmp_path, config, conflicting)


@pytest.mark.parametrize("drift_level", [-0.01, 1.01])
def test_generation_rejects_invalid_drift(drift_level: float) -> None:
    with pytest.raises(ValueError, match="drift_level"):
        GenerationConfig(date(2026, 8, 2), drift_level=drift_level)
