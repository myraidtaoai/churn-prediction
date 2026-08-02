"""Validated customer state used as input to event simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CustomerState:
    """Small, storage-independent customer snapshot for event generation."""

    customer_id: str
    tenure_months: int
    monthly_charge: float
    contract: str
    internet_service: str
    payment_method: str
    tech_support: str
    churn_signal: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> CustomerState:
        customer_id = str(values.get("customer_id") or "").strip()
        if not customer_id:
            raise ValueError("customer_id is required")

        tenure_months = int(values["tenure"])
        monthly_charge = float(values["monthly_charges"])
        churn_signal = int(values["churn_label"])
        if tenure_months < 0:
            raise ValueError("tenure must be non-negative")
        if monthly_charge < 0:
            raise ValueError("monthly_charges must be non-negative")
        if churn_signal not in {0, 1}:
            raise ValueError("churn_label must be 0 or 1")

        return cls(
            customer_id=customer_id,
            tenure_months=tenure_months,
            monthly_charge=monthly_charge,
            contract=str(values.get("contract") or "Unknown").strip(),
            internet_service=str(values.get("internet_service") or "Unknown").strip(),
            payment_method=str(values.get("payment_method") or "Unknown").strip(),
            tech_support=str(values.get("tech_support") or "Unknown").strip(),
            churn_signal=churn_signal,
        )
