# Data contract — customer events

**Version:** 1
**Owner:** `src/churn_pipeline/ingestion/generate_events.py` (producer) · `src/churn_pipeline/ingestion/ingest_events.py` (consumer)
**Source of truth:** `src/churn_pipeline/contracts.py`

Both producer and consumer import from `contracts.py`. Changing the schema in only one place is a build error, not a deployment surprise.

## Event schema (v1)

| Field | Type | Nullable | Description |
|---|---|---|---|
| `schema_version` | integer | no | Contract version; currently `1` |
| `event_id` | string | no | Deterministic SHA-256 event identifier |
| `generation_id` | string | yes | SHA-256 of the batch configuration (date + seed + drift + version) |
| `event_type` | string | no | One of: `billing`, `cancellation`, `complaint`, `payment`, `plan_change`, `support_call`, `usage` |
| `event_timestamp` | string | no | UTC ISO-8601 event time (`YYYY-MM-DDTHH:MM:SSZ`) |
| `event_date` | string | no | UTC business date (`YYYY-MM-DD`), used for landing-path partitioning |
| `customer_id` | string | no | Source customer identifier (matches `telco_silver.customer_id`) |
| `amount` | double | yes | Billing or payment amount; `null` for non-financial events |
| `usage_gb` | double | yes | Internet usage in GB; `0.0` for customers without internet service |
| `payment_status` | string | yes | `succeeded` or `failed`; `null` for non-payment events |
| `support_topic` | string | yes | `billing`, `connectivity`, `service_quality`, or `technical` |
| `plan_from` | string | yes | Contract type before a plan change |
| `plan_to` | string | yes | Contract type after a plan change |
| `complaint_severity` | string | yes | `medium` or `high` |
| `cancellation_reason` | string | yes | `competitor`, `price`, `service`, or `unknown` |

## Identifiers

**`event_id`** is a SHA-256 of `(schema_version, generation_id, customer_id, event_type, event_timestamp)`. It is deterministic: the same event configuration always produces the same ID. This is the dedup key at Bronze.

**`generation_id`** is a SHA-256 of `(schema_version, event_date, seed, drift_level)`. It identifies the batch configuration, not individual events.

## Constraints enforced at ingestion

These are applied by the quality framework (`quality.py`) rather than schema validation alone.

| Constraint | Severity | Behavior on violation |
|---|---|---|
| `event_id` is not null and non-empty | quarantine | Row preserved in `telco_quarantine`; batch continues |
| `customer_id` is not null and non-empty | quarantine | Row preserved in `telco_quarantine`; batch continues |
| `event_timestamp` parses and is within 2020–2030 | quarantine | Row preserved in `telco_quarantine`; batch continues |
| `event_type` is in the contract enum | quarantine | Row preserved in `telco_quarantine`; batch continues |
| `amount >= 0` when present | warn | Logged to `data_quality_metrics`; row continues |
| `usage_gb >= 0` when present | warn | Logged to `data_quality_metrics`; row continues |
| `schema_version` is present | warn | Logged to `data_quality_metrics`; row continues |
| Zero rows survive | fail | Run aborts |

## Schema evolution policy

**Additive change** (new nullable field):

1. Add the `FieldDef` to `EVENT_FIELDS` in `contracts.py`.
2. Bump `CONTRACT_VERSION` and add the old version to `SUPPORTED_VERSIONS`.
3. Update the generator to emit the new field.
4. Deploy. Auto Loader rescue mode means the consumer already handles the transition — old files without the field are valid, new files with the field are read normally.
5. Update this document.

**Breaking change** (rename, type change, remove required field):

1. Bump `CONTRACT_VERSION` and add the old version to `SUPPORTED_VERSIONS`.
2. Update both `contracts.py` and the generator in the same PR.
3. If old data must be reprocessed, update the ingestion job to handle both schema versions.
4. Deploy and reprocess affected dates via `src/churn_pipeline/ops/backfill.py`.
5. Update this document.

**Rescue behavior**: Auto Loader is configured with `schemaEvolutionMode=rescue`. Unexpected fields land in a `_rescued_data` column rather than failing the batch. Any run that rescues data should emit a warning for investigation — it means the producer is ahead of the consumer's contract.

## Landing layout

```
/Volumes/<catalog>/<schema>/landing/events/
  year=YYYY/
    month=MM/
      day=DD/
        events-seed=<seed>-drift_bps=<basis-points>.jsonl
```

Each file is immutable. The generator refuses to overwrite a file with different content. An identical rerun is a no-op. This invariant, combined with Auto Loader's checkpoint, makes the landing zone a durable, replayable source of truth.

## Validation for producers

The `contracts.py` module exports a `validate_event(event_dict)` function that returns a list of violations. The generator should call this as a self-check:

```python
from contracts import validate_event

errors = validate_event(event)
if errors:
    raise ValueError(f"Contract violation: {errors}")
```

This catches drift at generation time, before events reach the landing zone.
