# Customer event generation

The customer-event generator is the first stage of continuous ingestion. It runs inside Databricks, reads the validated `telco_silver` customer snapshot, and produces a daily newline-delimited JSON batch in the managed `landing` Volume. It deliberately does not write a Delta table; the next Auto Loader stage owns ingestion and Bronze metadata.

## Landing layout

Each configuration has one immutable destination:

```text
/Volumes/<catalog>/<schema>/landing/events/
  year=YYYY/
    month=MM/
      day=DD/
        events-seed=<seed>-drift_bps=<basis-points>.jsonl
```

The `event_date`, seed, schema version, and drift level determine `generation_id`. Those inputs plus customer and event identity determine every `event_id`, timestamp, probability decision, and value. Repeating the same configuration therefore produces byte-identical output. If the file already contains that output, the task reports `unchanged`; if it differs, the task fails instead of overwriting history.

If `event_date` is empty, the generator uses the previous UTC date so it never creates future-dated events during a start-of-day run.

## Event contract

All records contain the same versioned fields. Event-specific values remain `null` when they do not apply.

| Field | Purpose |
| --- | --- |
| `schema_version` | Integer event-contract version; currently `1` |
| `event_id` | Deterministic SHA-256 event identifier |
| `generation_id` | Deterministic identifier for the complete batch configuration |
| `event_type` | `billing`, `payment`, `support_call`, `plan_change`, `usage`, `complaint`, or `cancellation` |
| `event_timestamp` | UTC ISO-8601 event time |
| `event_date` | UTC business date used for partitioning |
| `customer_id` | Source customer identifier |
| `amount` | Billing or payment amount |
| `usage_gb` | Simulated internet usage; zero for customers without internet service |
| `payment_status` | `succeeded` or `failed` |
| `support_topic` | Support-call category |
| `plan_from`, `plan_to` | Contract transition |
| `complaint_severity` | `medium` or `high` |
| `cancellation_reason` | Simulated cancellation category |

Every customer receives a daily usage record. Each customer has one stable billing day in a 28-day cycle, when matching billing and payment events are emitted. Other event types use deterministic probability draws informed by the seed customer state and configured drift.

The historical `churn_label` is used only as a hidden simulation signal that increases adverse-event probability. It is not written to the event stream. A later delayed-label stage will derive labels from observed cancellations after the configured outcome horizon.

## Controlled drift

`drift_level` must be between `0.0` and `1.0`:

- `0.0`: baseline population behavior.
- Increasing values: more payment failures, support calls, complaints, plan changes, cancellations, and billing pressure, with reduced usage.
- `1.0`: maximum deterministic stress scenario.

Use the same date and seed when comparing drift levels so behavioral changes are attributable to the drift configuration rather than a different random population.

## Run and verify

After deploying the bundle, run:

```bash
databricks bundle run churn_event_generator \
  -t dev \
  --var="warehouse_id=<warehouse-id>" \
  --params event_date=2026-08-02,seed=20260801,drift_level=0.0
```

The task prints a JSON summary containing the status, destination, generation ID, customer count, and event count. Verify the file from a Databricks notebook or SQL editor with:

```sql
SELECT *
FROM read_files(
  '/Volumes/<catalog>/<schema>/landing/events/year=2026/month=08/day=02/*.jsonl',
  format => 'json'
)
LIMIT 20;
```

The adapter intentionally caps a batch at 100,000 customers to prevent accidental driver exhaustion. The current Telco snapshot is roughly 7,000 customers. Before scaling beyond that guardrail, partition generation across Spark tasks or move the same storage-independent simulation logic to an external event producer.
