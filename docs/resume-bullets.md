# Resume bullets

Working file. Bullets marked **[pending N.N]** are not yet true — they become true when that plan task lands. Do not put a pending bullet on a resume.

Numbers in `<>` are placeholders to fill from actual runs. Every number you write here must be one you can reproduce on demand, because you will be asked to.

---

## Primary set — Data Engineer

Use four of these. Lead with ingestion and idempotency.

> **Built an incremental ingestion pipeline on Databricks** processing ~`<N>` customer events per day from a managed Volume into a Delta medallion architecture using Auto Loader with checkpointed exactly-once semantics and rescue-mode schema evolution. **[pending 1.2]**

> **Designed idempotent, replayable ETL** using deterministic SHA-256 event keys and Delta `MERGE`, enabling full historical backfill over any date range through the same code path as the scheduled run — no divergent backfill logic. **[pending 1.1]**

> **Implemented a versioned data contract enforced in code** and shared by producer and consumer, with three-tier validation that quarantines rejected records with their violation reason instead of failing the batch — `<X>`% batch survivability under `<Y>`% malformed input. **[pending 1.3, 1.4]**

> **Eliminated target leakage** by separating event time from processing time and materializing point-in-time feature snapshots with a configurable `<30>`-day label-maturity horizon, verified by a test asserting no feature derives from an event later than its snapshot date. **[pending 1.5]**

> **Automated deployment with Databricks Asset Bundles and GitHub Actions**, using short-lived GitHub OIDC credentials with no long-lived tokens in the repository — lint, test, and bundle validation on every PR, automatic dev deployment with an end-to-end smoke run on merge, and SHA-pinned, approval-gated production release. ✅

> **Instrumented the platform for operations** with per-run lineage, data-quality metrics, and drift tables surfaced in an AI/BI dashboard, plus a runbook documenting failure modes and first response for every job. **[pending 1.6, 2.3]**

## Secondary set — ML Engineer / MLOps

Swap two of these in when the posting is ML-weighted.

> **Built a quality-gated model promotion pipeline** where candidates are evaluated against the incumbent Champion on PR-AUC improvement and bounded recall degradation; rejection is a logged, first-class outcome rather than a pipeline failure. ✅

> **Made every prediction auditable** through an append-only prediction history tagged with model version and scoring run ID, with the operational view derived from it — enabling single-command alias rollback that validates version, inference contract, and immutable metrics before switching. ✅

> **Replaced driver-side inference with a distributed MLflow Spark UDF**, cutting scoring time from `<A>`s to `<B>`s at 1M rows and removing the driver-memory ceiling; a regression test fails the build if `toPandas()` is reintroduced into the scoring path. ✅ *(benchmark pending 1.7)*

> **Versioned the inference contract** so batch scoring refuses an incompatible Champion rather than silently producing wrong output — a failed run instead of a quiet data-quality incident. ✅

## One-line project header

> **Telco Churn Data Platform** — Databricks · Delta Lake · Auto Loader · Unity Catalog · MLflow · Asset Bundles · GitHub Actions · PySpark
> Production-style batch platform: incremental ingestion, medallion refinement, point-in-time features, gated model promotion, immutable audit trail, one-command rollback. `github.com/<user>/churn-prediction`

---

## Numbers to capture

Fill these in as the phases land. Record where each came from so you can defend it.

| Placeholder | What to measure | Source | Value |
|---|---|---|---|
| `<N>` events/day | Row count from one generator run | Generator task JSON summary | |
| `<X>`% survivability | Rows passing / rows in, on a corrupted batch | `data_quality_metrics` | |
| `<Y>`% malformed | Corruption rate you injected | Test fixture | |
| `<A>`s → `<B>`s | Scoring wall time, driver vs. distributed, 1M rows | Job run history | |
| Backfill span | Days replayed in one command | `src/churn_pipeline/ops/backfill.py` run | |
| Test count | `pytest` collected | CI log | |
| PR-AUC lift | Champion PR-AUC vs. no-skill baseline | `model_validation_metrics` | |

---

## Writing rules

**Mechanism, not adjective.** "Robust pipeline" is unfalsifiable and reads as filler. "Idempotent via deterministic keys and `MERGE`" invites the follow-up question you want.

**Name the tradeoff when there is room.** "Batch, not streaming, because the consumer is a daily campaign" shows judgment. Listing only technologies shows exposure.

**Never claim a number you cannot reproduce.** Every number here should be recoverable from a job run or a test in under two minutes. Assume the interviewer asks how you measured it — because for a portfolio project, that is the most natural question in the world.

**Do not hide that the data is synthetic.** Say "deterministic event simulator" in the bullet or the project header. Getting caught softening it costs more than the honesty does, and controlled drift is genuinely a better demo than a static dump.
