# Telco Churn — a production-style data platform on Databricks

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Serverless](https://img.shields.io/badge/Compute-Serverless-6F42C1?style=flat-square&logo=apache-spark&logoColor=white)
![Asset Bundles](https://img.shields.io/badge/IaC-Asset%20Bundles-FF3621?style=flat-square&logo=databricks&logoColor=white)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20%2B%20OIDC-2088FF?style=flat-square&logo=githubactions&logoColor=white)

A batch data platform that ingests simulated customer events, refines them through a Bronze/Silver/Gold medallion into point-in-time feature snapshots, and serves a governed churn model on top of them. Every stage is deployed as a Databricks Asset Bundle, governed by Unity Catalog, and released through GitHub Actions with short-lived OIDC credentials. Model candidates are gated against the incumbent Champion on objective metric thresholds and promoted only when they win; every score ever produced is retained in an immutable, model-versioned prediction history, and the Champion can be rolled back with a single command.

The point of this repository is the operational surface, not the model. It is built to answer *"how would you run a machine learning system in production?"* rather than *"how do you train a model?"*

## Architecture

```mermaid
flowchart LR
    GEN["Event generator<br/>deterministic · drift-controlled"] --> VOL["Landing Volume<br/>append-only JSONL"]
    VOL --> AL["Auto Loader<br/>checkpointed"]
    AL --> BR["Bronze<br/>raw + dedup"]
    AL -.rejected.-> QR["Quarantine"]
    BR --> SV["Silver<br/>MERGE · contract-enforced"]
    SV -.rejected.-> QR
    SV --> GD["Gold<br/>point-in-time snapshots<br/>+ delayed labels"]
    GD --> TR["Train candidate"]
    TR --> GATE{"Quality gates"}
    GATE -->|reject| LOG["Decision logged<br/>Champion unchanged"]
    GATE -->|promote| REG["Champion alias"]
    REG --> SC["Batch scoring<br/>Spark UDF"]
    GD --> SC
    SC --> PH["Prediction history<br/>immutable · versioned"]
    PH --> DASH["AI/BI dashboard"]
    REG -.rollback.-> REG
```

Full diagram, layer contracts, design decisions, and the failure/recovery matrix: **[docs/architecture.md](docs/architecture.md)**.

## Dashboard

![Telco Churn exploration and model results dashboard](docs/images/telco-churn-dashboard-snapshot.png)

[View the full dashboard PDF →](outputs/Telco%20Churn%20%E2%80%94%20Exploration%20%26%20Model%20Results.pdf)

## What this demonstrates

| Capability | How | Where |
|---|---|---|
| Infrastructure as code | Schema, Volume, experiment, registered model, dashboard, and six jobs declared in one bundle | `databricks.yml`, `resources/churn_workflow.yml` |
| CI/CD with no long-lived secrets | Lint, test, and bundle validation on PR; auto-deploy to dev on merge; SHA-pinned, approval-gated prod | `.github/workflows/` |
| Governed data and models | Unity Catalog schema, managed Volume, registered model with aliases | `resources/churn_workflow.yml` |
| Objective model promotion | PR-AUC improvement and recall-degradation thresholds; rejection is a logged outcome, not a failure | `src/churn_pipeline/modeling/promotion_policy.py`, `promote.py` |
| Auditable scoring | Append-only `prediction_history` tagged with model version and run ID; `current_customer_churn_scores` is a view over it | `src/churn_pipeline/modeling/score.py` |
| Safe rollback | Validates version, inference contract, and metrics before restoring the alias; appends a `ROLLBACK` event | `src/churn_pipeline/modeling/rollback.py` |
| Distributed inference | MLflow Spark UDF — no `toPandas()` collection to the driver, enforced by a regression test | `src/churn_pipeline/modeling/inference.py`, `mlflow_compat.py` |
| Versioned inference contract | Scoring refuses an incompatible Champion instead of silently producing wrong output | `src/churn_pipeline/modeling/inference.py` |
| Reproducible synthetic data | Same date + seed + drift level yields byte-identical events; conflicting rewrites are rejected | `src/churn_pipeline/ingestion/generate_events.py` |
| Model explainability | SHAP over a held-out sample, published to a table the dashboard reads | `src/churn_pipeline/modeling/train.py` |
| Tested | 11 test modules covering thresholds, promotion guardrails, inference contract, bundle/dashboard contracts | `tests/` |

## Roadmap

The platform is being hardened along a documented plan: **[plans/implementation-plan.md](plans/implementation-plan.md)**.

- [x] Asset Bundles, Unity Catalog, dev/prod targets
- [x] CI/CD with OIDC and approval-gated production
- [x] Quality-gated promotion, alias rollback, immutable prediction history
- [x] Distributed inference with a versioned contract
- [x] Deterministic event generator with controlled drift
- [ ] **Idempotent, replayable ETL** — deterministic keys, `MERGE`, date-windowed backfill
- [ ] **Auto Loader incremental ingestion** — the events currently land but are not yet consumed
- [ ] **Data quality with quarantine** — reject rows, not batches
- [ ] **Versioned data contract enforced in code** — shared by producer and consumer
- [ ] **Point-in-time features and delayed labels** — leakage prevention proven by test
- [ ] Operational tables, alerting, and runbook
- [ ] Drift monitoring

## Deliberate non-goals

- **Real-time serving.** Churn scores drive a daily retention campaign. Sub-second latency buys nothing and costs a continuously running endpoint. The `availableNow` batch trigger is the right tool; streaming becomes correct when a consumer acts on individual events in seconds.
- **Real customer data.** The source is the public Telco Customer Churn dataset plus a deterministic event simulator. Synthetic generation is a feature here: it makes drift controllable, backfills unlimited, and the whole pipeline reproducible by anyone who clones the repo.
- **Automated retraining on drift.** Drift raises an alert. It does not silently promote a new model — that is how you ship a bad model quickly.
- **Multi-tenant catalog design and cost attribution.** Out of scope at this size; sketched in the plan appendix.

---

## Running it

<details>
<summary><b>Deploy</b></summary>

Authenticate the Databricks CLI, then validate and deploy the development target:

```bash
databricks auth login --host <workspace-url>
databricks warehouses list
databricks bundle validate -t dev --var="warehouse_id=<warehouse-id>"
databricks bundle deploy -t dev --var="warehouse_id=<warehouse-id>"
```

Deployment packages the repository CSV and creates the schema and `landing` Volume. Run `churn_seed_bronze` once to copy the packaged CSV to the resolved Volume path. This correctly handles the schema prefix that development mode adds, so no manual `databricks fs cp` command is required.

Pushes to `main` that pass CI are automatically validated, deployed to Dev, and exercised with the end-to-end workflow. Production deployment is manual, accepts only a commit SHA that passed the Dev deployment, and uses the protected `prod` GitHub environment for approval. Both stages authenticate with short-lived GitHub OIDC credentials. Complete the one-time environment and service-principal configuration in [Deployment and identity setup](docs/deployment.md).

Override variables to use another catalog or schema. Use the same values for deploy and run:

```bash
databricks bundle deploy -t dev --var="warehouse_id=<warehouse-id>" --var="catalog=my_catalog" --var="schema=churn_alex"
databricks bundle run churn_data_pipeline -t dev --var="warehouse_id=<warehouse-id>" --var="catalog=my_catalog" --var="schema=churn_alex"
```

</details>

<details>
<summary><b>Run the workflows</b></summary>

Run the complete workflow with one command:

```bash
databricks bundle run churn_end_to_end -t dev --var="warehouse_id=<warehouse-id>"
```

Generate a reproducible daily customer-event batch in the managed landing Volume:

```bash
databricks bundle run churn_event_generator -t dev --var="warehouse_id=<warehouse-id>" --params event_date=2026-08-02,seed=20260801,drift_level=0.0
```

The generator reads the validated `telco_silver` snapshot and writes append-only JSONL under `landing/events/year=YYYY/month=MM/day=DD`. The same date, seed, and drift level always produce the same event IDs and payload; an identical rerun is a no-op, while a conflicting rewrite is rejected. Increase `drift_level` from `0.0` toward `1.0` to simulate more payment failures, support calls, complaints, plan changes, cancellations, higher charges, and reduced usage. See [Customer event generation](docs/event-generation.md) for the data contract, drift behavior, and verification query.

To seed the Bronze table with the historical CSV (one-time, idempotent):

```bash
databricks bundle run churn_seed_bronze -t dev --var="warehouse_id=<warehouse-id>"
```

For targeted reruns or troubleshooting, run the stage workflows in dependency order. The data pipeline starts with Auto Loader event ingestion (incremental, checkpointed, exactly-once):

```bash
databricks bundle run churn_data_pipeline -t dev --var="warehouse_id=<warehouse-id>"
databricks bundle run churn_model_pipeline -t dev --var="warehouse_id=<warehouse-id>"
databricks bundle run churn_batch_score -t dev --var="warehouse_id=<warehouse-id>"
```

To roll the Champion alias back to a previously validated registered-model version, run the manual rollback job with the target version:

```bash
databricks bundle run churn_model_rollback -t dev --var="warehouse_id=<warehouse-id>" --params target_version=<model-version>
```

Rollback validates the model version, inference contract, and immutable candidate metrics before changing the alias. It verifies the new alias and appends a `ROLLBACK` event to `model_promotion_history`; it never edits model metrics or prediction tables.

The first deployment of the versioned inference contract must run the model pipeline before batch scoring. Training logs a probability-producing MLflow `pyfunc` model, promotion replaces a legacy Champion only when model quality is preserved, and batch scoring rejects an incompatible Champion instead of silently producing incorrect output.

</details>

<details>
<summary><b>The model</b></summary>

The training task compares `BalancedRandomForestClassifier`, XGBoost, LightGBM, and Extra Trees. It uses a stratified train/validation/test split, applies each model's appropriate class-imbalance method, and selects the candidate with the best validation PR-AUC. It picks a classification threshold from validation data and publishes PR-AUC, ROC-AUC, precision, recall, F1, and balanced accuracy for the dashboard. It also calculates Shapley values on a representative held-out sample and writes global feature impact to `shap_feature_importance`.

PR-AUC rather than ROC-AUC is the selection metric because churn is imbalanced and the operational question is precision within the flagged population, not ranking across the whole base.

</details>

<details>
<summary><b>Tests</b></summary>

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The unit tests cover validation-threshold selection, production risk-segment boundaries, promotion guardrails, and the versioned probability-output contract. Regression tests verify distributed scoring remains free of driver-side Pandas collection, as well as Champion selection, metric ranges, portfolio completeness, risk ordering, and the published SHAP snapshot. These dependencies are for local testing only and are not installed into the Databricks serverless job environments.

</details>

<details>
<summary><b>Dashboard details</b></summary>

The bundle deploys an AI/BI dashboard definition from `dashboards/churn_exploration_and_model_results.lvdash.json`. It combines customer exploration with model results using:

- `<catalog>.<schema>.telco_silver` and `<catalog>.<schema>.current_customer_churn_scores` for customer counts, observed churn, risk segmentation, contract analysis, and the high-risk action list.
- `<catalog>.<schema>.prediction_history` for immutable, versioned batch-scoring history.
- `<catalog>.<schema>.model_validation_metrics` for the Champion model's ROC-AUC and PR-AUC.
- `<catalog>.<schema>.model_comparison_metrics` for validation PR-AUC across all four candidate classifiers.
- `<catalog>.<schema>.shap_feature_importance` for the model's most influential churn drivers.

The dashboard is organized as one continuous, PDF-friendly **Churn Intelligence Report**. It flows from the executive summary into customer exploration, model evidence, and retention actions without switching pages. The report uses counters, colored bars, an area trend, a calibration line, a risk-composition pie, a contract-level bubble scatter, horizontal SHAP bars, and an operational table. It covers tenure, payment, internet-service and charge-band patterns, calibration, candidate trade-offs, test metrics, SHAP explanation, and risk-value concentration.

Run the workflow once before opening the dashboard, so its source tables exist. The dashboard is deployed as a draft; open it from **Dashboards**, verify the configured Free Edition SQL warehouse, then publish it when the data looks correct.

To pull an existing remote Databricks dashboard definition down into the local bundle (for example, after edits made in the Databricks UI), run:

```bash
databricks bundle generate dashboard -t dev --existing-id <dashboard-id>
```

Review the generated local dashboard file before redeploying it. This lets the remote UI version become the starting point for subsequent bundle-managed changes.

</details>

<details>
<summary><b>Compute, permissions, and Free Edition</b></summary>

Bronze and Silver use an untouched serverless environment-4 base. Training and scoring use a separate environment that installs only the missing ML libraries. NumPy, pandas, SciPy, scikit-learn, PyArrow, and Databricks Connect come from the coherent Databricks base environment and must not be reinstalled in a task or notebook. This separation exists so that compiled ML wheels cannot break Spark Connect initialization in the ETL path.

The bundle is configured for Free Edition's serverless-only compute. The end-to-end job invokes the stage jobs sequentially and does not require a custom VM type or cluster configuration. Keep its weekly schedule paused until `churn_end_to_end` succeeds interactively; Free Edition compute is quota-limited and is intended for learning and non-commercial projects. Free Edition also has constrained Model Serving capacity, so a real-time API endpoint should be treated as optional experimentation rather than the primary delivery mechanism.

The identity running `bundle deploy` needs permission to create a schema, managed Volume, registered model, dashboard, and an MLflow experiment directly in its workspace home. The workflow run identity needs `READ VOLUME` and `WRITE VOLUME` on the landing Volume and permission to create or replace tables in the deployed schema.

The end-to-end job owns the weekly schedule, which is intentionally paused. Resume it in the Databricks job UI after the first successful end-to-end run, or change `pause_status` to `UNPAUSED` and redeploy. The stage jobs remain unscheduled so they cannot accidentally overlap with orchestration.

</details>

<details>
<summary><b>Troubleshooting</b></summary>

- `dashboard warehouse_id is required`: pass `--var="warehouse_id=<warehouse-id>"` to `bundle validate`, `bundle deploy`, and `bundle run`.
- `Parent directory does not exist` while creating the experiment: deploy the latest bundle. The experiment now lives directly in `/Users/<deployer>/<bundle>-<target>-churn-training`, whose parent already exists.
- `spark should be initialized with the first notebook command`: use the committed Python tasks, which explicitly create a serverless `DatabricksSession`; do not replace that helper with a notebook-injected global `spark`.
- `numpy.dtype size changed`: deploy the latest bundle and start a completely new workflow run—not a repair of the failed run. Verify `ingest_events` and `transform_silver_and_gold` use `etl_v4`, which has no added dependencies. Do not install or upgrade NumPy, pandas, SciPy, scikit-learn, PyArrow, or Databricks Connect from a notebook cell.
- `InvalidVersion: Invalid version: '18.x-aarch64-photon-scala2'`: deploy the latest bundle and start a new `churn_batch_score` run. The scorer contains a narrow MLflow 2.22 compatibility retry for Free Edition's uncut Databricks Runtime 18 image label.
- Landing-file errors: deploy the latest bundle and start a new run. The CSV is synchronized to `${workspace.file_path}/data` and automatically copied into the resolved `/Volumes/<catalog>/<deployed-schema>/landing` path. Confirm the run identity has `WRITE VOLUME`.

</details>

## Documentation

| Document | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Full diagram, layer contracts, design decisions, failure/recovery matrix |
| [Implementation plan](plans/implementation-plan.md) | Current-state audit, phased tasks with acceptance criteria, execution order |
| [Event generation](docs/event-generation.md) | Simulation contract, landing layout, drift behavior |
| [Deployment](docs/deployment.md) | OIDC and service-principal setup |
