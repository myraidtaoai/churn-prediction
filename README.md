# Telco Churn — Databricks Declarative Automation Bundle

![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Serverless](https://img.shields.io/badge/Compute-Serverless-6F42C1?style=flat-square&logo=apache-spark&logoColor=white)
![Classification](https://img.shields.io/badge/ML-Classification-2EA44F?style=flat-square&logo=scikitlearn&logoColor=white)

This bundle deploys the Unity Catalog schema, managed landing Volume, MLflow experiment, registered model, AI/BI dashboard, three independently runnable stage workflows, and an end-to-end orchestration workflow.

The training task compares `BalancedRandomForestClassifier`, XGBoost, LightGBM, and Extra Trees. It uses a stratified train/validation/test split, applies each model's appropriate class-imbalance method, and promotes the candidate with the best validation PR-AUC. It selects a classification threshold from validation data and publishes PR-AUC, ROC-AUC, precision, recall, F1, and balanced accuracy for the dashboard. It also calculates Shapley values on a representative held-out sample and writes global feature impact to `shap_feature_importance`.

## Dashboard output

[View the full dashboard PDF](outputs/Telco%20Churn%20%E2%80%94%20Exploration%20%26%20Model%20Results.pdf)

![Telco Churn exploration and model results dashboard](docs/images/telco-churn-dashboard-snapshot.png)

Bronze and Silver use an untouched serverless environment-4 base. Training and scoring use a separate environment that installs only the missing ML libraries. NumPy, pandas, SciPy, scikit-learn, PyArrow, and Databricks Connect come from the coherent Databricks base environment and must not be reinstalled in a task or notebook.

## Deploy

Authenticate the Databricks CLI, then validate and deploy the development target:

```bash
databricks auth login --host <workspace-url>
databricks warehouses list
databricks bundle validate -t dev --var="warehouse_id=<warehouse-id>"
databricks bundle deploy -t dev --var="warehouse_id=<warehouse-id>"
```

Deployment packages the repository CSV and creates the schema and `landing` Volume. On the first run, `ingest_bronze` automatically copies the packaged CSV to the resolved Volume path. This correctly handles the schema prefix that development mode adds, so no manual `databricks fs cp` command is required.

Pushes to `main` that pass CI are automatically validated, deployed to Dev, and exercised with the end-to-end workflow. Production deployment is manual, accepts only a commit SHA that passed the Dev deployment, and uses the protected `prod` GitHub environment for approval. Both stages authenticate with short-lived GitHub OIDC credentials. Complete the one-time environment and service-principal configuration in [Deployment and identity setup](docs/deployment.md).

Run the complete workflow with one command:

```bash
databricks bundle run churn_end_to_end -t dev --var="warehouse_id=<warehouse-id>"
```

Generate a reproducible daily customer-event batch in the managed landing Volume:

```bash
databricks bundle run churn_event_generator -t dev --var="warehouse_id=<warehouse-id>" --params event_date=2026-08-02,seed=20260801,drift_level=0.0
```

The generator reads the validated `telco_silver` snapshot and writes append-only JSONL under `landing/events/year=YYYY/month=MM/day=DD`. The same date, seed, and drift level always produce the same event IDs and payload; an identical rerun is a no-op, while a conflicting rewrite is rejected. Increase `drift_level` from `0.0` toward `1.0` to simulate more payment failures, support calls, complaints, plan changes, cancellations, higher charges, and reduced usage. Auto Loader ingestion is intentionally the next independent pipeline stage. See [Customer event generation](docs/event-generation.md) for the data contract, drift behavior, and verification query.

For targeted reruns or troubleshooting, run the stage workflows in dependency order:

```bash
databricks bundle run churn_data_pipeline -t dev --var="warehouse_id=<warehouse-id>"
databricks bundle run churn_model_pipeline -t dev --var="warehouse_id=<warehouse-id>"
databricks bundle run churn_batch_score -t dev --var="warehouse_id=<warehouse-id>"
```

To roll the Champion alias back to a previously validated registered-model version,
run the manual rollback job with the target version:

```bash
databricks bundle run churn_model_rollback -t dev --var="warehouse_id=<warehouse-id>" --params target_version=<model-version>
```

Rollback validates the model version, inference contract, and immutable candidate
metrics before changing the alias. It verifies the new alias and appends a `ROLLBACK`
event to `model_promotion_history`; it never edits model metrics or prediction tables.

The first deployment of the versioned inference contract must run the model pipeline before batch scoring. Training logs a probability-producing MLflow `pyfunc` model, promotion replaces a legacy Champion only when model quality is preserved, and batch scoring rejects an incompatible Champion instead of silently producing incorrect output.

Override variables to use another catalog or schema. Use the same values for deploy and run:

```bash
databricks bundle deploy -t dev --var="warehouse_id=<warehouse-id>" --var="catalog=my_catalog" --var="schema=churn_alex"
databricks bundle run churn_data_pipeline -t dev --var="warehouse_id=<warehouse-id>" --var="catalog=my_catalog" --var="schema=churn_alex"
databricks bundle run churn_model_pipeline -t dev --var="warehouse_id=<warehouse-id>" --var="catalog=my_catalog" --var="schema=churn_alex"
databricks bundle run churn_batch_score -t dev --var="warehouse_id=<warehouse-id>" --var="catalog=my_catalog" --var="schema=churn_alex"
```

## Databricks Free Edition

The bundle is configured for Free Edition's serverless-only compute. The end-to-end job invokes the three stage jobs sequentially and does not require a custom VM type or cluster configuration. Keep its weekly schedule paused until `churn_end_to_end` succeeds interactively; Free Edition compute is quota-limited and is intended for learning and non-commercial projects.

Batch inference uses an MLflow Spark UDF, so customer rows remain distributed instead of being collected into driver memory. Use `prediction_history` as the append-only scoring record and `current_customer_churn_scores` as the latest-run view used by the dashboard. The Free Edition has constrained Model Serving capacity, so a real-time API endpoint should be treated as optional experimentation rather than the primary delivery mechanism.

## Permissions

The identity running `bundle deploy` needs permission to create a schema, managed Volume, registered model, dashboard, and an MLflow experiment directly in its workspace home. The workflow run identity needs `READ VOLUME` and `WRITE VOLUME` on the landing Volume and permission to create or replace tables in the deployed schema.

The end-to-end job owns the weekly schedule, which is intentionally paused. Resume it in the Databricks job UI after the first successful end-to-end run, or change `pause_status` to `UNPAUSED` and redeploy. The three stage jobs remain unscheduled so they cannot accidentally overlap with orchestration.

## Tests

Install the local development dependencies and run the pytest suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -v
```

The unit tests cover validation-threshold selection, production risk-segment boundaries, promotion guardrails, and the versioned probability-output contract. Regression tests verify distributed scoring remains free of driver-side Pandas collection, as well as Champion selection, metric ranges, portfolio completeness, risk ordering, and the published SHAP snapshot. These dependencies are for local testing only and are not installed into the Databricks serverless job environments.

## Dashboard

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
databricks bundle generate dashboard \
  -t dev \
  --existing-id <dashboard-id>
```

Review the generated local dashboard file before redeploying it. This lets the remote UI version become the starting point for subsequent bundle-managed changes.

## Troubleshooting

- `dashboard warehouse_id is required`: pass `--var="warehouse_id=<warehouse-id>"` to `bundle validate`, `bundle deploy`, and `bundle run`.
- `Parent directory does not exist` while creating the experiment: deploy the latest bundle. The experiment now lives directly in `/Users/<deployer>/<bundle>-<target>-churn-training`, whose parent already exists.
- `spark should be initialized with the first notebook command`: use the committed Python tasks, which explicitly create a serverless `DatabricksSession`; do not replace that helper with a notebook-injected global `spark`.
- `numpy.dtype size changed`: deploy the latest bundle and start a completely new workflow run—not a repair of the failed run. Verify `ingest_bronze` and `transform_silver_and_gold` use `etl_v4`, which has no added dependencies. Do not install or upgrade NumPy, pandas, SciPy, scikit-learn, PyArrow, or Databricks Connect from a notebook cell.
- `InvalidVersion: Invalid version: '18.x-aarch64-photon-scala2'`: deploy the latest bundle and start a new `churn_batch_score` run. The scorer contains a narrow MLflow 2.22 compatibility retry for Free Edition's uncut Databricks Runtime 18 image label.
- Landing-file errors: deploy the latest bundle and start a new run. The CSV is synchronized to `${workspace.file_path}/data` and automatically copied into the resolved `/Volumes/<catalog>/<deployed-schema>/landing` path. Confirm the run identity has `WRITE VOLUME`.
