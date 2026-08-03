# Deployment and identity setup

The repository deploys the same Databricks bundle through two controlled stages using personal access token (PAT) authentication, compatible with Databricks Free Edition.

1. `CI` tests every pull request and push to `main` (ruff, pytest, bundle validate).
2. A successful `CI` push to `main` automatically starts `Deploy Dev`.
3. Dev validates and deploys the bundle, then runs `churn_end_to_end` as an integration smoke test.
4. An operator manually starts `Deploy Production` with the full commit SHA from a successful Dev deployment and a deployment reason.
5. The production workflow verifies that the exact SHA passed Dev, validates the bundle, and deploys it. It does not run the production data pipeline automatically.

## 1. Create a Databricks personal access token

In your Databricks workspace, go to **Settings → Developer → Access tokens** and generate a new token. The token must have the `all-apis` scope. Copy the token value immediately — it cannot be retrieved later.

## 2. Add GitHub secrets

In **Repository settings → Secrets and variables → Actions → Secrets**, create:

| Secret | Value |
| --- | --- |
| `DATABRICKS_HOST` | Workspace URL, e.g. `https://dbc-...cloud.databricks.com` |
| `DATABRICKS_TOKEN` | The PAT created in step 1 |
| `DATABRICKS_WAREHOUSE_ID` | SQL warehouse ID used by the dashboard |

All three secrets are used by both the Dev and Production deployment workflows.

Optionally, set the repository variable `DATABRICKS_ENABLED=true` under **Settings → Secrets and variables → Actions → Variables** to enable the `bundle validate` CI job.

## 3. Test Dev deployment

Merge a tested change to `main`, or manually run **Actions → Deploy Dev → Run workflow**. The workflow must complete all three controls:

- Bundle validation
- Bundle deployment
- Successful `churn_end_to_end` integration smoke test

Copy the full 40-character commit SHA from the successful run. A failed smoke test prevents that revision from being eligible for the production workflow.

## 4. Approve Production deployment

Open **Actions → Deploy Production → Run workflow** and provide:

- `release_sha`: the full SHA from the successful `Deploy Dev` run
- `deployment_reason`: the release/change identifier and reason

The workflow checks GitHub Actions for a successful Dev deployment of that SHA. Production deployment ends with `databricks bundle summary`. Run production workflows separately after reviewing the deployed resources; this prevents a deployment approval from also authorizing a production data mutation.

## Free Edition limitations

Databricks Free Edition does not support OIDC workload identity federation, service principals, or multiple workspaces. The deployment uses a single PAT stored in GitHub secrets. If you upgrade to a paid tier, consider migrating to OIDC authentication for short-lived credentials without stored secrets — see [Databricks CI/CD with GitHub Actions](https://docs.databricks.com/aws/en/dev-tools/ci-cd/github).

## Troubleshooting

- **Authentication fails with scope error:** regenerate the PAT with the `all-apis` scope. Tokens created via the UI default to a narrower scope.
- **No successful Deploy Dev run exists:** enter a full SHA from a completed green `Deploy Dev` run, not merely a green `CI` run.
- **Bundle validation reports `warehouse_id` missing:** add `DATABRICKS_WAREHOUSE_ID` to GitHub secrets, and set the `DATABRICKS_ENABLED` repository variable to `true`.
- **Deploy Dev does not trigger after CI:** confirm CI ran on a `push` to `main`, not just a `pull_request`. The `workflow_run` trigger only fires for pushes.
- **Databricks returns permission errors:** confirm the PAT belongs to a user with workspace access and permissions to manage jobs, schemas, tables, the Volume, the MLflow experiment, and the registered model.
