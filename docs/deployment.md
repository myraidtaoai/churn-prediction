# Deployment and identity setup

The repository deploys the same Databricks bundle through two controlled stages:

1. `CI` tests every pull request and push to `main`.
2. A successful `CI` push to `main` automatically starts `Deploy Dev`.
3. Dev validates and deploys the bundle, then runs `churn_end_to_end` as an integration smoke test.
4. An operator manually starts `Deploy Production` with the full commit SHA from a successful Dev deployment and a deployment reason.
5. The production workflow verifies that the exact SHA passed Dev, waits for the `prod` environment approval, validates the bundle, and deploys it. It does not run the production data pipeline automatically.

Both deployment workflows use GitHub OpenID Connect (OIDC), so no long-lived Databricks token is stored in GitHub.

## 1. Create the GitHub environments

In **Repository settings → Environments**, create environments named exactly `dev` and `prod`.

Add these environment variables to both environments, using the appropriate workspace values for each stage:

| Variable | Value |
| --- | --- |
| `DATABRICKS_HOST` | Workspace URL, such as `https://dbc-...cloud.databricks.com` |
| `DATABRICKS_CLIENT_ID` | Application/client ID of the Databricks service principal |
| `DATABRICKS_WAREHOUSE_ID` | SQL warehouse ID used by the dashboard |

For `prod`, add at least one required reviewer in **Deployment protection rules**. Enable prevention of self-review if your GitHub plan supports it. Restrict deployment branches to `main` for another protection layer.

Do not create a `DATABRICKS_TOKEN` secret. The workflows explicitly use `DATABRICKS_AUTH_TYPE=github-oidc`.

## 2. Configure Databricks workload identity federation

Create a Databricks service principal for automation, assign it to the workspace, then create a federation policy for each GitHub environment. Use:

- Issuer: `https://token.actions.githubusercontent.com`
- Audience: your Databricks account ID
- Dev subject: `repo:myraidtaoai/churn-prediction:environment:dev`
- Production subject: `repo:myraidtaoai/churn-prediction:environment:prod`

The subjects must match the GitHub owner, repository, and environment names exactly. Databricks documents both account-console and CLI methods in [Enable workload identity federation in GitHub Actions](https://docs.databricks.com/gcp/en/dev-tools/auth/provider-github).

Grant the service principal only the permissions required by this bundle:

- Workspace access and permission to deploy jobs, an MLflow experiment, and a dashboard.
- `USE CATALOG` and permission to create/use the target schemas and their tables, Volume, and registered model.
- `CAN USE` on the configured SQL warehouse.
- Read/write access to the bundle's landing Volume and model aliases.

Use separate service principals for Dev and Production if the two environments use different workspaces or access boundaries. In that case, set each environment's `DATABRICKS_CLIENT_ID` to its own principal.

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

The workflow checks GitHub Actions for a successful Dev deployment of that SHA. GitHub then pauses the job for the required `prod` reviewer before exposing environment variables and obtaining a Production OIDC token. GitHub's [deployment environments documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments) explains reviewer and branch protection settings.

Production deployment ends with `databricks bundle summary`. Run production workflows separately after reviewing the deployed resources; this prevents a deployment approval from also authorizing a production data mutation.

## Troubleshooting

- **OIDC authentication fails:** confirm the service-principal client ID, issuer, audience, and exact environment-specific subject. Confirm the workflow has `id-token: write`.
- **No successful Deploy Dev run exists:** enter a full SHA from a completed green `Deploy Dev` run, not merely a green `CI` run.
- **Bundle validation reports `warehouse_id` missing:** define `DATABRICKS_WAREHOUSE_ID` on that GitHub environment.
- **The Production job does not pause:** configure a required reviewer on the `prod` GitHub environment before using it.
- **Databricks returns permission errors:** compare the failing resource with the least-privilege grants listed above; do not replace OIDC with a personal access token.

The workflow structure follows Databricks' [CI/CD with GitHub Actions guidance](https://docs.databricks.com/aws/en/dev-tools/ci-cd/github).
