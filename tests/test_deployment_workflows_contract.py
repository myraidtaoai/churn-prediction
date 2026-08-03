from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _load(name: str) -> tuple[dict, str]:
    text = (WORKFLOWS / name).read_text()
    return yaml.safe_load(text), text


def _triggers(workflow: dict) -> dict:
    # PyYAML 1.1 parses the unquoted GitHub Actions key `on` as boolean true.
    return workflow.get("on", workflow.get(True))


def test_dev_deploys_only_after_ci_and_runs_integration_smoke() -> None:
    workflow, text = _load("deploy-dev.yml")
    triggers = _triggers(workflow)
    job = workflow["jobs"]["deploy-dev"]

    assert triggers["workflow_run"]["workflows"] == ["CI"]
    assert "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"contents": "read"}
    assert job["environment"] == "dev"
    assert "DATABRICKS_AUTH_TYPE" not in job["env"]
    assert "databricks bundle validate -t dev" in text
    assert "databricks bundle deploy -t dev" in text


def test_production_is_manual_gated_and_accepts_only_dev_verified_sha() -> None:
    workflow, text = _load("deploy-prod.yml")
    triggers = _triggers(workflow)
    dispatch = triggers["workflow_dispatch"]
    job = workflow["jobs"]["deploy-production"]

    assert set(triggers) == {"workflow_dispatch"}
    assert dispatch["inputs"]["release_sha"]["required"] is True
    assert dispatch["inputs"]["deployment_reason"]["required"] is True
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert job["needs"] == "verify-release"
    assert job["environment"] == "prod"
    assert "DATABRICKS_AUTH_TYPE" not in job["env"]
    assert "gh run list" in text
    assert "--workflow deploy-dev.yml" in text
    assert "--status success" in text
    assert "databricks bundle validate -t prod" in text
    assert "databricks bundle deploy -t prod" in text
    assert "databricks bundle run -t prod" not in text


def test_deployment_jobs_use_environment_scoped_pat_configuration() -> None:
    for filename, job_name in (
        ("deploy-dev.yml", "deploy-dev"),
        ("deploy-prod.yml", "deploy-production"),
    ):
        workflow, text = _load(filename)
        env = workflow["jobs"][job_name]["env"]

        assert env["DATABRICKS_HOST"] == "${{ vars.DATABRICKS_HOST }}"
        assert env["DATABRICKS_TOKEN"] == "${{ secrets.DATABRICKS_TOKEN }}"
        assert env["DATABRICKS_WAREHOUSE_ID"] == "${{ vars.DATABRICKS_WAREHOUSE_ID }}"
        assert "DATABRICKS_CLIENT_ID" not in env
        assert "DATABRICKS_AUTH_TYPE" not in env
        assert "uses: databricks/setup-cli@main" in text
        assert "uses: actions/checkout@v5" in text
