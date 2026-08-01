from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).parents[1] / "resources" / "churn_workflow.yml"
)
README_PATH = Path(__file__).parents[1] / "README.md"

EXPECTED_JOBS = {
    "churn_data_pipeline",
    "churn_model_pipeline",
    "churn_batch_score",
    "churn_end_to_end",
}


def load_jobs():
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return workflow["resources"]["jobs"]


def test_bundle_exposes_three_stages_and_one_orchestrator():
    jobs = load_jobs()

    assert set(jobs) == EXPECTED_JOBS


def test_job_tasks_preserve_required_execution_order():
    jobs = load_jobs()
    data_tasks = jobs["churn_data_pipeline"]["tasks"]
    model_tasks = jobs["churn_model_pipeline"]["tasks"]
    scoring_tasks = jobs["churn_batch_score"]["tasks"]

    assert [task["task_key"] for task in data_tasks] == [
        "ingest_bronze",
        "transform_silver_and_gold",
    ]
    assert data_tasks[1]["depends_on"] == [{"task_key": "ingest_bronze"}]

    assert [task["task_key"] for task in model_tasks] == [
        "train_and_register_model",
        "evaluate_and_promote_candidate",
    ]
    assert model_tasks[1]["depends_on"] == [
        {"task_key": "train_and_register_model"}
    ]

    assert [task["task_key"] for task in scoring_tasks] == [
        "batch_score_customers"
    ]


def test_readme_run_commands_match_bundle_job_keys():
    readme = README_PATH.read_text(encoding="utf-8")

    for job_key in EXPECTED_JOBS:
        assert f"databricks bundle run {job_key}" in readme


def test_end_to_end_job_orchestrates_stage_jobs_in_order():
    jobs = load_jobs()
    orchestrator = jobs["churn_end_to_end"]
    tasks = orchestrator["tasks"]

    assert [task["task_key"] for task in tasks] == [
        "run_data_pipeline",
        "run_model_pipeline",
        "run_batch_score",
    ]
    assert tasks[0]["run_job_task"]["job_id"] == (
        "${resources.jobs.churn_data_pipeline.id}"
    )
    assert tasks[1]["depends_on"] == [
        {"task_key": "run_data_pipeline"}
    ]
    assert tasks[1]["run_job_task"]["job_id"] == (
        "${resources.jobs.churn_model_pipeline.id}"
    )
    assert tasks[2]["depends_on"] == [
        {"task_key": "run_model_pipeline"}
    ]
    assert tasks[2]["run_job_task"]["job_id"] == (
        "${resources.jobs.churn_batch_score.id}"
    )


def test_only_orchestrator_owns_the_weekly_schedule():
    jobs = load_jobs()
    scheduled_jobs = {
        job_key for job_key, job in jobs.items() if "schedule" in job
    }

    assert scheduled_jobs == {"churn_end_to_end"}
    assert jobs["churn_end_to_end"]["schedule"]["pause_status"] == "PAUSED"
