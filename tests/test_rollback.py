from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from inference import INFERENCE_CONTRACT_TAG, INFERENCE_CONTRACT_VERSION
from mlflow.exceptions import MlflowException
from rollback import (
    RollbackValidationError,
    rollback_champion,
)


class FakeMlflowClient:
    def __init__(
        self,
        versions: dict[str, SimpleNamespace],
        champion_version: str,
        *,
        ignore_alias_updates: bool = False,
    ):
        self.versions = versions
        self.champion_version = champion_version
        self.ignore_alias_updates = ignore_alias_updates
        self.alias_updates: list[tuple[str, str, str]] = []

    def get_model_version(self, model_name, version):
        try:
            return self.versions[str(version)]
        except KeyError as exc:
            raise MlflowException(
                f"Unknown model version {version}",
                error_code="RESOURCE_DOES_NOT_EXIST",
            ) from exc

    def get_model_version_by_alias(self, model_name, alias):
        return self.versions[self.champion_version]

    def set_registered_model_alias(self, model_name, alias, version):
        version = str(version)
        self.alias_updates.append((model_name, alias, version))
        if not self.ignore_alias_updates:
            self.champion_version = version


def model_version(version: str, contract: str = INFERENCE_CONTRACT_VERSION):
    return SimpleNamespace(
        version=version,
        tags={INFERENCE_CONTRACT_TAG: contract},
    )


@pytest.fixture
def versions():
    return {
        "1": model_version("1"),
        "2": model_version("2"),
        "3": model_version("3", contract="legacy_contract"),
    }


@pytest.fixture
def metrics():
    return {
        "1": {"test_pr_auc": 0.68, "test_recall": 0.75},
        "2": {"test_pr_auc": 0.72, "test_recall": 0.77},
    }


def test_rollback_moves_alias_verifies_it_and_appends_audit(versions, metrics):
    client = FakeMlflowClient(versions, champion_version="2")
    original_metrics = deepcopy(metrics)
    audits = []

    result = rollback_champion(
        client=client,
        model_name="main.churn.telco_churn_model",
        target_version="1",
        read_metrics=metrics.get,
        write_audit=audits.append,
    )

    assert result.previous_version == "2"
    assert result.target_version == "1"
    assert client.champion_version == "1"
    assert client.alias_updates == [("main.churn.telco_churn_model", "Champion", "1")]
    assert audits[0]["decision"] == "ROLLBACK"
    assert audits[0]["previous_champion_version"] == "2"
    assert audits[0]["candidate_version"] == "1"
    assert metrics == original_metrics


def test_rollback_rejects_unknown_target_before_alias_mutation(versions, metrics):
    client = FakeMlflowClient(versions, champion_version="2")

    with pytest.raises(RollbackValidationError, match="does not exist"):
        rollback_champion(
            client=client,
            model_name="model",
            target_version="99",
            read_metrics=metrics.get,
            write_audit=lambda row: None,
        )

    assert client.alias_updates == []


def test_rollback_rejects_incompatible_inference_contract(versions, metrics):
    client = FakeMlflowClient(versions, champion_version="2")

    with pytest.raises(RollbackValidationError, match="incompatible"):
        rollback_champion(
            client=client,
            model_name="model",
            target_version="3",
            read_metrics=metrics.get,
            write_audit=lambda row: None,
        )

    assert client.alias_updates == []


def test_rollback_rejects_target_without_immutable_metrics(versions, metrics):
    client = FakeMlflowClient(versions, champion_version="2")

    with pytest.raises(RollbackValidationError, match="Metrics were not found"):
        rollback_champion(
            client=client,
            model_name="model",
            target_version="1",
            read_metrics=lambda version: None,
            write_audit=lambda row: None,
        )

    assert client.alias_updates == []


def test_rollback_rejects_current_champion_as_target(versions, metrics):
    client = FakeMlflowClient(versions, champion_version="2")

    with pytest.raises(RollbackValidationError, match="already points"):
        rollback_champion(
            client=client,
            model_name="model",
            target_version="2",
            read_metrics=metrics.get,
            write_audit=lambda row: None,
        )

    assert client.alias_updates == []


def test_rollback_restores_previous_alias_when_audit_write_fails(
    versions,
    metrics,
):
    client = FakeMlflowClient(versions, champion_version="2")

    def fail_audit(row):
        raise RuntimeError("Delta write failed")

    with pytest.raises(RuntimeError, match="previous Champion alias was restored"):
        rollback_champion(
            client=client,
            model_name="model",
            target_version="1",
            read_metrics=metrics.get,
            write_audit=fail_audit,
        )

    assert client.champion_version == "2"
    assert client.alias_updates == [
        ("model", "Champion", "1"),
        ("model", "Champion", "2"),
    ]


def test_rollback_restores_previous_alias_when_verification_fails(
    versions,
    metrics,
):
    client = FakeMlflowClient(
        versions,
        champion_version="2",
        ignore_alias_updates=True,
    )

    with pytest.raises(RuntimeError, match="previous Champion alias was restored"):
        rollback_champion(
            client=client,
            model_name="model",
            target_version="1",
            read_metrics=metrics.get,
            write_audit=lambda row: None,
        )

    assert client.champion_version == "2"
