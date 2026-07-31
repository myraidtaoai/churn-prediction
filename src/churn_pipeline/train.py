"""Compare imbalance-aware classifiers and register the best churn model."""

from __future__ import annotations

from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from common import base_parser, get_spark, table
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from mlflow.models import infer_signature
from model_utils import choose_threshold
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 996


def positive_class_shap_values(classifier, matrix) -> np.ndarray:
    """Normalize TreeExplainer output to rows by features for class 1."""
    raw_values = shap.TreeExplainer(classifier).shap_values(matrix)
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[1] if len(raw_values) > 1 else raw_values[0])
    else:
        values = np.asarray(getattr(raw_values, "values", raw_values))
        if values.ndim == 3:
            if values.shape[-1] == 2:
                values = values[:, :, 1]
            elif values.shape[0] == 2:
                values = values[1]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")
    return values


spark = get_spark()
parser = base_parser("Train and register the best churn classifier.")
parser.add_argument("--experiment-name", required=True)
parser.add_argument("--model-name", required=True)
args = parser.parse_args()

pdf = spark.table(table(args.catalog, args.schema, "telco_silver")).toPandas()
if pdf.empty:
    raise ValueError("telco_silver is empty.")
if pdf["churn_label"].nunique() != 2:
    raise ValueError("Training requires both churn and non-churn examples.")

feature_columns = [
    column
    for column in pdf.columns
    if column not in {"customer_id", "churn", "churn_label"}
]
X = pdf[feature_columns]
y = pdf["churn_label"].astype(int)
categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
numeric = [column for column in X.columns if column not in categorical]

preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ),
                ]
            ),
            categorical,
        ),
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric,
        ),
    ]
)

# 64% training, 16% validation, 20% final test. The test fold is evaluated
# only after the validation winner has been selected.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=RANDOM_STATE,
)

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
if positive_count == 0:
    raise ValueError("The training fold contains no churn examples.")
scale_pos_weight = negative_count / positive_count

candidate_estimators = {
    "balanced_random_forest": (
        BalancedRandomForestClassifier(
            n_estimators=300,
            sampling_strategy="all",
            replacement=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "balanced_bootstrap",
    ),
    "xgboost": (
        XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "scale_pos_weight",
    ),
    "lightgbm": (
        LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "scale_pos_weight",
    ),
    "extra_trees": (
        ExtraTreesClassifier(
            n_estimators=350,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "class_weight_balanced",
    ),
}

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(args.experiment_name)

candidates = []
for algorithm, (estimator, imbalance_method) in candidate_estimators.items():
    model = Pipeline(
        [("preprocessor", clone(preprocessor)), ("classifier", estimator)]
    )
    with mlflow.start_run(run_name=algorithm) as run:
        model.fit(X_train, y_train)
        validation_probability = model.predict_proba(X_validation)[:, 1]
        threshold, validation_best_f1 = choose_threshold(
            y_validation, validation_probability
        )
        validation_pr_auc = float(
            average_precision_score(y_validation, validation_probability)
        )
        mlflow.log_metrics(
            {
                "validation_pr_auc": validation_pr_auc,
                "validation_best_f1": validation_best_f1,
                "classification_threshold": threshold,
            }
        )
        mlflow.log_params(
            {
                "algorithm": algorithm,
                "imbalance_method": imbalance_method,
                "scale_pos_weight": float(scale_pos_weight),
                "feature_count": len(feature_columns),
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
            }
        )
        example = X_train.head(5)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=example,
            signature=infer_signature(example, model.predict(example)),
        )
        candidates.append(
            {
                "algorithm": algorithm,
                "imbalance_method": imbalance_method,
                "model": model,
                "run_id": run.info.run_id,
                "validation_pr_auc": validation_pr_auc,
                "validation_best_f1": validation_best_f1,
                "threshold": threshold,
            }
        )

best = max(
    candidates,
    key=lambda candidate: (
        candidate["validation_pr_auc"],
        candidate["validation_best_f1"],
    ),
)

# Evaluate only the selected candidate on the untouched test fold.
test_probability = best["model"].predict_proba(X_test)[:, 1]
test_prediction = (test_probability >= best["threshold"]).astype(int)
test_metrics = {
    "test_roc_auc": float(roc_auc_score(y_test, test_probability)),
    "test_pr_auc": float(average_precision_score(y_test, test_probability)),
    "test_precision": float(
        precision_score(y_test, test_prediction, zero_division=0)
    ),
    "test_recall": float(recall_score(y_test, test_prediction, zero_division=0)),
    "test_f1": float(f1_score(y_test, test_prediction, zero_division=0)),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_prediction)
    ),
}
with mlflow.start_run(run_id=best["run_id"]):
    mlflow.log_metrics(test_metrics)
    mlflow.set_tag("selected_champion", "true")

# Explain a validation sample, keeping the final test fold reserved for metrics.
best_preprocessor = best["model"].named_steps["preprocessor"]
best_classifier = best["model"].named_steps["classifier"]
explain_sample = X_validation.sample(
    n=min(500, len(X_validation)), random_state=RANDOM_STATE
)
explain_matrix = best_preprocessor.transform(explain_sample)
if hasattr(explain_matrix, "toarray"):
    explain_matrix = explain_matrix.toarray()
shap_values = positive_class_shap_values(best_classifier, explain_matrix)
feature_names = [
    name.replace("categorical__", "").replace("numeric__", "")
    for name in best_preprocessor.get_feature_names_out()
]
if shap_values.shape[1] != len(feature_names):
    raise ValueError(
        "SHAP feature count does not match the fitted preprocessing output."
    )
shap_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "directional_mean_shap": shap_values.mean(axis=0),
        "selected_algorithm": best["algorithm"],
    }
).sort_values("mean_abs_shap", ascending=False)

registered = mlflow.register_model(
    f"runs:/{best['run_id']}/model",
    args.model_name,
    await_registration_for=300,
)
MlflowClient().set_registered_model_alias(
    args.model_name,
    "Candidate",
    registered.version,
)

trained_at = datetime.now(timezone.utc)
comparison = pd.DataFrame(
    [
        {
            "algorithm": candidate["algorithm"],
            "imbalance_method": candidate["imbalance_method"],
            "run_id": candidate["run_id"],
            "validation_pr_auc": candidate["validation_pr_auc"],
            "validation_best_f1": candidate["validation_best_f1"],
            "classification_threshold": candidate["threshold"],
            "selected": candidate["run_id"] == best["run_id"],
            "trained_at": trained_at,
        }
        for candidate in candidates
    ]
)
(
    spark.createDataFrame(comparison)
    .write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(table(args.catalog, args.schema, "model_comparison_metrics"))
)

(
    spark.createDataFrame(shap_importance)
    .write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(
        table(args.catalog, args.schema, "shap_feature_importance")
    )
)

metrics_row = {
    "run_id": best["run_id"],
    "model_name": args.model_name,
    "model_version": str(registered.version),
    "model_alias": "Candidate",
    "selected_algorithm": best["algorithm"],
    "selection_metric": "validation_pr_auc",
    "validation_pr_auc": best["validation_pr_auc"],
    "validation_best_f1": best["validation_best_f1"],
    "trained_at": trained_at,
    **test_metrics,
    "test_rows": int(len(y_test)),
    "classification_threshold": float(best["threshold"]),
}
(
    spark.createDataFrame(pd.DataFrame([metrics_row]))
    .write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(
        table(args.catalog, args.schema, "model_candidate_metrics")
    )
)