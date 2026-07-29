# Public dashboard exports

The committed CSV files are a reproducible local training snapshot from the same source data, split, candidate models, threshold selection, SHAP calculation, and risk logic as the Databricks workflow. They contain aggregate-only results and can be used as a publication-safe reference for the Databricks dashboard.

When a newer Databricks Champion is available, replace these snapshots with the following query results downloaded as CSV from Databricks SQL. This keeps the public dashboard current with the registered production model.

Do not export `customer_id`, individual churn probabilities, or the `customer_churn_scores` table.

Run these queries in the Databricks SQL editor, replacing `<catalog>` and `<schema>` with the deployed names. Download each result as CSV using the SQL editor's download control.

To reproduce the committed local snapshot, use the project virtual environment:

```bash
.venv/bin/python tools/generate_public_exports.py
```

## `model_validation_metrics.csv`

```sql
SELECT
  selected_algorithm,
  model_name,
  model_version,
  validation_pr_auc,
  test_roc_auc,
  test_pr_auc,
  test_precision,
  test_recall,
  test_f1,
  test_balanced_accuracy,
  classification_threshold
FROM <catalog>.<schema>.model_validation_metrics;
```

## `model_comparison_metrics.csv`

```sql
SELECT
  algorithm,
  imbalance_method,
  validation_pr_auc,
  validation_best_f1,
  classification_threshold,
  selected
FROM <catalog>.<schema>.model_comparison_metrics;
```

## `shap_feature_importance.csv`

```sql
SELECT feature, mean_abs_shap, directional_mean_shap, selected_algorithm
FROM <catalog>.<schema>.shap_feature_importance
ORDER BY mean_abs_shap DESC;
```

## `risk_segment_summary.csv`

```sql
SELECT
  risk_segment,
  COUNT(*) AS customers,
  AVG(churn_probability) AS average_churn_probability,
  SUM(annual_revenue_at_risk) AS annual_revenue_at_risk
FROM <catalog>.<schema>.customer_churn_scores
GROUP BY risk_segment;
```
