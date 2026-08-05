# Power BI Dashboard Spec — Telco Churn ML Operations

**Connection:** DirectQuery to Databricks SQL Warehouse → `main.churn`

---

## Page 1 — Exploratory Data Analysis

**Source tables:** `main.churn.telco_silver`, `main.churn.gold_feature_snapshot`, `main.churn.training_dataset`

### KPI Cards (top row)

| Card | Field / DAX |
|---|---|
| Total Customers | `DISTINCTCOUNT(telco_silver[customer_id])` |
| Avg Tenure (months) | `AVERAGE(telco_silver[tenure])` |
| Avg Monthly Charges | `AVERAGE(telco_silver[monthly_charges])` |
| Churn Rate (labeled) | `DIVIDE(COUNTROWS(FILTER(training_dataset, training_dataset[churned] = 1)), COUNTROWS(training_dataset))` — format as % |

### Visuals — Customer Demographics

| Visual | Type | Fields |
|---|---|---|
| Tenure Distribution | Histogram (column chart) | X: `telco_silver[tenure]` binned (0–12, 12–24, 24–36, 36–48, 48–60, 60–72) Y: Count of `customer_id` Color: split by `churn_label` (churned=red, retained=blue) |
| Monthly Charges Distribution | Histogram | X: `telco_silver[monthly_charges]` binned ($0–20, 20–40, 40–60, 60–80, 80–100, 100+) Y: Count of `customer_id` Color: split by `churn_label` |
| Contract Type Breakdown | Stacked bar | X: `telco_silver[contract]` (Month-to-month, One year, Two year) Y: Count of `customer_id` Legend: `churn_label` Insight: month-to-month should show much higher churn |
| Internet Service by Churn | Clustered bar | X: `telco_silver[internet_service]` (DSL, Fiber optic, No) Y: Count of `customer_id` Legend: `churn_label` |
| Payment Method by Churn | Clustered bar | X: `telco_silver[payment_method]` Y: Count of `customer_id` Legend: `churn_label` |
| Senior Citizen Churn Rate | Donut chart | Legend: `senior_citizen` (0, 1) Values: churn rate per group |

### Visuals — Behavioral Features

| Visual | Type | Fields |
|---|---|---|
| Support Calls vs Churn | Scatter plot | X: `gold_feature_snapshot[support_count_30d]` Y: `gold_feature_snapshot[complaint_count_30d]` Size: `monthly_charges` Color: `churn_label` from training_dataset (join on customer_id, snapshot_date) |
| Feature Correlation Matrix | Matrix / heatmap (use Python visual or R visual) | Features: `tenure`, `monthly_charges`, `support_count_30d`, `complaint_count_30d`, `payment_failures_30d`, `latest_usage_gb`, `days_since_last_activity` Values: Pearson correlation |
| Usage Trend | Box plot or violin | X: `churn_label` Y: `latest_usage_gb` from gold_feature_snapshot Shows usage distribution for churned vs retained |
| Days Since Last Activity | Histogram | X: `gold_feature_snapshot[days_since_last_activity]` binned Y: Count Color: `churn_label` Insight: churned customers tend to have higher inactivity |
| Complaint Severity Impact | Stacked bar | X: churn_label Y: Count Legend: has_high_complaint (derived: `complaint_high_30d > 0`) |

### Visuals — Label & Class Balance

| Visual | Type | Fields |
|---|---|---|
| Class Balance | Donut chart | Source: `training_dataset` Values: Count of `customer_id` Legend: `churned` (0=Retained, 1=Churned) Show: percentage labels |
| Churn Rate by Snapshot Date | Line chart | X: `training_dataset[snapshot_date]` Y: `DIVIDE(COUNTROWS(FILTER(training_dataset, training_dataset[churned] = 1)), COUNTROWS(training_dataset))` Shows label stability across backfill dates |

### Slicers

- `contract` (dropdown)
- `internet_service` (dropdown)
- `payment_method` (dropdown)
- `senior_citizen` (dropdown: 0, 1)
- `snapshot_date` (date range — for feature snapshot visuals)

---

## Page 2 — Churn Predictions Overview

**Source table:** `main.churn.customer_churn_scores`

### KPI Cards (top row)

| Card | Field / DAX Measure |
|---|---|
| Total Customers | `COUNTROWS(customer_churn_scores)` |
| Overall Churn Rate | `DIVIDE(COUNTROWS(FILTER(customer_churn_scores, customer_churn_scores[churn_prediction] = 1)), COUNTROWS(customer_churn_scores))` |
| Avg Churn Probability | `AVERAGE(customer_churn_scores[churn_probability])` |
| High-Risk Customers | `COUNTROWS(FILTER(customer_churn_scores, customer_churn_scores[risk_segment] = "High"))` |
| Total Revenue at Risk | `SUM(customer_churn_scores[annual_revenue_at_risk])` |

### Visuals

| Visual | Type | Fields |
|---|---|---|
| Probability Distribution | Histogram (column chart) | X: `churn_probability` binned into 10 buckets (0.0–0.1, 0.1–0.2, …) Y: Count of customers |
| Risk Segment Breakdown | Donut chart | Legend: `risk_segment` Values: Count of `customer_id` Colors: High=#E74C3C, Medium=#F39C12, Low=#27AE60 |
| Top 20 At-Risk Customers | Table | Columns: `customer_id`, `churn_probability`, `risk_segment`, `contract`, `tenure`, `monthly_charges`, `annual_revenue_at_risk` Sort: `churn_probability` DESC, Top N = 20 Conditional formatting: `churn_probability` color scale (green→red) |
| Churn by Contract Type | Stacked bar | X: `contract` (Month-to-month, One year, Two year) Y: Count of `customer_id` Legend: `risk_segment` |

### Slicers

- `risk_segment` (dropdown)
- `contract` (dropdown)
- `selected_algorithm` (dropdown)
- `scoring_run_id` (dropdown — lets user compare scoring runs)

---

## Page 3 — Feature & Prediction Drift

**Source table:** `main.churn.drift_metrics`

### KPI Cards (top row)

| Card | DAX Measure |
|---|---|
| Features Monitored | `COUNTROWS(FILTER(drift_metrics, drift_metrics[feature_type] <> "prediction"))` |
| Significant Drift | `COUNTROWS(FILTER(drift_metrics, drift_metrics[alert_level] = "significant"))` |
| Moderate Drift | `COUNTROWS(FILTER(drift_metrics, drift_metrics[alert_level] = "moderate"))` |
| Latest Snapshot | `MAX(drift_metrics[snapshot_date])` |

### Visuals

| Visual | Type | Fields |
|---|---|---|
| PSI by Feature | Bar chart | X: `feature_name` Y: `psi` Sort: `psi` DESC Reference lines: 0.10 (yellow, dashed, label "Moderate"), 0.25 (red, dashed, label "Significant") Conditional formatting: bars colored by `alert_level` (none=#27AE60, moderate=#F39C12, significant=#E74C3C) |
| Drift Alert Summary | Table | Columns: `feature_name`, `feature_type`, `psi`, `ks_statistic`, `alert_level`, `baseline_mean`, `current_mean`, `baseline_count`, `current_count` Conditional formatting: `alert_level` cell background (none=green, moderate=yellow, significant=red) Sort: `psi` DESC |
| PSI Trend Over Time | Line chart | X: `snapshot_date` Y: `psi` Legend: `feature_name` (filter to top 5 by max PSI) Reference line: 0.25 (red) |
| Prediction Drift Card | Card / KPI | Filter: `feature_name = "_prediction_churn_probability"` Value: `psi` Subtitle: `alert_level` Color: conditional on alert_level |

### Slicers

- `snapshot_date` (date range)
- `feature_type` (dropdown: numeric, categorical, prediction)
- `alert_level` (dropdown)

---

## Page 4 — Pipeline Operations

**Source tables:** `main.churn.pipeline_runs`, `main.churn.data_quality_metrics`

### KPI Cards (top row)

| Card | DAX Measure |
|---|---|
| Total Runs | `COUNTROWS(pipeline_runs)` |
| Success Rate | `DIVIDE(COUNTROWS(FILTER(pipeline_runs, pipeline_runs[status] = "succeeded")), COUNTROWS(pipeline_runs))` — format as % |
| Avg Duration (sec) | `AVERAGE(pipeline_runs[duration_seconds])` |
| Failed Runs | `COUNTROWS(FILTER(pipeline_runs, pipeline_runs[status] = "failed"))` |

### Visuals — Pipeline Runs

| Visual | Type | Fields |
|---|---|---|
| Runs by Task & Status | Stacked bar chart | X: `started_at` (date hierarchy — day) Y: Count of `run_id` Legend: `status` Colors: succeeded=#27AE60, failed=#E74C3C, skipped=#95A5A6 |
| Task Duration Trend | Line chart | X: `started_at` (date) Y: `duration_seconds` Legend: `task_name` |
| Recent Runs | Table | Columns: `task_name`, `run_id`, `status`, `started_at`, `finished_at`, `duration_seconds`, `error_message` Sort: `started_at` DESC, Top N = 50 Conditional formatting: `status` cell (succeeded=green, failed=red, skipped=gray) |

### Visuals — Data Quality

| Visual | Type | Fields |
|---|---|---|
| Quality Pass Rate | Gauge or KPI | Value: `DIVIDE(SUM(data_quality_metrics[passing_rows]), SUM(data_quality_metrics[total_rows]))` Target: 0.99 |
| Violations by Rule | Bar chart | X: `rule_name` Y: `SUM(violating_rows)` Color: by `severity` (fail=red, quarantine=orange, warn=yellow) |
| Quality Over Time | Line chart | X: `evaluated_at` (date) Y: `DIVIDE(SUM(passing_rows), SUM(total_rows))` — pass rate |

### Slicers

- `task_name` (dropdown)
- `status` (dropdown)
- `started_at` (date range)
- `stage` (dropdown — for data quality: ingest, transform)

---

## Page 5 — Model Governance

**Source tables:** `main.churn.model_promotion_history`, `main.churn.model_validation_metrics`

### KPI Cards (top row)

| Card | Field / DAX |
|---|---|
| Current Champion Version | `LOOKUPVALUE(model_promotion_history[candidate_version], model_promotion_history[evaluated_at], MAX(model_promotion_history[evaluated_at]))` filtered to `decision = "APPROVED"` |
| Latest PR-AUC | Latest approved `candidate_pr_auc` |
| Latest Recall | Latest approved `candidate_recall` |
| Total Promotions | `COUNTROWS(FILTER(model_promotion_history, model_promotion_history[decision] = "APPROVED"))` |

### Visuals

| Visual | Type | Fields |
|---|---|---|
| Promotion History | Table | Columns: `candidate_version`, `previous_champion_version`, `candidate_pr_auc`, `champion_pr_auc`, `candidate_recall`, `champion_recall`, `decision`, `decision_reason`, `evaluated_at` Sort: `evaluated_at` DESC Conditional formatting: `decision` (APPROVED=green, REJECTED=red) |
| PR-AUC Over Versions | Line + clustered column | X: `candidate_version` Column: `candidate_pr_auc` (blue) Line: `champion_pr_auc` (gray dashed) — shows improvement over time |
| Recall Over Versions | Line chart | X: `candidate_version` Y: `candidate_recall` Reference line: minimum acceptable recall threshold |
| Contract Compatibility | Table | Columns: `candidate_version`, `candidate_inference_contract`, `previous_champion_inference_contract`, `contract_upgrade` Conditional formatting: `contract_upgrade` (true=yellow flag) |

### Slicers

- `decision` (dropdown: APPROVED, REJECTED)
- `evaluated_at` (date range)
- `model_name` (dropdown)

---

## Theme & Formatting

- **Color palette:** #2C3E50 (dark blue-gray background), #ECF0F1 (light text), #27AE60 (green/healthy), #F39C12 (yellow/warning), #E74C3C (red/alert), #3498DB (blue/accent)
- **Font:** Segoe UI, 10pt body, 24pt KPI values
- **Header:** each page has a title bar with page name + last refreshed timestamp
- **Report title:** "Telco Churn — ML Operations Dashboard"
- **Number formatting:** probabilities as 0.00, percentages as 0.0%, durations as #,##0.0 sec, revenue as $#,##0
