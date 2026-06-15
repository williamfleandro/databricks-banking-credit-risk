#!/usr/bin/env bash
set -euo pipefail
databricks fs cp data/raw/credit_risk_dataset.csv dbfs:/Volumes/mlops_dev/banking/raw/credit_risk_dataset.csv --overwrite
