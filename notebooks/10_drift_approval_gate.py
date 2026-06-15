# Databricks notebook source
# MAGIC %md
# MAGIC # 10 Drift Approval Gate
# MAGIC
# MAGIC Projeto: Databricks Banking — Credit Risk / Loan Default

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

DOMAIN = "credit_risk"
TARGET_COLUMN = "loan_status"
RAW_FILE = "credit_risk_dataset.csv"

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Domain: {DOMAIN}")
print(f"Target: {TARGET_COLUMN}")
print("TODO: implementar este notebook seguindo o padrão do projeto Databricks MLOps Churn Lab.")
