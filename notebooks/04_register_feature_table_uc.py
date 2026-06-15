# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Register Feature Table UC

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

SOURCE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_feature_table"
UC_FEATURE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_features_uc"

source_df = spark.table(SOURCE_TABLE)

(
    source_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(UC_FEATURE_TABLE)
)

try:
    spark.sql(f"ALTER TABLE {UC_FEATURE_TABLE} ALTER COLUMN loan_id SET NOT NULL")
except Exception as exc:
    print(f"Could not set NOT NULL on loan_id: {exc}")

try:
    spark.sql(f"""
        ALTER TABLE {UC_FEATURE_TABLE}
        ADD CONSTRAINT credit_risk_features_uc_pk
        PRIMARY KEY (loan_id)
    """)
except Exception as exc:
    print(f"Primary key may already exist or is not supported: {exc}")

try:
    spark.sql(f"""
        ALTER TABLE {UC_FEATURE_TABLE}
        SET TBLPROPERTIES (
            'domain' = 'banking',
            'use_case' = 'credit_risk',
            'business_owner' = 'mlops',
            'feature_table_type' = 'formal_uc_feature_table'
        )
    """)
except Exception as exc:
    print(f"Could not set table properties: {exc}")

feature_engineering_client_available = False
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    feature_engineering_client_available = True
    fe = FeatureEngineeringClient()
except Exception as exc:
    print(f"FeatureEngineeringClient not available. Continuing with UC Delta Feature Table. Details: {exc}")

record_count = spark.table(UC_FEATURE_TABLE).count()
message = (
    f"FORMAL_FEATURE_TABLE_READY: {UC_FEATURE_TABLE}, "
    f"records={record_count}, "
    f"feature_engineering_client_available={feature_engineering_client_available}"
)

dbutils.notebook.exit(message)
