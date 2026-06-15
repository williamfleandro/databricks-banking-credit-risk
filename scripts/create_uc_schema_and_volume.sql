CREATE SCHEMA IF NOT EXISTS mlops_dev.banking;
CREATE SCHEMA IF NOT EXISTS mlops_acc.banking;
CREATE SCHEMA IF NOT EXISTS mlops_production.banking;

CREATE VOLUME IF NOT EXISTS mlops_dev.banking.raw;
CREATE VOLUME IF NOT EXISTS mlops_acc.banking.raw;
CREATE VOLUME IF NOT EXISTS mlops_production.banking.raw;
