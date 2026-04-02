# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial Python port of R CDMConnector and omopgenerics.
- `Cdm` reference object with table access (`cdm.person`, `cdm["person"]`).
- `cdm_from_con()` and `cdm_from_tables()` to build a CDM from an Ibis connection or in-memory tables.
- `cdm_name()`, `cdm_version()`, `cdm_select()`, `cdm_disconnect()`, `cdm_write_schema()`, `cdm_con()`, `list_tables()`.
- Source operations: `list_source_tables()`, `read_source_table()`, `insert_table()`, `drop_table()`.
- Cohort utilities: `cohort_count()`, `attrition()`, `record_cohort_attrition()`, `new_cohort_table()`.
- Eunomia: `download_eunomia_data()`, `eunomia_dir()`, `eunomia_is_available()`, `example_datasets()`, `require_eunomia()`.
- OMOP schemas: `omop_tables()`, `cohort_columns()`.
- Ibis-based lazy SQL (replacing dplyr/dbplyr).

## [0.1.0] - 2025-01-31

### Added

- First release.
- DuckDB backend supported; Postgres/Snowflake/BigQuery via optional extras.
