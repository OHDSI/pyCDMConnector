# Style Guide

This document defines the code and naming conventions for `pyCDMConnector`.

The short version:

- Use Python-style `snake_case` for functions, variables, modules, file names, table names, and column names we control.
- Prefer lowercase identifiers in code and in database objects.
- Keep the public Python API Pythonic, even when porting ideas from R packages.

## General principles

- Follow standard Python conventions first.
- Prefer consistency over cleverness.
- If an R API is being ported, preserve the behavior and concepts where practical, but express them in Python naming style.
- New public APIs should be designed in `snake_case`, not camelCase.

## Python naming

### Functions and methods

- Use `snake_case`.
- Examples:
  - `cdm_from_con`
  - `generate_cohort_set`
  - `generate_concept_cohort_set`
  - `new_cohort_table`
  - `cohort_collapse`

### Variables

- Use `snake_case`.
- Prefer descriptive names over abbreviations unless the abbreviation is already standard in the project.
- Examples:
  - `cohort_definition_set`
  - `write_schema`
  - `required_observation`
  - `concept_set`

### Classes

- Use `PascalCase`.
- Examples:
  - `Cdm`
  - `Result`
  - `SummarisedResult`
  - `Codelist`

### Constants

- Use `UPPER_SNAKE_CASE`.
- Examples:
  - `COHORT_TABLE_COLUMNS`
  - `EXAMPLE_DATASETS`

### Module and file names

- Use lowercase `snake_case`.
- Examples:
  - `patient_profiles.py`
  - `characteristics.py`
  - `styleguide.md`

## Database object naming

### Table names

- Use lowercase `snake_case`.
- Avoid mixed case, spaces, and punctuation.
- Examples:
  - `person`
  - `observation_period`
  - `condition_occurrence`
  - `gibleed_cohort`
  - `cohort_set`
  - `cohort_attrition`

### Column names

- Use lowercase `snake_case`.
- Follow OMOP names as-is when they already match the convention.
- New derived columns should also use lowercase `snake_case`.
- Examples:
  - `cohort_definition_id`
  - `subject_id`
  - `cohort_start_date`
  - `cohort_end_date`
  - `prior_observation`
  - `future_observation`

### Schema and write-table names

- Prefer lowercase names where the target backend allows it.
- Treat names as case-insensitive unless a specific backend requires otherwise.
- Avoid quoted mixed-case identifiers.

### Temporary tables

- Temporary or helper table names should also be lowercase `snake_case`.
- Prefer a clear prefix such as `tmp_`.

## API design

- Public Python APIs should remain `snake_case` even when the R equivalent is camelCase.
- Examples:
  - Use `new_cohort_table`, not `newCohortTable`
  - Use `generate_concept_cohort_set`, not `generateConceptCohortSet`
  - Use `cohort_collapse`, not `cohortCollapse`

- R package names may still be referenced in prose using their original names:
  - `PatientProfiles`
  - `CohortCharacteristics`
  - `visOmopResults`

## Cohort naming conventions

- Cohort table names should start with a letter.
- Cohort table names should contain only lowercase letters, numbers, and underscores.
- Related cohort metadata tables should follow the standard suffix pattern:
  - `<name>`
  - `<name>_set`
  - `<name>_attrition`

## Docstrings and documentation

- Use sentence case in prose.
- Use backticks for code identifiers in Markdown and docstrings.
- Prefer Python names in docs, examples, and tutorials.
- If mentioning the R equivalent, do so as a reference, not as the primary API name.

## Tests

- Test function names should be `snake_case`.
- Prefer names that describe behavior.
- Examples:
  - `test_new_cohort_table_requires_db_cdm`
  - `test_generate_concept_cohort_set_limit_all`
  - `test_cohort_collapse_merges_overlapping_periods`

## Environment variables

- Use `UPPER_SNAKE_CASE`.
- Preserve established names for external compatibility.
- Examples:
  - `EUNOMIA_DATA_FOLDER`
  - `CDM5_POSTGRESQL_DBNAME`
  - `SNOWFLAKE_ACCOUNT`
  - `DATABRICKS_HTTPPATH`

## SQL and backend compatibility

- Prefer Ibis expressions over handwritten SQL when possible.
- When SQL must be written or rendered, keep placeholders and generated object names lowercase where practical.
- Avoid backend-specific quoting or case-sensitive identifiers unless required.

## When in doubt

- Choose the Pythonic `snake_case` version.
- Choose lowercase for database object names we create.
- Match existing OMOP/CDM naming where the schema already defines it.
