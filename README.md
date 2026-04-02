# CDMConnector (Python)

> **Early development — not production-ready.** This package is an AI-generated port (see [Attribution](#attribution)) and is under active development. You are welcome to install it, try it out, and [open issues](https://github.com/OHDSI/pyCDMConnector/issues) with bug reports or feedback. Please do not use it in production or for research that depends on correct results.

<!-- badges: start -->
[![duckdb status](https://github.com/OHDSI/pyCDMConnector/workflows/duckdb-test/badge.svg)](https://github.com/OHDSI/pyCDMConnector/actions?query=workflow%3Aduckdb-test)
[![Postgres status](https://github.com/OHDSI/pyCDMConnector/workflows/postgres-test/badge.svg)](https://github.com/OHDSI/pyCDMConnector/actions?query=workflow%3Apostgres-test)
[![SQL Server status](https://github.com/OHDSI/pyCDMConnector/workflows/sqlserver-test/badge.svg)](https://github.com/OHDSI/pyCDMConnector/actions?query=workflow%3Asqlserver-test)
[![Redshift status](https://github.com/OHDSI/pyCDMConnector/workflows/redshift-test/badge.svg)](https://github.com/OHDSI/pyCDMConnector/actions?query=workflow%3Aredshift-test)
[![Snowflake status](https://github.com/OHDSI/pyCDMConnector/workflows/snowflake-test/badge.svg)](https://github.com/OHDSI/pyCDMConnector/actions?query=workflow%3Asnowflake-test)
[![Spark status](https://github.com/OHDSI/pyCDMConnector/workflows/spark-test/badge.svg)](https://github.com/OHDSI/pyCDMConnector/actions?query=workflow%3Aspark-test)
[![codecov](https://codecov.io/gh/OHDSI/pyCDMConnector/graph/badge.svg)](https://codecov.io/gh/OHDSI/pyCDMConnector)
<!-- badges: end -->

AI-generated Python port of several R packages from the [DARWIN EU](https://darwin-eu.org/) ecosystem, primarily [CDMConnector](https://github.com/darwin-eu/CDMConnector), [omopgenerics](https://github.com/darwin-eu/omopgenerics), [PatientProfiles](https://github.com/darwin-eu/PatientProfiles), [CohortCharacteristics](https://github.com/darwin-eu/CohortCharacteristics), and [visOmopResults](https://github.com/darwin-eu/visOmopResults). Connects to an **OMOP Common Data Model** using [Ibis](https://ibis-project.org/) for lazy, backend-agnostic SQL (replacing dplyr/dbplyr from R). All transformations are lazy until you call **`collect()`** (to a pandas DataFrame) or **`compute()`** (to a table).

**Requirements:** Python 3.10+. Ibis with DuckDB is included; 

## Features

- **Cdm** — Single object holding OMOP table references (Ibis tables) plus metadata (name, version, schemas).
- **Lazy-by-default** — Build queries with Ibis; materialize only with `collect()` or `compute()`.
- **Cohort tables** — OMOP cohort structure, attrition, and `generate_cohort_set` from CIRCE definitions.
- **add_{} functions** — `add_demographics`, `add_age`, `add_sex`, ect to add columns to cohort tables.
- **Eunomia example datasets** — Helpers to download and use example OMOP datasets (e.g. GiBleed).
- **Schema-aware** — `cdm_schema`, `write_schema`, optional prefix; works with Ibis backends (DuckDB, Postgres, etc.).

## Install

Install directly from GitHub:

```bash
pip install git+https://github.com/OHDSI/pyCDMConnector.git
```

For development (editable install with test/doc dependencies):

```bash
git clone https://github.com/OHDSI/pyCDMConnector.git
cd pyCDMConnector
pip install -e ".[dev]"
```

To use a database backend other than DuckDB, install the corresponding extra:

```bash
pip install "cdmconnector[postgres]"    # PostgreSQL
pip install "cdmconnector[snowflake]"   # Snowflake
pip install "cdmconnector[bigquery]"    # BigQuery
```

## Setting up `EUNOMIA_DATA_FOLDER`

The **`EUNOMIA_DATA_FOLDER`** environment variable tells pyCDMConnector where to cache downloaded example datasets (zip files and DuckDB databases). It must be set before using `eunomia_dir()` or any of the example datasets.

```bash
# Set for the current session
export EUNOMIA_DATA_FOLDER=/path/to/eunomia_data

# Or add to your shell profile (~/.zshrc, ~/.bashrc) for persistence
echo 'export EUNOMIA_DATA_FOLDER=/path/to/eunomia_data' >> ~/.zshrc
```

On first use, `eunomia_dir()` downloads the dataset zip, extracts the parquet files, and builds a DuckDB database in this folder. Subsequent calls use the cached data. The function always returns a path to a *copy* of the DB so the cache is never modified.

> **Note:** Python and R CDMConnector can share the same `EUNOMIA_DATA_FOLDER`. The zip files are shared, and the DuckDB files use different names (`GiBleed_5.3_py.duckdb` for Python vs `GiBleed_5.3.duckdb` for R) so they don't conflict.

## Eunomia example datasets

```python
import cdmconnector as cc

# List all available datasets
cc.example_datasets()
# ('GiBleed', 'synthea-allergies-10k', 'synthea-anemia-10k', ..., 'synpuf-1k', 'empty_cdm', ...)
```

Most datasets are available in CDM version 5.3. A few (`synpuf-1k`, `empty_cdm`, `Synthea27NjParquet`) are also available in 5.4.

## Quick start with Eunomia GiBleed

```python
import cdmconnector as cc
import ibis

path = cc.eunomia_dir("GiBleed", cdm_version="5.3")  # prints data folder path
con = ibis.duckdb.connect(path)
cdm = cc.cdm_from_con(
    con,
    cdm_schema="main",
    write_schema="main",
    cdm_name="eunomia",
)
print(cdm)
```
```
# OMOP CDM reference (duckdb) of eunomia

  omop tables: care_site, cdm_source, concept, concept_ancestor, concept_relationship, concept_synonym, condition_era, condition_occurrence, cost, death, device_exposure, dose_era, drug_era, drug_exposure, drug_strength, fact_relationship, location, measurement, note, note_nlp, observation, observation_period, payer_plan_period, person, procedure_occurrence, provider, specimen, visit_detail, visit_occurrence, vocabulary
  cohort tables: -
  achilles tables: -
  other tables: -
```

```python
# List CDM tables, snapshot, run a query
print(cc.cdm_tables(cdm))
snap = cdm.snapshot()
print(snap[["cdm_name", "person_count", "cdm_version"]].to_string())
df = cc.collect(cdm.person.limit(5))
print(df)
```

**If you see `CDMValidationError: No CDM tables found in the given schema`:** (1) Ensure you have the latest package: `pip install -e .` from the repo root (Eunomia needs tables persisted to the DuckDB file). (2) If you already had Eunomia data, delete the existing `GiBleed_5.3.duckdb` (and optionally `GiBleed_5.3.zip`) in that folder so it is rebuilt with tables.

**To see what `eunomia_dir` is doing** (data folder, zip/DB existence, download/build): run `import logging; logging.basicConfig(level=logging.INFO)` before calling `eunomia_dir`.

## Overview of main functions

### Core: CDM reference and execution

| Function | Description |
|----------|-------------|
| **`cdm_from_con(con, cdm_schema, write_schema=None, ...)`** | Build a `Cdm` from an Ibis connection (e.g. DuckDB). |
| **`cdm_from_tables(tables, cdm_name, cdm_version="5.3", ...)`** | Build a `Cdm` from a dict of table names → Ibis tables or DataFrames (uses in-memory DuckDB if needed). |
| **`cdm_tables(cdm)`** | Return list of available CDM table names (logical). |
| **`cdm.snapshot()`** | Execute and return a one-row **DataFrame** of CDM metadata (counts, version, etc.). |
| **`collect(expr, limit=None)`** | Materialize an Ibis expression to a pandas DataFrame. *Only* place that “pulls” data. |
| **`compute(cdm, expr, name, overwrite=True)`** | Materialize an Ibis expression into a table in the write schema and return an Ibis table reference. |
| **`cdm.disconnect()`** | Disconnect the CDM’s source. |
| **`cdm.subset(person_id)`** | Return a new CDM with all tables filtered to the given person IDs (requires write_schema). |
| **`cdm.subset_cohort(cohort_table="cohort", cohort_id=None, verbose=False)`** | Return a new CDM filtered to persons in the given cohort table (optionally by cohort_id). |
| **`cdm.sample(n, seed=None, name="person_sample")`** | Return a new CDM with a random sample of *n* persons (adds the sample table to the CDM). |
| **`cdm.flatten(domain=None, include_concept_name=True)`** | Return a lazy single observation table (union of selected domain tables); use **`collect()`** to materialize. |

### Cohorts

| Function | Description |
|----------|-------------|
| **`generate_cohort_set(cdm, cohort_definition_set, name="cohort", ...)`** | Create cohort tables from a CIRCE cohort definition set (e.g. from `read_cohort_set`). |

### Patient profiles

| Function | Description |
|----------|-------------|
| **`add_demographics(table, cdm, index_date="cohort_start_date", ...)`** | Add age, sex, prior/future observation (and optional date of birth) by joining person and observation_period. Returns an Ibis table expression. |
| **`add_age(table, cdm, ...)`** | Add age at index date. |
| **`add_sex(table, cdm, ...)`** | Add sex from person. |

### Lazy vs materialized

- **Lazy:** `cdm.person`, `add_demographics(...)` return Ibis expressions. No query runs until you call **`collect()`** or **`compute()`**.
- **Materialize:** Use **`collect(expr)`** to get a pandas DataFrame, or **`compute(cdm, expr, name)`** to write a table. **`cdm.snapshot()`** executes immediately and returns a one-row DataFrame.

## Examples using GiBleed

All of the following assume you have already run the Quick start with Eunomia GiBleed above (so `cdm` and `con` are defined).

### List tables and inspect person

```python
print(cc.cdm_tables(cdm))
person_df = cc.collect(cdm.person.limit(10))
print(person_df.head())
```

### Add demographics to a cohort-like table and collect

Build a simple “cohort” from person, add demographics, then materialize:

```python
cohort = (
    cdm.person
    .mutate(
        cohort_definition_id=1,
        subject_id=cdm.person.person_id,
        cohort_start_date=ibis.date(2020, 1, 1),
        cohort_end_date=ibis.date(2020, 6, 1),
    )
)
with_demographics = cc.add_demographics(cohort, cdm, index_date="cohort_start_date")
df = cc.collect(with_demographics.limit(20))
print(df[["person_id", "age", "sex", "prior_observation", "future_observation"]].head())
```

### Persist a result with compute

```python
with_demographics = cc.add_demographics(cohort, cdm, index_date="cohort_start_date")
tbl = cc.compute(cdm, with_demographics, "my_demographics", overwrite=True)
# tbl is an Ibis table reference to the new table
print(cc.collect(tbl.limit(5)))
```

### Snapshot and export

```python
snap = cdm.snapshot()  # one-row DataFrame (cdm_name, person_count, dates, etc.)
print(snap[["cdm_name", "person_count", "cdm_version"]])
# To persist as a table: cc.compute(cdm, ibis.memtable(snap.to_dict("records")), "cdm_snapshot_table", overwrite=True)
```

### Example datasets

```python
from cdmconnector.eunomia import example_datasets
print(example_datasets())  # e.g. ('GiBleed', 'synpuf-1k', ...)
```

## Optional backends

Default install includes DuckDB. For other databases:

```bash
pip install -e ".[postgres]"    # PostgreSQL
pip install -e ".[snowflake]"   # Snowflake
pip install -e ".[bigquery]"    # Google BigQuery
```

Then use the matching Ibis connection (e.g. `ibis.postgres.connect(...)`) with **`cdm_from_con()`**.

## Documentation

- [Architecture](https://darwin-eu.github.io/CDMConnector/architecture/): evaluation model, public API, table naming (prefix/schema).
- Full docs: [https://darwin-eu.github.io/CDMConnector/](https://darwin-eu.github.io/CDMConnector/).

### Build locally

Source lives in `docs-src/` (Quarto `.qmd`, notebooks, assets). The built site is written to `docs/` (for GitHub Pages). To build:

```bash
pip install -e ".[docs]"
cd docs-src && quartodoc build
quarto render docs-src/
```

Or use the Makefile: `make docs` (runs quartodoc + quarto render and ensures `docs/.nojekyll` exists). **Quarto CLI** must be installed separately ([quarto.org](https://quarto.org)); it is not installed via pip.

### Publish (GitHub Pages from `/docs`)

GitHub Pages is configured to serve from the **`/docs` folder** on the default branch. Publishing:

1. Run `make docs` (or `quartodoc build` then `quarto render docs-src/`).
2. Commit the updated `docs/` folder (optional; the [Docs workflow](.github/workflows/docs.yml) also builds and deploys on push to `main`).
3. Push to the default branch.

**Note:** The output directory is `docs/` (built HTML and assets). `docs-src/` contains only the Quarto source.

## Development

```bash
pip install -e ".[dev]"
pytest
```

**Live DB tests (integration):** Tests for `generate_cohort_set` run against a database selected by **`CDMCONNECTOR_TEST_DB`** (default: `duckdb`). Run them with e.g. `CDMCONNECTOR_TEST_DB=duckdb pytest tests/test_generate_cohort_set_live.py -v -m integration`. Omit `-m integration` to run only unit tests.

## Attribution

This Python package is a port of the following R packages developed as part of [DARWIN EU](https://darwin-eu.org/). We acknowledge the original authors and maintainers and their source work:

| R package | Description | Links |
|-----------|-------------|--------|
| **[CDMConnector](https://github.com/darwin-eu/CDMConnector)** | Connect to an OMOP Common Data Model (dplyr/dbplyr) | [Docs](https://darwin-eu.github.io/CDMConnector/) · [Repo](https://github.com/darwin-eu/CDMConnector) |
| **[omopgenerics](https://github.com/darwin-eu/omopgenerics)** | Core classes and methods for OMOP CDM pipelines | [Docs](https://darwin-eu.github.io/omopgenerics/) · [Repo](https://github.com/darwin-eu/omopgenerics) |
| **[PatientProfiles](https://github.com/darwin-eu/PatientProfiles)** | Patient-level demographics and cohort/concept/table intersections | [Docs](https://darwin-eu.github.io/PatientProfiles/) · [Repo](https://github.com/darwin-eu/PatientProfiles) |
| **[CohortCharacteristics](https://github.com/darwin-eu/CohortCharacteristics)** | Summarise and visualise cohort characteristics | [Docs](https://darwin-eu.github.io/CohortCharacteristics/) · [Repo](https://github.com/darwin-eu/CohortCharacteristics) |
| **[visOmopResults](https://github.com/darwin-eu/visOmopResults)** | Publication-ready tables and plots for OMOP results | [Docs](https://darwin-eu.github.io/visOmopResults/) · [Repo](https://github.com/darwin-eu/visOmopResults) |

The design of CDM references, cohort tables, summarised results, and documentation in this project follows the conventions established by these packages. All authors listed in their DESCRIPTION files are credited as authors of this Python package (see [pyproject.toml](pyproject.toml)).

## License

Apache 2.0. See [LICENSE](LICENSE).
