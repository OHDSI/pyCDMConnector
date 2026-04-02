# Tutorial notebooks (pyCdmConnector)

This folder contains **Jupyter notebook skeletons** that teach the `cdmconnector` Python package through a story-driven onboarding path. Each notebook follows a consistent structure: **Setup → Explore → Build → Interpret → Exercises → What we learned**.

## Running notebooks locally

### Environment setup

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install the package** with DuckDB (required for Eunomia):

   ```bash
   pip install -e ".[duckdb]"
   ```

   Or from PyPI:

   ```bash
   pip install CDMConnector
   pip install "ibis-framework[duckdb]"
   ```

3. **Optional**: For cohort SQL from JSON (notebook 06), Circepy is a dependency of the package. For pretty tables (notebooks 07, 09), install great-tables:

   ```bash
   pip install great-tables
   ```

4. **Start Jupyter** from the repository root (so paths like `docs-src/assets/cohort_json` resolve):

   ```bash
   jupyter notebook docs-src/notebooks/
   ```

   Or open individual `.ipynb` files from your IDE.

### Datasets used

| Notebook(s) | Dataset(s) |
|-------------|------------|
| 00, 01 | **GiBleed** (small, narrative-friendly) |
| 02, 04, 07, 08 | **synpuf-1k** (richer, realistic distributions) |
| 03, 09 | **GiBleed** |
| 05 | **empty_cdm** (plus optional synthetic inserts into write schema) |
| 06 | **GiBleed** and/or **synpuf-1k**; cohort JSONs in `docs-src/assets/cohort_json/` |

Eunomia data is downloaded on first use via `cc.eunomia_dir("GiBleed")` etc., or you can run `cc.download_eunomia_data("GiBleed")` in advance.

## Rendering notebooks on the website

- **Quarto** (this repo): The docs are built from `docs-src/` with Quarto. Notebooks under `docs-src/notebooks/` are rendered to HTML when you run `quarto render docs-src/`; they appear at `notebooks/*.html` in the deployed site. The nav links in the sidebar point to these rendered notebooks.
- Run `make docs` (or `quartodoc build` then `quarto render docs-src/`) from the repo root to rebuild the full site.

## Optional extras

- **great-tables**: Used in notebooks 07 (Table 1) and 09 (export HTML). If not installed, code falls back to plain pandas DataFrames / CSV.
- **CIRCE / Circepy**: Notebook 06 uses `build_cohort_query` and `render_cohort_sql` from `cdmconnector._circe`; these require the Circepy dependency (shipped with the package). If Circepy is missing, the notebook shows a `NotImplementedError` and explains how to install.
- **SQLGlot**: Notebook 06 (and the WASM tutorial) show how to use **SQLGlot** to translate SQL between dialects (e.g. DuckDB → Postgres). Queries are expressed in **Ibis** by default; use SQLGlot when you have existing SQL strings and need dialect-specific output.

## Notebook list

| # | Notebook | Topic |
|---|----------|--------|
| 00 | [Welcome: Your first CDM](00_welcome_your_first_cdm.ipynb) | Connect, list tables, person count, preview |
| 01 | [Lazy queries and pipelines](01_lazy_queries_and_pipelines.ipynb) | filter/select/join, compile SQL |
| 02 | [OMOP people, time, domains](02_understanding_omop_people_time_domains.ipynb) | Demographics, observation period, visits |
| 03 | [Story: GiBleed end-to-end](03_story_gibleed_end_to_end.ipynb) | Cohort, index date, windowed analysis |
| 04 | [Measurement distributions](04_measurement_distributions_with_guardrails.ipynb) | Numeric stats, missingness, guardrails |
| 05 | [Cohorts 101](05_cohorts_101_tables_attrition_counts.ipynb) | Tables, attrition, counts |
| 06 | [Cohorts from JSON (CIRCE/Atlas)](06_cohorts_from_json_circe_atlas.ipynb) | Load JSON, render SQL, materialise cohort |
| 07 | [Cohort characterization (Table 1)](07_cohort_characterization_table1.ipynb) | Demographics, baseline comorbidities |
| 08 | [Performance: pushdown and materialization](08_performance_pushdown_and_materialization.ipynb) | Lazy pipelines, compile, temp tables |
| 09 | [Exporting results and tables](09_exporting_results_and_tables.ipynb) | CSV/Parquet/HTML, results bundle |

## Troubleshooting

- **Dataset not found**: Run `cc.download_eunomia_data("GiBleed")` (or the dataset name you need) with network access. Ensure `cc.eunomia_dir(...)` is called from a context where the package can write to its cache.
- **Path errors in notebook 06**: Run Jupyter from the **repository root** so `docs-src/assets/cohort_json` exists. Alternatively set `COHORT_JSON_DIR` in the notebook to the full path of `docs-src/assets/cohort_json`.
- **Import errors**: Ensure `cdmconnector` and `ibis` are installed (`pip install -e .` or `pip install CDMConnector` and `ibis-framework[duckdb]`).
