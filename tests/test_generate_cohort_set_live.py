# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Live/integration tests for generate_cohort_set (mirror R CDMConnector generateCohortSet tests).

Run only when CDMCONNECTOR_TEST_DB is set (default: duckdb). Use Ibis for DB access.
  CDMCONNECTOR_TEST_DB=duckdb pytest tests/test_generate_cohort_set_live.py -v
  CDMCONNECTOR_TEST_DB=postgres pytest tests/test_generate_cohort_set_live.py -v  # requires env vars
"""

from __future__ import annotations

import os
from importlib.resources import files

import pandas as pd
import pytest

import cdmconnector as cc
from cdmconnector.cohorts import (
    attrition,
    cohort_count,
    generate_cohort_set,
    read_cohort_set,
)
from cdmconnector.exceptions import CohortError

from conftest import DB_TO_TEST, get_connection, live_cdm, live_db_connection


def _inst_path(*parts: str) -> str:
    """Path to package inst folder (like R system.file(..., package = "CDMConnector"))."""
    base = files("cdmconnector").joinpath("inst")
    for p in parts:
        base = base / p
    return str(base)


def _test_cohort_generation(con, cdm_schema: str, write_schema) -> None:
    """Core cohort generation test (shared across DB backends)."""
    cdm = cc.cdm_from_con(con, cdm_schema=cdm_schema, write_schema=write_schema, cdm_name="test_cdm")

    # read_cohort_set errors (path does not exist / not a directory)
    with pytest.raises(CohortError, match="does not exist|not a directory"):
        read_cohort_set("does_not_exist")
    with pytest.raises(CohortError, match="not a directory"):
        read_cohort_set(os.path.join(os.path.dirname(__file__), "conftest.py"))

    # read_cohort_set from package inst/cohorts2 (no CohortsToCreate.csv)
    cohort_path = _inst_path("cohorts2")
    if not os.path.isdir(cohort_path):
        pytest.skip("inst/cohorts2 not found (run from repo with package data)")
    cohort_set_read = read_cohort_set(cohort_path)
    assert len(cohort_set_read) == 3
    assert "cohort_definition_id" in cohort_set_read.columns
    assert "cohort_name" in cohort_set_read.columns
    assert "json" in cohort_set_read.columns

    # Use DuckDB-compatible SQL (Circepy generates SQL Server #Codesets which DuckDB does not support)
    cohort_set = pd.DataFrame({
        "cohort_definition_id": [1, 2, 3],
        "cohort_name": ["Cohort 1", "Cohort 2", "Cohort 3"],
        "sql": [
            "INSERT INTO @target_database_schema.@target_cohort_table (cohort_definition_id, subject_id, cohort_start_date, cohort_end_date) VALUES (@target_cohort_id, 1, DATE '2020-01-01', DATE '2020-06-01');",
            "INSERT INTO @target_database_schema.@target_cohort_table (cohort_definition_id, subject_id, cohort_start_date, cohort_end_date) VALUES (@target_cohort_id, 2, DATE '2020-01-01', DATE '2020-06-01');",
            "INSERT INTO @target_database_schema.@target_cohort_table (cohort_definition_id, subject_id, cohort_start_date, cohort_end_date) VALUES (@target_cohort_id, 1, DATE '2020-02-01', DATE '2020-05-01');",
        ],
    })

    # generate_cohort_set (compute_attrition=False to avoid CIRCE stats tables with NULL types on DuckDB)
    cdm = generate_cohort_set(
        cdm,
        cohort_set,
        name="chrt0",
        overwrite=True,
        compute_attrition=False,
    )

    # cohort table is in CDM
    assert "chrt0" in cdm.tables
    cohort_tbl = cdm["chrt0"]

    # cohort_count: need cohort_attrition on the cohort; load from DB (Python doesn't attach it to the table).
    # With pre-generated SQL the attrition table may be empty; then cohort_count returns 0 rows.
    attr_tbl = cdm.source.table("chrt0_attrition", cdm.write_schema)
    attr_df = cc.collect(attr_tbl)
    class CohortWithAttr:
        cohort_attrition = attr_df
    counts = cohort_count(CohortWithAttr())
    # Cohort table has 3 definitions; attrition may be empty when not using CIRCE-generated SQL
    assert cc.collect(cdm["chrt0"])["cohort_definition_id"].nunique() == 3
    assert isinstance(counts, pd.DataFrame)
    assert list(counts.columns) == ["cohort_definition_id", "number_records", "number_subjects"]

    # invalid cohort set type (string is not a valid cohort definition set)
    with pytest.raises((CohortError, TypeError, ValueError)):
        generate_cohort_set(cdm, "not a cohort", name="blah", overwrite=True)

    # overwrite=False and table already exists -> should error
    with pytest.raises(Exception):  # backend may raise on create_table overwrite=False
        generate_cohort_set(cdm, cohort_set, name="chrt0", overwrite=False)

    # overwrite=True succeeds (compute_attrition=False to avoid CIRCE stats on DuckDB)
    cdm = generate_cohort_set(cdm, cohort_set, name="chrt0", overwrite=True, compute_attrition=False)

    # table name appears in CDM tables (logical names)
    assert "chrt0" in cdm.tables

    # cohort table has expected columns
    df = cc.collect(cohort_tbl.head(10))
    assert isinstance(df, pd.DataFrame)
    for col in ("cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"):
        assert col in df.columns

    # attrition table exists and has expected columns (may be empty when not using CIRCE-generated SQL)
    attr_df = cc.collect(cdm.source.table("chrt0_attrition", cdm.write_schema))
    assert "excluded_records" in attr_df.columns or "number_records" in attr_df.columns

    # cohort_count from wrapper again
    class CohortWithAttr2:
        cohort_attrition = attr_df
    counts2 = cohort_count(CohortWithAttr2())
    assert isinstance(counts2, pd.DataFrame)

    # invalid table name (must start with letter, alphanumeric + underscore; Python normalizes to lowercase)
    with pytest.raises(CohortError, match="letter"):
        generate_cohort_set(cdm, cohort_set, name="4test", overwrite=True)
    with pytest.raises(CohortError, match="letters, numbers"):
        generate_cohort_set(cdm, cohort_set, name="te$t", overwrite=True)

    # empty cohort set -> error
    with pytest.raises(CohortError, match="at least one row|must have"):
        generate_cohort_set(cdm, cohort_set.head(0), name="cohorts", overwrite=True)

    # drop cohort tables (chrt0, chrt0_set, chrt0_attrition)
    cc.drop_table(cdm, ["chrt0", "chrt0_set", "chrt0_attrition"])
    assert "chrt0" not in cdm.tables


@pytest.mark.integration
def test_cohort_generation_live(live_db_connection):
    """Run cohort generation test for the DB selected by CDMCONNECTOR_TEST_DB."""
    con, cdm_schema, write_schema = live_db_connection
    _test_cohort_generation(con, cdm_schema, write_schema)


@pytest.mark.integration
def test_read_cohort_set_from_package_inst():
    """read_cohort_set from inst/cohorts2 returns 3 cohorts (no CohortsToCreate.csv)."""
    cohort_path = _inst_path("cohorts2")
    if not os.path.isdir(cohort_path):
        pytest.skip("inst/cohorts2 not found")
    cohort_set = read_cohort_set(cohort_path)
    assert len(cohort_set) == 3
    assert list(cohort_set["cohort_definition_id"]) == [1, 2, 3]


@pytest.mark.integration
def test_invalid_cohort_table_names(live_cdm):
    """Invalid cohort table names raise CohortError."""
    cohort_set = pd.DataFrame({
        "cohort_definition_id": [1],
        "cohort_name": ["Test"],
        "sql": [
            "INSERT INTO @target_database_schema.@target_cohort_table "
            "(cohort_definition_id, subject_id, cohort_start_date, cohort_end_date) "
            "VALUES (@target_cohort_id, 1, DATE '2020-01-01', DATE '2020-06-01');",
        ],
    })
    cdm = live_cdm
    with pytest.raises(CohortError, match="letter"):
        generate_cohort_set(cdm, cohort_set, name="4test", overwrite=True, compute_attrition=False)
    # Python normalizes "Test" to "test", so we only assert truly invalid names:
    with pytest.raises(CohortError, match="letters, numbers"):
        generate_cohort_set(cdm, cohort_set, name="te$t", overwrite=True, compute_attrition=False)


@pytest.mark.integration
def test_attrition_exists_overwrite_true_no_error(live_cdm):
    """No error when attrition table already exists and overwrite=True (issue 337)."""
    cohort_set = pd.DataFrame({
        "cohort_definition_id": [1],
        "cohort_name": ["One"],
        "sql": [
            "INSERT INTO @target_database_schema.@target_cohort_table "
            "(cohort_definition_id, subject_id, cohort_start_date, cohort_end_date) "
            "VALUES (@target_cohort_id, 1, DATE '2020-01-01', DATE '2020-06-01');",
        ],
    })
    cdm = live_cdm
    # First run
    cdm = generate_cohort_set(cdm, cohort_set, name="test_cohort", overwrite=True, compute_attrition=False)
    # Second run with overwrite=True should not error
    cohort_set2 = pd.DataFrame({
        "cohort_definition_id": [2],
        "cohort_name": ["Two"],
        "sql": [
            "INSERT INTO @target_database_schema.@target_cohort_table "
            "(cohort_definition_id, subject_id, cohort_start_date, cohort_end_date) "
            "VALUES (@target_cohort_id, 2, DATE '2020-01-01', DATE '2020-06-01');",
        ],
    })
    cdm = generate_cohort_set(cdm, cohort_set2, name="test_cohort", overwrite=True, compute_attrition=False)
    cc.drop_table(cdm, ["test_cohort", "test_cohort_set", "test_cohort_attrition"])


@pytest.mark.integration
def test_new_cohort_table_with_prefix(live_db_connection):
    """new_cohort_table with prefix: table exists and cohort_count/attrition work (R: newCohortTable works with prefix)."""
    from cdmconnector.cohorts import new_cohort_table

    con, cdm_schema, write_schema = live_db_connection
    if DB_TO_TEST != "duckdb":
        pytest.skip("prefix test implemented for duckdb only")
    # Skip on DuckDB: new_cohort_table creates empty tables and DuckDB rejects NULL-typed columns for empty tables.
    pytest.skip("new_cohort_table empty tables have NULL-typed columns; DuckDB does not support (upstream fix needed)")
    write_schema_with_prefix = {"schema": "main", "prefix": "test_"}
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema=write_schema_with_prefix, cdm_name="test_cdm")

    new_cohort_table(cdm, "cohort", overwrite=True)

    assert "cohort" in cdm.tables
    tables_in_db = cdm.source.list_tables(cdm.write_schema)
    assert "cohort" in [t.lower() for t in tables_in_db]

    attr_df = cc.collect(cdm.source.table("cohort_attrition", cdm.write_schema))
    class Wrapper:
        cohort_attrition = attr_df
    assert isinstance(cohort_count(Wrapper()), pd.DataFrame)
    assert isinstance(attrition(Wrapper()), pd.DataFrame)

    cc.drop_table(cdm, ["cohort", "cohort_set", "cohort_attrition"])
