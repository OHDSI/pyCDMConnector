# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Live database tests that run across all backends in CDMCONNECTOR_TEST_DB.

These tests mirror the R CDMConnector pattern of looping ``for (dbtype in dbToTest)``.
Each test is parametrised via the ``db_connection`` / ``db_cdm`` fixtures from conftest.py.

Run examples::

    # DuckDB only (default)
    pytest tests/test_cdm_live.py -m integration -v

    # DuckDB + PostgreSQL
    CDMCONNECTOR_TEST_DB=duckdb,postgres pytest tests/test_cdm_live.py -m integration -v

    # PostgreSQL only
    CDMCONNECTOR_TEST_DB=postgres pytest tests/test_cdm_live.py -m integration -v
"""

from __future__ import annotations

import pandas as pd
import pytest

import cdmconnector as cc


# ---------------------------------------------------------------------------
# cdm_from_con: basic construction and table access
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cdm_from_con(db_connection):
    """cdm_from_con creates a CDM with person and observation_period tables."""
    dbms, con, cdm_schema, write_schema = db_connection
    cdm = cc.cdm_from_con(
        con, cdm_schema=cdm_schema, write_schema=write_schema, cdm_name="test_cdm"
    )
    assert "person" in cdm.tables
    assert "observation_period" in cdm.tables


@pytest.mark.integration
def test_cdm_table_access(db_cdm):
    """CDM tables are accessible via attribute and item access."""
    dbms, cdm = db_cdm
    person = cdm.person
    assert person is not None
    person2 = cdm["person"]
    assert person2 is not None


@pytest.mark.integration
def test_cdm_person_collect(db_cdm):
    """Collecting person table returns a DataFrame with required columns."""
    dbms, cdm = db_cdm
    df = cc.collect(cdm.person)
    assert isinstance(df, pd.DataFrame)
    assert "person_id" in df.columns
    assert "gender_concept_id" in df.columns
    assert "year_of_birth" in df.columns
    assert len(df) > 0


@pytest.mark.integration
def test_cdm_observation_period_collect(db_cdm):
    """Collecting observation_period returns a DataFrame with date columns."""
    dbms, cdm = db_cdm
    df = cc.collect(cdm.observation_period)
    assert isinstance(df, pd.DataFrame)
    assert "observation_period_start_date" in df.columns
    assert "observation_period_end_date" in df.columns
    assert len(df) > 0


# ---------------------------------------------------------------------------
# CDM metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cdm_name(db_cdm):
    """CDM name is set correctly."""
    dbms, cdm = db_cdm
    assert cdm.name == "test_cdm"


@pytest.mark.integration
def test_cdm_version(db_cdm):
    """CDM version is a string (5.3 or 5.4)."""
    dbms, cdm = db_cdm
    assert cdm.version in ("5.3", "5.4", None)


@pytest.mark.integration
def test_cdm_snapshot(db_cdm):
    """snapshot() returns a DataFrame summarising the CDM (requires cdm_source table)."""
    dbms, cdm = db_cdm
    if "cdm_source" not in cdm.tables:
        pytest.skip("cdm_source table not present (minimal test data)")
    snap = cdm.snapshot()
    assert isinstance(snap, pd.DataFrame)


# ---------------------------------------------------------------------------
# Table listing and schema introspection
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cdm_tables_list(db_cdm):
    """cdm.tables returns a list containing at least person."""
    dbms, cdm = db_cdm
    tables = cdm.tables
    assert isinstance(tables, list)
    assert "person" in tables


@pytest.mark.integration
def test_source_list_tables(db_connection):
    """source.list_tables returns tables from the CDM schema."""
    dbms, con, cdm_schema, write_schema = db_connection
    cdm = cc.cdm_from_con(
        con, cdm_schema=cdm_schema, write_schema=write_schema, cdm_name="test_cdm"
    )
    db_tables = cdm.source.list_tables(cdm_schema)
    assert isinstance(db_tables, list)
    # person should be in the physical table list (case-insensitive)
    assert any(t.lower() == "person" for t in db_tables)


# ---------------------------------------------------------------------------
# Lazy evaluation contracts
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_lazy_head(db_cdm):
    """head() returns an Ibis expression (lazy), collect materialises it."""
    dbms, cdm = db_cdm
    expr = cdm.person.head(2)
    # Should be an Ibis table expression, not a DataFrame yet
    assert not isinstance(expr, pd.DataFrame)
    df = cc.collect(expr)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2


@pytest.mark.integration
def test_lazy_filter(db_cdm):
    """Filtering returns a lazy expression."""
    dbms, cdm = db_cdm
    expr = cdm.person.filter(cdm.person.year_of_birth > 1980)
    assert not isinstance(expr, pd.DataFrame)
    df = cc.collect(expr)
    assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Write schema operations
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_insert_and_drop_table(db_connection):
    """insert_table + drop_table round-trip works on the write schema."""
    import pyarrow as pa

    dbms, con, cdm_schema, write_schema = db_connection
    cdm = cc.cdm_from_con(
        con, cdm_schema=cdm_schema, write_schema=write_schema, cdm_name="test_cdm"
    )
    test_data = pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    cdm.source.insert_table("_test_live_tbl", test_data, overwrite=True)

    # Table should be readable
    tbl = cdm.source.table("_test_live_tbl", write_schema)
    df = cc.collect(tbl)
    assert len(df) == 3

    # Clean up
    cdm.source.drop_table("_test_live_tbl")
