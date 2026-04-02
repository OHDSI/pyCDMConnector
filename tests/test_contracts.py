# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests enforcing lazy-by-default and naming contracts."""

import datetime
import pytest
import ibis
import pandas as pd
import pyarrow as pa

import cdmconnector as cc
from cdmconnector.source import DbCdmSource, full_table_name, resolve_prefix
from cdmconnector.utils import parse_cdm_version, resolve_schema_name, to_dataframe


# ---- Version parsing ----

@pytest.mark.parametrize("ver,expected", [
    ("5.3", "5.3"),
    ("5.4.1", "5.4"),
    ("v5.4", "5.4"),
    ("5.10.2", "5.10"),
    (" 5.4 ", "5.4"),
    ("v5.10", "5.10"),
])
def test_parse_cdm_version(ver, expected):
    assert parse_cdm_version(ver) == expected


def test_parse_cdm_version_invalid():
    assert parse_cdm_version("") == "5.3"
    assert parse_cdm_version("x.y") == "5.3"


def test_parse_cdm_version_none_or_non_string():
    """parse_cdm_version returns 5.3 for None or non-string."""
    assert parse_cdm_version(None) == "5.3"
    assert parse_cdm_version(5) == "5.3"


def test_parse_cdm_version_minor_non_int():
    """parse_cdm_version treats non-int minor as 0 (e.g. '5.x' -> '5.0')."""
    assert parse_cdm_version("5.x") == "5.0"
    assert parse_cdm_version("5.1a") == "5.0"


# ---- Prefix / schema resolution ----

def test_resolve_prefix():
    assert resolve_prefix(None, None) is None
    assert resolve_prefix("cdm_", None) == "cdm_"
    assert resolve_prefix(None, {"schema": "main", "prefix": "cdm_"}) == "cdm_"
    assert resolve_prefix("x_", {"prefix": "cdm_"}) == "cdm_"


def test_full_table_name():
    assert full_table_name("person", None) == "person"
    assert full_table_name("person", "cdm_") == "cdm_person"
    assert full_table_name("cdm_person", "cdm_") == "cdm_person"


def test_resolve_schema_name():
    """resolve_schema_name returns schema string from str or dict."""
    assert resolve_schema_name(None) is None
    assert resolve_schema_name("main") == "main"
    assert resolve_schema_name({"schema": "myschema"}) == "myschema"
    assert resolve_schema_name({"schema_name": "other"}) == "other"
    assert resolve_schema_name({"schema": "", "schema_name": "fallback"}) == "fallback"
    assert resolve_schema_name({}) is None


def test_to_dataframe():
    """to_dataframe materializes Ibis or list/dict to DataFrame."""
    df = to_dataframe([1, 2, 3])
    assert hasattr(df, "columns")
    assert len(df) == 3
    df2 = to_dataframe(pd.DataFrame({"a": [1]}))
    pd.testing.assert_frame_equal(df2, pd.DataFrame({"a": [1]}))


def test_prefix_schema_list_tables_table(duckdb_con):
    """Create tables with prefix cdm_; list_tables returns logical names; table('person') finds cdm_person."""
    con = duckdb_con
    person = pa.table({"person_id": [1, 2], "year_of_birth": [1990, 1985]})
    con.create_table("cdm_person", obj=person, overwrite=True)
    schema = {"schema": "main", "prefix": "cdm_"}
    src = DbCdmSource(con, schema, prefix="cdm_")
    names = src.list_tables()
    assert "person" in names
    tbl = src.table("person")
    assert tbl is not None
    df = cc.collect(tbl)
    assert len(df) == 2


# ---- Lazy contract: no execute outside collect/compute/Result ----

def test_collect_returns_dataframe(cdm_from_duckdb):
    """collect(expr.limit(5)) returns pandas with expected columns."""
    cdm = cdm_from_duckdb
    expr = cdm.person.limit(5)
    df = cc.collect(expr)
    assert isinstance(df, pd.DataFrame)
    assert "person_id" in df.columns
    assert len(df) <= 5


def test_compute_writes_table(cdm_from_duckdb):
    """compute(cdm, expr, 'x') writes a physical table and returns Ibis table."""
    from cdmconnector.cdm import drop_table

    cdm = cdm_from_duckdb
    expr = cdm.person.select("person_id").limit(3)
    out = cc.compute(cdm, expr, "contract_test_table", overwrite=True)
    assert out is not None
    assert "contract_test_table" in cdm.tables
    df = cc.collect(out)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    drop_table(cdm, "contract_test_table")


def test_add_demographics_no_execute(cdm_from_tables_fixture):
    """add_demographics returns Ibis expression; no execution until collect."""
    cdm = cdm_from_tables_fixture
    cohort = (
        cdm.person
        .mutate(
            cohort_definition_id=1,
            subject_id=cdm.person.person_id,
            cohort_start_date=ibis.date(2020, 1, 1),
            cohort_end_date=ibis.date(2020, 6, 1),
        )
    )
    tbl = cc.add_demographics(cohort, cdm, index_date="cohort_start_date")
    assert tbl is not None
    assert hasattr(tbl, "schema")
    df = cc.collect(tbl.limit(5))
    assert isinstance(df, pd.DataFrame)
    assert "person_id" in df.columns or "age" in df.columns
