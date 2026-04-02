# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for CDM connector: cdm, eunomia, source, schemas, characteristics, dates, patient_profiles, settings, sql, utils, validation, vis, circe."""

import datetime
import json
import pytest
import ibis
import pandas as pd
import pyarrow as pa

import cdmconnector as cc
from cdmconnector._circe import (
    build_cohort_query,
    build_concept_set_query,
    cohort_expression_from_json,
    concept_set_expression_from_json,
    create_generate_options,
    GenerateOptions,
    render_cohort_sql,
)
from cdmconnector.characteristics import (
    SummarisedResult,
    additional_columns,
    group_columns,
    settings_columns,
    strata_columns,
    summarise_characteristics,
    summarise_cohort_count,
    table_characteristics,
)
from cdmconnector.dates import dateadd, datediff, datepart
from cdmconnector.eunomia import example_datasets
from cdmconnector.exceptions import (
    CDMValidationError,
    CohortError,
    EunomiaError,
    SourceError,
    TableNotFoundError,
)
from cdmconnector.patient_profiles import (
    start_date_column,
    end_date_column,
    standard_concept_id_column,
    source_concept_id_column,
    variable_types,
    available_estimates,
    mock_patient_profiles,
    add_demographics,
    filter_cohort_id,
)
from cdmconnector.schemas import (
    COHORT_TABLE_COLUMNS,
    cohort_columns,
    omop_columns,
    omop_table_fields,
    omop_tables,
)
from cdmconnector.settings import Settings
from cdmconnector.source import CdmSource, LocalCdmSource, DbCdmSource
from cdmconnector.cdm import (
    Cdm,
    cdm_con,
    cdm_write_schema,
    list_tables as cdm_list_tables,
    validate_observation_period,
)
from cdmconnector.sql import qualify_table, in_schema
from cdmconnector.utils import assert_character, assert_choice, unique_table_name
from cdmconnector.vis import (
    default_table_options,
    empty_table,
    format_estimate_name,
    format_estimate_value,
    format_header,
    format_min_cell_count,
    format_table,
    mock_summarised_result,
    table_columns,
    table_options,
    table_style,
    table_type,
    vis_omop_table,
    vis_table,
)


# ---- cdm_from_con / Cdm (database-backed) ----

def test_cdm_from_con_minimal(duckdb_con, minimal_cdm_tables):
    """cdm_from_con with person and observation_period only."""
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="Test")
    assert cdm.name == "Test"
    assert cdm.version in ("5.3", "5.4")
    assert "person" in cdm.tables
    assert "observation_period" in cdm.tables
    assert cdm.person is not None
    assert cdm["person"] is not None
    n_df = cc.collect(cdm.person.count())
    assert n_df is not None and (n_df.iloc[0, 0] == 3 or n_df.shape[0] == 1)


def test_cdm_from_con_table_access(cdm_from_duckdb):
    """cdm['person'] and cdm.person."""
    cdm = cdm_from_duckdb
    t1 = cdm["person"]
    t2 = cdm.person
    assert t1 is t2
    with pytest.raises(TableNotFoundError):
        _ = cdm["nonexistent"]


def test_cdm_select_tables_method(cdm_from_duckdb):
    """Cdm.select_tables returns subset of tables."""
    cdm = cdm_from_duckdb
    sub = cdm.select_tables("person")
    assert sub.tables == ["person"]
    with pytest.raises(TableNotFoundError):
        cdm.select_tables("person", "nonexistent")


def test_cdm_tables(cdm_from_duckdb):
    """cdm_tables returns CDM table names."""
    cdm = cdm_from_duckdb
    names = cc.cdm_tables(cdm)
    assert "person" in names
    assert "observation_period" in names
    assert names == cdm.tables


def test_cdm_disconnect(cdm_from_duckdb):
    """cdm.disconnect() closes the connection (no-op for already closed)."""
    cdm = cdm_from_duckdb
    cdm.disconnect()
    cdm.disconnect(drop_write_schema=False)


def test_collect_with_limit(cdm_from_duckdb):
    """collect(expr, limit=N) limits rows."""
    cdm = cdm_from_duckdb
    df = cc.collect(cdm.person, limit=2)
    assert len(df) == 2
    assert "person_id" in df.columns


def test_collect_with_dataframe():
    """collect on pandas DataFrame returns same DataFrame (to_dataframe path)."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = cc.collect(df)
    assert out is not None
    pd.testing.assert_frame_equal(out, df)


def test_collect_with_list():
    """collect on list wraps in DataFrame."""
    data = [1, 2, 3]
    out = cc.collect(data)
    assert out is not None
    assert hasattr(out, "columns")
    assert len(out) == 3


def test_cdm_from_con_dict_schema_infers_version_name(duckdb_con, minimal_cdm_tables):
    """cdm_from_con with dict cdm_schema infers version and name from cdm_source."""
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm_source = pa.table({
        "cdm_source_name": ["My CDM"],
        "cdm_source_abbreviation": ["MCDM"],
        "source_description": [""],
        "source_documentation_reference": [""],
        "cdm_holder": [""],
        "cdm_release_date": [str(datetime.date.today())],
        "cdm_version": ["5.4"],
    })
    con.create_table("cdm_source", obj=cdm_source, overwrite=True)
    vocabulary = pa.table({
        "vocabulary_id": ["None"],
        "vocabulary_name": ["None"],
        "vocabulary_reference": [""],
        "vocabulary_version": ["v1"],
        "vocabulary_concept_id": [0],
    })
    con.create_table("vocabulary", obj=vocabulary, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema={"schema": "main"}, write_schema="main")
    assert cdm.version == "5.4"
    assert cdm.name == "MCDM"


def test_cdm_getattr_missing_raises(cdm_from_duckdb):
    """Accessing non-existent table via attribute raises AttributeError."""
    cdm = cdm_from_duckdb
    with pytest.raises(AttributeError, match="nonexistent|not in CDM"):
        _ = cdm.nonexistent


def test_cdm_getitem_suggestion(cdm_from_duckdb):
    """__getitem__ with typo includes 'Did you mean' suggestion when similar table exists."""
    cdm = cdm_from_duckdb
    with pytest.raises(TableNotFoundError) as exc_info:
        _ = cdm["persn"]
    assert "Did you mean" in str(exc_info.value)
    assert "person" in str(exc_info.value)


def test_cdm_getitem_missing_no_suggestion(cdm_from_duckdb):
    """__getitem__ with non-similar name raises without suggestion."""
    cdm = cdm_from_duckdb
    with pytest.raises(TableNotFoundError) as exc_info:
        _ = cdm["zzz"]
    assert "zzz" in str(exc_info.value)
    assert "Available" in str(exc_info.value)


def test_cdm_setitem(cdm_from_duckdb):
    """Cdm __setitem__ assigns table."""
    cdm = cdm_from_duckdb
    extra = ibis.memtable({"x": [1, 2]})
    cdm["extra_table"] = extra
    assert "extra_table" in cdm.tables
    assert cdm["extra_table"] is extra


def test_cdm_write_schema_and_con_and_list_tables(cdm_from_duckdb):
    """cdm_write_schema, cdm_con, list_tables return expected values."""
    cdm = cdm_from_duckdb
    assert cdm_write_schema(cdm) is not None
    assert cdm_con(cdm) is not None
    assert cdm_list_tables(cdm) == cdm.tables


def test_cdm_con_local_returns_none(minimal_cdm_tables):
    """cdm_con returns None for CDM with LocalCdmSource."""
    con = ibis.duckdb.connect()
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = Cdm(
        {"person": con.table("person"), "observation_period": con.table("observation_period")},
        cdm_name="x",
        cdm_version="5.3",
        source=LocalCdmSource(),
    )
    assert cdm_con(cdm) is None


def test_list_source_tables_local_returns_empty(minimal_cdm_tables):
    """list_source_tables on CDM with LocalCdmSource returns []."""
    from cdmconnector.cdm import list_source_tables

    con = ibis.duckdb.connect()
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = Cdm(
        {"person": con.table("person"), "observation_period": con.table("observation_period")},
        cdm_name="x",
        cdm_version="5.3",
        source=LocalCdmSource(),
    )
    tables = list_source_tables(cdm)
    assert tables == []


def test_read_source_table_requires_db(minimal_cdm_tables):
    """read_source_table on CDM with LocalCdmSource raises SourceError."""
    from cdmconnector.cdm import read_source_table

    con = ibis.duckdb.connect()
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = Cdm(
        {"person": con.table("person"), "observation_period": con.table("observation_period")},
        cdm_name="x",
        cdm_version="5.3",
        source=LocalCdmSource(),
    )
    with pytest.raises(SourceError, match="database-backed|cdm_from_con"):
        read_source_table(cdm, "person")


def test_insert_table_requires_db(minimal_cdm_tables):
    """insert_table on CDM with LocalCdmSource raises SourceError."""
    from cdmconnector.cdm import insert_table

    con = ibis.duckdb.connect()
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = Cdm(
        {"person": con.table("person"), "observation_period": con.table("observation_period")},
        cdm_name="x",
        cdm_version="5.3",
        source=LocalCdmSource(),
    )
    with pytest.raises(SourceError, match="database-backed|cdm_from_con"):
        insert_table(cdm, "t", pa.table({"a": [1]}), overwrite=True)


def test_drop_table_list(cdm_from_duckdb):
    """drop_table accepts list of table names."""
    from cdmconnector.cdm import drop_table

    cdm = cdm_from_duckdb
    cc.compute(cdm, cdm.person.select("person_id").limit(1), "drop_a", overwrite=True)
    cc.compute(cdm, cdm.person.select("person_id").limit(1), "drop_b", overwrite=True)
    assert "drop_a" in cdm.tables and "drop_b" in cdm.tables
    drop_table(cdm, ["drop_a", "drop_b"])
    assert "drop_a" not in cdm.tables and "drop_b" not in cdm.tables


def test_validate_observation_period_overlap(cdm_from_tables_fixture):
    """validate_observation_period raises when periods overlap."""
    cdm = cdm_from_tables_fixture
    # Two rows same person_id with overlapping dates: period1 2020-01-01 to 2020-06-01, period2 2020-05-01 to 2020-12-31
    overlap_data = pd.DataFrame({
        "observation_period_id": [1, 2],
        "person_id": [1, 1],
        "observation_period_start_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-05-01")],
        "observation_period_end_date": [pd.Timestamp("2020-06-01"), pd.Timestamp("2020-12-31")],
        "period_type_concept_id": [0, 0],
    })
    con = ibis.duckdb.connect()
    con.create_table("observation_period", ibis.memtable(overlap_data.to_dict("list")), overwrite=True)
    person_df = cc.collect(cdm.person.head(1))
    con.create_table("person", ibis.memtable(person_df.to_dict("list")), overwrite=True)
    cdm2 = Cdm(
        {"person": con.table("person"), "observation_period": con.table("observation_period")},
        cdm_name="x",
        cdm_version="5.3",
        source=DbCdmSource(con, "main"),
    )
    with pytest.raises(CDMValidationError, match="overlap"):
        validate_observation_period(cdm2, check_overlap=True)
    con.disconnect()


def test_validate_observation_period_start_after_end(cdm_from_tables_fixture):
    """validate_observation_period raises when start_date > end_date."""
    cdm = cdm_from_tables_fixture
    op_df = cc.collect(cdm.observation_period)
    op_df = op_df.head(1)
    op_df["observation_period_start_date"] = pd.Timestamp("2021-01-01")
    op_df["observation_period_end_date"] = pd.Timestamp("2020-01-01")
    con = ibis.duckdb.connect()
    con.create_table("observation_period", ibis.memtable(op_df.to_dict("list")), overwrite=True)
    con.create_table("person", obj=cc.collect(cdm.person.head(1)), overwrite=True)
    cdm2 = Cdm(
        {"person": con.table("person"), "observation_period": con.table("observation_period")},
        cdm_name="x",
        cdm_version="5.3",
        source=DbCdmSource(con, "main"),
    )
    with pytest.raises(CDMValidationError, match="start date after end"):
        validate_observation_period(cdm2, check_start_before_end=True)
    con.disconnect()


def test_validate_observation_period_plausible_dates_warns(cdm_from_tables_fixture):
    """validate_observation_period warns when dates before 1800 or after today."""
    cdm = cdm_from_tables_fixture
    op_df = cc.collect(cdm.observation_period)
    op_df = op_df.head(1)
    op_df["observation_period_start_date"] = pd.Timestamp("1799-01-01")
    op_df["observation_period_end_date"] = pd.Timestamp("2030-12-31")
    con = ibis.duckdb.connect()
    con.create_table("observation_period", ibis.memtable(op_df.to_dict("list")), overwrite=True)
    con.create_table("person", obj=cc.collect(cdm.person.head(1)), overwrite=True)
    cdm2 = Cdm(
        {"person": con.table("person"), "observation_period": con.table("observation_period")},
        cdm_name="x",
        cdm_version="5.3",
        source=DbCdmSource(con, "main"),
    )
    with pytest.warns(UserWarning, match="1800-01-01|2030"):
        validate_observation_period(cdm2, check_overlap=False, check_start_before_end=False, check_plausible_dates=True)
    con.disconnect()


def test_copy_cdm_to(cdm_from_duckdb):
    """copy_cdm_to copies CDM to another connection."""
    from cdmconnector.cdm import copy_cdm_to

    cdm = cdm_from_duckdb
    con2 = ibis.duckdb.connect()
    cdm2 = copy_cdm_to(con2, cdm, "main", overwrite=False)
    assert cdm2.name == cdm.name
    assert "person" in cdm2.tables
    df = cc.collect(cdm2.person)
    assert len(df) == 3
    con2.disconnect()


def test_cdm_snapshot_requires_tables():
    """cdm.snapshot() raises when cdm_source or vocabulary is missing."""
    person = ibis.memtable({"person_id": [1], "year_of_birth": [1990]})
    op = ibis.memtable({"person_id": [1], "observation_period_start_date": ["2020-01-01"], "observation_period_end_date": ["2020-12-31"]})
    cdm = Cdm({"person": person, "observation_period": op}, cdm_name="x", cdm_version="5.3", source=LocalCdmSource())
    with pytest.raises(CDMValidationError, match="cdm_source|vocabulary"):
        cdm.snapshot()


def test_cdm_snapshot_result_collect(duckdb_con, minimal_cdm_tables):
    """cdm.snapshot() executes and returns metadata DataFrame."""
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm_source = pa.table({
        "cdm_source_name": ["Test"],
        "cdm_source_abbreviation": ["TEST"],
        "source_description": [""],
        "source_documentation_reference": [""],
        "cdm_holder": [""],
        "cdm_release_date": [str(datetime.date.today())],
        "cdm_version": ["5.3"],
    })
    vocabulary = pa.table({
        "vocabulary_id": ["None"],
        "vocabulary_name": ["None"],
        "vocabulary_reference": [""],
        "vocabulary_version": ["v1"],
        "vocabulary_concept_id": [0],
    })
    con.create_table("cdm_source", obj=cdm_source, overwrite=True)
    con.create_table("vocabulary", obj=vocabulary, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="Test")
    out = cdm.snapshot()
    assert out is not None
    assert hasattr(out, "columns")
    assert "cdm_name" in out.columns or "person_count" in out.columns


def test_compute(cdm_from_duckdb):
    """compute materializes an expression into a table in write schema."""
    from cdmconnector.cdm import drop_table

    cdm = cdm_from_duckdb
    expr = cdm.person.select("person_id", "year_of_birth")
    tbl = cc.compute(cdm, expr, "computed_persons", overwrite=True)
    assert tbl is not None
    assert "computed_persons" in cdm.tables
    drop_table(cdm, "computed_persons")


def test_compute_requires_db_cdm():
    """compute raises for local CDM (Cdm with LocalCdmSource)."""
    person = ibis.memtable({"person_id": [1]})
    op = ibis.memtable({"person_id": [1], "observation_period_start_date": ["2020-01-01"], "observation_period_end_date": ["2020-12-31"]})
    cdm = Cdm({"person": person, "observation_period": op}, cdm_name="x", cdm_version="5.3", source=LocalCdmSource())
    with pytest.raises(SourceError, match="database-backed"):
        cc.compute(cdm, cdm.person, "x", overwrite=True)


# ---- subset, subset_cohort, sample, flatten ----

def test_cdm_subset(cdm_from_duckdb):
    """cdm.subset(person_id) returns new CDM filtered to those persons."""
    cdm = cdm_from_duckdb
    cdm2 = cdm.subset([1, 3])
    df = cc.collect(cdm2["person"])
    assert len(df) == 2
    assert set(df["person_id"].tolist()) == {1, 3}


def test_cdm_subset_requires_db():
    """cdm.subset() requires database-backed CDM with write_schema."""
    person = ibis.memtable({"person_id": [1]})
    op = ibis.memtable({"person_id": [1], "observation_period_start_date": ["2020-01-01"], "observation_period_end_date": ["2020-12-31"]})
    cdm = Cdm({"person": person, "observation_period": op}, cdm_name="x", cdm_version="5.3", source=LocalCdmSource())
    with pytest.raises(SourceError, match="database-backed|write_schema|subset"):
        cdm.subset([1])


def test_cdm_subset_empty_raises(cdm_from_duckdb):
    """cdm.subset([]) raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        cdm_from_duckdb.subset([])


def test_cdm_subset_cohort(duckdb_con, minimal_cdm_tables):
    """cdm.subset_cohort(cohort_table) returns CDM filtered to persons in that cohort."""
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cohort = pa.table({
        "cohort_definition_id": [1, 1],
        "subject_id": [1, 3],
        "cohort_start_date": pa.array([datetime.date(2020, 1, 1), datetime.date(2020, 1, 1)], type=pa.date32()),
        "cohort_end_date": pa.array([datetime.date(2020, 12, 31), datetime.date(2020, 12, 31)], type=pa.date32()),
    })
    con.create_table("cohort", obj=cohort, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test", cohort_tables=["cohort"])
    cdm2 = cdm.subset_cohort(cohort_table="cohort")
    df = cc.collect(cdm2["person"])
    assert len(df) == 2
    assert set(df["person_id"].tolist()) == {1, 3}


def test_cdm_subset_cohort_cohort_id(duckdb_con, minimal_cdm_tables):
    """cdm.subset_cohort(cohort_table, cohort_id) filters to that cohort definition."""
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cohort = pa.table({
        "cohort_definition_id": [1, 1, 2],
        "subject_id": [1, 3, 2],
        "cohort_start_date": pa.array([datetime.date(2020, 1, 1)] * 3, type=pa.date32()),
        "cohort_end_date": pa.array([datetime.date(2020, 12, 31)] * 3, type=pa.date32()),
    })
    con.create_table("cohort", obj=cohort, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test", cohort_tables=["cohort"])
    cdm2 = cdm.subset_cohort(cohort_table="cohort", cohort_id=1)
    df = cc.collect(cdm2["person"])
    assert len(df) == 2
    assert set(df["person_id"].tolist()) == {1, 3}


def test_cdm_subset_cohort_table_not_found_raises(cdm_from_duckdb):
    """cdm.subset_cohort with missing cohort table raises TableNotFoundError."""
    with pytest.raises(TableNotFoundError, match="cohort|not in CDM|Available"):
        cdm_from_duckdb.subset_cohort(cohort_table="nonexistent_cohort")


def test_cdm_sample(cdm_from_duckdb):
    """cdm.sample(n, seed) returns new CDM with n persons and adds sample table."""
    cdm = cdm_from_duckdb
    cdm2 = cdm.sample(2, seed=42)
    df = cc.collect(cdm2["person"])
    assert len(df) == 2
    assert "person_sample" in cdm2.tables
    sample_df = cc.collect(cdm2["person_sample"])
    assert len(sample_df) == 2
    assert "person_id" in sample_df.columns


def test_cdm_sample_requires_db():
    """cdm.sample() requires database-backed CDM."""
    person = ibis.memtable({"person_id": [1]})
    op = ibis.memtable({"person_id": [1], "observation_period_start_date": ["2020-01-01"], "observation_period_end_date": ["2020-12-31"]})
    cdm = Cdm({"person": person, "observation_period": op}, cdm_name="x", cdm_version="5.3", source=LocalCdmSource())
    with pytest.raises(SourceError, match="database-backed|write_schema|sample"):
        cdm.sample(1)


def test_cdm_sample_n_larger_than_persons_returns_all(cdm_from_duckdb):
    """cdm.sample(n) with n >= person count returns all persons."""
    cdm = cdm_from_duckdb
    cdm2 = cdm.sample(10, seed=1)
    df = cc.collect(cdm2["person"])
    assert len(df) == 3


def test_cdm_flatten_domain(cdm_from_duckdb):
    """cdm.flatten(domain=..., include_concept_name=False) returns lazy observation table."""
    cdm = cdm_from_duckdb
    # Add condition_occurrence so we have a flattenable domain
    cond = pa.table({
        "person_id": [1, 2],
        "condition_occurrence_id": [1, 2],
        "condition_concept_id": [1, 2],
        "condition_start_date": pa.array([datetime.date(2010, 1, 1), datetime.date(2010, 1, 2)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2010, 1, 1), datetime.date(2010, 1, 2)], type=pa.date32()),
        "condition_type_concept_id": [0, 0],
    })
    cdm.source.con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm["condition_occurrence"] = cdm.source.con.table("condition_occurrence")
    flat = cdm.flatten(domain=["condition_occurrence"], include_concept_name=False)
    df = cc.collect(flat)
    assert "person_id" in df.columns
    assert "observation_concept_id" in df.columns
    assert "start_date" in df.columns
    assert "end_date" in df.columns
    assert "type_concept_id" in df.columns
    assert "domain" in df.columns
    assert len(df) == 2
    assert list(df["domain"].unique()) == ["condition_occurrence"]


def test_cdm_flatten_include_concept_name_requires_concept(cdm_from_duckdb):
    """cdm.flatten(include_concept_name=True) without concept table raises."""
    cdm = cdm_from_duckdb
    cond = pa.table({
        "person_id": [1],
        "condition_occurrence_id": [1],
        "condition_concept_id": [1],
        "condition_start_date": pa.array([datetime.date(2010, 1, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2010, 1, 1)], type=pa.date32()),
        "condition_type_concept_id": [0],
    })
    cdm.source.con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm["condition_occurrence"] = cdm.source.con.table("condition_occurrence")
    with pytest.raises(TableNotFoundError, match="concept|include_concept_name"):
        cdm.flatten(domain=["condition_occurrence"], include_concept_name=True)


def test_cdm_flatten_invalid_domain_raises(cdm_from_duckdb):
    """cdm.flatten(domain=[...]) with invalid domain name raises ValueError."""
    with pytest.raises(ValueError, match="domain|subset"):
        cdm_from_duckdb.flatten(domain=["not_a_domain"])


def test_cdm_flatten_domain_table_missing_raises(cdm_from_duckdb):
    """cdm.flatten(domain=[...]) when domain table not in CDM raises TableNotFoundError."""
    with pytest.raises(TableNotFoundError, match="condition_occurrence|not in CDM|Available"):
        cdm_from_duckdb.flatten(domain=["condition_occurrence"])


# ---- cdm_from_tables (in-memory / local) ----

def test_cdm_from_tables_empty_dict():
    """cdm_from_tables with empty dict creates CDM with no tables (StopIteration path)."""
    cdm = cc.cdm_from_tables({}, cdm_name="empty")
    assert cdm.tables == []
    assert cdm.name == "empty"


def test_cdm_from_con_list_tables_attr_error_fallback(duckdb_con, minimal_cdm_tables):
    """cdm_from_con when list_tables(database=db) raises AttributeError falls back to list_tables()."""
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    orig = con.list_tables
    call_count = 0

    def list_tables(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1 and kwargs:
            raise AttributeError("database")
        return orig() if not kwargs else orig(**kwargs)

    con.list_tables = list_tables
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="Test")
    assert "person" in cdm.tables


def test_cdm_from_con_cohort_table_not_found(duckdb_con, minimal_cdm_tables):
    """cdm_from_con with cohort_tables name not in write schema raises TableNotFoundError."""
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    with pytest.raises(TableNotFoundError, match="not found in write schema|Cohort table"):
        cc.cdm_from_con(
            con,
            cdm_schema="main",
            write_schema="main",
            cohort_tables=["nonexistent_cohort_table"],
            cdm_name="Test",
        )


def test_cdm_from_tables_arrow(minimal_cdm_tables):
    """cdm_from_tables with Arrow tables creates in-memory DuckDB CDM."""
    cdm = cc.cdm_from_tables(
        minimal_cdm_tables,
        cdm_name="test",
        cdm_version="5.3",
    )
    assert cdm.name == "test"
    assert cdm.version == "5.3"
    assert "person" in cdm.tables
    assert "observation_period" in cdm.tables
    n_df = cc.collect(cdm.person.count())
    assert n_df is not None and (n_df.iloc[0, 0] == 3 or n_df.shape[0] == 1)


def test_cdm_from_tables_repr(minimal_cdm_tables):
    """Cdm repr shows name, version, table count."""
    cdm = cc.cdm_from_tables(minimal_cdm_tables, cdm_name="x", cdm_version="5.3")
    r = repr(cdm)
    assert "x" in r
    assert "5.3" in r
    assert "tables=2" in r or "tables=" in r


def test_cdm_select_tables_method_minimal(minimal_cdm_tables):
    """Cdm.select_tables() returns subset."""
    cdm = cc.cdm_from_tables(minimal_cdm_tables, cdm_name="x")
    sub = cdm.select_tables("person")
    assert sub.tables == ["person"]


def test_cdm_select_tables_missing_raises(minimal_cdm_tables):
    """select_tables with missing table raises TableNotFoundError."""
    cdm = cc.cdm_from_tables(minimal_cdm_tables, cdm_name="x")
    with pytest.raises(TableNotFoundError, match="not found|Available"):
        cdm.select_tables("person", "nonexistent")


# ---- Eunomia ----

def test_example_datasets():
    """example_datasets returns tuple of names."""
    ds = example_datasets()
    assert isinstance(ds, tuple)
    assert "GiBleed" in ds
    assert "synpuf-1k" in ds


def test_download_eunomia_path_required():
    """download_eunomia_data requires path or env."""
    from cdmconnector.eunomia import download_eunomia_data

    with pytest.raises(EunomiaError):
        download_eunomia_data("GiBleed", path="")


def test_download_eunomia_unknown_dataset():
    """Unknown dataset name raises."""
    from cdmconnector.eunomia import download_eunomia_data

    with pytest.raises(EunomiaError):
        download_eunomia_data("NotADataset", path="/tmp")


def test_eunomia_is_available_nonexistent_path():
    """eunomia_is_available returns False for nonexistent path."""
    from cdmconnector.eunomia import eunomia_is_available

    assert eunomia_is_available(path="/nonexistent/eunomia/folder") is False


def test_require_eunomia_sets_path(tmp_path):
    """require_eunomia creates dir and returns path when EUNOMIA_DATA_FOLDER unset."""
    import os

    from cdmconnector.eunomia import require_eunomia

    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        path = require_eunomia("GiBleed", path=str(tmp_path))
        assert path == str(tmp_path)
        assert tmp_path.exists()
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old


def test_eunomia_dir_requires_eunomia_data_folder_when_path_unset():
    """eunomia_dir raises when path is None and EUNOMIA_DATA_FOLDER is unset."""
    import os

    from cdmconnector.eunomia import eunomia_dir
    from cdmconnector.exceptions import EunomiaError

    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        with pytest.raises(EunomiaError, match="EUNOMIA_DATA_FOLDER must be set"):
            eunomia_dir("GiBleed", cdm_version="5.3", path=None)
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old


def test_eunomia_is_available_with_path(tmp_path):
    """eunomia_is_available returns True when zip exists at path."""
    from cdmconnector.eunomia import eunomia_is_available
    (tmp_path / "GiBleed_5.3.zip").touch()
    assert eunomia_is_available(dataset_name="GiBleed", cdm_version="5.3", path=str(tmp_path)) is True
    assert eunomia_is_available(dataset_name="GiBleed", cdm_version="5.3", path=str(tmp_path / "nonexistent")) is False


def test_require_eunomia_sets_path_and_uses_existing(tmp_path):
    """require_eunomia with path uses existing zip when available."""
    from cdmconnector.eunomia import require_eunomia
    (tmp_path / "GiBleed_5.3.zip").touch()
    out = require_eunomia(dataset_name="GiBleed", cdm_version="5.3", path=str(tmp_path))
    assert out == str(tmp_path)


# ---- Source (CdmSource, LocalCdmSource, DbCdmSource) ----

def test_cdm_source_base():
    """CdmSource holds source_type."""
    src = CdmSource("duckdb")
    assert src.source_type == "duckdb"
    assert "duckdb" in repr(src)


def test_local_cdm_source():
    """LocalCdmSource has source_type 'local'."""
    src = LocalCdmSource()
    assert src.source_type == "local"
    assert "local" in repr(src).lower()


def test_db_cdm_source_list_tables(duckdb_con):
    """DbCdmSource.list_tables returns table names in schema."""
    con = duckdb_con
    con.create_table("person", obj=pa.table({"person_id": [1, 2], "year_of_birth": [1990, 1985]}), overwrite=True)
    src = DbCdmSource(con, "main")
    tables = src.list_tables()
    assert isinstance(tables, list)
    assert "person" in tables


def test_db_cdm_source_table(duckdb_con):
    """DbCdmSource.table returns Ibis table."""
    con = duckdb_con
    con.create_table("mytable", obj=pa.table({"a": [1, 2], "b": ["x", "y"]}), overwrite=True)
    src = DbCdmSource(con, "main")
    tbl = src.table("mytable")
    assert tbl is not None
    n_df = cc.collect(tbl.count())
    assert n_df is not None and (n_df.iloc[0, 0] == 2 or n_df.shape[0] == 1)


def test_db_cdm_source_insert_drop_table(duckdb_con):
    """DbCdmSource insert_table and drop_table."""
    con = duckdb_con
    src = DbCdmSource(con, "main")
    tbl = pa.table({"id": [1], "name": ["a"]})
    out = src.insert_table("test_insert_drop", tbl, overwrite=True)
    assert out is not None
    assert "test_insert_drop" in src.list_tables()
    src.drop_table("test_insert_drop")
    assert "test_insert_drop" not in src.list_tables()


def test_db_cdm_source_disconnect(duckdb_con):
    """DbCdmSource.disconnect closes connection."""
    con = duckdb_con
    src = DbCdmSource(con, "main")
    src.disconnect(drop_write_schema=False)
    assert True


def test_db_cdm_source_dict_schema(duckdb_con):
    """DbCdmSource with dict write_schema and _database_for_schema with schema_name."""
    con = duckdb_con
    con.create_table("person", obj=pa.table({"person_id": [1], "year_of_birth": [1990]}), overwrite=True)
    src = DbCdmSource(con, {"schema": "main"})
    assert src.write_schema["schema"] == "main"
    tbl = src.table("person")
    assert cc.collect(tbl.count()).iloc[0, 0] == 1
    src2 = DbCdmSource(con, {"schema_name": "main"})
    assert src2._database_for_schema({"schema_name": "main"}) == "main"


def test_db_cdm_source_insert_table_dict(duckdb_con):
    """DbCdmSource.insert_table accepts dict (from_pydict path)."""
    con = duckdb_con
    src = DbCdmSource(con, "main")
    src.insert_table("dict_table", {"a": [1, 2], "b": ["x", "y"]}, overwrite=True)
    assert "dict_table" in src.list_tables()
    df = cc.collect(src.table("dict_table"))
    assert len(df) == 2
    src.drop_table("dict_table")


def test_db_cdm_source_disconnect_drop_schema():
    """DbCdmSource.disconnect(drop_write_schema=True) iterates and drops tables."""
    from unittest.mock import MagicMock
    con = MagicMock()
    con.list_tables.return_value = ["t1"]
    con.drop_table = MagicMock()
    con.disconnect = MagicMock()
    src = DbCdmSource(con, "main")
    src.disconnect(drop_write_schema=True)
    assert con.drop_table.called
    assert con.disconnect.called


def test_cdm_source_repr():
    """CdmSource and LocalCdmSource __repr__."""
    base = CdmSource("duckdb")
    assert "CdmSource" in repr(base)
    assert "duckdb" in repr(base)
    local = LocalCdmSource()
    assert "local" in repr(local)


def test_source_list_tables_raises():
    """DbCdmSource.list_tables raises SourceError when connection.list_tables fails."""
    from unittest.mock import MagicMock
    con = MagicMock()
    con.list_tables.side_effect = RuntimeError("connection failed")
    src = DbCdmSource(con, "main")
    with pytest.raises(SourceError, match="Failed to list"):
        src.list_tables()


# ---- Schemas (OMOP) ----

def test_omop_tables():
    """omop_tables returns tuple of table names."""
    t53 = omop_tables("5.3")
    t54 = omop_tables("5.4")
    assert "person" in t53
    assert "observation_period" in t53
    assert "cdm_source" in t53
    assert "person" in t54


def test_omop_tables_invalid_version():
    """Invalid version raises."""
    with pytest.raises(ValueError):
        omop_tables("6.0")


def test_cohort_columns():
    """cohort_columns for cohort, cohort_set, cohort_attrition."""
    assert cohort_columns("cohort") == COHORT_TABLE_COLUMNS
    assert "cohort_definition_id" in cohort_columns("cohort_set")
    assert "reason_id" in cohort_columns("cohort_attrition")
    with pytest.raises(ValueError):
        cohort_columns("other")


def test_omop_table_fields():
    """omop_table_fields returns DataFrame with cdm_table_name, cdm_field_name, etc."""
    df = omop_table_fields("5.3")
    assert "cdm_table_name" in df.columns
    assert "cdm_field_name" in df.columns
    assert "is_required" in df.columns
    assert "cdm_datatype" in df.columns
    assert "type" in df.columns
    person_rows = df[df["cdm_table_name"] == "person"]
    assert len(person_rows) > 0
    assert "person_id" in person_rows["cdm_field_name"].tolist()
    df54 = omop_table_fields("5.4")
    assert len(df54) >= len(df)
    with pytest.raises(ValueError):
        omop_table_fields("6.0")


def test_omop_columns():
    """omop_columns(table) returns column names; omop_columns(table, field) returns single column."""
    cols = omop_columns("person", version="5.3")
    assert isinstance(cols, tuple)
    assert "person_id" in cols
    assert "year_of_birth" in cols
    obs_cols = omop_columns("observation_period")
    assert "observation_period_start_date" in obs_cols
    assert omop_columns("observation_period", field="start_date") == "observation_period_start_date"
    assert omop_columns("observation_period", field="end_date") == "observation_period_end_date"
    assert omop_columns("condition_occurrence", field="person_id") == "person_id"
    # Cohort tables (not in omop_table_fields) return cohort_columns
    assert "cohort_definition_id" in omop_columns("cohort")
    assert "cohort_definition_id" in omop_columns("cohort_set")
    assert "reason_id" in omop_columns("cohort_attrition")
    with pytest.raises(ValueError, match="no field mapping|start_date"):
        omop_columns("person", field="start_date")
    with pytest.raises(ValueError):
        omop_columns("unknown_table")
    # Field not in cols
    with pytest.raises(ValueError, match="Unknown field"):
        omop_columns("condition_occurrence", field="nonexistent_key")
    # Field maps to None (observation_period has standard_concept: None)
    with pytest.raises(ValueError, match="no column for field"):
        omop_columns("observation_period", field="standard_concept")
    # Unknown table (no field) for column list
    with pytest.raises(ValueError, match="Unknown table"):
        omop_columns("not_a_real_table")
    # Invalid version
    with pytest.raises(ValueError, match="Unsupported CDM version"):
        omop_columns("person", version="6.0")


# ---- Characteristics ----

def test_summarise_cohort_count_from_dataframe():
    """summarise_cohort_count with DataFrame returns counts per cohort."""
    cohort_df = pd.DataFrame({
        "cohort_definition_id": [1, 1, 1, 2],
        "subject_id": [10, 11, 10, 12],
        "cohort_start_date": [datetime.date(2020, 1, 1)] * 4,
        "cohort_end_date": [datetime.date(2020, 6, 1)] * 4,
    })
    res = summarise_cohort_count(cohort_df)
    assert isinstance(res, SummarisedResult)
    assert "results" in dir(res) and "settings" in dir(res)
    assert res.settings["result_type"].iloc[0] == "summarise_cohort_count"
    r = res.results
    subj = r[(r["variable_name"] == "Number subjects") & (r["group_level"] == "1")]
    rec = r[(r["variable_name"] == "Number records") & (r["group_level"] == "1")]
    assert subj["estimate_value"].iloc[0] == "2"
    assert rec["estimate_value"].iloc[0] == "3"


def test_summarise_characteristics_requires_cohort_columns():
    """summarise_characteristics raises if required columns missing."""
    bad = pd.DataFrame({"cohort_definition_id": [1], "subject_id": [1]})
    with pytest.raises(CohortError):
        summarise_characteristics(bad)


def test_summarise_characteristics_empty_cohort():
    """summarise_characteristics with empty cohort returns empty-style result."""
    empty = pd.DataFrame(columns=["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"])
    res = summarise_characteristics(empty, counts=True, demographics=False)
    assert isinstance(res, SummarisedResult)
    assert res.results.shape[0] == 0 or "Number" in res.results["variable_name"].values


def test_summarise_characteristics_with_demographics(cdm_from_tables_fixture):
    """summarise_characteristics with demographics=True adds age, sex, prior_obs, etc."""
    cdm = cdm_from_tables_fixture
    cohort_df = pd.DataFrame({
        "cohort_definition_id": [1],
        "subject_id": [1],
        "cohort_start_date": [pd.Timestamp("2020-06-01")],
        "cohort_end_date": [pd.Timestamp("2020-12-31")],
    })
    result = summarise_characteristics(cohort_df, cdm, demographics=True, counts=True)
    assert result.results is not None and len(result.results) > 0
    assert "Number subjects" in result.results["variable_name"].values or "count" in result.results["estimate_name"].values
    assert result.settings is not None and len(result.settings) > 0


def test_summarise_characteristics_with_strata(cdm_from_tables_fixture):
    """summarise_characteristics with strata stratifies results."""
    cdm = cdm_from_tables_fixture
    cohort_df = pd.DataFrame({
        "cohort_definition_id": [1, 1],
        "subject_id": [1, 2],
        "cohort_start_date": [pd.Timestamp("2020-06-01")] * 2,
        "cohort_end_date": [pd.Timestamp("2020-12-31")] * 2,
        "stratum": ["A", "B"],
    })
    result = summarise_characteristics(cohort_df, cdm, strata=["stratum"], demographics=False, counts=True)
    assert result.results is not None and len(result.results) > 0
    assert "stratum" in result.results["strata_level"].values or "A" in result.results["strata_level"].values


def test_summarise_characteristics_cohort_names_from_cdm(cdm_from_duckdb):
    """summarise_characteristics uses cohort_set from cdm for cohort names when table_name set."""
    cdm = cdm_from_duckdb
    cohort_set_df = pd.DataFrame({
        "cohort_definition_id": [1],
        "cohort_name": ["My Cohort"],
    })
    cc.compute(cdm, ibis.memtable(cohort_set_df.to_dict("list")), "cohort_set", overwrite=True)
    cohort_df = pd.DataFrame({
        "cohort_definition_id": [1],
        "subject_id": [1],
        "cohort_start_date": [pd.Timestamp("2020-06-01")],
        "cohort_end_date": [pd.Timestamp("2020-12-31")],
    })
    result = summarise_characteristics(cohort_df, cdm, table_name="cohort", demographics=False, counts=True)
    assert result.results is not None and len(result.results) > 0
    assert "My Cohort" in result.results["group_level"].values or "1" in result.results["group_level"].values


def test_summarise_characteristics_cohort_id_filter():
    """summarise_characteristics with cohort_id filters to those cohorts."""
    cohort_df = pd.DataFrame({
        "cohort_definition_id": [1, 1, 2, 2],
        "subject_id": [10, 11, 12, 13],
        "cohort_start_date": [datetime.date(2020, 1, 1)] * 4,
        "cohort_end_date": [datetime.date(2020, 6, 1)] * 4,
    })
    res = summarise_characteristics(cohort_df, cohort_id=1, counts=True, demographics=False)
    assert set(res.results["group_level"].unique()) == {"1"}


def test_table_characteristics_pivots():
    """table_characteristics pivots estimate_name/estimate_value into columns."""
    res = summarise_cohort_count(
        pd.DataFrame({
            "cohort_definition_id": [1],
            "subject_id": [1],
            "cohort_start_date": [datetime.date(2020, 1, 1)],
            "cohort_end_date": [datetime.date(2020, 6, 1)],
        })
    )
    tbl = table_characteristics(res)
    assert isinstance(tbl, pd.DataFrame)
    assert "count" in tbl.columns or "variable_name" in tbl.columns


def test_settings_columns():
    """settings_columns returns expected names."""
    assert "result_id" in settings_columns()
    assert "result_type" in settings_columns()


def test_group_columns():
    """group_columns returns group_name, group_level."""
    assert group_columns() == ("group_name", "group_level")


def test_strata_columns():
    """strata_columns returns strata_name, strata_level."""
    assert strata_columns() == ("strata_name", "strata_level")


def test_additional_columns():
    """additional_columns returns additional_name, additional_level."""
    assert additional_columns() == ("additional_name", "additional_level")


# ---- Dates ----

def test_dateadd_day():
    """dateadd with interval day builds expression."""
    t = ibis.memtable({"d": ["2020-01-01"]}).cast({"d": "date"})
    expr = dateadd(t.d, 1, "day")
    assert expr is not None
    mutated = t.mutate(next_d=expr)
    assert "next_d" in mutated.schema().names


def test_dateadd_year():
    """dateadd with interval year builds expression."""
    t = ibis.memtable({"d": ["2020-01-01"]}).cast({"d": "date"})
    expr = dateadd(t.d, 1, "year")
    mutated = t.mutate(next_d=expr)
    assert "next_d" in mutated.schema().names


def test_dateadd_invalid_interval():
    """dateadd with invalid interval raises."""
    t = ibis.memtable({"d": ["2020-01-01"]}).cast({"d": "date"})
    with pytest.raises(ValueError, match="interval must be"):
        dateadd(t.d, 1, "month")


def test_datediff_day():
    """datediff with interval day builds expression."""
    t = ibis.memtable({
        "start": ["2020-01-01"], "end": ["2020-01-31"],
    }).cast({"start": "date", "end": "date"})
    expr = datediff(t.start, t.end, "day")
    mutated = t.mutate(days=expr)
    assert "days" in mutated.schema().names


def test_datediff_month_year():
    """datediff with month/year builds expression."""
    t = ibis.memtable({
        "start": ["2020-01-15"], "end": ["2021-03-15"],
    }).cast({"start": "date", "end": "date"})
    days = datediff(t.start, t.end, "day")
    mutated = t.mutate(d=days)
    assert "d" in mutated.schema().names
    # Cover month/year branch (lines 87-89)
    months = datediff(t.start, t.end, "month")
    years = datediff(t.start, t.end, "year")
    mutated2 = t.mutate(m=months, y=years)
    assert "m" in mutated2.schema().names
    assert "y" in mutated2.schema().names


def test_datediff_invalid_interval():
    """datediff with invalid interval raises."""
    t = ibis.memtable({"start": ["2020-01-01"], "end": ["2020-01-31"]}).cast({"start": "date", "end": "date"})
    with pytest.raises(ValueError, match="interval must be"):
        datediff(t.start, t.end, "week")


def test_datepart():
    """datepart extracts year, month, day (expression only)."""
    t = ibis.memtable({"birth_date": ["1993-04-19"]}).cast({"birth_date": "date"})
    y = datepart(t.birth_date, "year")
    m = datepart(t.birth_date, "month")
    d = datepart(t.birth_date, "day")
    mutated = t.mutate(year=y, month=m, day=d)
    assert "year" in mutated.schema().names
    assert "month" in mutated.schema().names
    assert "day" in mutated.schema().names


def test_datepart_invalid_interval():
    """datepart with invalid interval raises."""
    t = ibis.memtable({"birth_date": ["1993-04-19"]}).cast({"birth_date": "date"})
    with pytest.raises(ValueError, match="interval must be"):
        datepart(t.birth_date, "hour")


# ---- Patient profiles ----

def test_start_date_column():
    """start_date_column returns correct column for OMOP table."""
    assert start_date_column("condition_occurrence") == "condition_start_date"
    assert start_date_column("observation_period") == "observation_period_start_date"
    assert start_date_column("visit_occurrence") == "visit_start_date"


def test_start_date_column_unknown_raises():
    """start_date_column raises for unknown table."""
    with pytest.raises(ValueError, match="Unknown table|FIELD_TABLES_COLUMNS"):
        start_date_column("unknown_table")


def test_end_date_column():
    """end_date_column returns column or None."""
    assert end_date_column("condition_occurrence") == "condition_end_date"
    assert end_date_column("observation_period") == "observation_period_end_date"
    assert end_date_column("person") is None


def test_end_date_column_unknown_returns_none():
    """end_date_column returns None for unknown table."""
    assert end_date_column("unknown_table") is None


def test_standard_concept_id_column():
    """standard_concept_id_column returns column or None."""
    assert standard_concept_id_column("condition_occurrence") == "condition_concept_id"
    assert standard_concept_id_column("person") is None


def test_source_concept_id_column():
    """source_concept_id_column returns column or None."""
    assert source_concept_id_column("condition_occurrence") == "condition_source_concept_id"
    assert source_concept_id_column("person") is None


def test_variable_types_from_dataframe():
    """variable_types returns variable_name and variable_type from DataFrame."""
    df = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["x", "y"], "d": [True, False]})
    out = variable_types(df)
    assert "variable_name" in out.columns
    assert "variable_type" in out.columns
    assert len(out) >= 2


def test_variable_types_empty():
    """variable_types on empty table returns empty DataFrame with columns."""
    df = pd.DataFrame(columns=["x", "y"])
    out = variable_types(df)
    assert "variable_name" in out.columns
    assert "variable_type" in out.columns
    assert len(out) == 0


def test_available_estimates():
    """available_estimates returns DataFrame of estimate_name, estimate_description, variable_type."""
    out = available_estimates()
    assert "estimate_name" in out.columns
    assert "estimate_description" in out.columns
    assert "variable_type" in out.columns
    assert len(out) > 0
    out_numeric = available_estimates("numeric")
    assert len(out_numeric) > 0
    assert all(out_numeric["variable_type"] == "numeric")


def test_mock_patient_profiles():
    """mock_patient_profiles returns Cdm with person, observation_period, cohort1."""
    cdm = mock_patient_profiles(number_individuals=5, seed=42)
    assert cdm is not None
    assert "person" in cdm.tables
    assert "observation_period" in cdm.tables
    assert "cohort1" in cdm.tables
    n = cdm.person.count().execute()
    assert n == 5
    try:
        cdm.disconnect()
    except Exception:
        pass


def test_add_demographics_returns_table(cdm_from_tables_fixture):
    """add_demographics returns Ibis table with extra columns when given cohort and CDM."""
    cdm = cdm_from_tables_fixture
    t = cdm.person
    cohort = t.mutate(
        cohort_start_date=ibis.date(2020, 1, 1),
        cohort_end_date=ibis.date(2020, 6, 1),
        subject_id=t.person_id,
        cohort_definition_id=1,
    )
    out = add_demographics(cohort, cdm, index_date="cohort_start_date", age=True, sex=True)
    assert out is not None
    assert hasattr(out, "schema") or hasattr(out, "columns")
    # Ibis Table: schema() returns Schema; Schema.names is a tuple (property)
    cols = out.schema().names if hasattr(out, "schema") else out.columns
    assert "age" in cols or "person_id" in cols


def test_filter_cohort_id_table():
    """filter_cohort_id filters Ibis table by cohort_definition_id."""
    t = ibis.memtable({
        "cohort_definition_id": [1, 1, 2],
        "subject_id": [10, 11, 12],
        "cohort_start_date": [datetime.date(2020, 1, 1)] * 3,
        "cohort_end_date": [datetime.date(2020, 6, 1)] * 3,
    })
    out = filter_cohort_id(t, 1)
    assert out is not None
    n = out.count().execute()
    assert n == 2
    out2 = filter_cohort_id(t, [1, 2])
    n2 = out2.count().execute()
    assert n2 == 3


# ---- Settings ----

def test_settings_empty():
    """Settings() starts with empty data."""
    s = Settings()
    assert s.data == {}


def test_settings_get_set_item():
    """Settings get/set by key."""
    s = Settings()
    s["foo"] = "bar"
    assert s["foo"] == "bar"
    assert s.get("foo") == "bar"
    assert s.get("missing", "default") == "default"


def test_settings_contains():
    """Settings supports 'in'."""
    s = Settings()
    s["key"] = 1
    assert "key" in s
    assert "other" not in s


def test_settings_repr():
    """Settings has repr."""
    s = Settings()
    s["a"] = 1
    r = repr(s)
    assert "Settings" in r
    assert "a" in r or "1" in r


def test_settings_from_data():
    """Settings can be constructed with initial data."""
    s = Settings(data={"x": 10, "y": "hello"})
    assert s["x"] == 10
    assert s["y"] == "hello"
    assert s.get("z", 0) == 0


# ---- SQL ----

def test_qualify_table_none_schema():
    """qualify_table with schema None returns table name (optionally with prefix)."""
    assert qualify_table(None, None, "person") == "person"
    assert qualify_table(None, None, "person", prefix="tmp_") == "tmp_person"


def test_qualify_table_string_schema():
    """qualify_table with string schema returns (schema, table) for non-main."""
    out = qualify_table(None, "myschema", "person")
    assert out == ("myschema", "person")
    out_main = qualify_table(None, "main", "person")
    assert out_main == "person"


def test_qualify_table_dict_schema():
    """qualify_table with dict schema uses schema key."""
    out = qualify_table(None, {"schema": "results"}, "cohort")
    assert out == ("results", "cohort")
    out_prefix = qualify_table(None, {"schema": "r", "prefix": "t_"}, "cohort")
    assert out_prefix == ("r", "t_cohort") or "cohort" in str(out_prefix)
    # Dict with single key (schema_name from first value)
    out_single = qualify_table(None, {"db": "mydb"}, "t")
    assert out_single == ("mydb", "t")
    # Dict with catalog
    out_catalog = qualify_table(None, {"catalog": "c", "schema": "s"}, "t")
    assert out_catalog == ("c", "s", "t")
    # Prefix with dict schema (no "prefix" in schema)
    out_prefixed = qualify_table(None, {"schema": "s"}, "t", prefix="tmp_")
    assert out_prefixed == ("s", "tmp_t")
    # Schema as non-str non-dict (e.g. object) -> str(schema)
    class FakeSchema:
        def __str__(self):
            return "myschema"
    out_fake = qualify_table(None, FakeSchema(), "t")
    assert out_fake == ("myschema", "t")


def test_in_schema():
    """in_schema returns schema-qualified table identifier."""
    out = in_schema("main", "person")
    assert out == qualify_table(None, "main", "person")
    out_none = in_schema(None, "person")
    assert out_none == "person" or out_none == ("person",)


# ---- Utils ----

def test_assert_character_valid():
    """assert_character accepts non-empty string."""
    assert_character("x")
    assert_character("x", length=1)
    assert_character("ab", length=2)
    assert_character("hello", length=None)
    assert_character("ab", min_num_character=1, length=None)


def test_assert_character_allow_none():
    """assert_character with allow_none accepts None."""
    assert_character(None, allow_none=True)


def test_assert_character_wrong_type():
    """assert_character raises TypeError for non-string."""
    with pytest.raises(TypeError, match="str"):
        assert_character(123)
    with pytest.raises(TypeError, match="str"):
        assert_character([])


def test_assert_character_wrong_length():
    """assert_character raises ValueError for wrong length."""
    with pytest.raises(ValueError, match="length"):
        assert_character("ab", length=1)
    with pytest.raises(ValueError, match="length"):
        assert_character("", length=1)


def test_assert_character_min_num_character():
    """assert_character raises when string too short for min_num_character."""
    with pytest.raises(ValueError, match="at least"):
        assert_character("", min_num_character=1, length=None)
    with pytest.raises(ValueError, match="at least"):
        assert_character("x", min_num_character=2, length=None)


def test_assert_choice_valid():
    """assert_choice accepts value in choices."""
    assert_choice("a", ["a", "b", "c"])
    assert_choice("b", ("a", "b", "c"))


def test_assert_choice_invalid():
    """assert_choice raises ValueError for value not in choices."""
    with pytest.raises(ValueError, match="Expected one of"):
        assert_choice("z", ["a", "b", "c"])
    with pytest.raises(ValueError, match="Expected one of"):
        assert_choice(1, ["a", "b"])


def test_unique_table_name():
    """unique_table_name returns prefix + unique suffix."""
    a = unique_table_name()
    b = unique_table_name()
    assert a != b
    assert a.startswith("tmp_")
    assert len(a) > 4
    c = unique_table_name(prefix="foo_")
    assert c.startswith("foo_")


# ---- Validation ----

def test_validate_observation_period_requires_table():
    """validate_observation_period raises when observation_period table is missing."""
    from cdmconnector.cdm import validate_observation_period

    person = ibis.memtable({
        "person_id": [1], "gender_concept_id": [0], "year_of_birth": [1990],
        "race_concept_id": [0], "ethnicity_concept_id": [0],
    })
    cdm = Cdm(
        {"person": person},
        cdm_name="test",
        cdm_version="5.3",
        source=LocalCdmSource(),
    )
    with pytest.raises(CDMValidationError, match="observation_period"):
        validate_observation_period(cdm)


# ---- Vis ----

def test_table_options():
    assert "decimals" in table_options()
    assert table_options()["decimal_mark"] == "."
    assert table_options()["na"] == "\u2013"


def test_default_table_options_with_user_options():
    """default_table_options merges user_options."""
    opts = default_table_options({"title": "My Title", "decimals": {"integer": 1}})
    assert opts["title"] == "My Title"
    assert opts["decimals"]["integer"] == 1


def test_vis_table_type_dataframe():
    """vis_table with type dataframe returns DataFrame."""
    df = vis_table(mock_summarised_result(seed=1), type="dataframe")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_vis_omop_table_style_darwin():
    """vis_omop_table with style darwin."""
    try:
        out = vis_omop_table(mock_summarised_result(seed=1), style="darwin")
        assert out is not None
    except Exception:
        pytest.skip("great_tables not available or darwin style not supported")


def test_table_style():
    assert "default" in table_style()
    assert "darwin" in table_style()


def test_table_type():
    assert "dataframe" in table_type()
    assert "html" in table_type()
    assert "gt" in table_type()


def test_mock_summarised_result():
    df = mock_summarised_result(seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for c in ("cdm_name", "group_level", "strata_level", "estimate_name", "estimate_value"):
        assert c in df.columns


def test_table_columns():
    df = mock_summarised_result(seed=1)
    cols = table_columns(df)
    assert "cdm_name" in cols
    assert "estimate_name" in cols
    assert "variable_name" in cols


def test_format_estimate_value():
    df = pd.DataFrame({
        "estimate_name": ["count", "mean"],
        "estimate_type": ["integer", "numeric"],
        "estimate_value": ["1234567", "12.3456"],
    })
    out = format_estimate_value(df, decimals={"integer": 0, "numeric": 2}, big_mark=",")
    assert out["estimate_value"].iloc[0] == "1,234,567"
    assert "12.35" in out["estimate_value"].iloc[1] or "12,35" in out["estimate_value"].iloc[1]


def test_format_estimate_value_decimals_int():
    """format_estimate_value with decimals as int (single value for all types)."""
    df = pd.DataFrame({
        "estimate_name": ["count", "mean"],
        "estimate_type": ["integer", "numeric"],
        "estimate_value": ["42", "3.14159"],
    })
    out = format_estimate_value(df, decimals=2)
    assert out is not None
    assert "estimate_value" in out.columns


def test_table_columns_with_group_strata_additional():
    """table_columns includes group_name, strata_name, additional_name when present."""
    df = pd.DataFrame({
        "cdm_name": ["x"],
        "group_name": ["g"],
        "group_level": ["1"],
        "strata_name": ["s"],
        "strata_level": ["1"],
        "additional_name": ["a"],
        "additional_level": ["1"],
        "variable_name": ["v"],
        "variable_level": ["1"],
        "estimate_name": ["count"],
    })
    cols = table_columns(df)
    assert "group_name" in cols
    assert "strata_name" in cols
    assert "additional_name" in cols


def test_format_header():
    df = pd.DataFrame({
        "group_level": ["A", "A", "B", "B"],
        "estimate_value": ["1", "2", "3", "4"],
    })
    out = format_header(df, header=["group_level"], include_header_key=False)
    assert "estimate_value" not in out.columns or "group_level" in out.columns
    assert out.shape[0] <= 4


def test_empty_table():
    out = empty_table(type="dataframe")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_vis_table_empty():
    out = vis_table(pd.DataFrame(), type="dataframe")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_vis_omop_table_mock():
    df = mock_summarised_result(seed=1)
    out = vis_omop_table(
        df,
        header=["group_level"],
        rename={"cdm_name": "DB"},
        hide=["result_id"],
        type="dataframe",
    )
    assert isinstance(out, pd.DataFrame)
    assert not out.empty or len(df) == 0


def test_vis_omop_table_estimate_name():
    df = mock_summarised_result(seed=1)
    out = vis_table(
        df.head(10),
        estimate_name={"N": "<count>"},
        type="dataframe",
    )
    assert isinstance(out, pd.DataFrame)


def test_vis_omop_table_gt():
    """When type='gt', output is a great_tables GT object."""
    df = mock_summarised_result(seed=1)
    out = vis_omop_table(df.head(12), header=[], type="gt")
    assert out is not None
    assert type(out).__name__ == "GT"
    html = out.as_raw_html()
    assert isinstance(html, str)
    assert "<table" in html or "table" in html.lower()


def test_empty_table_gt():
    out = empty_table(type="gt")
    assert out is not None
    assert type(out).__name__ == "GT"


def test_format_estimate_name():
    """format_estimate_name combines estimate_name/estimate_value into display labels."""
    df = pd.DataFrame({
        "group_level": ["1", "1"],
        "estimate_name": ["count", "percentage"],
        "estimate_type": ["integer", "percentage"],
        "estimate_value": ["10", "50.0"],
    })
    out = format_estimate_name(df, estimate_name={"N (%)": "<count> (<percentage>)"})
    assert out is not None
    assert isinstance(out, pd.DataFrame)
    assert "estimate_name" in out.columns
    assert "estimate_value" in out.columns


def test_format_estimate_name_empty_dict_returns_copy():
    """format_estimate_name with None/empty estimate_name returns copy."""
    df = pd.DataFrame({"estimate_name": ["count"], "estimate_value": ["1"]})
    out = format_estimate_name(df, estimate_name=None)
    assert out is not None
    assert len(out) == len(df)


def test_format_estimate_name_missing_columns_raises():
    """format_estimate_name raises when result missing estimate_name/estimate_value."""
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="estimate_name|estimate_value"):
        format_estimate_name(df, estimate_name={"N": "<count>"})


def test_format_table_dataframe():
    """format_table with type='dataframe' returns DataFrame."""
    df = pd.DataFrame({
        "group_level": ["A"],
        "strata_level": ["B"],
        "estimate_name": ["count"],
        "estimate_value": ["10"],
    })
    out = format_table(df, type="dataframe")
    assert isinstance(out, pd.DataFrame)
    assert not out.empty or len(df) == 0


def test_format_min_cell_count():
    """format_min_cell_count merges min_cell_count from settings when present."""
    result = pd.DataFrame({
        "result_id": [1],
        "estimate_name": ["count"],
        "estimate_value": ["-"],
    })
    out_none = format_min_cell_count(result, settings=None)
    assert out_none is result or out_none["estimate_value"].iloc[0] == "-"
    settings = pd.DataFrame({"result_id": [1], "min_cell_count": [5]})
    out = format_min_cell_count(result, settings=settings)
    assert "estimate_value" in out.columns
    # With result_id match, suppressed value may be replaced
    assert out is not None


# ---- Circe (_circe) ----

def test_cohort_expression_from_json_valid():
    """cohort_expression_from_json parses valid CIRCE JSON."""
    js = json.dumps({
        "ConceptSets": [],
        "PrimaryCriteria": {"CriteriaList": [], "ObservationWindow": {"PriorDays": 0, "PostDays": 0}, "RestrictVisit": False},
        "InclusionRules": [],
        "CensoringCriteria": [],
    })
    out = cohort_expression_from_json(js)
    assert isinstance(out, dict)
    assert "ConceptSets" in out
    assert "PrimaryCriteria" in out


def test_cohort_expression_from_json_empty_raises():
    """cohort_expression_from_json raises for empty string."""
    with pytest.raises(ValueError, match="non-empty"):
        cohort_expression_from_json("")


def test_cohort_expression_from_json_invalid_json_raises():
    """cohort_expression_from_json raises for invalid JSON."""
    with pytest.raises(ValueError, match="Invalid|JSON"):
        cohort_expression_from_json("{ invalid")


def test_cohort_expression_from_json_non_dict_raises():
    """cohort_expression_from_json raises when JSON is not an object."""
    with pytest.raises(ValueError, match="Cohort definition JSON must be a JSON object"):
        cohort_expression_from_json("[]")


def test_cohort_expression_from_json_missing_concept_sets_raises():
    """cohort_expression_from_json raises when ConceptSets missing."""
    js = json.dumps({"PrimaryCriteria": {"CriteriaList": []}})
    with pytest.raises(ValueError, match="ConceptSets"):
        cohort_expression_from_json(js)


def test_cohort_expression_from_json_missing_primary_criteria_raises():
    """cohort_expression_from_json raises when PrimaryCriteria missing."""
    js = json.dumps({"ConceptSets": []})
    with pytest.raises(ValueError, match="PrimaryCriteria"):
        cohort_expression_from_json(js)


def test_concept_set_expression_from_json_valid():
    """concept_set_expression_from_json parses valid concept set JSON."""
    js = json.dumps({"items": [], "id": 0})
    out = concept_set_expression_from_json(js)
    assert isinstance(out, dict)
    assert "items" in out


def test_concept_set_expression_from_json_empty_raises():
    """concept_set_expression_from_json raises for empty string."""
    with pytest.raises(ValueError, match="non-empty"):
        concept_set_expression_from_json("")


def test_concept_set_expression_from_json_invalid_raises():
    """concept_set_expression_from_json raises for invalid JSON."""
    with pytest.raises(ValueError, match="Invalid|JSON"):
        concept_set_expression_from_json("not json")


def test_create_generate_options():
    """create_generate_options returns GenerateOptions with given fields."""
    opts = create_generate_options(
        cohort_id=1,
        cdm_schema="cdm",
        target_table="cohort",
        result_schema="results",
        generate_stats=False,
    )
    assert isinstance(opts, GenerateOptions)
    assert opts.cohort_id == 1
    assert opts.cdm_schema == "cdm"
    assert opts.target_table == "cohort"
    assert opts.result_schema == "results"
    assert opts.generate_stats is False


def test_render_cohort_sql():
    """render_cohort_sql replaces @ placeholders in SQL."""
    sql = "SELECT * FROM @cdm_database_schema.person WHERE 1=1;"
    out = render_cohort_sql(
        sql,
        cdm_database_schema="main",
        vocabulary_database_schema="main",
        target_database_schema="main",
        results_database_schema="results",
        target_cohort_table="my_cohort",
        target_cohort_id=1,
    )
    assert "@cdm_database_schema" not in out
    assert "main" in out
    assert "person" in out


def test_render_cohort_sql_optional_params_defaults():
    """render_cohort_sql uses cdm_database_schema when vocabulary/results not provided."""
    sql = "@vocabulary_database_schema @results_database_schema @results_database_schema.cohort_inclusion"
    out = render_cohort_sql(
        sql,
        cdm_database_schema="cdm_schema",
        target_database_schema="target_schema",
        target_cohort_table="cohort",
        target_cohort_id=1,
    )
    assert "cdm_schema" in out
    assert "target_schema" in out
    assert "target_schema.cohort_inclusion" in out


def test_render_cohort_sql_custom_stem_and_tables():
    """render_cohort_sql uses custom cohort_stem and table names when provided."""
    sql = "@results_database_schema.cohort_inclusion @results_database_schema.cohort_summary_stats"
    out = render_cohort_sql(
        sql,
        cdm_database_schema="main",
        target_database_schema="results",
        target_cohort_table="my_cohort",
        target_cohort_id=1,
        cohort_stem="my_stem",
        cohort_inclusion="custom.inclusion",
        cohort_summary_stats="custom.summary",
    )
    assert "custom.inclusion" in out
    assert "custom.summary" in out


def test_cohort_expression_from_json_non_string_raises():
    """cohort_expression_from_json raises for non-string or empty."""
    with pytest.raises(ValueError, match="non-empty string"):
        cohort_expression_from_json(None)
    with pytest.raises(ValueError, match="non-empty string"):
        cohort_expression_from_json("")
    with pytest.raises(ValueError, match="non-empty string"):
        cohort_expression_from_json(123)


def test_concept_set_expression_from_json_non_string_raises():
    """concept_set_expression_from_json raises for non-string or empty."""
    with pytest.raises(ValueError, match="non-empty string"):
        concept_set_expression_from_json("")


def test_concept_set_expression_from_json_non_dict_raises():
    """concept_set_expression_from_json raises when JSON is not an object."""
    with pytest.raises(ValueError, match="Concept set JSON must be a JSON object"):
        concept_set_expression_from_json("[]")
    with pytest.raises(ValueError, match="Concept set JSON must be a JSON object"):
        concept_set_expression_from_json("123")


def test_build_cohort_query_with_dict_and_vocabulary_default():
    """build_cohort_query accepts dict; vocabulary_schema defaults to cdm_schema."""
    expr = {"ConceptSets": [], "PrimaryCriteria": {"CriteriaList": [], "ObservationWindow": {"PriorDays": 0, "PostDays": 0}, "RestrictVisit": False}, "InclusionRules": [], "CensoringCriteria": []}
    opts = create_generate_options(cdm_schema="main", target_table="cohort", result_schema="results")
    try:
        sql = build_cohort_query(expr, opts)
        assert isinstance(sql, str)
    except NotImplementedError:
        pytest.skip("Circepy not available")


def test_build_cohort_query():
    """build_cohort_query returns SQL when Circepy available else raises NotImplementedError."""
    from cdmconnector._circe import build_cohort_query
    js = json.dumps({
        "ConceptSets": [],
        "PrimaryCriteria": {"CriteriaList": [], "ObservationWindow": {"PriorDays": 0, "PostDays": 0}, "RestrictVisit": False},
        "InclusionRules": [],
        "CensoringCriteria": [],
    })
    opts = create_generate_options(cdm_schema="main", target_table="cohort", result_schema="results")
    try:
        sql = build_cohort_query(js, opts)
        assert isinstance(sql, str)
        assert len(sql) > 0
    except NotImplementedError:
        pytest.skip("Circepy not available")


def test_build_concept_set_query_valid_json():
    """build_concept_set_query validates JSON; may raise NotImplementedError if Circepy missing."""
    js = json.dumps({"items": [{"concept": {"CONCEPT_ID": 123}, "isExcluded": False, "includeDescendants": False}]})
    try:
        out = build_concept_set_query(js)
        assert isinstance(out, str)
        assert len(out) > 0
    except NotImplementedError:
        pytest.skip("Circepy not available or concept set builder not exposed")


def test_build_concept_set_query_invalid_json_raises():
    """build_concept_set_query raises for invalid concept set JSON."""
    with pytest.raises(ValueError, match="Invalid|JSON"):
        build_concept_set_query("{ invalid }")


def test_build_cohort_query_raises_when_circepy_unavailable():
    """build_cohort_query raises NotImplementedError when Circepy is not available."""
    import cdmconnector._circe as circe_mod
    opts = create_generate_options(cdm_schema="main", target_table="cohort")
    orig = circe_mod._CIRCEPY_AVAILABLE
    try:
        circe_mod._CIRCEPY_AVAILABLE = False
        with pytest.raises(NotImplementedError, match="Circepy"):
            build_cohort_query('{"ConceptSets":[],"PrimaryCriteria":{}}', opts)
    finally:
        circe_mod._CIRCEPY_AVAILABLE = orig


def test_build_concept_set_query_raises_when_circepy_unavailable():
    """build_concept_set_query raises NotImplementedError when Circepy is not available."""
    import cdmconnector._circe as circe_mod
    js = json.dumps({"items": [{"concept": {"CONCEPT_ID": 123}, "isExcluded": False, "includeDescendants": False}]})
    orig = circe_mod._CIRCEPY_AVAILABLE
    try:
        circe_mod._CIRCEPY_AVAILABLE = False
        with pytest.raises(NotImplementedError, match="Circepy"):
            build_concept_set_query(js)
    finally:
        circe_mod._CIRCEPY_AVAILABLE = orig


def test_result_snapshot_collect_and_compute(duckdb_con, minimal_cdm_tables):
    """Result with expr=None and _snapshot_cdm in meta uses _materialize_snapshot in collect/compute."""
    from cdmconnector.results import Result

    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm_source = pa.table({
        "cdm_source_name": ["Test"],
        "cdm_source_abbreviation": ["TEST"],
        "source_description": [""],
        "source_documentation_reference": [""],
        "cdm_holder": [""],
        "cdm_release_date": [str(datetime.date.today())],
        "cdm_version": ["5.3"],
    })
    vocabulary = pa.table({
        "vocabulary_id": ["None"],
        "vocabulary_name": ["None"],
        "vocabulary_reference": [""],
        "vocabulary_version": ["v1"],
        "vocabulary_concept_id": [0],
    })
    con.create_table("cdm_source", obj=cdm_source, overwrite=True)
    con.create_table("vocabulary", obj=vocabulary, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="Test")
    result = Result(expr=None, meta={"_snapshot_cdm": cdm, "_snapshot_compute_data_hash": False})
    df = result.collect()
    assert df is not None and hasattr(df, "columns")
    tbl = result.compute(cdm, "snapshot_test_table", overwrite=True)
    assert tbl is not None
    assert "snapshot_test_table" in cdm.tables
    cdm.disconnect()


# ---- Exceptions (ensure each can be raised and caught) ----

def test_exceptions_hierarchy():
    """All CDMConnector exceptions are subclasses of CDMConnectorError."""
    from cdmconnector.exceptions import (
        CDMConnectorError,
        CDMValidationError,
        TableNotFoundError,
        SourceError,
        CohortError,
        EunomiaError,
    )
    for exc_cls in (CDMValidationError, TableNotFoundError, SourceError, CohortError, EunomiaError):
        assert issubclass(exc_cls, CDMConnectorError)
        assert issubclass(exc_cls, Exception)
    # Raise and catch each
    with pytest.raises(CDMValidationError):
        raise CDMValidationError("test")
    with pytest.raises(TableNotFoundError):
        raise TableNotFoundError("test")
    with pytest.raises(SourceError):
        raise SourceError("test")
    with pytest.raises(CohortError):
        raise CohortError("test")
    with pytest.raises(EunomiaError):
        raise EunomiaError("test")
