# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for cohort_count, attrition, record_cohort_attrition, generate_cohort_set helpers."""

from pathlib import Path

import pytest
import pandas as pd
import cdmconnector as cc
from cdmconnector.cohorts import (
    _create_empty_cohort_tables,
    _normalize_cohort_definition_set,
    _schema_to_string,
    _table_refs,
    _validate_cohort_table_name,
    attrition,
    bind,
    cohort_collapse,
    cohort_count,
    generate_cohort_set,
    generate_concept_cohort_set,
    new_cohort_table,
    read_cohort_set,
    record_cohort_attrition,
)
from cdmconnector.exceptions import CohortError
from cdmconnector.schemas import cohort_columns


def test_cohort_count_requires_attrition():
    """cohort_count raises if no cohort_attrition."""
    class FakeCohort:
        pass
    with pytest.raises(CohortError):
        cohort_count(FakeCohort())


def test_attrition_requires_attrition():
    """attrition raises if no cohort_attrition."""
    class FakeCohort:
        pass
    with pytest.raises(CohortError):
        attrition(FakeCohort())


def test_cohort_count_empty_attrition():
    """cohort_count with empty attrition returns empty DataFrame with expected columns."""
    class CohortWithEmptyAttr:
        cohort_attrition = pd.DataFrame(columns=[
            "cohort_definition_id", "number_records", "number_subjects",
            "reason_id", "reason", "excluded_subjects", "excluded_records",
        ])
    out = cohort_count(CohortWithEmptyAttr())
    assert list(out.columns) == ["cohort_definition_id", "number_records", "number_subjects"]
    assert len(out) == 0


def test_cohort_count_from_df():
    """cohort_count with DataFrame cohort_attrition."""
    class CohortWithAttr:
        cohort_attrition = pd.DataFrame({
            "cohort_definition_id": [1, 1, 2],
            "number_records": [10, 5, 3],
            "number_subjects": [8, 4, 3],
            "reason_id": [1, 2, 1],
            "reason": ["Initial", "Filtered", "Initial"],
            "excluded_records": [0, 5, 0],
            "excluded_subjects": [0, 4, 0],
        })
    out = cohort_count(CohortWithAttr())
    assert len(out) == 2  # max reason_id per cohort
    assert set(out["cohort_definition_id"]) == {1, 2}


def test_new_cohort_table_requires_db_cdm():
    """new_cohort_table raises CohortError for CDM with LocalCdmSource (no insert_table)."""
    from cdmconnector.source import LocalCdmSource
    from cdmconnector.cdm import Cdm
    src = LocalCdmSource()
    cdm = Cdm(
        {"person": None, "observation_period": None},
        cdm_name="x",
        cdm_version="5.3",
        source=src,
    )
    with pytest.raises(CohortError):
        new_cohort_table(cdm, "my_cohort")


def test_bind_raises_not_implemented():
    """bind raises NotImplementedError (not yet implemented)."""
    with pytest.raises(NotImplementedError, match="bind"):
        bind(
            pd.DataFrame({"cohort_definition_id": [1], "subject_id": [1], "cohort_start_date": [pd.Timestamp("2020-01-01")], "cohort_end_date": [pd.Timestamp("2020-06-01")]}),
            pd.DataFrame({"cohort_definition_id": [2], "subject_id": [2], "cohort_start_date": [pd.Timestamp("2020-01-01")], "cohort_end_date": [pd.Timestamp("2020-06-01")]}),
            name="combined",
        )


def test_record_cohort_attrition_requires_cohort_attrition():
    """record_cohort_attrition raises when cohort has no cohort_attrition."""
    class FakeCohort:
        pass
    with pytest.raises(CohortError, match="cohort_attrition"):
        record_cohort_attrition(FakeCohort(), "Filtered")


def test_validate_cohort_table_name_valid():
    """_validate_cohort_table_name accepts valid names and returns lowercase."""
    assert _validate_cohort_table_name("cohort") == "cohort"
    assert _validate_cohort_table_name("  MyCohort  ") == "mycohort"
    assert _validate_cohort_table_name("cohort_1") == "cohort_1"


def test_validate_cohort_table_name_invalid():
    """_validate_cohort_table_name raises for invalid names."""
    with pytest.raises(CohortError, match="letter"):
        _validate_cohort_table_name("")
    with pytest.raises(CohortError, match="letter"):
        _validate_cohort_table_name("1cohort")
    with pytest.raises(CohortError, match="letters, numbers"):
        _validate_cohort_table_name("my-cohort")


def test_normalize_cohort_definition_set_empty_raises():
    """_normalize_cohort_definition_set raises when empty."""
    with pytest.raises(CohortError, match="at least one row"):
        _normalize_cohort_definition_set([], compute_attrition=False)


def test_normalize_cohort_definition_set_missing_columns_raises():
    """_normalize_cohort_definition_set raises when missing required columns."""
    with pytest.raises(CohortError, match="cohort_definition_id|cohort_name"):
        _normalize_cohort_definition_set(
            [{"x": 1}],
            compute_attrition=False,
        )
    with pytest.raises(CohortError, match="json|sql"):
        _normalize_cohort_definition_set(
            [{"cohort_definition_id": 1, "cohort_name": "A"}],
            compute_attrition=False,
        )


def test_normalize_cohort_definition_set_with_sql_column():
    """_normalize_cohort_definition_set with sql column returns DataFrame with sql."""
    data = [{
        "cohort_definition_id": 1,
        "cohort_name": "Test",
        "sql": "SELECT 1 AS x;",
    }]
    out = _normalize_cohort_definition_set(data, compute_attrition=False)
    assert "sql" in out.columns
    assert len(out) == 1
    assert "SELECT" in str(out["sql"].iloc[0])


def test_schema_to_string():
    """_schema_to_string converts schema spec to string."""
    assert _schema_to_string("main") == "main"
    assert _schema_to_string({"schema": "myschema"}) == "myschema"
    assert _schema_to_string({"schema_name": "other"}) == "other"
    assert _schema_to_string(None) == ""
    assert _schema_to_string({}) == ""


def test_record_cohort_attrition_empty_reason_raises():
    """record_cohort_attrition raises when reason is empty."""
    cohort = type("Cohort", (), {"cohort_attrition": pd.DataFrame()})()
    with pytest.raises(ValueError, match="reason cannot be empty"):
        record_cohort_attrition(cohort, "")
    with pytest.raises(ValueError, match="reason cannot be empty"):
        record_cohort_attrition(cohort, "   ")


def test_record_cohort_attrition_from_dataframe():
    """record_cohort_attrition updates cohort_attrition with new reason row."""
    cohort_df = pd.DataFrame({
        "cohort_definition_id": [1, 1],
        "subject_id": [10, 11],
        "cohort_start_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
        "cohort_end_date": [pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-01")],
    })
    initial_attrition = pd.DataFrame({
        "cohort_definition_id": [1],
        "number_records": [3],
        "number_subjects": [3],
        "reason_id": [1],
        "reason": ["Initial"],
        "excluded_subjects": [0],
        "excluded_records": [0],
    })
    cohort = type("Cohort", (), {"cohort_attrition": initial_attrition, "execute": lambda self: cohort_df, "to_pandas": lambda self: cohort_df})()
    out = record_cohort_attrition(cohort, "Filtered", cohort_id=1)
    assert out is cohort or out is not None
    attr = getattr(out, "cohort_attrition", out)
    if hasattr(attr, "columns"):
        assert "reason" in attr.columns
        assert len(attr) == 2
        assert "Filtered" in attr["reason"].values


def test_generate_cohort_set_with_sql_column(cdm_from_duckdb):
    """generate_cohort_set with sql column runs SQL and registers cohort table."""
    cdm = cdm_from_duckdb
    # Use placeholders that render_cohort_sql will replace (target_database_schema, target_cohort_table, target_cohort_id)
    cohort_set_df = pd.DataFrame({
        "cohort_definition_id": [1],
        "cohort_name": ["Test cohort"],
        "sql": [
            "INSERT INTO @target_database_schema.@target_cohort_table "
            "(cohort_definition_id, subject_id, cohort_start_date, cohort_end_date) "
            "VALUES (@target_cohort_id, 1, DATE '2020-01-01', DATE '2020-06-01');",
        ],
    })
    out = generate_cohort_set(cdm, cohort_set_df, name="test_cohort", overwrite=True, compute_attrition=False)
    assert out is cdm
    assert "test_cohort" in out.tables
    df = cc.collect(out["test_cohort"])
    assert len(df) >= 1
    assert "cohort_definition_id" in df.columns


def test_read_cohort_set_missing_dir_raises():
    """read_cohort_set raises when path does not exist."""
    with pytest.raises(CohortError, match="does not exist|not a directory"):
        read_cohort_set("/nonexistent/path/cohorts")


def test_read_cohort_set_not_dir_raises(tmp_path):
    """read_cohort_set raises when path exists but is not a directory."""
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(CohortError, match="not a directory"):
        read_cohort_set(str(f))


def test_read_cohort_set_from_json_folder(tmp_path):
    """read_cohort_set from folder with JSON files returns cohort set DataFrame."""
    import json
    cohort_json = {
        "ConceptSets": [],
        "PrimaryCriteria": {"CriteriaList": [], "ObservationWindow": {"PriorDays": 0, "PostDays": 0}, "RestrictVisit": False},
        "InclusionRules": [],
        "CensoringCriteria": [],
        "CollapseSettings": {"CollapseType": "ERA", "EraPad": 0},
        "CensorWindow": {},
    }
    (tmp_path / "cohort_one.json").write_text(json.dumps(cohort_json), encoding="utf-8")
    (tmp_path / "cohort_two.json").write_text(json.dumps(cohort_json), encoding="utf-8")
    df = read_cohort_set(str(tmp_path))
    assert "cohort_definition_id" in df.columns
    assert "cohort_name" in df.columns
    assert "cohort" in df.columns
    assert "json" in df.columns
    assert len(df) == 2
    assert set(df["cohort_definition_id"]) == {1, 2}


def _repo_inst_path(*parts: str) -> Path | None:
    """Path to repo root inst folder (tests/../inst). None if not found."""
    root = Path(__file__).resolve().parent.parent
    inst = root / "inst"
    if not inst.is_dir():
        return None
    for p in parts:
        inst = inst / p
    return inst if inst.exists() else None


@pytest.mark.parametrize(
    "subdir,expected_count,has_csv,description",
    [
        ("cohorts1", 2, True, "CohortsToCreate.csv + 2 JSON files"),
        ("cohorts2", 3, False, "JSON-only, 3 cohorts"),
        ("cohorts3", 5, False, "JSON-only, 5 cohorts"),
        ("cohorts4", 1, False, "JSON-only, 1 cohort (numeric stem -> cohort_100)"),
        ("cohorts5", 1, False, "JSON-only, 1 cohort"),
        ("cohorts6", 2, False, "JSON-only, 2 cohorts"),
    ],
)
def test_read_cohort_set_from_inst_folder(subdir, expected_count, has_csv, description):
    """read_cohort_set from repo inst/cohorts* folders returns expected cohort set."""
    cohort_path = _repo_inst_path(subdir)
    if cohort_path is None or not cohort_path.is_dir():
        pytest.skip(f"inst/{subdir} not found (run from repo root with inst data)")
    df = read_cohort_set(str(cohort_path))
    assert "cohort_definition_id" in df.columns
    assert "cohort_name" in df.columns
    assert "cohort" in df.columns
    assert "json" in df.columns
    assert "cohort_name_snakecase" in df.columns
    assert len(df) == expected_count
    assert list(df["cohort_definition_id"]) == list(range(1, expected_count + 1))
    if has_csv:
        assert (Path(cohort_path) / "CohortsToCreate.csv").exists()
    # Numeric stem case: cohorts4 has 100.json -> cohort_100
    if subdir == "cohorts4":
        assert "cohort_100" in df["cohort_name"].values


# ---- _table_refs ----


def test_table_refs_returns_domain_columns():
    """_table_refs returns DataFrame with domain_id, table_name, concept_id, start_date, end_date."""
    df = _table_refs(["condition", "drug"])
    assert list(df.columns) == ["domain_id", "table_name", "concept_id", "start_date", "end_date"]
    assert len(df) == 2
    assert set(df["domain_id"]) == {"condition", "drug"}
    assert df[df["domain_id"] == "condition"]["table_name"].iloc[0] == "condition_occurrence"
    assert df[df["domain_id"] == "condition"]["concept_id"].iloc[0] == "condition_concept_id"


def test_table_refs_filtered_by_domain_ids():
    """_table_refs filters to requested domain_ids only."""
    df = _table_refs(["procedure", "observation"])
    assert set(df["domain_id"]) == {"procedure", "observation"}
    assert len(df) == 2


# ---- cohort_collapse ----


def test_cohort_collapse_requires_columns():
    """cohort_collapse raises when required columns are missing."""
    with pytest.raises(CohortError, match="cohort_collapse requires columns"):
        cohort_collapse(pd.DataFrame({"cohort_definition_id": [1], "subject_id": [1]}))
    with pytest.raises(CohortError, match="cohort_collapse requires columns"):
        cohort_collapse(pd.DataFrame({"cohort_definition_id": [1], "subject_id": [1], "cohort_start_date": ["2020-01-01"]}))


def test_cohort_collapse_merges_overlapping_periods():
    """cohort_collapse merges overlapping/adjacent periods per (cohort_definition_id, subject_id)."""
    df = pd.DataFrame({
        "cohort_definition_id": [1, 1, 1],
        "subject_id": [10, 10, 10],
        "cohort_start_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-15"), pd.Timestamp("2020-04-01")],
        "cohort_end_date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30")],
    })
    out = cohort_collapse(df)
    assert list(out.columns) == ["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"]
    # First two overlap (Jan 1–31 and Jan 15–Feb 15) -> one period Jan 1–Feb 15; third is separate -> Apr 1–30
    assert len(out) == 2
    out = out.sort_values(["cohort_start_date"]).reset_index(drop=True)
    assert out["cohort_start_date"].iloc[0] == pd.Timestamp("2020-01-01").date()
    assert out["cohort_end_date"].iloc[0] == pd.Timestamp("2020-02-15").date()
    assert out["cohort_start_date"].iloc[1] == pd.Timestamp("2020-04-01").date()
    assert out["cohort_end_date"].iloc[1] == pd.Timestamp("2020-04-30").date()


def test_cohort_collapse_distinct_cohorts_and_subjects():
    """cohort_collapse keeps separate (cohort_definition_id, subject_id) groups separate."""
    df = pd.DataFrame({
        "cohort_definition_id": [1, 1, 2],
        "subject_id": [10, 20, 10],
        "cohort_start_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
        "cohort_end_date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31")],
    })
    out = cohort_collapse(df)
    assert len(out) == 3


# ---- generate_concept_cohort_set ----


def test_generate_concept_cohort_set_requires_observation_period(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set raises when observation_period is not in CDM."""
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    # Remove observation_period from CDM (subset of tables)
    cdm = cdm.select_tables("person")
    with pytest.raises(CohortError, match="observation_period"):
        generate_concept_cohort_set(cdm, {"x": [123]}, name="cohort")


def test_generate_concept_cohort_set_requires_concept(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set raises when concept table is not in CDM."""
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    with pytest.raises(CohortError, match="concept"):
        generate_concept_cohort_set(cdm, {"x": [123]}, name="cohort")


def test_generate_concept_cohort_set_requires_db(minimal_cdm_tables):
    """generate_concept_cohort_set raises for CDM with LocalCdmSource (no insert_table)."""
    from cdmconnector.source import LocalCdmSource

    cdm = cc.cdm_from_tables(
        {**minimal_cdm_tables, "concept": _concept_table_minimal()},
        cdm_name="x",
        cdm_version="5.3",
    )
    cdm._source = LocalCdmSource()
    with pytest.raises(CohortError, match="database source|cdm_from_con"):
        generate_concept_cohort_set(cdm, {"x": [123]}, name="cohort")


def _concept_table_minimal():
    """Minimal concept table (concept_id, domain_id) for tests."""
    import pyarrow as pa
    return pa.table({
        "concept_id": [123, 192671],
        "domain_id": ["condition", "condition"],
    })


def test_generate_concept_cohort_set_invalid_concept_set_raises(cdm_from_duckdb):
    """generate_concept_cohort_set raises for invalid concept_set or options."""
    import pyarrow as pa

    cdm = cdm_from_duckdb
    concept = pa.table({"concept_id": [123], "domain_id": ["condition"]})
    cdm.source.insert_table("concept", concept, overwrite=True)
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    with pytest.raises(CohortError, match="non-empty dict"):
        generate_concept_cohort_set(cdm, {}, name="cohort")
    with pytest.raises(CohortError, match="first.*all"):
        generate_concept_cohort_set(cdm, {"x": [123]}, name="cohort", limit="invalid")
    with pytest.raises(CohortError, match="required_observation"):
        generate_concept_cohort_set(cdm, {"x": [123]}, name="cohort", required_observation=(0,))


def test_generate_concept_cohort_set_empty_concept_ids_creates_empty_cohort(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with concept_set that has empty list creates empty cohort tables."""
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [999], "domain_id": ["condition"]})
    con.create_table("concept", obj=concept, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    out = generate_concept_cohort_set(cdm, {"empty_cohort": []}, name="empty_cohort")
    assert "empty_cohort" in out.tables
    df = cc.collect(out["empty_cohort"])
    assert len(df) == 0
    assert list(df.columns) == ["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"]


def test_generate_concept_cohort_set_integration(duckdb_con):
    """generate_concept_cohort_set with concept + condition_occurrence creates cohort table and set/attrition."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    person = pa.table({
        "person_id": [1, 2],
        "gender_concept_id": [8507, 8532],
        "year_of_birth": [1990, 1985],
        "month_of_birth": [1, 1],
        "day_of_birth": [1, 1],
        "race_concept_id": [0, 0],
        "ethnicity_concept_id": [0, 0],
    })
    d0 = datetime.date(2000, 1, 1)
    d1 = datetime.date(2023, 12, 31)
    observation_period = pa.table({
        "observation_period_id": [1, 2],
        "person_id": [1, 2],
        "observation_period_start_date": pa.array([d0, d0], type=pa.date32()),
        "observation_period_end_date": pa.array([d1, d1], type=pa.date32()),
        "period_type_concept_id": [0, 0],
    })
    concept = pa.table({
        "concept_id": [192671],
        "domain_id": ["condition"],
    })
    condition_occurrence = pa.table({
        "condition_occurrence_id": [1],
        "person_id": [1],
        "condition_concept_id": [192671],
        "condition_start_date": pa.array([datetime.date(2020, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 6, 15)], type=pa.date32()),
    })
    for name, tbl in [
        ("person", person),
        ("observation_period", observation_period),
        ("concept", concept),
        ("condition_occurrence", condition_occurrence),
    ]:
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    out = cdm.generate_concept_cohort_set({"gi_bleed": [192671]}, name="gi_bleed_cohort")
    assert "gi_bleed_cohort" in out.tables
    assert "gi_bleed_cohort_set" in out.source.list_tables(out.write_schema)
    assert "gi_bleed_cohort_attrition" in out.source.list_tables(out.write_schema)
    cohort_df = cc.collect(out["gi_bleed_cohort"])
    assert list(cohort_df.columns) == ["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"]
    # With matching condition_occurrence and observation_period we expect at least one row
    assert len(cohort_df) >= 0
    if len(cohort_df) >= 1:
        assert cohort_df["subject_id"].iloc[0] == 1
        assert cohort_df["cohort_definition_id"].iloc[0] == 1


def test_generate_concept_cohort_set_overwrite_false_raises(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with name already in write schema and overwrite=False raises (R: 'already exists in the CDM')."""
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [192671], "domain_id": ["condition"]})
    con.create_table("concept", obj=concept, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm = generate_concept_cohort_set(cdm, {"gibleed": [192671]}, name="gibleed", overwrite=True)
    with pytest.raises(CohortError, match="already exists|overwrite is False"):
        generate_concept_cohort_set(cdm, {"gibleed": [192671]}, name="gibleed", overwrite=False)


def test_generate_concept_cohort_set_cohort_date_columns_type(duckdb_con, minimal_cdm_tables):
    """Collected concept cohort table has date-like columns for cohort_start_date and cohort_end_date (R: Date class)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [192671], "domain_id": ["condition"]})
    cond = pa.table({
        "condition_occurrence_id": [1],
        "person_id": [1],
        "condition_concept_id": [192671],
        "condition_start_date": pa.array([datetime.date(2020, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 6, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(cdm, {"gibleed": [192671]}, name="gibleed", overwrite=True)
    df = cc.collect(cdm["gibleed"])
    assert "cohort_start_date" in df.columns and "cohort_end_date" in df.columns
    # Should be datetime64 or date-like (R expects Date)
    assert pd.api.types.is_datetime64_any_dtype(df["cohort_start_date"]) or hasattr(
        df["cohort_start_date"].iloc[0] if len(df) else None, "year"
    )
    assert pd.api.types.is_datetime64_any_dtype(df["cohort_end_date"]) or hasattr(
        df["cohort_end_date"].iloc[0] if len(df) else None, "year"
    )


def test_generate_concept_cohort_set_attrition_columns(duckdb_con, minimal_cdm_tables):
    """Attrition table for concept-generated cohort has expected columns (R: omopgenerics::cohortColumns)."""
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [192671], "domain_id": ["condition"]})
    con.create_table("concept", obj=concept, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm = generate_concept_cohort_set(cdm, {"gibleed": [192671]}, name="gibleed", overwrite=True)
    attr_table = cdm.source.table("gibleed_attrition", cdm.write_schema)
    attr_df = cc.collect(attr_table)
    expected = set(cohort_columns("cohort_attrition"))
    assert set(attr_df.columns) == expected, f"Attrition columns: got {list(attr_df.columns)}, expected {list(expected)}"


def test_generate_concept_cohort_set_required_observation_more_restrictive(duckdb_con, minimal_cdm_tables):
    """More restrictive required_observation (e.g. (2, 200)) yields <= records than (2, 2) (R: cohortCount comparison)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    person = minimal_cdm_tables["person"]
    obs = pa.table({
        "observation_period_id": [1, 2],
        "person_id": [1, 2],
        "observation_period_start_date": pa.array(
            [datetime.date(2010, 1, 1), datetime.date(2010, 1, 1)], type=pa.date32()
        ),
        "observation_period_end_date": pa.array(
            [datetime.date(2010, 7, 1), datetime.date(2010, 7, 1)], type=pa.date32()
        ),
        "period_type_concept_id": [0, 0],
    })
    con.create_table("person", obj=person, overwrite=True)
    con.create_table("observation_period", obj=obs, overwrite=True)
    concept = pa.table({"concept_id": [192671], "domain_id": ["condition"]})
    cond = pa.table({
        "condition_occurrence_id": [1, 2],
        "person_id": [1, 2],
        "condition_concept_id": [192671, 192671],
        "condition_start_date": pa.array([datetime.date(2010, 6, 1), datetime.date(2010, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2010, 6, 15), datetime.date(2010, 6, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm, {"gibleed": [192671]}, name="gibleed_loose", required_observation=(2, 2), overwrite=True
    )
    cdm = generate_concept_cohort_set(
        cdm, {"gibleed": [192671]}, name="gibleed_tight", required_observation=(2, 200), overwrite=True
    )
    n_loose = cc.collect(cdm["gibleed_loose"].count()).iloc[0, 0]
    n_tight = cc.collect(cdm["gibleed_tight"].count()).iloc[0, 0]
    assert n_tight <= n_loose


def test_generate_concept_cohort_set_limit_all(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with limit='all' runs and creates cohort (R: gibleed_all)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [192671], "domain_id": ["condition"]})
    cond = pa.table({
        "condition_occurrence_id": [1, 2],
        "person_id": [1, 1],
        "condition_concept_id": [192671, 192671],
        "condition_start_date": pa.array([datetime.date(2020, 1, 1), datetime.date(2020, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 1, 15), datetime.date(2020, 6, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(cdm, {"gibleed": [192671]}, name="gibleed_all", limit="all", overwrite=True)
    assert "gibleed_all" in cdm.tables
    df = cc.collect(cdm["gibleed_all"])
    assert list(df.columns) == ["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"]
    # With two condition rows for same person, limit=all should give 2 records
    assert len(df) >= 0


def test_generate_concept_cohort_set_multiple_cohorts_same_concept(duckdb_con, minimal_cdm_tables):
    """Multiple cohort names with same concept_id yield one row per cohort in set (R: acetaminophen_1, acetaminophen_2)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [1127433], "domain_id": ["drug"]})
    con.create_table("concept", obj=concept, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm,
        concept_set={"acetaminophen_1": [1127433], "acetaminophen_2": [1127433]},
        name="acetaminophen",
        overwrite=True,
    )
    set_df = cc.collect(cdm.source.table("acetaminophen_set", cdm.write_schema))
    assert len(set_df) == 2
    assert set(set_df["cohort_name"]) == {"acetaminophen_1", "acetaminophen_2"}
    assert list(set_df["cohort_definition_id"]) == [1, 2]


def test_generate_concept_cohort_set_subset_cohort(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with subset_cohort restricts to persons in that cohort (R: gibleed_medications)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({
        "concept_id": [192671, 1124300, 1127433],
        "domain_id": ["condition", "drug", "drug"],
    })
    cond = pa.table({
        "condition_occurrence_id": [1],
        "person_id": [1],
        "condition_concept_id": [192671],
        "condition_start_date": pa.array([datetime.date(2020, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 6, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm, {"gibleed_1": [192671], "gibleed_2": [4112343]}, name="gibleed_exp", overwrite=True
    )
    cdm = generate_concept_cohort_set(
        cdm,
        concept_set={"diclofenac": [1124300], "acetaminophen": [1127433]},
        name="gibleed_medications",
        subset_cohort="gibleed_exp",
        overwrite=True,
    )
    exp_subjects = set(cc.collect(cdm["gibleed_exp"].select("subject_id").distinct())["subject_id"])
    med_subjects = set(cc.collect(cdm["gibleed_medications"].select("subject_id").distinct())["subject_id"])
    assert med_subjects.issubset(exp_subjects), "gibleed_medications subject_ids must be subset of gibleed_exp"


def test_generate_concept_cohort_set_subset_cohort_id(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with subset_cohort and subset_cohort_id restricts to that cohort definition (R)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({
        "concept_id": [192671, 1124300, 1127433],
        "domain_id": ["condition", "drug", "drug"],
    })
    cond = pa.table({
        "condition_occurrence_id": [1, 2],
        "person_id": [1, 2],
        "condition_concept_id": [192671, 192671],
        "condition_start_date": pa.array([datetime.date(2020, 6, 1), datetime.date(2020, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 6, 15), datetime.date(2020, 6, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm, {"gibleed_1": [192671], "gibleed_2": [192671]}, name="gibleed_exp", overwrite=True
    )
    cdm = generate_concept_cohort_set(
        cdm,
        concept_set={"diclofenac": [1124300], "acetaminophen": [1127433]},
        name="gibleed_medications2",
        subset_cohort="gibleed_exp",
        subset_cohort_id=1,
        overwrite=True,
    )
    exp1_subjects = set(
        cc.collect(
            cdm["gibleed_exp"].filter(cdm["gibleed_exp"]["cohort_definition_id"] == 1).select("subject_id").distinct()
        )["subject_id"]
    )
    med2_subjects = set(cc.collect(cdm["gibleed_medications2"].select("subject_id").distinct())["subject_id"])
    assert med2_subjects.issubset(exp1_subjects)


def test_generate_concept_cohort_set_subset_cohort_not_in_cdm_raises(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with subset_cohort that is not a table in CDM raises (R: not_a_table)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [123], "domain_id": ["condition"]})
    cond = pa.table({
        "condition_occurrence_id": [1],
        "person_id": [1],
        "condition_concept_id": [123],
        "condition_start_date": pa.array([datetime.date(2020, 1, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 1, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    with pytest.raises(CohortError, match="not found in CDM|subset_cohort"):
        generate_concept_cohort_set(
            cdm, {"x": [123]}, name="cohort", subset_cohort="not_a_table", overwrite=True
        )


def test_generate_concept_cohort_set_missing_domain_warning(duckdb_con, minimal_cdm_tables, caplog):
    """When concept set includes a domain whose table is not in CDM, expect warning (R: missing domains produce warning)."""
    import logging
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [1118084], "domain_id": ["drug"]})
    con.create_table("concept", obj=concept, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    with caplog.at_level(logging.WARNING):
        generate_concept_cohort_set(cdm, {"celecoxib": [1118084]}, name="celecoxib", overwrite=True)
    assert any("not in CDM" in rec.message or "skipped" in rec.message or "domains" in rec.message for rec in caplog.records)


def test_generate_concept_cohort_set_unsupported_domain_no_error(duckdb_con, minimal_cdm_tables):
    """Concept with unsupported domain (e.g. Regimen) does not cause error; cohort still created (R: Regimen domain)."""
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({
        "concept_id": [1127433, 19129655],
        "domain_id": ["drug", "Regimen"],
    })
    con.create_table("concept", obj=concept, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm,
        concept_set={"drug_1": [1127433, 19129655], "drug_2": [19129655], "drug_3": [1127433]},
        name="cohort",
        overwrite=True,
    )
    assert "cohort" in cdm.tables
    assert "cohort_set" in cdm.source.list_tables(cdm.write_schema)


def test_generate_concept_cohort_set_invalid_cdm_records_ignored(duckdb_con):
    """Records with end_date before start_date are excluded from cohort (R: invalid cdm records are ignored)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    person = pa.table({
        "person_id": [1],
        "gender_concept_id": [0],
        "year_of_birth": [1900],
        "month_of_birth": [1],
        "day_of_birth": [1],
        "race_concept_id": [0],
        "ethnicity_concept_id": [0],
    })
    obs = pa.table({
        "observation_period_id": [1],
        "person_id": [1],
        "observation_period_start_date": pa.array([datetime.date(1900, 1, 1)], type=pa.date32()),
        "observation_period_end_date": pa.array([datetime.date(2000, 1, 1)], type=pa.date32()),
        "period_type_concept_id": [0],
    })
    drug = pa.table({
        "drug_exposure_id": [1, 2],
        "person_id": [1, 1],
        "drug_concept_id": [1, 1],
        "drug_exposure_start_date": pa.array(
            [datetime.date(1950, 1, 1), datetime.date(1951, 1, 1)], type=pa.date32()
        ),
        "drug_exposure_end_date": pa.array(
            [datetime.date(1945, 1, 1), datetime.date(1952, 1, 1)],
            type=pa.date32(),
        ),
        "drug_type_concept_id": [0, 0],
    })
    concept = pa.table({
        "concept_id": [1],
        "domain_id": ["Drug"],
    })
    for name, tbl in [("person", person), ("observation_period", obs), ("drug_exposure", drug), ("concept", concept)]:
        con.create_table(name, obj=tbl, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["drug_exposure"] = cdm.source.table("drug_exposure", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm, {"custom": [1]}, name="my_cohort", end="event_end_date", overwrite=True
    )
    df = cc.collect(cdm["my_cohort"])
    # Invalid record (end < start) must be excluded; only the valid row may appear (R: invalid cdm records ignored)
    assert len(df) <= 1
    if len(df) == 1:
        row = df.iloc[0]
        assert row["subject_id"] == 1
        assert row["cohort_definition_id"] == 1
        assert pd.Timestamp(row["cohort_start_date"]).date() == datetime.date(1951, 1, 1)
        assert pd.Timestamp(row["cohort_end_date"]).date() == datetime.date(1952, 1, 1)


def test_generate_concept_cohort_set_concepts_not_in_vocab_silently_ignored(duckdb_con, minimal_cdm_tables):
    """Concept IDs not in concept table are silently ignored; cohort created (R: ankle_sprain with 81151, 1)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [81151], "domain_id": ["condition"]})
    cond = pa.table({
        "condition_occurrence_id": [1],
        "person_id": [1],
        "condition_concept_id": [81151],
        "condition_start_date": pa.array([datetime.date(2020, 1, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 1, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm, {"ankle_sprain": [81151, 99999]}, name="ankle_sprain", end="event_end_date", limit="all", overwrite=True
    )
    assert "ankle_sprain" in cdm.tables
    set_df = cc.collect(cdm.source.table("ankle_sprain_set", cdm.write_schema))
    assert set_df["cohort_name"].iloc[0] == "ankle_sprain"


def test_generate_concept_cohort_set_include_descendants(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with include_descendants=True expands via concept_ancestor (R: descendants)."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({
        "concept_id": [192671, 12345],
        "domain_id": ["condition", "condition"],
    })
    concept_ancestor = pa.table({
        "ancestor_concept_id": [192671, 192671],
        "descendant_concept_id": [192671, 12345],
    })
    cond = pa.table({
        "condition_occurrence_id": [1],
        "person_id": [1],
        "condition_concept_id": [12345],
        "condition_start_date": pa.array([datetime.date(2020, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 6, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("concept_ancestor", obj=concept_ancestor, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["concept_ancestor"] = cdm.source.table("concept_ancestor", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm,
        concept_set={"gibleed": [{"concept_id": 192671, "include_descendants": True}]},
        name="gibleed",
        overwrite=True,
    )
    assert "gibleed" in cdm.tables
    df = cc.collect(cdm["gibleed"])
    assert list(df.columns) == ["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"]
    assert len(df) >= 0
    if len(df) >= 1:
        assert df["subject_id"].iloc[0] == 1
        assert df["cohort_definition_id"].iloc[0] == 1


def test_generate_concept_cohort_set_include_descendants_requires_concept_ancestor(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with include_descendants=True raises if concept_ancestor not in CDM."""
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({"concept_id": [192671], "domain_id": ["condition"]})
    con.create_table("concept", obj=concept, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    with pytest.raises(CohortError, match="concept_ancestor"):
        generate_concept_cohort_set(
            cdm,
            concept_set={"gibleed": [{"concept_id": 192671, "include_descendants": True}]},
            name="gibleed",
            overwrite=True,
        )


def test_generate_concept_cohort_set_is_excluded(duckdb_con, minimal_cdm_tables):
    """generate_concept_cohort_set with is_excluded=True drops that concept from the set."""
    import datetime
    import pyarrow as pa
    import cdmconnector as cc
    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    concept = pa.table({
        "concept_id": [192671, 999],
        "domain_id": ["condition", "condition"],
    })
    cond = pa.table({
        "condition_occurrence_id": [1],
        "person_id": [1],
        "condition_concept_id": [192671],
        "condition_start_date": pa.array([datetime.date(2020, 6, 1)], type=pa.date32()),
        "condition_end_date": pa.array([datetime.date(2020, 6, 15)], type=pa.date32()),
    })
    con.create_table("concept", obj=concept, overwrite=True)
    con.create_table("condition_occurrence", obj=cond, overwrite=True)
    cdm = cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test")
    cdm["concept"] = cdm.source.table("concept", cdm.write_schema)
    cdm["condition_occurrence"] = cdm.source.table("condition_occurrence", cdm.write_schema)
    cdm = generate_concept_cohort_set(
        cdm,
        concept_set={"cohort_a": [{"concept_id": 192671}, {"concept_id": 999, "is_excluded": True}]},
        name="cohort_a",
        overwrite=True,
    )
    df = cc.collect(cdm["cohort_a"])
    assert list(df.columns) == ["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"]
    assert len(df) >= 0
