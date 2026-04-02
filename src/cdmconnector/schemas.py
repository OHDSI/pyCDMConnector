# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""OMOP CDM table and column definitions (schemas).

Loads CSV field definitions from the package's inst/csv/ directory using
``importlib.resources``.
"""

from __future__ import annotations

import importlib.resources
import re
from functools import lru_cache
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Cohort table constants
# ---------------------------------------------------------------------------

COHORT_TABLE_COLUMNS: tuple[str, ...] = (
    "cohort_definition_id",
    "subject_id",
    "cohort_start_date",
    "cohort_end_date",
)

COHORT_SET_COLUMNS: tuple[str, ...] = (
    "cohort_definition_id",
    "cohort_name",
)

COHORT_ATTRITION_COLUMNS: tuple[str, ...] = (
    "cohort_definition_id",
    "number_subjects",
    "number_records",
    "reason_id",
    "reason",
    "excluded_subjects",
    "excluded_records",
)

# ---------------------------------------------------------------------------
# Field-table column mapping (table -> start_date, end_date, etc.)
# ---------------------------------------------------------------------------

FIELD_TABLES_COLUMNS: dict[str, dict[str, str | None]] = {
    "observation_period": {
        "start_date": "observation_period_start_date",
        "end_date": "observation_period_end_date",
        "standard_concept": None,
        "source_concept": None,
        "type_concept": "period_type_concept_id",
        "unique_id": "observation_period_id",
        "domain_id": None,
        "person_id": "person_id",
    },
    "visit_occurrence": {
        "start_date": "visit_start_date",
        "end_date": "visit_end_date",
        "standard_concept": "visit_concept_id",
        "source_concept": "visit_source_concept_id",
        "type_concept": "visit_type_concept_id",
        "unique_id": "visit_occurrence_id",
        "domain_id": "visit",
        "person_id": "person_id",
    },
    "condition_occurrence": {
        "start_date": "condition_start_date",
        "end_date": "condition_end_date",
        "standard_concept": "condition_concept_id",
        "source_concept": "condition_source_concept_id",
        "type_concept": "condition_type_concept_id",
        "unique_id": "condition_occurrence_id",
        "domain_id": "condition",
        "person_id": "person_id",
    },
    "drug_exposure": {
        "start_date": "drug_exposure_start_date",
        "end_date": "drug_exposure_end_date",
        "standard_concept": "drug_concept_id",
        "source_concept": "drug_source_concept_id",
        "type_concept": "drug_type_concept_id",
        "unique_id": "drug_exposure_id",
        "domain_id": "drug",
        "person_id": "person_id",
    },
    "death": {
        "start_date": "death_date",
        "end_date": "death_date",
        "standard_concept": "cause_concept_id",
        "source_concept": "cause_source_concept_id",
        "type_concept": "death_type_concept_id",
        "unique_id": "person_id",
        "domain_id": None,
        "person_id": "person_id",
    },
    "procedure_occurrence": {
        "start_date": "procedure_date",
        "end_date": "procedure_date",
        "standard_concept": "procedure_concept_id",
        "source_concept": "procedure_source_concept_id",
        "type_concept": "procedure_type_concept_id",
        "unique_id": "procedure_occurrence_id",
        "domain_id": "procedure",
        "person_id": "person_id",
    },
    "device_exposure": {
        "start_date": "device_exposure_start_date",
        "end_date": "device_exposure_end_date",
        "standard_concept": "device_concept_id",
        "source_concept": "device_source_concept_id",
        "type_concept": "device_type_concept_id",
        "unique_id": "device_exposure_id",
        "domain_id": "device",
        "person_id": "person_id",
    },
    "measurement": {
        "start_date": "measurement_date",
        "end_date": "measurement_date",
        "standard_concept": "measurement_concept_id",
        "source_concept": "measurement_source_concept_id",
        "type_concept": "measurement_type_concept_id",
        "unique_id": "measurement_id",
        "domain_id": "measurement",
        "person_id": "person_id",
    },
    "observation": {
        "start_date": "observation_date",
        "end_date": "observation_date",
        "standard_concept": "observation_concept_id",
        "source_concept": "observation_source_concept_id",
        "type_concept": "observation_type_concept_id",
        "unique_id": "observation_id",
        "domain_id": "observation",
        "person_id": "person_id",
    },
    "specimen": {
        "start_date": "specimen_date",
        "end_date": "specimen_date",
        "standard_concept": "specimen_concept_id",
        "source_concept": "specimen_source_concept_id",
        "type_concept": "specimen_type_concept_id",
        "unique_id": "specimen_id",
        "domain_id": "specimen",
        "person_id": "person_id",
    },
    "visit_detail": {
        "start_date": "visit_detail_start_date",
        "end_date": "visit_detail_end_date",
        "standard_concept": "visit_detail_concept_id",
        "source_concept": "visit_detail_source_concept_id",
        "type_concept": "visit_detail_type_concept_id",
        "unique_id": "visit_detail_id",
        "domain_id": "visit_detail",
        "person_id": "person_id",
    },
    "episode": {
        "start_date": "episode_start_date",
        "end_date": "episode_end_date",
        "standard_concept": "episode_concept_id",
        "source_concept": "episode_source_concept_id",
        "type_concept": "episode_type_concept_id",
        "unique_id": "episode_id",
        "domain_id": "episode",
        "person_id": "person_id",
    },
}

# Mapping from OMOP domain_id (lowercase) to OMOP table name
DOMAIN_TABLE_MAP: dict[str, str] = {
    "condition": "condition_occurrence",
    "drug": "drug_exposure",
    "procedure": "procedure_occurrence",
    "device": "device_exposure",
    "measurement": "measurement",
    "observation": "observation",
    "specimen": "specimen",
    "visit": "visit_occurrence",
    "visit_detail": "visit_detail",
    "episode": "episode",
}

# Achilles tables (optional)
ACHILLES_TABLES: tuple[str, ...] = (
    "achilles_analysis",
    "achilles_results",
    "achilles_results_dist",
)

# ---------------------------------------------------------------------------
# Internal CSV loading helpers
# ---------------------------------------------------------------------------

_SUPPORTED_VERSIONS = ("5.3", "5.4")


def _csv_package():
    """Return the importlib.resources anchor for the CSV data directory."""
    return importlib.resources.files("cdmconnector.inst.csv")


def _map_cdm_datatype(dtype: str) -> str:
    """Map a raw cdmDatatype string to a simplified type label."""
    if not isinstance(dtype, str):
        return "varchar"
    d = dtype.strip().lower()
    if d == "integer":
        return "integer"
    if d in ("date",):
        return "date"
    if d in ("datetime", "timestamp"):
        return "datetime"
    if d in ("float", "numeric", "real", "double"):
        return "float"
    # varchar(...), text, etc.
    return "varchar"


@lru_cache(maxsize=4)
def _load_field_level(version: str) -> pd.DataFrame:
    """Load and cache a Field_Level CSV for the given CDM version."""
    filename = f"OMOP_CDMv{version}_Field_Level.csv"
    csv_path = _csv_package() / filename
    with importlib.resources.as_file(csv_path) as path:
        df = pd.read_csv(path)
    # Normalise column names to snake_case
    df = df.rename(columns={
        "cdmTableName": "cdm_table_name",
        "cdmFieldName": "cdm_field_name",
        "isRequired": "is_required",
        "cdmDatatype": "cdm_datatype",
    })
    # Coerce is_required to bool
    df["is_required"] = df["is_required"].map(
        lambda v: str(v).strip().upper() == "TRUE"
    )
    # Derive simplified 'type' column from cdm_datatype
    df["type"] = df["cdm_datatype"].apply(_map_cdm_datatype)
    return df


@lru_cache(maxsize=4)
def _load_table_level(version: str) -> pd.DataFrame:
    """Load and cache a Table_Level CSV for the given CDM version."""
    filename = f"OMOP_CDMv{version}_Table_Level.csv"
    csv_path = _csv_package() / filename
    with importlib.resources.as_file(csv_path) as path:
        df = pd.read_csv(path)
    df = df.rename(columns={"cdmTableName": "cdm_table_name"})
    return df


def _check_version(version: str) -> None:
    """Raise ValueError if version is not supported."""
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unsupported CDM version: {version}. Use '5.3' or '5.4'."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cohort_columns(table_type: str) -> tuple[str, ...]:
    """Return column names for cohort / cohort_set / cohort_attrition.

    Parameters
    ----------
    table_type : str
        One of ``"cohort"``, ``"cohort_set"``, ``"cohort_attrition"``.

    Returns
    -------
    tuple[str, ...]
        Column names for that table.

    Raises
    ------
    ValueError
        If *table_type* is not recognised.
    """
    if table_type == "cohort":
        return COHORT_TABLE_COLUMNS
    if table_type == "cohort_set":
        return COHORT_SET_COLUMNS
    if table_type == "cohort_attrition":
        return COHORT_ATTRITION_COLUMNS
    raise ValueError(
        f"Unknown cohort table: {table_type}. "
        "Use 'cohort', 'cohort_set', or 'cohort_attrition'."
    )


def omop_tables(version: str = "5.3") -> tuple[str, ...]:
    """Return OMOP CDM table names for the given version.

    Parameters
    ----------
    version : str, optional
        CDM version (``"5.3"`` or ``"5.4"``); default ``"5.3"``.

    Returns
    -------
    tuple[str, ...]
        OMOP table names read from the Table_Level CSV.

    Raises
    ------
    ValueError
        If *version* is not ``"5.3"`` or ``"5.4"``.
    """
    _check_version(version)
    df = _load_table_level(version)
    return tuple(df["cdm_table_name"].tolist())


def omop_table_fields(version: str = "5.3") -> pd.DataFrame:
    """Return OMOP table/field metadata as a DataFrame.

    The returned DataFrame has columns:

    * ``cdm_table_name``
    * ``cdm_field_name``
    * ``is_required`` (bool)
    * ``cdm_datatype`` (original datatype string from the CSV)
    * ``type`` (simplified: ``"integer"``, ``"date"``, ``"datetime"``,
      ``"float"``, or ``"varchar"``)

    Parameters
    ----------
    version : str, optional
        CDM version (``"5.3"`` or ``"5.4"``); default ``"5.3"``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If *version* is not ``"5.3"`` or ``"5.4"``.
    """
    _check_version(version)
    return _load_field_level(version).copy()


def omop_columns(
    table: str,
    version: str = "5.3",
    field: str | None = None,
) -> tuple[str, ...] | str:
    """Return column names for an OMOP table, or a specific field mapping.

    Parameters
    ----------
    table : str
        CDM table name (e.g. ``"person"``, ``"condition_occurrence"``) or a
        cohort table (``"cohort"``, ``"cohort_set"``, ``"cohort_attrition"``).
    version : str, optional
        CDM version (``"5.3"`` or ``"5.4"``); default ``"5.3"``.
    field : str or None, optional
        If *None*, return a tuple of all column names for the table.
        If given, return the mapped column name for that logical field.
        Supported field keys for domain tables: ``"start_date"``,
        ``"end_date"``, ``"person_id"``, ``"standard_concept"``,
        ``"source_concept"``, ``"type_concept"``, ``"unique_id"``,
        ``"domain_id"``.

    Returns
    -------
    tuple[str, ...] or str
        Tuple of column names when *field* is None, or a single column name
        string when *field* is given.

    Raises
    ------
    ValueError
        If the table is unknown, the version is unsupported, the field key is
        unknown, or the field maps to ``None`` for that table.
    """
    _check_version(version)
    table_lower = table.lower()

    # Cohort tables delegate to cohort_columns()
    if table_lower in ("cohort", "cohort_set", "cohort_attrition"):
        if field is not None:
            raise ValueError(
                f"Field mapping is not supported for cohort table {table!r}."
            )
        return cohort_columns(table_lower)

    if field is not None:
        # Field mapping via FIELD_TABLES_COLUMNS
        if table_lower not in FIELD_TABLES_COLUMNS:
            raise ValueError(
                f"Table {table!r} has no field mapping. "
                "Use omop_columns(table) without 'field' for column list, "
                "or a table in FIELD_TABLES_COLUMNS."
            )
        cols = FIELD_TABLES_COLUMNS[table_lower]
        if field not in cols:
            raise ValueError(
                f"Unknown field {field!r}. Choose from: {list(cols.keys())}"
            )
        val = cols[field]
        if val is None:
            raise ValueError(
                f"Table {table!r} has no column for field {field!r}."
            )
        return val

    # Return all columns for this table from the Field_Level CSV
    df = _load_field_level(version)
    subset = df[df["cdm_table_name"] == table_lower]
    if len(subset) == 0:
        raise ValueError(
            f"Unknown table {table!r}. Use omop_tables() for CDM table names, "
            "or 'cohort', 'cohort_set', 'cohort_attrition' for cohort tables."
        )
    return tuple(subset["cdm_field_name"].tolist())


# ---------------------------------------------------------------------------
# Achilles helpers
# ---------------------------------------------------------------------------


def achilles_tables(version: str = "5.3") -> tuple[str, ...]:
    """Return Achilles analysis result table names.

    Parameters
    ----------
    version : str, optional
        CDM version (currently unused; Achilles tables are the same for all
        versions).

    Returns
    -------
    tuple[str, ...]
        Achilles table names.
    """
    return ACHILLES_TABLES


def achilles_columns(table: str, version: str = "5.3") -> tuple[str, ...]:
    """Return column names for an Achilles table.

    Parameters
    ----------
    table : str
        One of "achilles_analysis", "achilles_results", "achilles_results_dist".
    version : str, optional
        CDM version (currently unused).

    Returns
    -------
    tuple[str, ...]
        Column names.

    Raises
    ------
    ValueError
        If table is not a valid Achilles table.
    """
    if table not in ACHILLES_TABLES:
        raise ValueError(
            f"Unknown Achilles table: {table!r}. "
            f"Use one of: {ACHILLES_TABLES}"
        )
    df = omop_table_fields(version)
    subset = df[df["cdm_table_name"] == table]
    return tuple(subset["cdm_field_name"].tolist())


def cohort_tables(cdm: Any) -> list[str]:
    """List cohort table names registered in a CDM.

    Identifies cohort tables by checking which tables in the CDM have the
    required cohort columns (cohort_definition_id, subject_id, etc.).

    Parameters
    ----------
    cdm : Cdm
        CDM reference.

    Returns
    -------
    list[str]
        Names of tables that appear to be cohort tables.
    """
    required = set(COHORT_TABLE_COLUMNS)
    result = []
    tables = getattr(cdm, "tables", [])
    for name in tables:
        try:
            tbl = cdm[name]
            cols = set()
            if hasattr(tbl, "columns"):
                cols = set(tbl.columns)
            elif hasattr(tbl, "schema"):
                cols = set(tbl.schema().names)
            if required.issubset(cols):
                result.append(name)
        except Exception:
            continue
    return sorted(result)


def omop_data_folder() -> str:
    """Return the path to the bundled OMOP data folder.

    Returns
    -------
    str
        Path to the inst/ directory containing OMOP reference data.
    """
    from pathlib import Path

    inst = Path(__file__).parent / "inst"
    return str(inst)
