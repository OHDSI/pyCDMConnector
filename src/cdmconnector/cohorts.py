# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Cohort table utilities: cohort_count, attrition, record_cohort_attrition, new_cohort_table."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cdmconnector.exceptions import CohortError
from cdmconnector.logging_config import get_logger
from cdmconnector.schemas import COHORT_TABLE_COLUMNS

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass


def cohort_count(cohort: Any) -> Any:
    """
    Get cohort counts from a cohort table (or cohort_table wrapper).

    Expects cohort to have a cohort_attrition attribute (table or DataFrame) with
    cohort_definition_id, number_records, number_subjects. Returns the latest
    reason_id row per cohort_definition_id.

    Returns
    -------
    DataFrame or table with columns cohort_definition_id, number_records, number_subjects.

    Raises
    ------
    CohortError
        If cohort has no cohort_attrition attribute.
    """
    from cdmconnector.utils import to_dataframe

    attr = getattr(cohort, "cohort_attrition", None)
    if attr is None:
        logger.error("Cohort count requires cohort_attrition attribute.")
        raise CohortError("Cohort count requires cohort_attrition attribute.")
    logger.debug("Computing cohort count from cohort_attrition.")
    df = to_dataframe(attr) if (hasattr(attr, "schema") or hasattr(attr, "op")) else attr
    import pandas as pd
    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        return pd.DataFrame(columns=["cohort_definition_id", "number_records", "number_subjects"])
    if not hasattr(df, "groupby"):
        df = pd.DataFrame(df)
    latest = df.loc[df.groupby("cohort_definition_id")["reason_id"].idxmax()]
    return latest[["cohort_definition_id", "number_records", "number_subjects"]].sort_values("cohort_definition_id")


def attrition(cohort: Any) -> Any:
    """
    Get full attrition table from a cohort (cohort_attrition attribute).

    Returns
    -------
    Table or DataFrame with cohort_definition_id, reason_id, reason,
    number_subjects, number_records, excluded_subjects, excluded_records.

    Raises
    ------
    CohortError
        If cohort has no cohort_attrition attribute.
    """
    from cdmconnector.utils import to_dataframe

    attr = getattr(cohort, "cohort_attrition", None)
    if attr is None:
        logger.error("Attrition requires cohort_attrition attribute.")
        raise CohortError("Attrition requires cohort_attrition attribute.")
    logger.debug("Returning full attrition table.")
    return to_dataframe(attr) if (hasattr(attr, "schema") or hasattr(attr, "op")) else attr


def settings(cohort: Any) -> Any:
    """
    Get cohort settings (cohort_set attribute).

    Returns
    -------
    DataFrame with cohort_definition_id, cohort_name, and any additional columns.

    Raises
    ------
    CohortError
        If cohort has no cohort_set attribute.
    """
    from cdmconnector.utils import to_dataframe

    cs = getattr(cohort, "cohort_set", None)
    if cs is None:
        logger.error("settings() requires cohort_set attribute.")
        raise CohortError("settings() requires cohort_set attribute.")
    return to_dataframe(cs) if (hasattr(cs, "schema") or hasattr(cs, "op")) else cs


def cohort_codelist(cohort: Any) -> Any:
    """
    Get concept lists used in cohort definitions (cohort_codelist attribute).

    Returns
    -------
    DataFrame with cohort_definition_id, codelist_name, concept_id, codelist_type.
    Returns empty DataFrame if no cohort_codelist attribute exists.
    """
    import pandas as pd

    from cdmconnector.utils import to_dataframe

    cl = getattr(cohort, "cohort_codelist", None)
    if cl is None:
        return pd.DataFrame(
            columns=["cohort_definition_id", "codelist_name", "concept_id", "codelist_type"]
        )
    return to_dataframe(cl) if (hasattr(cl, "schema") or hasattr(cl, "op")) else cl


def get_cohort_id(cohort: Any, cohort_name: str | list[str]) -> dict[str, int]:
    """
    Get cohort_definition_id(s) from cohort name(s).

    Parameters
    ----------
    cohort : Cohort table with cohort_set attribute.
    cohort_name : str or list[str]
        Cohort name(s) to look up.

    Returns
    -------
    dict[str, int]
        Mapping of cohort_name to cohort_definition_id.

    Raises
    ------
    CohortError
        If cohort has no cohort_set attribute.
    ValueError
        If a name is not found.
    """
    cs = settings(cohort)
    names = [cohort_name] if isinstance(cohort_name, str) else list(cohort_name)
    result = {}
    for name in names:
        match = cs[cs["cohort_name"] == name]
        if match.empty:
            available = cs["cohort_name"].tolist()
            raise ValueError(
                f"Cohort name '{name}' not found. Available: {available}"
            )
        result[name] = int(match["cohort_definition_id"].iloc[0])
    return result


def get_cohort_name(cohort: Any, cohort_id: int | list[int]) -> dict[int, str]:
    """
    Get cohort name(s) from cohort_definition_id(s).

    Parameters
    ----------
    cohort : Cohort table with cohort_set attribute.
    cohort_id : int or list[int]
        Cohort definition ID(s) to look up.

    Returns
    -------
    dict[int, str]
        Mapping of cohort_definition_id to cohort_name.

    Raises
    ------
    CohortError
        If cohort has no cohort_set attribute.
    ValueError
        If an ID is not found.
    """
    cs = settings(cohort)
    ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
    result = {}
    for cid in ids:
        match = cs[cs["cohort_definition_id"] == cid]
        if match.empty:
            available = cs["cohort_definition_id"].tolist()
            raise ValueError(
                f"Cohort ID {cid} not found. Available: {available}"
            )
        result[cid] = str(match["cohort_name"].iloc[0])
    return result


def record_cohort_attrition(
    cohort: Any,
    reason: str,
    cohort_id: int | list[int] | None = None,
) -> Any:
    """
    Record a new attrition step: update cohort_attrition with a new reason row.

    Parameters
    ----------
    cohort : Cohort table (Ibis table or wrapper with cohort_attrition attribute).
    reason : Description of the attrition step.
    cohort_id : cohort_definition_id(s) to update; if None, all.

    Returns
    -------
    Updated cohort (if it has cohort_attrition attribute) or combined attrition DataFrame.

    Raises
    ------
    CohortError
        If cohort has no cohort_attrition attribute.
    ValueError
        If reason is empty.
    """
    import pandas as pd

    from cdmconnector.cdm import collect
    from cdmconnector.utils import to_dataframe

    if not reason or not str(reason).strip():
        logger.error("Attrition reason cannot be empty.")
        raise ValueError("Attrition reason cannot be empty.")

    attr = getattr(cohort, "cohort_attrition", None)
    if attr is None:
        logger.error("record_cohort_attrition requires cohort_attrition attribute.")
        raise CohortError("record_cohort_attrition requires cohort_attrition attribute.")
    logger.info("Recording cohort attrition: reason=%s, cohort_id=%s", reason[:50] if reason else "", cohort_id)

    old_attr = to_dataframe(attr) if (hasattr(attr, "schema") or hasattr(attr, "op")) else attr
    if not isinstance(old_attr, pd.DataFrame):
        old_attr = pd.DataFrame(old_attr)

    cohort_df = to_dataframe(cohort) if (
        hasattr(cohort, "schema") or hasattr(cohort, "op") or hasattr(cohort, "execute") or hasattr(cohort, "to_pandas")
    ) else cohort
    if not isinstance(cohort_df, pd.DataFrame):
        cohort_df = pd.DataFrame(cohort_df)

    if isinstance(cohort_id, list):
        ids = cohort_id
    elif cohort_id is not None:
        ids = [cohort_id]
    else:
        ids = cohort_df["cohort_definition_id"].unique().tolist()
    max_reason = old_attr.groupby("cohort_definition_id")["reason_id"].max()
    new_rows = []
    for cid in ids:
        prev = old_attr[(old_attr["cohort_definition_id"] == cid) & (old_attr["reason_id"] == max_reason.get(cid, 0))]
        prev_rec = prev["number_records"].iloc[0] if len(prev) else 0
        prev_sub = prev["number_subjects"].iloc[0] if len(prev) else 0
        curr = cohort_df[cohort_df["cohort_definition_id"] == cid]
        n_rec = len(curr)
        n_sub = curr["subject_id"].nunique() if "subject_id" in curr.columns else n_rec
        new_rows.append({
            "cohort_definition_id": cid,
            "number_subjects": int(n_sub),
            "number_records": int(n_rec),
            "reason_id": int(max_reason.get(cid, 0) + 1),
            "reason": reason,
            "excluded_subjects": int(prev_sub - n_sub),
            "excluded_records": int(prev_rec - n_rec),
        })
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([old_attr, new_df], ignore_index=True)
    # Attach back to cohort if it's a wrapper; otherwise return combined attrition
    if hasattr(cohort, "cohort_attrition"):
        cohort.cohort_attrition = combined
        return cohort
    return combined


def new_cohort_table(
    cdm: Any,
    name: str,
    *,
    overwrite: bool = True,
) -> Any:
    """
    Create an empty cohort table in the CDM's write schema and register it.

    Creates tables: name (cohort), name_set (cohort_set), name_attrition (cohort_attrition).
    Returns the CDM with the new cohort table added.

    Parameters
    ----------
    cdm : Cdm (database-backed, from cdm_from_con).
    name : Table name (lowercase recommended).
    overwrite : If True, replace existing tables.

    Returns
    -------
    The same Cdm with the new cohort table added.

    Raises
    ------
    CohortError
        If cdm has no database source (e.g. from cdm_from_tables without a backend).
    """
    import pyarrow as pa

    cohort_schema = pa.schema([
        ("cohort_definition_id", pa.int64()),
        ("subject_id", pa.int64()),
        ("cohort_start_date", pa.date32()),
        ("cohort_end_date", pa.date32()),
    ])
    empty_cohort = pa.table({c: pa.array([], type=cohort_schema.field(c).type) for c in COHORT_TABLE_COLUMNS})
    src = getattr(cdm, "source", None)
    if src is None or not hasattr(src, "insert_table"):
        logger.error("new_cohort_table requires a CDM with a database source (cdm_from_con).")
        raise CohortError("new_cohort_table requires a CDM with a database source (cdm_from_con).")
    logger.info("Creating new cohort table: name=%s, overwrite=%s", name, overwrite)
    src.insert_table(name, empty_cohort, overwrite=overwrite)
    set_name = f"{name}_set"
    attr_name = f"{name}_attrition"
    set_schema = pa.schema([("cohort_definition_id", pa.int64()), ("cohort_name", pa.string())])
    attr_schema = pa.schema([
        ("cohort_definition_id", pa.int64()),
        ("number_subjects", pa.int64()),
        ("number_records", pa.int64()),
        ("reason_id", pa.int64()),
        ("reason", pa.string()),
        ("excluded_subjects", pa.int64()),
        ("excluded_records", pa.int64()),
    ])
    empty_set = pa.table({k: pa.array([], type=set_schema.field(k).type) for k in set_schema.names})
    empty_attr = pa.table({k: pa.array([], type=attr_schema.field(k).type) for k in attr_schema.names})
    src.insert_table(set_name, empty_set, overwrite=overwrite)
    src.insert_table(attr_name, empty_attr, overwrite=overwrite)
    cdm[name] = src.table(name, cdm.write_schema)
    logger.debug("Registered cohort table %s and %s_set, %s_attrition.", name, name, name)
    return cdm


def _create_circe_stats_tables(src: Any, name: str, overwrite: bool = True) -> None:
    """Create empty CIRCE stats tables (inclusion, inclusion_result, etc.) for cohort SQL.

    Parameters
    ----------
    src : Any
        DbCdmSource with insert_table.
    name : str
        Cohort table stem (e.g. "cohort" -> cohort_inclusion, cohort_inclusion_result, ...).
    overwrite : bool, optional
        If True, replace existing tables (default True).
    """
    import pyarrow as pa

    tables_schema = [
        (f"{name}_inclusion", [("cohort_definition_id", pa.int64()), ("rule_sequence", pa.int64()), ("name", pa.string()), ("description", pa.string())]),
        (f"{name}_inclusion_result", [("cohort_definition_id", pa.int64()), ("inclusion_rule_mask", pa.int64()), ("person_count", pa.int64()), ("mode_id", pa.int64())]),
        (f"{name}_inclusion_stats", [("cohort_definition_id", pa.int64()), ("rule_sequence", pa.int64()), ("person_count", pa.int64()), ("gain_count", pa.int64()), ("person_total", pa.int64()), ("mode_id", pa.int64())]),
        (f"{name}_summary_stats", [("cohort_definition_id", pa.int64()), ("base_count", pa.int64()), ("final_count", pa.int64()), ("mode_id", pa.int64())]),
        (f"{name}_censor_stats", [("cohort_definition_id", pa.int64()), ("lost_count", pa.int64())]),
    ]
    for table_name, fields in tables_schema:
        empty = pa.table({f[0]: [] for f in fields})
        src.insert_table(table_name, empty, overwrite=overwrite)


def read_cohort_set(path: str | Path) -> Any:
    """
    Read a set of cohort definitions from a folder (mirrors R readCohortSet).

    A cohort set is a collection of CIRCE cohort definitions. On disk this is
    a folder with either:
    - A CohortsToCreate.csv with columns cohortId, cohortName, jsonPath; or
    - No CSV: all .json files in the folder are used, with cohort_definition_id
      assigned in alphabetical order and cohort_name from file names (sanitized).

    Parameters
    ----------
    path : str or Path
        Path to a folder containing CIRCE cohort definition JSON files and
        optionally CohortsToCreate.csv.

    Returns
    -------
    pandas.DataFrame
        With columns cohort_definition_id, cohort_name, cohort (list of dicts),
        json (str per row), and cohort_name_snakecase. Ready for generate_cohort_set.

    Raises
    ------
    CohortError
        If path is not a directory or required files are missing.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        logger.error("read_cohort_set: directory does not exist: %s", path)
        raise CohortError(f"The directory {path} does not exist.")
    if not path.is_dir():
        logger.error("read_cohort_set: path is not a directory: %s", path)
        raise CohortError(f"{path} is not a directory.")

    logger.info("Reading cohort set from path: %s", path)
    csv_file = path / "CohortsToCreate.csv"
    if csv_file.exists():
        cohorts_df = pd.read_csv(csv_file)
        if "cohortId" in cohorts_df.columns and "cohort_definition_id" not in cohorts_df.columns:
            cohorts_df["cohort_definition_id"] = cohorts_df["cohortId"]
        if "cohortName" in cohorts_df.columns and "cohort_name" not in cohorts_df.columns:
            cohorts_df["cohort_name"] = cohorts_df["cohortName"]
        if "jsonPath" in cohorts_df.columns:
            json_paths = [path / p for p in cohorts_df["jsonPath"]]
        else:
            logger.error("CohortsToCreate.csv must contain jsonPath.")
            raise CohortError("CohortsToCreate.csv must contain jsonPath (or cohortId, cohortName, jsonPath).")
        logger.debug("Loaded %d cohorts from CohortsToCreate.csv", len(cohorts_df))
        cohorts_df["json_path"] = json_paths
        cohorts_df["json"] = [p.read_text(encoding="utf-8", errors="replace") for p in json_paths]
        cohorts_df["cohort"] = [json.loads(s) for s in cohorts_df["json"]]
        cohorts_df["cohort_name_snakecase"] = cohorts_df["cohort_name"].str.lower().str.replace(r"\s+", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    else:
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            logger.error("No .json files found in %s", path)
            raise CohortError(f"No .json files found in {path}.")
        cohort_definition_id = list(range(1, len(json_files) + 1))
        cohort_name = []
        for f in json_files:
            stem = f.stem.lower().replace(" ", "_")
            stem = re.sub(r"[^a-z0-9_]", "", stem)
            if stem.isdigit():
                cohort_name.append(f"cohort_{stem}")
            else:
                cohort_name.append(stem or f"cohort_{f.stem}")
        json_paths = list(json_files)
        json_strs = [p.read_text(encoding="utf-8", errors="replace") for p in json_paths]
        cohorts_df = pd.DataFrame({
            "cohort_definition_id": cohort_definition_id,
            "cohort_name": cohort_name,
            "json_path": json_paths,
            "json": json_strs,
        })
        cohorts_df["cohort"] = [json.loads(s) for s in cohorts_df["json"]]
        if cohorts_df["cohort_definition_id"].nunique() != len(cohorts_df) or cohorts_df["cohort_name"].nunique() != len(cohorts_df):
            raise CohortError("Cohort IDs and names derived from file names must be unique.")

    # cohort_name_snakecase for column/filename use
    cohorts_df["cohort_name_snakecase"] = cohorts_df["cohort_name"].str.lower().str.replace(r"\s+", "_", regex=True).str.replace(r"[^a-z0-9_]", "", regex=True)
    # Require names to start with a letter
    for i, n in enumerate(cohorts_df["cohort_name"]):
        if n and not n[0].isalpha():
            logger.error("Cohort name must start with a letter: %r", n)
            raise CohortError(f"Cohort names must start with a letter; got {n!r}. Rename the JSON file or use CohortsToCreate.csv.")
    logger.debug("read_cohort_set returning %d cohort definitions", len(cohorts_df))
    return cohorts_df[["cohort_definition_id", "cohort_name", "cohort", "json", "cohort_name_snakecase"]]


def _execute_cohort_sql(cdm: Any, statements: list[str]) -> None:
    """Execute a list of SQL statements on the CDM's backend. Uses raw_sql when available.

    Parameters
    ----------
    cdm : Any
        CDM reference with database source (source.con.raw_sql).
    statements : list[str]
        SQL statements (semicolons added if missing).
    """
    src = getattr(cdm, "source", None)
    if src is None or not hasattr(src, "con"):
        raise CohortError("generate_cohort_set requires a CDM with a database source (cdm_from_con).")
    con = src.con
    raw_sql = getattr(con, "raw_sql", None)
    if raw_sql is None:
        logger.error("Backend does not expose raw_sql; cohort SQL execution requires it.")
        raise CohortError(
            "Cohort SQL execution requires a backend with raw_sql (e.g. DuckDB). "
            "Your Ibis connection does not expose raw_sql."
        )
    logger.debug("Executing %d cohort SQL statement(s)", len(statements))
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt or stmt.startswith("--"):
            continue
        if not stmt.endswith(";"):
            stmt = stmt + ";"
        raw_sql(stmt)


def _validate_cohort_table_name(name: str) -> str:
    """Validate and return normalized cohort table name (lowercase, letter-start, alphanumeric + underscore).

    Parameters
    ----------
    name : str
        Desired cohort table name.

    Returns
    -------
    str
        Normalized name (lowercase, letters/numbers/underscores).

    Raises
    ------
    CohortError
        If name does not start with a letter or contains invalid characters.
    """
    name = name.strip().lower()
    if not name or not name[0].isalpha():
        raise CohortError("Cohort table name must start with a letter.")
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise CohortError("Cohort table name must contain only letters, numbers, and underscores.")
    return name


def _normalize_cohort_definition_set(
    cohort_definition_set: Any,
    compute_attrition: bool,
) -> Any:
    """Normalize cohort_definition_set to a DataFrame with cohort_definition_id, cohort_name, cohort, json, sql.

    Parameters
    ----------
    cohort_definition_set : Any
        DataFrame or object with __dataframe__; must have cohort_definition_id, cohort_name, and json or sql.
    compute_attrition : bool
        If True, generate SQL via Circepy when sql column missing.

    Returns
    -------
    pandas.DataFrame
        With columns cohort_definition_id, cohort_name, cohort, json, sql.
    """
    import pandas as pd

    from cdmconnector import _circe

    # Always produce a pandas DataFrame for .empty / .columns (avoid __dataframe__ interchange)
    cohort_set_df = pd.DataFrame(cohort_definition_set)
    if cohort_set_df.empty:
        logger.error("cohort_definition_set must have at least one row.")
        raise CohortError("cohort_definition_set must have at least one row.")
    required = {"cohort_definition_id", "cohort_name"}
    if not required.issubset(cohort_set_df.columns):
        logger.error("cohort_definition_set missing required columns: cohort_definition_id, cohort_name.")
        raise CohortError("cohort_definition_set must have columns: cohort_definition_id, cohort_name.")
    if "json" not in cohort_set_df.columns and "sql" not in cohort_set_df.columns:
        logger.error("cohort_definition_set must have 'json' or 'sql' column.")
        raise CohortError("cohort_definition_set must have 'json' (CIRCE JSON) or 'sql' (pre-generated SQL).")
    if "cohort" not in cohort_set_df.columns and "json" in cohort_set_df.columns:
        cohort_set_df = cohort_set_df.copy()
        cohort_set_df["cohort"] = [json.loads(s) for s in cohort_set_df["json"]]
    if "json" not in cohort_set_df.columns and "sql" in cohort_set_df.columns:
        cohort_set_df = cohort_set_df.copy()
        cohort_set_df["json"] = [""] * len(cohort_set_df)
    if "sql" not in cohort_set_df.columns:
        cohort_set_df = cohort_set_df.copy()
        sql_list = []
        opts = _circe.create_generate_options(generate_stats=compute_attrition)
        for _, row in cohort_set_df.iterrows():
            try:
                sql_list.append(_circe.build_cohort_query(row["json"], opts))
            except NotImplementedError:
                raise NotImplementedError(
                    "Cohort SQL generation requires Circepy. Reinstall CDMConnector to ensure the circepy dependency is installed, "
                    "or provide cohort_definition_set with a 'sql' column (pre-generated SQL)."
                ) from None
        cohort_set_df["sql"] = sql_list
    else:
        cohort_set_df["sql"] = cohort_set_df["sql"].astype(str)
    return cohort_set_df


def _schema_to_string(schema: Any) -> str:
    """Convert schema spec (str or dict) to a single string for SQL (e.g. schema name or dotted).

    Parameters
    ----------
    schema : str, dict, or None
        Schema spec.

    Returns
    -------
    str
        Schema name string, or "" if None.
    """
    from cdmconnector.utils import resolve_schema_name

    s = resolve_schema_name(schema)
    if s:
        return str(s)
    if isinstance(schema, dict):
        return ".".join(str(v) for v in schema.values() if v)
    return str(schema) if schema is not None else ""


def _create_empty_cohort_tables(
    src: Any,
    name: str,
    *,
    overwrite: bool,
    compute_attrition: bool,
) -> tuple[str, str]:
    """Create empty cohort, cohort_set, cohort_attrition (and CIRCE stats tables if compute_attrition).

    Parameters
    ----------
    src : Any
        DbCdmSource with insert_table.
    name : str
        Cohort table name (e.g. "cohort").
    overwrite : bool
        If True, replace existing tables.
    compute_attrition : bool
        If True, also create CIRCE inclusion/stats tables.

    Returns
    -------
    tuple[str, str]
        (set_name, attr_name) e.g. ("cohort_set", "cohort_attrition").
    """
    import ibis
    import pyarrow as pa

    # Explicit schema so DuckDB accepts empty tables (no NULL-typed columns).
    # Pass Ibis schema to create_table; otherwise Ibis infers NULL for empty date columns.
    cohort_schema_pa = pa.schema([
        ("cohort_definition_id", pa.int64()),
        ("subject_id", pa.int64()),
        ("cohort_start_date", pa.date32()),
        ("cohort_end_date", pa.date32()),
    ])
    empty_cohort = cohort_schema_pa.empty_table()
    cohort_schema_ibis = ibis.schema({
        "cohort_definition_id": "int64",
        "subject_id": "int64",
        "cohort_start_date": "date",
        "cohort_end_date": "date",
    })
    src.insert_table(name, empty_cohort, overwrite=overwrite, schema=cohort_schema_ibis)
    set_name = f"{name}_set"
    attr_name = f"{name}_attrition"
    set_schema_pa = pa.schema([("cohort_definition_id", pa.int64()), ("cohort_name", pa.string())])
    attr_schema_pa = pa.schema([
        ("cohort_definition_id", pa.int64()),
        ("number_subjects", pa.int64()),
        ("number_records", pa.int64()),
        ("reason_id", pa.int64()),
        ("reason", pa.string()),
        ("excluded_subjects", pa.int64()),
        ("excluded_records", pa.int64()),
    ])
    empty_set = set_schema_pa.empty_table()
    empty_attr = attr_schema_pa.empty_table()
    set_schema_ibis = ibis.schema({"cohort_definition_id": "int64", "cohort_name": "string"})
    attr_schema_ibis = ibis.schema({
        "cohort_definition_id": "int64",
        "number_subjects": "int64",
        "number_records": "int64",
        "reason_id": "int64",
        "reason": "string",
        "excluded_subjects": "int64",
        "excluded_records": "int64",
    })
    src.insert_table(set_name, empty_set, overwrite=overwrite, schema=set_schema_ibis)
    src.insert_table(attr_name, empty_attr, overwrite=overwrite, schema=attr_schema_ibis)
    if compute_attrition:
        _create_circe_stats_tables(src, name, overwrite=overwrite)
    return set_name, attr_name


def _run_cohort_sql_for_each_row(
    cdm: Any,
    cohort_set_df: Any,
    *,
    write_schema_str: str,
    cdm_schema_str: str,
    full_name: str,
    name: str,
) -> None:
    """Render and execute cohort SQL for each row in cohort_set_df.

    Parameters
    ----------
    cdm : Any
        CDM reference (database-backed).
    cohort_set_df : Any
        DataFrame with sql, cohort_definition_id columns.
    write_schema_str : str
        Write schema string for SQL placeholders.
    cdm_schema_str : str
        CDM schema string for SQL placeholders.
    full_name : str
        Full cohort table name (e.g. with prefix).
    name : str
        Cohort table stem (e.g. "cohort").
    """
    from cdmconnector import _circe

    for _, row in cohort_set_df.iterrows():
        rendered = _circe.render_cohort_sql(
            row["sql"],
            cdm_database_schema=cdm_schema_str,
            vocabulary_database_schema=cdm_schema_str,
            target_database_schema=write_schema_str,
            target_cohort_table=full_name,
            target_cohort_id=int(row["cohort_definition_id"]),
            cohort_stem=name,
        )
        statements = [
            s.strip() + ";" if s.strip() and not s.strip().startswith("--") else ""
            for s in re.split(r";\s*", rendered)
        ]
        statements = [s for s in statements if s and s != ";"]
        _execute_cohort_sql(cdm, statements)


def generate_cohort_set(
    cdm: Any,
    cohort_definition_set: Any,
    *,
    name: str = "cohort",
    overwrite: bool = True,
    compute_attrition: bool = True,
) -> Any:
    """
    Generate a cohort set from a cohort definition set (CIRCE JSON or equivalent).

    Uses internal CIRCE-style functions (cohort_expression_from_json,
    create_generate_options, build_cohort_query) backed by Circepy. SQL generation
    requires Circepy (a package dependency). Alternatively provide
    cohort_definition_set with a "sql" column (pre-generated SQL). Creates the
    cohort table, cohort_set, and cohort_attrition in the CDM write schema and
    runs the cohort SQL.

    Parameters
    ----------
    cdm : Cdm reference (from cdm_from_con with write_schema).
    cohort_definition_set : DataFrame with cohort_definition_id, cohort_name, and
        either "json" (CIRCE JSON strings) or "sql" (pre-generated SQL). May include
        "cohort" (parsed dicts). From read_cohort_set() or equivalent.
    name : Name of the cohort table (lowercase, letters/numbers/underscores). Default "cohort".
    overwrite : If True, overwrite existing cohort tables. Default True.
    compute_attrition : If True, CIRCE generates inclusion-rule stats (requires
        CIRCE SQL to create inclusion/inclusion_result tables). Default True.

    Returns
    -------
    Cdm
        CDM with the new cohort table and cohort_set/cohort_attrition populated.

    Raises
    ------
    CohortError
        If CDM has no database source, cohort_definition_set is invalid, or SQL execution fails.
    NotImplementedError
        If Circepy is not installed and no "sql" column is provided.
    """
    import pyarrow as pa

    name = _validate_cohort_table_name(name)
    logger.info("Generating cohort set: name=%s, overwrite=%s, compute_attrition=%s", name, overwrite, compute_attrition)
    cohort_set_df = _normalize_cohort_definition_set(cohort_definition_set, compute_attrition)
    logger.debug("Normalized cohort definition set: %d cohort(s)", len(cohort_set_df))

    src = getattr(cdm, "source", None)
    if src is None or not hasattr(src, "insert_table"):
        logger.error("generate_cohort_set requires a CDM with a database source (cdm_from_con).")
        raise CohortError("generate_cohort_set requires a CDM with a database source (cdm_from_con).")

    write_schema_str = _schema_to_string(cdm.write_schema)
    cdm_schema_str = _schema_to_string(cdm.cdm_schema)
    write_schema = cdm.write_schema
    prefix = write_schema.get("prefix", "") if isinstance(write_schema, dict) else ""
    prefix = prefix or ""
    full_name = f"{prefix}{name}"

    set_name, attr_name = _create_empty_cohort_tables(
        src, name, overwrite=overwrite, compute_attrition=compute_attrition
    )

    _run_cohort_sql_for_each_row(
        cdm,
        cohort_set_df,
        write_schema_str=write_schema_str,
        cdm_schema_str=cdm_schema_str,
        full_name=full_name,
        name=name,
    )

    set_data = pa.table({
        "cohort_definition_id": [int(x) for x in cohort_set_df["cohort_definition_id"]],
        "cohort_name": [str(x) for x in cohort_set_df["cohort_name"]],
    })
    src.insert_table(set_name, set_data, overwrite=True)
    cdm[name] = src.table(name, cdm.write_schema)
    logger.info("Cohort set generated: table=%s, %d cohort definition(s)", name, len(cohort_set_df))
    return cdm


def _table_refs(domain_ids: list[str]) -> Any:
    """Return domain_id -> table name and column names (concept_id, start_date, end_date).

    Parameters
    ----------
    domain_ids : list[str]
        Domain IDs (e.g. "condition", "drug", "procedure").

    Returns
    -------
    pandas.DataFrame
        Columns: domain_id, table_name, concept_id, start_date, end_date
        (concept_id/start_date/end_date are column name strings).
    """
    import pandas as pd

    refs = [
        ("condition", "condition_occurrence", "condition_concept_id", "condition_start_date", "condition_end_date"),
        ("drug", "drug_exposure", "drug_concept_id", "drug_exposure_start_date", "drug_exposure_end_date"),
        ("procedure", "procedure_occurrence", "procedure_concept_id", "procedure_date", "procedure_date"),
        ("observation", "observation", "observation_concept_id", "observation_date", "observation_date"),
        ("measurement", "measurement", "measurement_concept_id", "measurement_date", "measurement_date"),
        ("visit", "visit_occurrence", "visit_concept_id", "visit_start_date", "visit_end_date"),
        ("device", "device_exposure", "device_concept_id", "device_exposure_start_date", "device_exposure_end_date"),
    ]
    df = pd.DataFrame(
        refs,
        columns=["domain_id", "table_name", "concept_id", "start_date", "end_date"],
    )
    return df[df["domain_id"].isin([d.lower() for d in domain_ids])]


def cohort_collapse(x: Any) -> Any:
    """Collapse overlapping cohort periods per (cohort_definition_id, subject_id) into contiguous intervals.

    Expects columns cohort_definition_id, subject_id, cohort_start_date, cohort_end_date.
    Orders by cohort_start_date and duration, then merges overlapping/adjacent periods.

    Parameters
    ----------
    x : Ibis table or pandas DataFrame
        Cohort-like table with required columns.

    Returns
    -------
    pandas.DataFrame
        Collapsed cohort with columns cohort_definition_id, subject_id, cohort_start_date, cohort_end_date.
    """
    import pandas as pd

    from cdmconnector.utils import to_dataframe

    required = {"cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"}
    if hasattr(x, "schema") or hasattr(x, "op"):
        df = to_dataframe(x)
    else:
        df = pd.DataFrame(x)
    if not required.issubset(df.columns):
        raise CohortError(
            f"cohort_collapse requires columns {required}; got {list(df.columns)}"
        )
    df = df[list(required)].drop_duplicates()
    df["cohort_start_date"] = pd.to_datetime(df["cohort_start_date"]).dt.date
    df["cohort_end_date"] = pd.to_datetime(df["cohort_end_date"]).dt.date
    df["dur"] = (df["cohort_end_date"] - df["cohort_start_date"]).apply(lambda d: d.days if d is not None else 0)
    df = df.sort_values(
        ["cohort_definition_id", "subject_id", "cohort_start_date", "dur", "cohort_end_date"],
    ).reset_index(drop=True)
    df["_running_end"] = df.groupby(["cohort_definition_id", "subject_id"])["cohort_end_date"].transform(
        lambda s: s.cummax()
    )
    df["_prev_end"] = df.groupby(["cohort_definition_id", "subject_id"])["_running_end"].shift(1)
    # New group = 1 when start > prev_end (non-overlapping); 0 for first row or when overlapping
    df["_new_group"] = (
        (~df["_prev_end"].isna()) & (df["cohort_start_date"] > df["_prev_end"].astype("object"))
    ).astype(int)
    df["_groups"] = df.groupby(["cohort_definition_id", "subject_id"])["_new_group"].cumsum()
    collapsed = (
        df.groupby(["cohort_definition_id", "subject_id", "_groups"], as_index=False)
        .agg(
            cohort_start_date=("cohort_start_date", "min"),
            cohort_end_date=("cohort_end_date", "max"),
        )
    )
    return collapsed[["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"]]


def generate_concept_cohort_set(
    cdm: Any,
    concept_set: Any,
    *,
    name: str = "cohort",
    limit: str = "first",
    required_observation: tuple[int, int] = (0, 0),
    end: str | int = "observation_period_end_date",
    subset_cohort: str | None = None,
    subset_cohort_id: int | list[int] | None = None,
    overwrite: bool = True,
) -> Any:
    """
    Generate a cohort set from one or more concept sets (named list of concept IDs).

    Each concept set becomes one cohort; each row represents the time during which
    the concept was observed for that subject. Concepts are looked up in the CDM
    vocabulary and domain tables (condition_occurrence, drug_exposure, etc.).
    Concepts not in the vocabulary or in missing domain tables are silently skipped.
    If a domain has no end date (e.g. procedure, observation), start date is used as end date.

    Parameters
    ----------
    cdm : Cdm reference (from cdm_from_con with observation_period and concept table).
    concept_set : dict[str, list[int] | list[dict]]
        Named concept sets: name -> list of concept_id (int) or list of concept specs (dict).
        Each name becomes one cohort. Concept specs are dicts with:
        - "concept_id" (int, required)
        - "include_descendants" (bool, optional): if True, expand via concept_ancestor (requires concept_ancestor table). Default False.
        - "is_excluded" (bool, optional): if True, exclude this concept from the set. Default False.
        Simple form: {"cohort_a": [192671, 123]} uses no descendants and not excluded.
    name : str, optional
        Name of the cohort table (lowercase, letters/numbers/underscores). Default "cohort".
    limit : str, optional
        "first" (default) or "all": include only first occurrence per subject per cohort, or all.
    required_observation : tuple[int, int], optional
        (prior_days, future_days) required observation around the event. Default (0, 0).
    end : str or int, optional
        How to set cohort_end_date: "observation_period_end_date" (default), "event_end_date",
        or a fixed number of days from cohort_start_date.
    subset_cohort : str, optional
        If set, only persons in this cohort table are included.
    subset_cohort_id : int or list[int], optional
        If set with subset_cohort, only these cohort_definition_id(s) from the subset cohort.
    overwrite : bool, optional
        If True, overwrite existing cohort tables. Default True.

    Returns
    -------
    Cdm
        CDM with the new cohort table and cohort_set / cohort_attrition populated.

    Raises
    ------
    CohortError
        If CDM has no database source, name is invalid, or required tables are missing.
    """
    import pandas as pd

    from cdmconnector.cdm import collect
    from cdmconnector.dates import dateadd

    name = _validate_cohort_table_name(name)
    if limit not in ("first", "all"):
        raise CohortError(f'limit must be "first" or "all", got {limit!r}.')
    if not isinstance(required_observation, (tuple, list)) or len(required_observation) != 2:
        raise CohortError("required_observation must be a tuple or list of length 2 (prior_days, future_days).")
    prior_obs, future_obs = int(required_observation[0]), int(required_observation[1])
    if end not in ("observation_period_end_date", "event_end_date") and not (
        isinstance(end, (int, float)) and end >= 0
    ):
        raise CohortError(
            'end must be "observation_period_end_date", "event_end_date", or a non-negative number of days.'
        )
    if not isinstance(concept_set, dict) or len(concept_set) == 0:
        raise CohortError("concept_set must be a non-empty dict mapping cohort names to lists of concept_id or concept specs.")
    for k, v in concept_set.items():
        if not isinstance(k, str) or not k.strip():
            raise CohortError("concept_set keys must be non-empty strings.")
        if not isinstance(v, (list, tuple)):
            raise CohortError("concept_set values must be lists of concept_id (int) or dicts with 'concept_id' and optional 'include_descendants', 'is_excluded'.")
        for x in v:
            if isinstance(x, (int, float)):
                continue
            if isinstance(x, dict) and "concept_id" in x:
                continue
            raise CohortError(
                "Each concept_set value must be a list of int (concept_id) or dict with 'concept_id' and optional 'include_descendants', 'is_excluded'."
            )

    if "observation_period" not in getattr(cdm, "_tables", cdm.tables if hasattr(cdm, "tables") else []):
        raise CohortError("generate_concept_cohort_set requires observation_period in the CDM.")
    if "concept" not in getattr(cdm, "_tables", cdm.tables if hasattr(cdm, "tables") else []):
        raise CohortError("generate_concept_cohort_set requires concept table in the CDM.")

    src = getattr(cdm, "source", None)
    if src is None or not hasattr(src, "insert_table"):
        raise CohortError("generate_concept_cohort_set requires a CDM with a database source (cdm_from_con).")

    tables_list = getattr(cdm, "_tables", None) or (cdm.tables if hasattr(cdm, "tables") else [])
    existing = [t.lower() for t in (src.list_tables(cdm.write_schema) if hasattr(src, "list_tables") else [])]
    if hasattr(cdm, "write_schema"):
        try:
            existing = getattr(src, "list_tables", lambda s: [])(cdm.write_schema) or []
            existing = [str(x).lower() for x in existing]
        except Exception:
            existing = []
    if name in existing and not overwrite:
        raise CohortError(f"{name} already exists in the CDM write schema and overwrite is False.")

    # Build cohort_set_ref and concepts list with (cohort_definition_id, cohort_name, concept_id, include_descendants, is_excluded)
    cohort_defs = []
    concept_rows = []
    for idx, (cohort_name, spec_list) in enumerate(concept_set.items(), start=1):
        cohort_defs.append({
            "cohort_definition_id": idx,
            "cohort_name": str(cohort_name).strip(),
            "limit": limit,
            "prior_observation": prior_obs,
            "future_observation": future_obs,
            "end": end if isinstance(end, str) else int(end),
        })
        for spec in spec_list:
            if isinstance(spec, (int, float)):
                concept_rows.append({
                    "cohort_definition_id": idx,
                    "cohort_name": cohort_name,
                    "concept_id": int(spec),
                    "include_descendants": False,
                    "is_excluded": False,
                })
            else:
                cid = spec.get("concept_id")
                if cid is None:
                    raise CohortError("Concept spec dict must have 'concept_id'.")
                concept_rows.append({
                    "cohort_definition_id": idx,
                    "cohort_name": cohort_name,
                    "concept_id": int(cid),
                    "include_descendants": bool(spec.get("include_descendants", False)),
                    "is_excluded": bool(spec.get("is_excluded", False)),
                })
    if not concept_rows:
        logger.warning("Concept set has no concept IDs; creating empty cohort.")
        cohort_set_ref = pd.DataFrame(cohort_defs)
        set_name, attr_name = _create_empty_cohort_tables(src, name, overwrite=overwrite, compute_attrition=False)
        src.insert_table(set_name, pd.DataFrame({
            "cohort_definition_id": [r["cohort_definition_id"] for r in cohort_defs],
            "cohort_name": [r["cohort_name"] for r in cohort_defs],
        }), overwrite=True)
        attrition_rows = [
            {
                "cohort_definition_id": r["cohort_definition_id"],
                "number_subjects": 0,
                "number_records": 0,
                "reason_id": 1,
                "reason": "Initial qualifying events",
                "excluded_subjects": 0,
                "excluded_records": 0,
            }
            for r in cohort_defs
        ]
        src.insert_table(attr_name, pd.DataFrame(attrition_rows), overwrite=True)
        cdm[name] = src.table(name, cdm.write_schema)
        return cdm
    # Expand descendants via concept_ancestor when include_descendants is True
    if any(r.get("include_descendants", False) for r in concept_rows):
        if "concept_ancestor" not in getattr(cdm, "_tables", cdm.tables if hasattr(cdm, "tables") else []):
            raise CohortError("include_descendants requires concept_ancestor table in the CDM.")
        anc = collect(cdm["concept_ancestor"].select("ancestor_concept_id", "descendant_concept_id"))
        anc_df = anc.rename(columns={"ancestor_concept_id": "_anc", "descendant_concept_id": "concept_id"})
        expanded = []
        for r in concept_rows:
            if r.get("include_descendants", False):
                descendants = anc_df[anc_df["_anc"] == r["concept_id"]]["concept_id"].unique().tolist()
                for desc_cid in descendants:
                    expanded.append({
                        "cohort_definition_id": r["cohort_definition_id"],
                        "cohort_name": r["cohort_name"],
                        "concept_id": int(desc_cid),
                        "is_excluded": r.get("is_excluded", False),
                    })
            else:
                expanded.append({
                    "cohort_definition_id": r["cohort_definition_id"],
                    "cohort_name": r["cohort_name"],
                    "concept_id": r["concept_id"],
                    "is_excluded": r.get("is_excluded", False),
                })
        concept_rows = expanded
    # Filter excluded and build concepts_df (cohort_definition_id, cohort_name, concept_id)
    concepts_df = pd.DataFrame(concept_rows)
    if "is_excluded" in concepts_df.columns:
        concepts_df = concepts_df[concepts_df["is_excluded"] == False]
    concepts_df = concepts_df[["cohort_definition_id", "cohort_name", "concept_id"]].drop_duplicates()
    if concepts_df.empty:
        logger.warning("Concept set has no concept IDs after filtering; creating empty cohort.")
        cohort_set_ref = pd.DataFrame(cohort_defs)
        set_name, attr_name = _create_empty_cohort_tables(src, name, overwrite=overwrite, compute_attrition=False)
        src.insert_table(set_name, pd.DataFrame({
            "cohort_definition_id": [r["cohort_definition_id"] for r in cohort_defs],
            "cohort_name": [r["cohort_name"] for r in cohort_defs],
        }), overwrite=True)
        attrition_rows = [
            {
                "cohort_definition_id": r["cohort_definition_id"],
                "number_subjects": 0,
                "number_records": 0,
                "reason_id": 1,
                "reason": "Initial qualifying events",
                "excluded_subjects": 0,
                "excluded_records": 0,
            }
            for r in cohort_defs
        ]
        src.insert_table(attr_name, pd.DataFrame(attrition_rows), overwrite=True)
        cdm[name] = src.table(name, cdm.write_schema)
        return cdm

    # Join CDM concept table to get domain_id
    concept_tbl = cdm["concept"]
    concept_domains = collect(concept_tbl.select("concept_id", "domain_id"))
    concepts_df = concepts_df.merge(concept_domains, on="concept_id", how="inner")
    concepts_df["domain_id"] = concepts_df["domain_id"].astype(str).str.lower()
    supported = {"condition", "drug", "procedure", "observation", "measurement", "visit", "device"}
    domains_used = concepts_df["domain_id"].dropna().unique().tolist()
    domains_used = [d for d in domains_used if d in supported]
    table_names = list(cdm._tables.keys()) if hasattr(cdm, "_tables") and getattr(cdm, "_tables") else (cdm.tables if hasattr(cdm, "tables") else [])
    missing_tables = []
    table_refs_df = _table_refs(domains_used)
    for _, row in table_refs_df.iterrows():
        if row["table_name"] not in table_names:
            missing_tables.append(row["table_name"])
    if missing_tables:
        logger.warning(
            "Concept set includes concepts from tables %s not in CDM; those domains will be skipped.",
            list(set(missing_tables)),
        )
        table_refs_df = table_refs_df[~table_refs_df["table_name"].isin(missing_tables)]
        domains_used = table_refs_df["domain_id"].unique().tolist()
    if not len(domains_used):
        logger.warning("None of the input concept IDs map to available domain tables; creating empty cohort.")
        cohort_set_ref = pd.DataFrame(cohort_defs)
        set_name, attr_name = _create_empty_cohort_tables(src, name, overwrite=overwrite, compute_attrition=False)
        src.insert_table(set_name, pd.DataFrame({
            "cohort_definition_id": [r["cohort_definition_id"] for r in cohort_defs],
            "cohort_name": [r["cohort_name"] for r in cohort_defs],
        }), overwrite=True)
        attrition_rows = [
            {
                "cohort_definition_id": r["cohort_definition_id"],
                "number_subjects": 0,
                "number_records": 0,
                "reason_id": 1,
                "reason": "Initial qualifying events",
                "excluded_subjects": 0,
                "excluded_records": 0,
            }
            for r in cohort_defs
        ]
        src.insert_table(attr_name, pd.DataFrame(attrition_rows), overwrite=True)
        cdm[name] = src.table(name, cdm.write_schema)
        return cdm

    # Build cohort from domain tables (Ibis): union per-domain selections.
    # Upload concepts to a temp table so the join is same-backend (DuckDB-DuckDB etc.).
    import ibis

    from cdmconnector.utils import unique_table_name

    temp_concepts_name = unique_table_name("tmp_concepts_")
    src.insert_table(temp_concepts_name, concepts_df[["cohort_definition_id", "cohort_name", "concept_id", "domain_id"]], overwrite=True)
    concepts_tbl = src.table(temp_concepts_name, cdm.write_schema)
    try:

        def get_domain_cohort(domain_id: str):
            ref = table_refs_df[table_refs_df["domain_id"] == domain_id].iloc[0]
            tbl_name = ref["table_name"]
            concept_col = ref["concept_id"]
            start_col = ref["start_date"]
            end_col = ref["end_date"]
            if tbl_name not in cdm._tables:
                return None
            dom_tbl = cdm[tbl_name]
            dom_concepts_tbl = concepts_tbl.filter(concepts_tbl["domain_id"] == domain_id).select("cohort_definition_id", "concept_id")
            filtered = dom_tbl.filter(dom_tbl[concept_col].isin(dom_concepts_tbl["concept_id"]))
            merged = filtered.join(dom_concepts_tbl, filtered[concept_col] == dom_concepts_tbl["concept_id"], how="inner")
            start_expr = merged[start_col]
            end_expr = merged[end_col]
            cohort_end = end_expr if end_col != start_col else start_expr
            cohort_end = ibis.coalesce(cohort_end, dateadd(start_expr, 1, "day"))
            selected = merged.select(
                merged["cohort_definition_id"],
                subject_id=merged["person_id"],
                cohort_start_date=start_expr,
                cohort_end_date=cohort_end,
            )
            return selected.filter(selected["cohort_start_date"] <= selected["cohort_end_date"])

        cohort_parts = []
        for domain_id in domains_used:
            part = get_domain_cohort(domain_id)
            if part is not None:
                cohort_parts.append(part)
        if not cohort_parts:
            cohort_ibis = None
        else:
            cohort_ibis = cohort_parts[0]
            for p in cohort_parts[1:]:
                cohort_ibis = cohort_ibis.union(p)

        if cohort_ibis is None:
            cohort_set_ref = pd.DataFrame(cohort_defs)
            set_name, attr_name = _create_empty_cohort_tables(src, name, overwrite=overwrite, compute_attrition=False)
            src.insert_table(set_name, pd.DataFrame({
                "cohort_definition_id": [r["cohort_definition_id"] for r in cohort_defs],
                "cohort_name": [r["cohort_name"] for r in cohort_defs],
            }), overwrite=True)
            attrition_rows = [
                {"cohort_definition_id": r["cohort_definition_id"], "number_subjects": 0, "number_records": 0,
                 "reason_id": 1, "reason": "Initial qualifying events", "excluded_subjects": 0, "excluded_records": 0}
                for r in cohort_defs
            ]
            src.insert_table(attr_name, pd.DataFrame(attrition_rows), overwrite=True)
            cdm[name] = src.table(name, cdm.write_schema)
            return cdm

        # Join observation_period and apply filters / end rule
        op = cdm["observation_period"]
        obs = op.select(
            op["person_id"],
            observation_period_start_date=op["observation_period_start_date"],
            observation_period_end_date=op["observation_period_end_date"],
        )
        obs_renamed = obs.select(
            subject_id=obs["person_id"],
            observation_period_start_date=obs["observation_period_start_date"],
            observation_period_end_date=obs["observation_period_end_date"],
        )
        if subset_cohort is not None:
            if subset_cohort not in cdm._tables:
                raise CohortError(f"subset_cohort {subset_cohort!r} not found in CDM.")
            sub = cdm[subset_cohort].select("subject_id").distinct()
            if subset_cohort_id is not None:
                ids = [subset_cohort_id] if isinstance(subset_cohort_id, int) else list(subset_cohort_id)
                sub = cdm[subset_cohort].filter(cdm[subset_cohort]["cohort_definition_id"].isin(ids)).select("subject_id").distinct()
            obs_renamed = obs_renamed.join(sub, obs_renamed["subject_id"] == sub["subject_id"], how="inner").select(
                obs_renamed["subject_id"],
                obs_renamed["observation_period_start_date"],
                obs_renamed["observation_period_end_date"],
            )
        cohort_ibis = cohort_ibis.join(
            obs_renamed,
            [cohort_ibis["subject_id"] == obs_renamed["subject_id"]],
            how="inner",
        )
        cohort_ibis = cohort_ibis.filter(
            cohort_ibis["observation_period_start_date"] <= cohort_ibis["cohort_start_date"]
        ).filter(cohort_ibis["cohort_start_date"] <= cohort_ibis["observation_period_end_date"])
        if prior_obs > 0:
            cohort_ibis = cohort_ibis.filter(
                dateadd(cohort_ibis["observation_period_start_date"], prior_obs, "day")
                <= cohort_ibis["cohort_start_date"]
            )
        if future_obs > 0:
            cohort_ibis = cohort_ibis.filter(
                dateadd(cohort_ibis["cohort_start_date"], future_obs, "day")
                <= cohort_ibis["observation_period_end_date"]
            )
        if end == "observation_period_end_date":
            cohort_ibis = cohort_ibis.mutate(
                cohort_end_date=cohort_ibis["observation_period_end_date"],
            )
        elif isinstance(end, (int, float)):
            cohort_ibis = cohort_ibis.mutate(
                cohort_end_date=ibis.least(
                    dateadd(cohort_ibis["cohort_start_date"], int(end), "day"),
                    cohort_ibis["observation_period_end_date"],
                ),
            )
        else:
            # event_end_date: cap cohort_end_date at observation_period_end_date
            cohort_ibis = cohort_ibis.mutate(
                cohort_end_date=ibis.least(
                    cohort_ibis["cohort_end_date"],
                    cohort_ibis["observation_period_end_date"],
                ),
            )
        cohort_ibis = cohort_ibis.select(
            "cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date",
        )
        if limit == "first":
            w = ibis.window(
                group_by=[cohort_ibis["cohort_definition_id"], cohort_ibis["subject_id"]],
                order_by=cohort_ibis["cohort_start_date"],
            )
            cohort_ibis = cohort_ibis.mutate(_rn=ibis.row_number().over(w))
            cohort_ibis = cohort_ibis.filter(cohort_ibis["_rn"] == 1).select(
                "cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date",
            )
        cohort_df = collect(cohort_ibis)
        cohort_df = cohort_collapse(cohort_df)
        # Ensure date columns have a type DuckDB can create tables with (avoid NULL-typed columns)
        cohort_df["cohort_start_date"] = pd.to_datetime(cohort_df["cohort_start_date"])
        cohort_df["cohort_end_date"] = pd.to_datetime(cohort_df["cohort_end_date"])
        cohort_set_ref = pd.DataFrame(cohort_defs)
        set_name, attr_name = _create_empty_cohort_tables(src, name, overwrite=overwrite, compute_attrition=False)
        src.insert_table(name, cohort_df, overwrite=True)
        src.insert_table(set_name, cohort_set_ref[["cohort_definition_id", "cohort_name"]], overwrite=True)
        count_df = (
            cohort_df.groupby("cohort_definition_id")
            .agg(number_records=("subject_id", "count"), number_subjects=("subject_id", "nunique"))
            .reset_index()
        )
        attrition_df = cohort_set_ref[["cohort_definition_id"]].merge(
            count_df, on="cohort_definition_id", how="left"
        )
        attrition_df["number_records"] = attrition_df["number_records"].fillna(0).astype(int)
        attrition_df["number_subjects"] = attrition_df["number_subjects"].fillna(0).astype(int)
        attrition_df["reason_id"] = 1
        attrition_df["reason"] = "Initial qualifying events"
        attrition_df["excluded_subjects"] = 0
        attrition_df["excluded_records"] = 0
        src.insert_table(
            attr_name,
            attrition_df[
                [
                    "cohort_definition_id",
                    "number_subjects",
                    "number_records",
                    "reason_id",
                    "reason",
                    "excluded_subjects",
                    "excluded_records",
                ]
            ],
            overwrite=True,
        )
        cdm[name] = src.table(name, cdm.write_schema)
        logger.info("Generated concept cohort set: table=%s, %d cohort(s)", name, len(cohort_set_ref))
        return cdm
    finally:
        try:
            src.drop_table(temp_concepts_name)
        except Exception:
            pass


def bind(*cohort_tables: Any, name: str | None = None) -> Any:
    """
    Bind two or more cohort tables into one (same as R omopgenerics bind).

    Not fully implemented: in R, bind combines cohort_table objects with
    cohort_set and cohort_attrition. This stub raises NotImplementedError;
    a minimal implementation could combine DataFrames/tables with union.

    Parameters
    ----------
    *cohort_tables : Two or more cohort tables (Ibis tables or DataFrames with
        cohort_definition_id, subject_id, cohort_start_date, cohort_end_date).
    name : Name for the combined cohort (used when assigning to a CDM).

    Returns
    -------
    Combined cohort table (when implemented).

    Raises
    ------
    NotImplementedError
        Full bind (with cohort_set/attrition) is not yet implemented.
    """
    logger.warning("bind() for cohort tables is not implemented; combine tables manually.")
    raise NotImplementedError(
        "bind() for cohort tables is not yet implemented in Python. "
        "Combine tables manually with Ibis .union() or pandas concat, "
        "then assign to the CDM and use new_cohort_table/record_cohort_attrition as needed."
    )
