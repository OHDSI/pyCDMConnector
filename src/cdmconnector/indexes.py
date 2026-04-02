# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Database index operations for OMOP CDM tables.

Mirrors the omopgenerics index management functions for creating and
inspecting database indexes on CDM tables.

Functions
---------
expected_indexes
    Return recommended indexes for CDM tables.
create_indexes
    Create database indexes on CDM tables.
existing_indexes
    List existing indexes on CDM tables.
status_indexes
    Compare expected vs existing indexes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from cdmconnector.cdm import Cdm

# Recommended indexes: (table_name, column_name) pairs
_RECOMMENDED_INDEXES: list[tuple[str, str]] = [
    ("person", "person_id"),
    ("observation_period", "person_id"),
    ("observation_period", "observation_period_id"),
    ("visit_occurrence", "person_id"),
    ("visit_occurrence", "visit_occurrence_id"),
    ("visit_occurrence", "visit_concept_id"),
    ("visit_detail", "person_id"),
    ("visit_detail", "visit_detail_id"),
    ("condition_occurrence", "person_id"),
    ("condition_occurrence", "condition_occurrence_id"),
    ("condition_occurrence", "condition_concept_id"),
    ("drug_exposure", "person_id"),
    ("drug_exposure", "drug_exposure_id"),
    ("drug_exposure", "drug_concept_id"),
    ("procedure_occurrence", "person_id"),
    ("procedure_occurrence", "procedure_occurrence_id"),
    ("procedure_occurrence", "procedure_concept_id"),
    ("device_exposure", "person_id"),
    ("device_exposure", "device_exposure_id"),
    ("measurement", "person_id"),
    ("measurement", "measurement_id"),
    ("measurement", "measurement_concept_id"),
    ("observation", "person_id"),
    ("observation", "observation_id"),
    ("observation", "observation_concept_id"),
    ("death", "person_id"),
    ("concept", "concept_id"),
    ("concept_ancestor", "ancestor_concept_id"),
    ("concept_ancestor", "descendant_concept_id"),
    ("concept_relationship", "concept_id_1"),
    ("concept_relationship", "concept_id_2"),
    ("drug_strength", "drug_concept_id"),
    ("drug_strength", "ingredient_concept_id"),
    ("condition_era", "person_id"),
    ("condition_era", "condition_concept_id"),
    ("drug_era", "person_id"),
    ("drug_era", "drug_concept_id"),
]


def expected_indexes(cdm: Cdm | None = None) -> pd.DataFrame:
    """Return recommended indexes for OMOP CDM tables.

    Parameters
    ----------
    cdm : Cdm or None
        If provided, only returns indexes for tables present in the CDM.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``table_name`` and ``column_name``.
    """
    import pandas as pd

    df = pd.DataFrame(_RECOMMENDED_INDEXES, columns=["table_name", "column_name"])
    if cdm is not None:
        available = set(cdm.tables)
        df = df[df["table_name"].isin(available)].reset_index(drop=True)
    return df


def create_indexes(
    cdm: Cdm,
    indexes: pd.DataFrame | None = None,
) -> list[str]:
    """Create database indexes on CDM tables.

    Parameters
    ----------
    cdm : Cdm
        CDM reference (must be database-backed).
    indexes : pandas.DataFrame or None
        DataFrame with ``table_name`` and ``column_name`` columns.
        If None, uses expected_indexes(cdm).

    Returns
    -------
    list[str]
        SQL statements executed.

    Raises
    ------
    NotImplementedError
        If the CDM source does not support raw SQL execution.
    """
    if indexes is None:
        indexes = expected_indexes(cdm)

    source = getattr(cdm, "source", None)
    if source is None:
        raise NotImplementedError("create_indexes requires a database-backed CDM.")

    con = getattr(source, "con", None)
    if con is None:
        raise NotImplementedError("create_indexes requires a CDM with a database connection.")

    executed: list[str] = []
    for _, row in indexes.iterrows():
        table = row["table_name"]
        col = row["column_name"]
        idx_name = f"idx_{table}_{col}"
        # Qualify with schema if available
        schema = getattr(cdm, "cdm_schema", None)
        if schema:
            from cdmconnector.utils import resolve_schema_name

            schema_str = resolve_schema_name(schema)
            qualified = f"{schema_str}.{table}" if schema_str else table
        else:
            qualified = table

        sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {qualified} ({col})"
        try:
            con.raw_sql(sql)
            executed.append(sql)
        except Exception:
            # Some backends don't support CREATE INDEX IF NOT EXISTS
            try:
                sql_alt = f"CREATE INDEX {idx_name} ON {qualified} ({col})"
                con.raw_sql(sql_alt)
                executed.append(sql_alt)
            except Exception:
                pass  # Index may already exist

    return executed


def create_table_index(
    cdm: Cdm,
    table_name: str,
    column_name: str,
) -> str | None:
    """Create a single index on a CDM table.

    Parameters
    ----------
    cdm : Cdm
        CDM reference (must be database-backed).
    table_name : str
        Table to index.
    column_name : str
        Column to index.

    Returns
    -------
    str or None
        SQL statement executed, or None if creation failed.
    """
    import pandas as pd

    idx_df = pd.DataFrame([{
        "table_name": table_name,
        "column_name": column_name,
    }])
    results = create_indexes(cdm, indexes=idx_df)
    return results[0] if results else None


def existing_indexes(cdm: Cdm) -> pd.DataFrame:
    """List existing indexes on CDM tables.

    Parameters
    ----------
    cdm : Cdm
        CDM reference (database-backed).

    Returns
    -------
    pandas.DataFrame
        DataFrame with index information. Columns vary by backend but
        typically include ``table_name``, ``index_name``, ``column_name``.
        Returns empty DataFrame if indexes cannot be queried.
    """
    import pandas as pd

    source = getattr(cdm, "source", None)
    con = getattr(source, "con", None) if source else None
    if con is None:
        return pd.DataFrame(columns=["table_name", "index_name", "column_name"])

    backend = getattr(con, "name", "")

    try:
        if backend == "duckdb":
            result = con.raw_sql("SELECT * FROM duckdb_indexes()").fetchdf()
            return result
        elif backend in ("postgres", "postgresql"):
            sql = """
                SELECT tablename AS table_name,
                       indexname AS index_name,
                       indexdef AS column_name
                FROM pg_indexes
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            """
            result = con.raw_sql(sql).fetchdf()
            return result
    except Exception:
        pass

    return pd.DataFrame(columns=["table_name", "index_name", "column_name"])


def status_indexes(cdm: Cdm) -> pd.DataFrame:
    """Compare expected vs existing indexes.

    Parameters
    ----------
    cdm : Cdm
        CDM reference (database-backed).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``table_name``, ``column_name``, ``status``
        where status is "present" or "missing".
    """
    exp = expected_indexes(cdm)
    try:
        exist = existing_indexes(cdm)
        if exist.empty:
            exp["status"] = "unknown"
            return exp

        # Try to match by table_name and column_name
        existing_set: set[tuple[str, str]] = set()
        if "table_name" in exist.columns and "column_name" in exist.columns:
            for _, row in exist.iterrows():
                existing_set.add((str(row["table_name"]).lower(), str(row["column_name"]).lower()))

        statuses = []
        for _, row in exp.iterrows():
            key = (str(row["table_name"]).lower(), str(row["column_name"]).lower())
            statuses.append("present" if key in existing_set else "missing")
        exp["status"] = statuses
    except Exception:
        exp["status"] = "unknown"

    return exp
