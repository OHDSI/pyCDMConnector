# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around the Circepy (OHDSI/Circepy) package for CIRCE JSON → SQL.

This module provides a stable internal API for CDMConnector to convert CIRCE
cohort definition JSON into executable SQL.  All interaction with Circepy goes
through this module so that the rest of the package is insulated from upstream
API changes.

Functions
---------
cohort_expression_from_json
    Validate and parse a CIRCE cohort-definition JSON string → dict.
concept_set_expression_from_json
    Validate and parse a CIRCE concept-set JSON string → dict.
create_generate_options
    Build a :class:`GenerateOptions` with sensible defaults.
build_cohort_query
    Generate cohort SQL from a CIRCE expression (requires Circepy).
build_concept_set_query
    Generate concept-set SQL from a CIRCE expression (requires Circepy).
render_cohort_sql
    Replace ``@parameter`` placeholders in CIRCE-generated SQL.
translate_sql
    Translate rendered SQL from T-SQL to the target database dialect using SQLGlot.
ibis_to_sqlglot_dialect
    Map an Ibis backend name to a SQLGlot dialect string.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import sqlglot

# ---------------------------------------------------------------------------
# Try to import circepy (the OHDSI/Circepy package, imported as ``circe``)
# ---------------------------------------------------------------------------

try:
    import circe as circepy  # noqa: F401 – re-exported under canonical alias
    from circe import (
        CohortExpression as _CohortExpression,
        cohort_expression_from_json as _circe_expr_from_json,
        build_cohort_query as _circe_build_query,
    )
    from circe.cohortdefinition import (
        BuildExpressionQueryOptions as _BuildOpts,
        CohortExpressionQueryBuilder as _QueryBuilder,
    )
    from circe.vocabulary import (
        ConceptSetExpression as _ConceptSetExpression,
    )

    _CIRCEPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CIRCEPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# GenerateOptions dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerateOptions:
    """Options passed to :func:`build_cohort_query`.

    This is a plain dataclass that mirrors the fields of
    ``circe.cohortdefinition.BuildExpressionQueryOptions``.
    """

    cohort_id: Optional[int] = None
    cdm_schema: Optional[str] = None
    vocabulary_schema: Optional[str] = None
    target_table: Optional[str] = None
    result_schema: Optional[str] = None
    generate_stats: bool = False


def create_generate_options(
    *,
    cohort_id: Optional[int] = None,
    cdm_schema: Optional[str] = None,
    vocabulary_schema: Optional[str] = None,
    target_table: Optional[str] = None,
    result_schema: Optional[str] = None,
    generate_stats: bool = False,
) -> GenerateOptions:
    """Create a :class:`GenerateOptions` with the given parameters."""
    return GenerateOptions(
        cohort_id=cohort_id,
        cdm_schema=cdm_schema,
        vocabulary_schema=vocabulary_schema,
        target_table=target_table,
        result_schema=result_schema,
        generate_stats=generate_stats,
    )


# ---------------------------------------------------------------------------
# JSON validation / parsing helpers
# ---------------------------------------------------------------------------


def cohort_expression_from_json(json_str) -> dict:
    """Validate and parse a CIRCE cohort-definition JSON string.

    Parameters
    ----------
    json_str : str
        A JSON string representing a CIRCE cohort definition.

    Returns
    -------
    dict
        The parsed cohort definition as a plain dict.

    Raises
    ------
    ValueError
        If *json_str* is not a non-empty string, is not valid JSON, is not a
        JSON object, or is missing required keys (``ConceptSets``,
        ``PrimaryCriteria``).
    """
    if not isinstance(json_str, str) or not json_str.strip():
        raise ValueError("json_str must be a non-empty string")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Cohort definition JSON must be a JSON object")

    if "ConceptSets" not in data:
        raise ValueError("Cohort definition JSON must contain 'ConceptSets'")
    if "PrimaryCriteria" not in data:
        raise ValueError("Cohort definition JSON must contain 'PrimaryCriteria'")

    return data


def concept_set_expression_from_json(json_str) -> dict:
    """Validate and parse a CIRCE concept-set JSON string.

    Parameters
    ----------
    json_str : str
        A JSON string representing a CIRCE concept set expression.

    Returns
    -------
    dict
        The parsed concept set expression as a plain dict.

    Raises
    ------
    ValueError
        If *json_str* is not a non-empty string, is not valid JSON, or is not
        a JSON object.
    """
    if not isinstance(json_str, str) or not json_str.strip():
        raise ValueError("json_str must be a non-empty string")

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Concept set JSON must be a JSON object")

    return data


# ---------------------------------------------------------------------------
# SQL generation (requires Circepy)
# ---------------------------------------------------------------------------


def _require_circepy() -> None:
    if not _CIRCEPY_AVAILABLE:
        raise NotImplementedError(
            "Circepy is required for cohort SQL generation but is not installed. "
            "Install it with: pip install 'ohdsi-circe-python-alpha @ git+https://github.com/OHDSI/Circepy.git'"
        )


def build_cohort_query(
    expression,
    options: GenerateOptions,
) -> str:
    """Generate cohort SQL from a CIRCE expression using Circepy.

    Parameters
    ----------
    expression : str | dict
        Either a JSON string or an already-parsed dict representing the CIRCE
        cohort definition.
    options : GenerateOptions
        Generation options (schema names, cohort ID, etc.).

    Returns
    -------
    str
        The generated SQL query.

    Raises
    ------
    NotImplementedError
        If Circepy is not installed.
    """
    _require_circepy()

    # Parse JSON string → dict if needed
    if isinstance(expression, str):
        expression = cohort_expression_from_json(expression)

    # Convert dict → CohortExpression pydantic model
    cohort_expr = _circe_expr_from_json(json.dumps(expression))

    # Map GenerateOptions → Circepy BuildExpressionQueryOptions
    opts = _BuildOpts()
    if options.cohort_id is not None:
        opts.cohort_id = options.cohort_id
    if options.cdm_schema is not None:
        opts.cdm_schema = options.cdm_schema
    if options.vocabulary_schema is not None:
        opts.vocabulary_schema = options.vocabulary_schema
    elif options.cdm_schema is not None:
        opts.vocabulary_schema = options.cdm_schema
    if options.target_table is not None:
        opts.target_table = options.target_table
    if options.result_schema is not None:
        opts.result_schema = options.result_schema
    opts.generate_stats = options.generate_stats

    return _circe_build_query(cohort_expr, opts)


def build_concept_set_query(json_str: str) -> str:
    """Generate concept-set SQL from a CIRCE concept-set JSON string.

    Parameters
    ----------
    json_str : str
        A JSON string representing a CIRCE concept set expression.

    Returns
    -------
    str
        The generated SQL query.

    Raises
    ------
    NotImplementedError
        If Circepy is not installed.
    ValueError
        If *json_str* is not valid JSON.
    """
    _require_circepy()

    data = concept_set_expression_from_json(json_str)

    from circe.cohortdefinition import ConceptSetExpressionQueryBuilder

    items = data.get("items", [])

    # Classify items into standard concepts vs descendants vs mapped
    concepts = []
    descendant_concepts = []
    mapped_concepts = []
    mapped_descendant_concepts = []

    from circe.vocabulary import Concept as _Concept

    for item in items:
        concept_data = item.get("concept", {})
        concept = _Concept(
            concept_id=concept_data.get("CONCEPT_ID"),
            concept_name=concept_data.get("CONCEPT_NAME", ""),
        )
        is_excluded = item.get("isExcluded", False)
        if is_excluded:
            continue

        include_descendants = item.get("includeDescendants", False)
        include_mapped = item.get("includeMapped", False)

        if include_mapped:
            if include_descendants:
                mapped_descendant_concepts.append(concept)
            else:
                mapped_concepts.append(concept)
        else:
            if include_descendants:
                descendant_concepts.append(concept)
            else:
                concepts.append(concept)

    builder = ConceptSetExpressionQueryBuilder()
    return builder.build_concept_set_query(
        concepts, descendant_concepts, mapped_concepts, mapped_descendant_concepts
    )


# ---------------------------------------------------------------------------
# SQL rendering (placeholder replacement — no Circepy dependency)
# ---------------------------------------------------------------------------


def render_cohort_sql(
    sql: str,
    *,
    cdm_database_schema: str,
    vocabulary_database_schema: Optional[str] = None,
    target_database_schema: str,
    results_database_schema: Optional[str] = None,
    target_cohort_table: str,
    target_cohort_id: int,
    cohort_stem: Optional[str] = None,
    cohort_inclusion: Optional[str] = None,
    cohort_inclusion_result: Optional[str] = None,
    cohort_inclusion_stats: Optional[str] = None,
    cohort_summary_stats: Optional[str] = None,
    cohort_censor_stats: Optional[str] = None,
) -> str:
    """Replace ``@parameter`` placeholders in CIRCE-generated SQL.

    Mirrors the R ``SqlRender::render`` convention used by CirceR / CDMConnector.

    Parameters
    ----------
    sql : str
        SQL containing ``@param`` placeholders.
    cdm_database_schema : str
        Schema containing CDM tables.
    vocabulary_database_schema : str, optional
        Schema containing vocabulary tables (defaults to *cdm_database_schema*).
    target_database_schema : str
        Schema where cohort tables will be written.
    results_database_schema : str, optional
        Schema for results / stats tables (defaults to *target_database_schema*).
    target_cohort_table : str
        Name of the target cohort table.
    target_cohort_id : int
        Cohort definition ID to insert.
    cohort_stem : str, optional
        Prefix stem for stats table names.
    cohort_inclusion, cohort_inclusion_result, cohort_inclusion_stats,
    cohort_summary_stats, cohort_censor_stats : str, optional
        Override individual stats table names.

    Returns
    -------
    str
        The rendered SQL with all placeholders replaced.
    """
    if vocabulary_database_schema is None:
        vocabulary_database_schema = cdm_database_schema
    if results_database_schema is None:
        results_database_schema = target_database_schema

    stem = cohort_stem or target_cohort_table

    replacements = {
        "@cdm_database_schema": cdm_database_schema,
        "@vocabulary_database_schema": vocabulary_database_schema,
        "@target_database_schema": target_database_schema,
        "@results_database_schema": results_database_schema,
        "@target_cohort_table": target_cohort_table,
        "@target_cohort_id": str(target_cohort_id),
    }

    # Stats table overrides (longest-match first to avoid partial replacement)
    stats_tables = {
        "@results_database_schema.cohort_inclusion_result": (
            cohort_inclusion_result or f"{results_database_schema}.{stem}_inclusion_result"
        ),
        "@results_database_schema.cohort_inclusion_stats": (
            cohort_inclusion_stats or f"{results_database_schema}.{stem}_inclusion_stats"
        ),
        "@results_database_schema.cohort_summary_stats": (
            cohort_summary_stats or f"{results_database_schema}.{stem}_summary_stats"
        ),
        "@results_database_schema.cohort_censor_stats": (
            cohort_censor_stats or f"{results_database_schema}.{stem}_censor_stats"
        ),
        "@results_database_schema.cohort_inclusion": (
            cohort_inclusion or f"{results_database_schema}.{stem}_inclusion"
        ),
    }

    # Replace stats tables first (they contain @results_database_schema prefix)
    for placeholder, value in stats_tables.items():
        sql = sql.replace(placeholder, value)

    # Then replace remaining @parameters (sort by length descending to avoid partial matches)
    for placeholder in sorted(replacements, key=len, reverse=True):
        sql = sql.replace(placeholder, replacements[placeholder])

    return sql


# ---------------------------------------------------------------------------
# SQL dialect translation (Circepy T-SQL → target dialect via SQLGlot)
# ---------------------------------------------------------------------------

# Mapping from Ibis backend name → SQLGlot dialect identifier.
# Circepy always emits SQL Server (T-SQL) syntax.
_IBIS_TO_SQLGLOT: dict[str, str] = {
    "duckdb": "duckdb",
    "postgres": "postgres",
    "postgresql": "postgres",
    "redshift": "redshift",
    "mssql": "tsql",
    "sqlserver": "tsql",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "spark": "spark",
    "databricks": "databricks",
    "sqlite": "sqlite",
    "clickhouse": "clickhouse",
    "trino": "trino",
    "mysql": "mysql",
}

# The source dialect of SQL generated by Circepy (OHDSI CIRCE-BE = SQL Server).
_CIRCE_SOURCE_DIALECT = "tsql"


def ibis_to_sqlglot_dialect(ibis_backend_name: str) -> str:
    """Map an Ibis backend name to a SQLGlot dialect string.

    Parameters
    ----------
    ibis_backend_name : str
        The Ibis backend name (e.g. ``"duckdb"``, ``"postgres"``, ``"snowflake"``).
        Obtained at runtime from ``type(con).__module__.split('.')[1]`` or similar.

    Returns
    -------
    str
        The corresponding SQLGlot dialect identifier.

    Raises
    ------
    ValueError
        If the backend is not in the mapping.
    """
    key = ibis_backend_name.lower().strip()
    dialect = _IBIS_TO_SQLGLOT.get(key)
    if dialect is None:
        raise ValueError(
            f"Unsupported Ibis backend for SQL translation: {ibis_backend_name!r}. "
            f"Supported backends: {', '.join(sorted(_IBIS_TO_SQLGLOT))}"
        )
    return dialect


def _get_dialect_from_ibis_con(con) -> str:
    """Detect the SQLGlot dialect from a live Ibis connection object.

    Works by inspecting the Ibis backend module path
    (e.g. ``ibis.backends.duckdb`` → ``"duckdb"``).
    """
    backend_module = type(con).__module__  # e.g. "ibis.backends.duckdb"
    parts = backend_module.split(".")
    # ibis.backends.<name>  →  parts[2]
    if len(parts) >= 3 and parts[0] == "ibis" and parts[1] == "backends":
        return ibis_to_sqlglot_dialect(parts[2])
    # Fallback: try the class name
    cls_name = type(con).__name__.lower().replace("backend", "")
    return ibis_to_sqlglot_dialect(cls_name)


def translate_sql(
    sql: str,
    *,
    target_dialect: str | None = None,
    con=None,
) -> str:
    """Translate rendered CIRCE SQL from T-SQL to the target database dialect.

    Circepy generates SQL Server (T-SQL) syntax — ``#temp_tables``,
    ``DATEADD()``, ``SELECT … INTO``, etc.  This function uses **SQLGlot** to
    transpile each statement to the dialect of the target database.

    Provide either *target_dialect* (a SQLGlot dialect string like
    ``"duckdb"`` or ``"postgres"``) **or** *con* (an Ibis connection whose
    backend will be auto-detected).

    Parameters
    ----------
    sql : str
        One or more SQL statements (semicolon-separated) in T-SQL dialect,
        typically the output of :func:`render_cohort_sql`.
    target_dialect : str, optional
        SQLGlot target dialect (e.g. ``"duckdb"``, ``"postgres"``).
        Mutually exclusive with *con*.
    con : ibis.BaseBackend, optional
        An Ibis connection; the dialect is auto-detected from the backend.
        Mutually exclusive with *target_dialect*.

    Returns
    -------
    str
        The SQL translated to the target dialect, with statements separated by
        ``";\\n"``.

    Raises
    ------
    ValueError
        If neither *target_dialect* nor *con* is provided, or if the backend
        is not supported.

    Examples
    --------
    >>> rendered = render_cohort_sql(raw_sql, ...)
    >>> duckdb_sql = translate_sql(rendered, target_dialect="duckdb")
    >>> pg_sql = translate_sql(rendered, con=ibis.postgres.connect(...))
    """
    if target_dialect is None and con is None:
        raise ValueError("Provide either target_dialect or con.")
    if target_dialect is None:
        target_dialect = _get_dialect_from_ibis_con(con)

    # If source == target, no translation needed
    if target_dialect == _CIRCE_SOURCE_DIALECT:
        return sql

    # Split into individual statements, translate each, rejoin
    translated_parts: list[str] = []
    for statement in sqlglot.transpile(
        sql,
        read=_CIRCE_SOURCE_DIALECT,
        write=target_dialect,
        pretty=True,
    ):
        translated_parts.append(statement)

    return ";\n".join(translated_parts)
