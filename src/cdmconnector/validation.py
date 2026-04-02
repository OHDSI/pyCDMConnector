# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""CDM structure validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cdmconnector.schemas import omop_columns, omop_tables

if TYPE_CHECKING:
    from cdmconnector.cdm import Cdm


def validate_cdm_structure(cdm: Cdm, *, strict: bool = False) -> dict[str, list[str]]:
    """
    Validate CDM tables have required columns and types.

    Checks that each OMOP table in the CDM has the expected columns from the schema.
    Returns a dict mapping table name to a list of warning/error messages.

    Parameters
    ----------
    cdm : Cdm
        CDM reference to validate.
    strict : bool, default False
        If True, treat missing columns as errors; if False, as warnings.

    Returns
    -------
    dict[str, list[str]]
        Keys are table names; values are lists of issue messages (missing columns, etc.).
    """
    issues: dict[str, list[str]] = {}
    version = cdm.version
    try:
        expected_tables = set(omop_tables(version))
    except Exception:
        expected_tables = set(omop_tables("5.3"))
    for table_name in cdm.tables:
        if table_name not in expected_tables:
            continue
        try:
            expected_cols = set(omop_columns(table_name, version=version))
        except Exception:
            continue
        try:
            tbl = cdm[table_name]
            actual_cols = set(tbl.columns) if hasattr(tbl, "columns") else set()
        except Exception as e:
            issues[table_name] = [f"Cannot read table columns: {e}"]
            continue
        missing = expected_cols - actual_cols
        if missing:
            msg = f"Missing columns: {sorted(missing)}"
            issues.setdefault(table_name, []).append(msg)
    return issues


# ---------------------------------------------------------------------------
# Domain-specific validators (mirrors omopgenerics validate* functions)
# ---------------------------------------------------------------------------


def validate_cdm_argument(cdm: Any) -> None:
    """Validate that cdm is a valid Cdm object.

    Parameters
    ----------
    cdm : Any
        Object to validate.

    Raises
    ------
    TypeError
        If cdm is not a Cdm instance.
    CDMValidationError
        If cdm is missing required attributes.
    """
    from cdmconnector.cdm import Cdm
    from cdmconnector.exceptions import CDMValidationError

    if not isinstance(cdm, Cdm):
        raise TypeError(f"Expected Cdm, got {type(cdm).__name__}")
    if not hasattr(cdm, "name") or cdm.name is None:
        raise CDMValidationError("CDM must have a name.")
    if not hasattr(cdm, "tables") or not cdm.tables:
        raise CDMValidationError("CDM must have at least one table.")


def validate_cohort_argument(cohort: Any) -> None:
    """Validate that cohort has required cohort table structure.

    Checks for required columns (cohort_definition_id, subject_id,
    cohort_start_date, cohort_end_date).

    Parameters
    ----------
    cohort : Any
        Cohort table to validate.

    Raises
    ------
    CDMValidationError
        If required columns are missing.
    """
    from cdmconnector.exceptions import CDMValidationError
    from cdmconnector.schemas import COHORT_TABLE_COLUMNS

    cols = set()
    if hasattr(cohort, "columns"):
        cols = set(cohort.columns)
    elif hasattr(cohort, "schema"):
        try:
            cols = set(cohort.schema().names)
        except Exception:
            pass

    required = set(COHORT_TABLE_COLUMNS)
    missing = required - cols
    if missing:
        raise CDMValidationError(
            f"Cohort table missing required columns: {sorted(missing)}"
        )


def validate_cohort_id_argument(
    cohort: Any,
    cohort_id: int | list[int] | None,
) -> list[int]:
    """Validate and normalize cohort_id argument.

    Parameters
    ----------
    cohort : Any
        Cohort table with cohort_set attribute.
    cohort_id : int, list[int], or None
        Cohort ID(s) to validate. If None, returns all IDs.

    Returns
    -------
    list[int]
        Validated list of cohort_definition_ids.

    Raises
    ------
    ValueError
        If an ID is not found in the cohort_set.
    """
    cs = getattr(cohort, "cohort_set", None)
    if cs is not None:
        if hasattr(cs, "execute"):
            cs = cs.execute()
        available = {int(x) for x in cs["cohort_definition_id"]}
    else:
        available = None

    if cohort_id is None:
        return sorted(available) if available else []

    ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
    if available is not None:
        unknown = set(ids) - available
        if unknown:
            raise ValueError(
                f"Cohort IDs not found: {sorted(unknown)}. Available: {sorted(available)}"
            )
    return ids


def validate_result_argument(result: Any) -> None:
    """Validate that result is a valid SummarisedResult.

    Parameters
    ----------
    result : Any
        Object to validate.

    Raises
    ------
    TypeError
        If result is not a SummarisedResult.
    CDMValidationError
        If required columns are missing.
    """
    from cdmconnector.characteristics import _SUMMARISED_RESULT_COLUMNS, SummarisedResult
    from cdmconnector.exceptions import CDMValidationError

    if not isinstance(result, SummarisedResult):
        raise TypeError(f"Expected SummarisedResult, got {type(result).__name__}")

    required = set(_SUMMARISED_RESULT_COLUMNS)
    actual = set(result.results.columns)
    missing = required - actual
    if missing:
        raise CDMValidationError(
            f"SummarisedResult missing required columns: {sorted(missing)}"
        )


def validate_concept_set_argument(cs: Any) -> None:
    """Validate that cs is a valid ConceptSetExpression.

    Parameters
    ----------
    cs : Any
        Object to validate.

    Raises
    ------
    TypeError
        If cs is not a ConceptSetExpression.
    """
    from cdmconnector.codelist import ConceptSetExpression

    if not isinstance(cs, ConceptSetExpression):
        raise TypeError(f"Expected ConceptSetExpression, got {type(cs).__name__}")


def validate_name_argument(name: str) -> str:
    """Validate that name is a valid snake_case identifier.

    Parameters
    ----------
    name : str
        Name to validate.

    Returns
    -------
    str
        The validated name.

    Raises
    ------
    ValueError
        If name is not valid snake_case.
    """
    import re

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Name must be a non-empty string.")
    if not re.match(r"^[a-z][a-z0-9_]*$", name):
        raise ValueError(
            f"Name '{name}' must be snake_case (lowercase letters, digits, underscores, "
            "starting with a letter)."
        )
    return name


def validate_cdm_table(table: Any, name: str | None = None) -> None:
    """Validate that a table has the expected structure for a CDM table.

    Parameters
    ----------
    table : Any
        Table to validate (Ibis or DataFrame).
    name : str or None
        Optional table name for error messages.

    Raises
    ------
    CDMValidationError
        If the table is missing required columns.
    """
    from cdmconnector.exceptions import CDMValidationError

    cols = set()
    if hasattr(table, "columns"):
        cols = set(table.columns)
    elif hasattr(table, "schema"):
        try:
            cols = set(table.schema().names)
        except Exception:
            pass

    if not cols:
        ctx = f" ({name})" if name else ""
        raise CDMValidationError(f"Cannot determine columns for table{ctx}.")

    if name:
        try:
            from cdmconnector.schemas import omop_columns
            expected = set(omop_columns(name))
            required = set(expected)
            missing = required - cols
            if missing:
                raise CDMValidationError(
                    f"Table '{name}' missing columns: {sorted(missing)}"
                )
        except ValueError:
            pass  # Not a known OMOP table


def validate_omop_table(table: Any, name: str, version: str = "5.3") -> None:
    """Validate that table matches OMOP schema for the given table name.

    Parameters
    ----------
    table : Any
        Table to validate.
    name : str
        OMOP table name.
    version : str
        CDM version.

    Raises
    ------
    CDMValidationError
        If required columns are missing.
    """
    from cdmconnector.exceptions import CDMValidationError
    from cdmconnector.schemas import omop_table_fields

    cols = set()
    if hasattr(table, "columns"):
        cols = set(table.columns)
    elif hasattr(table, "schema"):
        try:
            cols = set(table.schema().names)
        except Exception:
            pass

    fields = omop_table_fields(version)
    required = fields[
        (fields["cdm_table_name"] == name.lower()) & (fields["is_required"])
    ]["cdm_field_name"].tolist()
    missing = set(required) - cols
    if missing:
        raise CDMValidationError(
            f"OMOP table '{name}' missing required columns: {sorted(missing)}"
        )


def validate_achilles_table(table: Any, name: str) -> None:
    """Validate that table matches Achilles table schema.

    Parameters
    ----------
    table : Any
        Table to validate.
    name : str
        Achilles table name.

    Raises
    ------
    CDMValidationError
        If required columns are missing.
    """
    from cdmconnector.exceptions import CDMValidationError
    from cdmconnector.schemas import achilles_columns

    cols = set()
    if hasattr(table, "columns"):
        cols = set(table.columns)
    elif hasattr(table, "schema"):
        try:
            cols = set(table.schema().names)
        except Exception:
            pass

    expected = set(achilles_columns(name))
    missing = expected - cols
    if missing:
        raise CDMValidationError(
            f"Achilles table '{name}' missing columns: {sorted(missing)}"
        )


def validate_name_level(name: str | list[str], level: str | list[str]) -> None:
    """Validate that name-level pairs are consistent.

    Parameters
    ----------
    name : str or list[str]
        Name(s) in compound format (may contain " &&& ").
    level : str or list[str]
        Corresponding level(s).

    Raises
    ------
    ValueError
        If name and level have different numbers of components.
    """
    names = name if isinstance(name, list) else [name]
    levels = level if isinstance(level, list) else [level]

    if len(names) != len(levels):
        raise ValueError(
            f"name and level must have the same length. Got {len(names)} names "
            f"and {len(levels)} levels."
        )

    for n, lv in zip(names, levels, strict=True):
        n_parts = n.split(" &&& ")
        lv_parts = lv.split(" &&& ")
        if len(n_parts) != len(lv_parts):
            raise ValueError(
                f"Name '{n}' has {len(n_parts)} parts but level '{lv}' has "
                f"{len(lv_parts)} parts."
            )


def validate_name_style(style: str) -> str:
    """Validate a naming style pattern (e.g. ``"{cohort_name}_{analysis}"``).

    Parameters
    ----------
    style : str
        Pattern string with ``{placeholder}`` tokens.

    Returns
    -------
    str
        The validated style.

    Raises
    ------
    ValueError
        If style is empty or contains no placeholders.
    """
    if not isinstance(style, str) or not style.strip():
        raise ValueError("Name style must be a non-empty string.")
    return style


def validate_new_column(
    name: str,
    existing_columns: list[str] | set[str],
) -> str:
    """Validate that a new column name doesn't conflict with existing columns.

    Parameters
    ----------
    name : str
        Proposed column name.
    existing_columns : list or set of str
        Existing column names.

    Returns
    -------
    str
        The validated column name.

    Raises
    ------
    ValueError
        If the name conflicts with an existing column.
    """
    if name in existing_columns:
        raise ValueError(
            f"Column '{name}' already exists. Choose a different name."
        )
    return name


def validate_age_group_argument(age_group: list | None) -> list | None:
    """Validate age group specification.

    Parameters
    ----------
    age_group : list of tuples/lists or None
        Each element is ``[lower, upper]`` defining an age range.

    Returns
    -------
    list or None
        Validated age groups.

    Raises
    ------
    ValueError
        If age groups are invalid.
    """
    if age_group is None:
        return None
    if not isinstance(age_group, list):
        raise ValueError("age_group must be a list of [lower, upper] pairs.")
    for group in age_group:
        if not isinstance(group, (list, tuple)) or len(group) != 2:
            raise ValueError(
                f"Each age group must be [lower, upper], got {group}"
            )
        lower, upper = group
        if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
            raise ValueError(f"Age bounds must be numeric, got {group}")
        if lower > upper:
            raise ValueError(
                f"Lower bound ({lower}) must be <= upper bound ({upper})"
            )
    return age_group


def validate_strata_argument(strata: list[str] | None) -> list[str]:
    """Validate strata column specification.

    Parameters
    ----------
    strata : list of str or None
        Column names to stratify by.

    Returns
    -------
    list[str]
        Validated strata columns (empty list if None).

    Raises
    ------
    TypeError
        If strata is not a list of strings.
    """
    if strata is None:
        return []
    if not isinstance(strata, list):
        raise TypeError(f"strata must be a list of strings, got {type(strata).__name__}")
    for s in strata:
        if not isinstance(s, str):
            raise TypeError(f"strata elements must be strings, got {type(s).__name__}")
    return strata


def validate_window_argument(window: list | tuple) -> tuple[int, int]:
    """Validate a time window argument.

    Parameters
    ----------
    window : list or tuple
        ``[lower, upper]`` where lower and upper are integers (days).
        Use ``float('inf')`` or ``None`` for unbounded.

    Returns
    -------
    tuple[int, int]
        Validated window as (lower, upper).

    Raises
    ------
    ValueError
        If window format is invalid.
    """
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError("Window must be [lower, upper] with 2 elements.")
    lower, upper = window
    if lower is None:
        lower = -999999
    if upper is None:
        upper = 999999
    if isinstance(lower, float) and lower == float("-inf"):
        lower = -999999
    if isinstance(upper, float) and upper == float("inf"):
        upper = 999999
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        raise ValueError(f"Window bounds must be numeric, got [{lower}, {upper}]")
    if lower > upper:
        raise ValueError(f"Lower bound ({lower}) must be <= upper ({upper})")
    return (int(lower), int(upper))


def validate_column(
    table: Any,
    column: str,
) -> str:
    """Validate that a column exists in a table.

    Parameters
    ----------
    table : Any
        Table with columns attribute or Ibis schema.
    column : str
        Column name to check.

    Returns
    -------
    str
        The validated column name.

    Raises
    ------
    ValueError
        If column does not exist in the table.
    """
    cols: set[str] = set()
    if hasattr(table, "columns"):
        cols = set(table.columns)
    elif hasattr(table, "schema"):
        try:
            cols = set(table.schema().names)
        except Exception:
            pass
    if column not in cols:
        raise ValueError(
            f"Column '{column}' not found. Available: {sorted(cols)}"
        )
