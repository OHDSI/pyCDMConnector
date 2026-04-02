# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Internal utilities."""

from __future__ import annotations

import os
import re
import time
import uuid
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from cdmconnector.typing import SchemaSpec


# ---------------------------------------------------------------------------
# Version / schema helpers
# ---------------------------------------------------------------------------


def parse_cdm_version(version) -> str:
    """Normalize a CDM version string to ``"M.m"`` format.

    Parameters
    ----------
    version : Any
        Version string (e.g. ``"5.4"``, ``"v5.4.1"``).  ``None`` or
        non-string values fall back to ``"5.3"``.

    Returns
    -------
    str
        ``"major.minor"`` (e.g. ``"5.4"``).  Invalid input yields ``"5.3"``.
    """
    if version is None or not isinstance(version, str):
        return "5.3"
    s = version.strip()
    if not s:
        return "5.3"
    if s.lower().startswith("v"):
        s = s[1:].strip()
    parts = re.split(r"[.\s]+", s)
    if not parts:
        return "5.3"
    try:
        major = int(parts[0])
    except ValueError:
        return "5.3"
    minor = 0
    if len(parts) > 1:
        try:
            minor = int(parts[1])
        except ValueError:
            pass
    return f"{major}.{minor}"


def resolve_schema_name(schema: SchemaSpec | None = None) -> str | None:
    """Extract schema name from a string or dict.

    Parameters
    ----------
    schema : str, dict, or None
        Schema spec: a plain string, a dict with ``"schema"`` or
        ``"schema_name"`` key, or ``None``.

    Returns
    -------
    str or None
        Schema name, or ``None`` if not determinable.
    """
    if schema is None:
        return None
    if isinstance(schema, str):
        return schema
    if isinstance(schema, dict):
        val = schema.get("schema")
        if val:
            return val
        return schema.get("schema_name") or None
    return None


# ---------------------------------------------------------------------------
# Type conversion
# ---------------------------------------------------------------------------


def to_dataframe(obj: Any) -> pd.DataFrame:
    """Convert various types to a :class:`pandas.DataFrame`.

    Parameters
    ----------
    obj : Any
        A ``pd.DataFrame``, list, dict, Ibis Table (with ``to_pandas``), or
        other array-like.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (list, dict)):
        return pd.DataFrame(obj)
    # Ibis Table or similar
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_character(
    value: Any,
    *,
    allow_none: bool = False,
    length: int | None = None,
    min_num_character: int | None = None,
) -> None:
    """Assert that *value* is a string, optionally with length constraints.

    Parameters
    ----------
    value : Any
        Value to check.
    allow_none : bool
        If ``True``, ``None`` is accepted without error.
    length : int or None
        If set, the string must have exactly this many characters.
    min_num_character : int or None
        If set, the string must have at least this many characters.

    Raises
    ------
    TypeError
        If *value* is not a string (and not ``None`` when allowed).
    ValueError
        If a length constraint is violated.
    """
    if allow_none and value is None:
        return
    if not isinstance(value, str):
        raise TypeError(f"Expected str, got {type(value).__name__}; value must be str")
    if length is not None and len(value) != length:
        raise ValueError(
            f"Expected string of length {length}, got {len(value)}"
        )
    if min_num_character is not None and len(value) < min_num_character:
        raise ValueError(
            f"Expected at least {min_num_character} characters, got {len(value)}"
        )


def assert_choice(value: Any, choices) -> None:
    """Assert that *value* is in *choices*.

    Parameters
    ----------
    value : Any
        Value to check.
    choices : iterable
        Allowed values.

    Raises
    ------
    ValueError
        If *value* is not found in *choices*.
    """
    if value not in choices:
        raise ValueError(f"Expected one of {choices}, got {value!r}")


# ---------------------------------------------------------------------------
# Naming utilities
# ---------------------------------------------------------------------------


_unique_counter = 0


def unique_table_name(prefix: str = "tmp_") -> str:
    """Generate a unique table name using *prefix*, a timestamp, and a counter.

    Parameters
    ----------
    prefix : str, optional
        Prefix for the name (default ``"tmp_"``).

    Returns
    -------
    str
        A name like ``"tmp_1712000000_12345_1"``.
    """
    global _unique_counter
    _unique_counter += 1
    return f"{prefix}{int(time.time() * 1000)}_{os.getpid()}_{_unique_counter}"


# ---------------------------------------------------------------------------
# Additional assertion helpers (mirrors omopgenerics assert* functions)
# ---------------------------------------------------------------------------


def assert_numeric(value: Any, *, allow_none: bool = False) -> None:
    """Check that *value* is numeric (int or float).

    Raises :class:`TypeError` if not.
    """
    if allow_none and value is None:
        return
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric, got {type(value).__name__}")


def assert_logical(value: Any, *, allow_none: bool = False) -> None:
    """Check that *value* is boolean.

    Raises :class:`TypeError` if not.
    """
    if allow_none and value is None:
        return
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value).__name__}")


def assert_class(obj: Any, cls: type, *, allow_none: bool = False) -> None:
    """Check that *obj* is an instance of *cls*.

    Raises :class:`TypeError` if not.
    """
    if allow_none and obj is None:
        return
    if not isinstance(obj, cls):
        raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")


def assert_list(value: Any, *, allow_none: bool = False) -> None:
    """Check that *value* is a list.

    Raises :class:`TypeError` if not.
    """
    if allow_none and value is None:
        return
    if not isinstance(value, list):
        raise TypeError(f"Expected list, got {type(value).__name__}")


def assert_date(value: Any, *, allow_none: bool = False) -> None:
    """Check that *value* is a date or datetime.

    Raises :class:`TypeError` if not.
    """
    import datetime

    if allow_none and value is None:
        return
    if not isinstance(value, (datetime.date, datetime.datetime)):
        raise TypeError(f"Expected date or datetime, got {type(value).__name__}")


def assert_true(condition: bool, msg: str = "Assertion failed") -> None:
    """Assert that *condition* is True.

    Parameters
    ----------
    condition : bool
        Condition to check.
    msg : str
        Error message if False.

    Raises
    ------
    ValueError
        If condition is not True.
    """
    if not condition:
        raise ValueError(msg)


def assert_table(value: Any, *, allow_none: bool = False) -> None:
    """Check that *value* is a table-like object (DataFrame or Ibis table).

    Raises :class:`TypeError` if not.
    """
    if allow_none and value is None:
        return
    if not (hasattr(value, "columns") or hasattr(value, "schema")):
        raise TypeError(
            f"Expected table-like object, got {type(value).__name__}"
        )


# ---------------------------------------------------------------------------
# Table utilities (mirrors omopgenerics)
# ---------------------------------------------------------------------------


def is_table_empty(table: Any) -> bool:
    """Check if a table (Ibis expression, DataFrame, or similar) has zero rows."""
    if table is None:
        return True
    if hasattr(table, "empty"):
        return bool(table.empty)
    if hasattr(table, "count"):
        try:
            cnt = table.count()
            if hasattr(cnt, "execute"):
                cnt = cnt.execute()
            return int(cnt) == 0
        except Exception:
            pass
    if hasattr(table, "__len__"):
        return len(table) == 0
    return False


def number_records(table: Any) -> int:
    """Count total records (rows) in a table."""
    if table is None:
        return 0
    if hasattr(table, "count"):
        cnt = table.count()
        if hasattr(cnt, "execute"):
            cnt = cnt.execute()
        return int(cnt)
    if hasattr(table, "__len__"):
        return len(table)
    return 0


def number_subjects(table: Any) -> int:
    """Count distinct subjects in a table.

    Looks for ``person_id`` or ``subject_id`` column.
    """
    pid_col = get_person_identifier(table)
    if pid_col is None:
        return 0
    if hasattr(table, "nunique"):
        try:
            cnt = table[pid_col].nunique()
            if hasattr(cnt, "execute"):
                cnt = cnt.execute()
            return int(cnt)
        except Exception:
            pass
    if hasattr(table, "columns"):
        if isinstance(table, pd.DataFrame) and pid_col in table.columns:
            return int(table[pid_col].nunique())
    return 0


def get_person_identifier(table: Any) -> str | None:
    """Return the person/subject ID column name in a table.

    Checks for ``person_id`` first, then ``subject_id``.
    """
    cols = _get_column_names(table)
    if "person_id" in cols:
        return "person_id"
    if "subject_id" in cols:
        return "subject_id"
    return None


def _get_column_names(table: Any) -> set[str]:
    """Extract column names from various table types."""
    if table is None:
        return set()
    if hasattr(table, "columns"):
        return set(table.columns)
    if hasattr(table, "schema"):
        try:
            return set(table.schema().names)
        except Exception:
            return set()
    return set()


# ---------------------------------------------------------------------------
# Naming utilities (mirrors omopgenerics)
# ---------------------------------------------------------------------------


def to_snake_case(name: str) -> str:
    """Convert a camelCase or PascalCase string to snake_case."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


def unique_id(n: int = 1) -> list[int]:
    """Generate *n* unique integer IDs."""
    import random

    return [random.randint(1, 2**31 - 1) for _ in range(n)]


def tmp_prefix() -> str:
    """Return the standard temporary table prefix (``"tmp_"``)."""
    return "tmp_"


def _first_scalar(df: Any, col: str | None = None) -> Any:
    """First cell of a 1-row DataFrame, or first element of column."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    if col is not None:
        return df[col].iloc[0] if col in getattr(df, "columns", []) else None
    if hasattr(df, "iloc"):
        return df.iloc[0, 0] if df.shape[1] == 1 else df.iloc[0].tolist()[0]
    return None


def _suggest_similar(
    name: str, candidates: list[str], max_suggestions: int = 3
) -> str | None:
    """Return the closest candidate name, or ``None``."""
    if not candidates or not name:
        return None
    name_lower = name.lower()

    def key(c: str) -> tuple[int, int]:
        c_lower = c.lower()
        prefix_ok = 1 if c_lower.startswith(name_lower[:1]) else 0
        same = sum(1 for a, b in zip(name_lower, c_lower) if a == b)
        return (-prefix_ok, -same)

    sorted_candidates = sorted(candidates, key=key)
    best = sorted_candidates[0] if sorted_candidates else None
    if best and best != name_lower:
        return best
    return None
