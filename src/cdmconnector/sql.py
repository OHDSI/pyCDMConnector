# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Schema qualification and SQL helpers for Ibis backends."""

from __future__ import annotations

from typing import Any


def qualify_table(
    con: Any,
    schema: Any | None,
    table: str,
    prefix: str | None = None,
) -> str | tuple[str, ...]:
    """Build a qualified table name for Ibis table lookup.

    Returns str or tuple depending on schema:
    - schema is None: return table (with prefix prepended if given)
    - schema is "main": return table (bare name, with prefix)
    - schema is a non-"main" str: return (schema, table)
    - schema is a dict:
      - If has "prefix" key: prepend prefix to table
      - If has "catalog" key: return (catalog, schema_val, table)
      - If has "schema" key: return (schema_val, table)
      - If has single key (like "db"): use value as schema -> (value, table)
    - schema is other type: return (str(schema), table)

    The *prefix* parameter is separate from a dict "prefix" key.
    If both are present, the dict prefix takes priority in the dict case.
    For non-dict schemas, the *prefix* parameter is prepended to table name.

    Parameters
    ----------
    con : Any
        Ibis connection (unused; kept for API compatibility).
    schema : str, dict, or None
        Schema spec (string, dict with schema/catalog/prefix, or None).
    table : str
        Table name.
    prefix : str or None, optional
        Optional table name prefix.

    Returns
    -------
    str or tuple[str, ...]
        Qualified name: str (e.g. "person"), (schema, table), or
        (catalog, schema, table).
    """
    # --- schema is None ---------------------------------------------------
    if schema is None:
        name = table
        if prefix:
            name = prefix + name
        return name

    # --- schema is a plain string -----------------------------------------
    if isinstance(schema, str):
        name = table
        if prefix:
            name = prefix + name
        if schema.lower() == "main":
            return name
        return (schema, name)

    # --- schema is a dict -------------------------------------------------
    if isinstance(schema, dict):
        catalog = schema.get("catalog")
        schema_name = schema.get("schema") or schema.get("schema_name") or ""

        # Single-key dict without recognised keys (e.g. {"db": "mydb"})
        if not schema_name and not catalog and len(schema) == 1:
            key = next(iter(schema))
            if key != "prefix":
                schema_name = schema[key]

        # Resolve name with prefix: dict prefix wins over param prefix
        name = table
        if "prefix" in schema:
            name = schema["prefix"] + name
        elif prefix:
            name = prefix + name

        if catalog:
            return (catalog, schema_name, name)
        if schema_name:
            return (schema_name, name)
        return name

    # --- schema is some other type ----------------------------------------
    name = table
    if prefix:
        name = prefix + name
    return (str(schema), name)


def in_schema(schema: Any | None, table: str) -> str | tuple[str, ...]:
    """Return schema-qualified table identifier.

    Wrapper around ``qualify_table(None, schema, table)``.

    Parameters
    ----------
    schema : str, dict, or None
        Schema spec.
    table : str
        Table name.

    Returns
    -------
    str or tuple[str, ...]
        Qualified identifier (string or tuple) for Ibis table resolution.
    """
    return qualify_table(None, schema, table)
