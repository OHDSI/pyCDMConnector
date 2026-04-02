# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""CDM source objects: CdmSource, LocalCdmSource, and DbCdmSource backed by Ibis."""

from __future__ import annotations

from typing import Any

from cdmconnector.exceptions import SourceError


def full_table_name(name: str, prefix: str | None) -> str:
    """Build full table name with prefix. If name already starts with prefix, return as-is.

    Parameters
    ----------
    name : str
        Logical table name (e.g. "person").
    prefix : str or None
        Optional prefix (e.g. "wrk_").

    Returns
    -------
    str
        Physical table name.
    """
    if not prefix:
        return name
    if name.lower().startswith(prefix.lower()):
        return name
    return prefix + name


def resolve_prefix(prefix: str | None, config: dict | None) -> str | None:
    """Resolve prefix from explicit value or config dict.

    Config dict prefix takes priority over explicit prefix.

    Parameters
    ----------
    prefix : str or None
        Default prefix.
    config : dict or None
        Schema config; if dict with "prefix" key, that value is returned.

    Returns
    -------
    str or None
        Resolved prefix string, or None.
    """
    if isinstance(config, dict) and "prefix" in config:
        return config["prefix"]
    return prefix


class CdmSource:
    """Base source class."""

    def __init__(self, source_type: str) -> None:
        """Store source type (e.g. duckdb, local).

        Parameters
        ----------
        source_type : str
            Backend identifier.
        """
        self.source_type = source_type

    def __repr__(self) -> str:
        return f"CdmSource(type={self.source_type!r})"


class LocalCdmSource(CdmSource):
    """In-memory source (no database connection)."""

    def __init__(self) -> None:
        """Create a local CDM source."""
        super().__init__("local")

    def __repr__(self) -> str:
        return "LocalCdmSource(local)"


class DbCdmSource(CdmSource):
    """Database-backed source wrapping an Ibis connection."""

    def __init__(
        self,
        con: Any,
        schema: Any,
        prefix: str | None = None,
        write_schema: Any | None = None,
    ) -> None:
        """Create a database-backed CDM source.

        Parameters
        ----------
        con : Any
            Ibis connection (e.g. ibis.duckdb.connect(...)).
        schema : str or dict
            CDM schema specification.
        prefix : str or None, optional
            Optional prefix for table names.
        write_schema : str, dict, or None, optional
            Schema for write operations. Defaults to *schema* if not given.
        """
        super().__init__(self._detect_dbms(con))
        self.con = con
        self._schema = schema
        self._prefix = prefix or resolve_prefix(None, schema)
        self._write_schema = write_schema if write_schema is not None else schema

    # ------------------------------------------------------------------
    # DBMS detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_dbms(con: Any) -> str:
        """Infer backend type from an Ibis connection.

        Parameters
        ----------
        con : Any
            Ibis connection.

        Returns
        -------
        str
            One of "duckdb", "postgres", "postgresql", "snowflake",
            "bigquery", "spark", "mssql", or "unknown".
        """
        name = type(con).__module__.split(".")
        for part in name:
            if part in (
                "duckdb",
                "postgres",
                "postgresql",
                "snowflake",
                "bigquery",
                "spark",
                "mssql",
            ):
                return part
        return "unknown"

    # ------------------------------------------------------------------
    # Write schema
    # ------------------------------------------------------------------

    @property
    def write_schema(self) -> Any:
        """Write schema spec (schema name, optional catalog, optional prefix)."""
        return self._write_schema

    @write_schema.setter
    def write_schema(self, value: Any) -> None:
        self._write_schema = value

    # ------------------------------------------------------------------
    # Table operations
    # ------------------------------------------------------------------

    def list_tables(self, schema: Any | None = None) -> list[str]:
        """List tables, optionally filtering by prefix.

        Parameters
        ----------
        schema : str, dict, or None, optional
            Schema to inspect. Used to resolve prefix overrides.

        Returns
        -------
        list[str]
            Logical table names (with prefix stripped).
        """
        try:
            all_tables = self.con.list_tables()
        except Exception as e:
            raise SourceError(f"Failed to list tables: {e}") from e

        prefix = self._prefix
        if isinstance(schema, dict) and "prefix" in schema:
            prefix = schema["prefix"]

        if prefix:
            matching = [t for t in all_tables if t.lower().startswith(prefix.lower())]
            return [t[len(prefix):] for t in matching]
        return list(all_tables)

    def table(self, name: str, schema: Any | None = None) -> Any:
        """Get an Ibis table reference.

        Parameters
        ----------
        name : str
            Logical table name.
        schema : str, dict, or None, optional
            Schema to use for prefix resolution.

        Returns
        -------
        ibis.expr.types.Table
            Ibis table expression.
        """
        prefix = self._prefix
        if isinstance(schema, dict) and "prefix" in schema:
            prefix = schema["prefix"]

        physical = full_table_name(name, prefix)
        return self.con.table(physical)

    def insert_table(
        self,
        name: str,
        data: Any,
        overwrite: bool = True,
        schema: Any = None,
    ) -> Any:
        """Insert data as a table. data can be Arrow table, pandas DataFrame, or dict.

        Parameters
        ----------
        name : str
            Logical table name.
        data : pyarrow.Table, pandas.DataFrame, dict, or Ibis expression
            Data to insert.
        overwrite : bool, optional
            Replace existing table (default True).

        Returns
        -------
        ibis.expr.types.Table
            Ibis table reference to the created table.
        """
        import pyarrow as pa

        prefix = self._prefix
        physical = full_table_name(name, prefix)

        if isinstance(data, dict):
            data = pa.table(data)

        return self.con.create_table(physical, obj=data, overwrite=overwrite)

    def drop_table(self, name: str) -> None:
        """Drop a table.

        Parameters
        ----------
        name : str
            Logical table name.
        """
        prefix = self._prefix
        physical = full_table_name(name, prefix)
        try:
            self.con.drop_table(physical)
        except Exception:
            pass

    def disconnect(self, drop_write_schema: bool = False) -> None:
        """Disconnect from database.

        Parameters
        ----------
        drop_write_schema : bool, optional
            If True, drop all tables before disconnecting (default False).
        """
        if drop_write_schema:
            try:
                for t in self.con.list_tables():
                    self.con.drop_table(t)
            except Exception:
                pass
        try:
            self.con.disconnect()
        except Exception:
            pass

    def _database_for_schema(self, schema: Any) -> str | None:
        """Extract database/schema name from schema spec.

        Parameters
        ----------
        schema : str, dict, or None
            Schema specification.

        Returns
        -------
        str or None
            Database/schema name.
        """
        if schema is None:
            return None
        if isinstance(schema, str):
            return schema
        if isinstance(schema, dict):
            return schema.get("schema") or schema.get("schema_name")
        return str(schema)

    def __repr__(self) -> str:
        return f"DbCdmSource(type={self.source_type!r}, schema={self._schema!r})"
