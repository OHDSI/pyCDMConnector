# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Result wrapper for lazy expressions: collect or compute explicitly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from cdmconnector.cdm import Cdm


@dataclass(frozen=True)
class Result:
    """
    Lazy result: holds an Ibis table expression and optional metadata.

    Use .collect(limit=None) to get a pandas DataFrame, or
    .compute(cdm, name, ...) to persist as a table and get an Ibis reference.
    """

    expr: Any  # ibis.Table
    meta: dict[str, Any] = field(default_factory=dict)

    def collect(self, limit: int | None = None) -> pd.DataFrame:
        """Materialize to a pandas DataFrame. Optionally limit rows.

        Parameters
        ----------
        limit : int or None, optional
            If set, limit number of rows (for table expressions).

        Returns
        -------
        pandas.DataFrame
            Materialized result (or CDM snapshot if this is a snapshot Result).
        """
        if self.expr is None and "_snapshot_cdm" in self.meta:
            from cdmconnector.cdm import _materialize_snapshot

            return _materialize_snapshot(
                self.meta["_snapshot_cdm"],
                self.meta.get("_snapshot_compute_data_hash", False),
            )
        from cdmconnector.cdm import collect

        return collect(self.expr, limit=limit)

    def compute(
        self,
        cdm: Cdm | Any,
        name: str,
        *,
        schema: str | dict | None = None,
        overwrite: bool = False,
    ) -> Any:
        """Persist to a table and return Ibis table reference.

        Parameters
        ----------
        cdm : Cdm or Any
            CDM reference (must be database-backed for table write).
        name : str
            Table name in write schema.
        schema : str or dict or None, optional
            Override schema for write (default: cdm.write_schema).
        overwrite : bool, optional
            If True, replace existing table (default False).

        Returns
        -------
        Any
            Ibis table reference to the new table.
        """
        if self.expr is None and "_snapshot_cdm" in self.meta:
            df = self.collect()
            from cdmconnector.cdm import insert_table

            return insert_table(cdm, name, df, overwrite=overwrite)
        from cdmconnector.cdm import compute

        return compute(cdm, self.expr, name, schema=schema, overwrite=overwrite)
