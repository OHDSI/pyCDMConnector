# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Type aliases and protocol hints for CDMConnector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import ibis
    from ibis.expr.types import Table

# Ibis table expression (lazy)
IbisTable = Union["Table", "ibis.expr.types.Table"]

# Schema can be a string (schema name) or dict with catalog/schema/prefix
SchemaSpec = Union[str, dict[str, str]]

# OMOP CDM versions we support
CdmVersionLiteral = str  # "5.3" | "5.4"

# Source type (backend name)
SourceTypeLiteral = str  # "duckdb" | "postgresql" | "snowflake" | "bigquery" | "spark" | "local"
