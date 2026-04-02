# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""CDMConnector exception hierarchy."""


class CDMConnectorError(Exception):
    """Base exception for all CDMConnector errors."""


class CDMValidationError(CDMConnectorError):
    """Raised when CDM validation fails (e.g. missing tables, bad schema)."""


class CohortError(CDMConnectorError):
    """Raised for cohort operation failures."""


class EunomiaError(CDMConnectorError):
    """Raised for Eunomia download/path failures."""


class SourceError(CDMConnectorError):
    """Raised for source/connection failures."""


class TableNotFoundError(CDMConnectorError):
    """Raised when a table is not found in the CDM."""
