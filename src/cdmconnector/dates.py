# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Date arithmetic: thin wrappers over Ibis temporal expressions.

Port of R CDMConnector dateadd(), datediff(), datepart() for use in Ibis pipelines.
"""

from __future__ import annotations

from typing import Any, Literal

import ibis


def dateadd(
    date_expr: Any,
    number: int | Any,
    interval: Literal["day", "year"] = "day",
) -> Any:
    """Add days or years to a date (Ibis expression).

    Parameters
    ----------
    date_expr : ibis DateValue or TimestampValue
        The date column or expression.
    number : int
        Number of units to add (positive or negative). For column-based values,
        use backend-specific expressions (e.g. date_expr + col * ibis.interval(days=1)).
    interval : "day" | "year"
        Unit to add. Default "day".

    Returns
    -------
    Ibis date/timestamp expression
        date_expr + number of given units.

    Examples
    --------
    >>> import ibis
    >>> from cdmconnector.dates import dateadd
    >>> t = ibis.memtable({"d": ["2020-01-01"]}).cast({"d": "date"})
    >>> t.mutate(next_d=dateadd(t.d, 1, "day"))
    """
    if interval == "day":
        delta = ibis.interval(days=number)
    elif interval == "year":
        delta = ibis.interval(years=number)
    else:
        raise ValueError(f"interval must be 'day' or 'year', got {interval!r}")
    return date_expr + delta


def datediff(
    start_expr: Any,
    end_expr: Any,
    interval: Literal["day", "month", "year"] = "day",
) -> Any:
    """Compute the difference between two dates (Ibis expression).

    Parameters
    ----------
    start_expr : ibis DateValue or TimestampValue
        Start date column or expression.
    end_expr : ibis DateValue or TimestampValue
        End date column or expression.
    interval : "day" | "month" | "year"
        Unit for the difference. Default "day".

    Returns
    -------
    Ibis integer expression
        Number of full units between start and end (end - start).

    Examples
    --------
    >>> import ibis
    >>> from cdmconnector.dates import datediff
    >>> t = ibis.memtable({
    ...     "start": ["2020-01-01"], "end": ["2020-01-31"]
    ... }).cast({"start": "date", "end": "date"})
    >>> t.mutate(days=datediff(t.start, t.end, "day"))
    """
    if interval == "day":
        # Ibis: (end - start) gives interval; extract days via delta
        return end_expr.delta(start_expr, unit="day")
    if interval in ("month", "year"):
        # Use delta with month or year unit
        return end_expr.delta(start_expr, unit=interval)
    raise ValueError(f"interval must be 'day', 'month', or 'year', got {interval!r}")


def datepart(
    date_expr: Any,
    interval: Literal["year", "month", "day"] = "year",
) -> Any:
    """Extract year, month, or day from a date (Ibis expression).

    Parameters
    ----------
    date_expr : ibis DateValue or TimestampValue
        The date column or expression.
    interval : "year" | "month" | "day"
        Part to extract. Default "year".

    Returns
    -------
    Ibis integer expression
        The extracted part.

    Examples
    --------
    >>> import ibis
    >>> from cdmconnector.dates import datepart
    >>> t = ibis.memtable({"birth_date": ["1993-04-19"]}).cast({"birth_date": "date"})
    >>> t.mutate(year=datepart(t.birth_date, "year"))
    """
    if interval == "year":
        return date_expr.year()
    if interval == "month":
        return date_expr.month()
    if interval == "day":
        return date_expr.day()
    raise ValueError(f"interval must be 'year', 'month', or 'day', got {interval!r}")
