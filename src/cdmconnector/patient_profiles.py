# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""
PatientProfiles: Add patient characteristics to OMOP CDM tables (Python with Ibis).

Port of R PatientProfiles. Adds demographics (age, sex), prior/future observation,
death flags, cohort/table/concept intersects, and summarised result helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ibis

from cdmconnector.dates import datediff
from cdmconnector.logging_config import get_logger
from cdmconnector.schemas import FIELD_TABLES_COLUMNS

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ibis.expr.types import Table

    from cdmconnector.cdm import Cdm


# --- Person / subject identifier ---

def _person_id_column(table: Any) -> str:
    """Return 'person_id' or 'subject_id' whichever is present in the table."""
    cols = [c.lower() for c in table.columns]
    if "subject_id" in cols:
        return "subject_id"
    if "person_id" in cols:
        return "person_id"
    logger.error("Table must have 'person_id' or 'subject_id'.")
    raise ValueError("Table must have 'person_id' or 'subject_id'.")


# --- Column name helpers (R: startDateColumn, endDateColumn, etc.) ---

def start_date_column(table_name: str) -> str:
    """Return the start date column name for an OMOP table (e.g. 'condition_occurrence' -> 'condition_start_date')."""
    tn = table_name.lower()
    if tn not in FIELD_TABLES_COLUMNS:
        logger.error("Unknown table for start_date_column: %r", table_name)
        raise ValueError(f"Unknown table {table_name!r}. Use a key from FIELD_TABLES_COLUMNS.")
    c = FIELD_TABLES_COLUMNS[tn].get("start_date")
    if c is None:
        raise ValueError(f"Table {table_name!r} has no start_date column.")
    return c


def end_date_column(table_name: str) -> str | None:
    """Return the end date column name for an OMOP table, or None."""
    tn = table_name.lower()
    if tn not in FIELD_TABLES_COLUMNS:
        return None
    return FIELD_TABLES_COLUMNS[tn].get("end_date")


def standard_concept_id_column(table_name: str) -> str | None:
    """Return the standard concept_id column for an OMOP table, or None."""
    tn = table_name.lower()
    if tn not in FIELD_TABLES_COLUMNS:
        return None
    return FIELD_TABLES_COLUMNS[tn].get("standard_concept")


def source_concept_id_column(table_name: str) -> str | None:
    """Return the source concept_id column for an OMOP table, or None."""
    tn = table_name.lower()
    if tn not in FIELD_TABLES_COLUMNS:
        return None
    return FIELD_TABLES_COLUMNS[tn].get("source_concept")


# --- Demographics (add_age, add_sex, add_demographics, prior/future observation, date_of_birth) ---

def add_demographics(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    age: bool = True,
    age_name: str = "age",
    age_missing_month: int = 1,
    age_missing_day: int = 1,
    age_impose_month: bool = False,
    age_impose_day: bool = False,
    age_group: list[tuple[str, tuple[int | float, int | float]]] | None = None,
    missing_age_group_value: str = "None",
    sex: bool = True,
    sex_name: str = "sex",
    missing_sex_value: str = "None",
    prior_observation: bool = True,
    prior_observation_name: str = "prior_observation",
    prior_observation_type: str = "days",
    future_observation: bool = True,
    future_observation_name: str = "future_observation",
    future_observation_type: str = "days",
    date_of_birth: bool = False,
    date_of_birth_name: str = "date_of_birth",
) -> Table:
    """Add demographic characteristics to a table: age, sex, prior/future observation, optional date_of_birth.

    Table must have person_id or subject_id and an index date column. Joins person and observation_period.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date column.
    cdm : Cdm
        CDM reference (person, observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    age, sex, prior_observation, future_observation, date_of_birth : bool, optional
        Which demographics to add.
    age_name, sex_name, prior_observation_name, future_observation_name, date_of_birth_name : str, optional
        Output column names.
    age_group : list or None, optional
        Optional age groups as (col_name, (low, high)) or (col_name, label, (low, high)).
    age_missing_month, age_missing_day : int, optional
        Default month/day when missing.
    age_impose_month, age_impose_day : bool, optional
        Force default month/day.
    missing_age_group_value, missing_sex_value : str, optional
        Value for missing/unknown.
    prior_observation_type, future_observation_type : str, optional
        "days" or "date".

    Returns
    -------
    ibis.expr.types.Table
        Table with added demographic columns.
    """
    return _add_demographics_query(
        table,
        cdm,
        index_date=index_date,
        age=age,
        age_name=age_name,
        age_missing_month=age_missing_month,
        age_missing_day=age_missing_day,
        age_impose_month=age_impose_month,
        age_impose_day=age_impose_day,
        age_group=age_group,
        missing_age_group_value=missing_age_group_value,
        sex=sex,
        sex_name=sex_name,
        missing_sex_value=missing_sex_value,
        prior_observation=prior_observation,
        prior_observation_name=prior_observation_name,
        prior_observation_type=prior_observation_type,
        future_observation=future_observation,
        future_observation_name=future_observation_name,
        future_observation_type=future_observation_type,
        date_of_birth=date_of_birth,
        date_of_birth_name=date_of_birth_name,
    )


def _join_observation_period_demographics(
    table: Any,
    cdm: Cdm,
    *,
    person_id_col: str,
    index_date: str,
    prior_observation: bool,
    prior_observation_name: str,
    prior_observation_type: str,
    future_observation: bool,
    future_observation_name: str,
    future_observation_type: str,
) -> Table:
    """Join observation period and add prior/future observation columns."""
    obs_period = cdm["observation_period"]
    op_sel = obs_period.select(
        obs_period.person_id.name(person_id_col),
        obs_period.observation_period_start_date.name("_obs_start"),
        obs_period.observation_period_end_date.name("_obs_end"),
    )
    distinct_subjects = table.select(person_id_col, index_date).distinct()
    join_obs = distinct_subjects.join(op_sel, person_id_col, how="inner")
    join_obs = join_obs.filter(
        (join_obs["_obs_start"] <= join_obs[index_date])
        & (join_obs[index_date] <= join_obs["_obs_end"])
    )
    if prior_observation:
        if prior_observation_type == "days":
            join_obs = join_obs.mutate(
                **{prior_observation_name: datediff(join_obs["_obs_start"], join_obs[index_date], "day")}
            )
        else:
            join_obs = join_obs.mutate(**{prior_observation_name: join_obs["_obs_start"]})
    if future_observation:
        if future_observation_type == "days":
            join_obs = join_obs.mutate(
                **{future_observation_name: datediff(join_obs[index_date], join_obs["_obs_end"], "day")}
            )
        else:
            join_obs = join_obs.mutate(**{future_observation_name: join_obs["_obs_end"]})
    keep = [c for c in join_obs.columns if c not in ("_obs_start", "_obs_end")]
    join_obs = join_obs.select(keep)
    return table.left_join(join_obs, [person_id_col, index_date])


def _join_person_for_demographics(table: Any, cdm: Cdm, person_id_col: str) -> Table:
    """Join person table to get birth and gender columns."""
    person = cdm["person"]
    person_sel = person.select(
        person.person_id,
        person.year_of_birth,
        person.month_of_birth,
        person.day_of_birth,
        person.gender_concept_id,
    )
    return table.left_join(person_sel, table[person_id_col] == person_sel.person_id)


def _add_date_of_birth_column(
    table: Any,
    *,
    date_of_birth_name: str,
    age_missing_month: int,
    age_missing_day: int,
    age_impose_month: bool,
    age_impose_day: bool,
) -> Table:
    """Add date_of_birth column from year/month/day of birth."""
    m = ibis.coalesce(table.month_of_birth, ibis.literal(age_missing_month))
    d = ibis.coalesce(table.day_of_birth, ibis.literal(age_missing_day))
    if age_impose_month:
        m = ibis.literal(age_missing_month)
    if age_impose_day:
        d = ibis.literal(age_missing_day)
    m_str = m.cast("string")
    d_str = d.cast("string")
    try:
        m_str = m_str.lpad(2, "0")
        d_str = d_str.lpad(2, "0")
    except Exception:
        pass
    dob_str = table.year_of_birth.cast("string") + "-" + m_str + "-" + d_str
    return table.mutate(**{date_of_birth_name: dob_str.cast("date")})


def _add_age_and_age_groups(
    table: Any,
    *,
    index_date: str,
    age_name: str,
    age_group: list[tuple[str, tuple[int | float, int | float]]] | None,
    missing_age_group_value: str,
) -> Table:
    """Add age column and optional age group columns."""
    from collections import defaultdict

    age_expr = table[index_date].year() - table.year_of_birth.cast("int32")
    table = table.mutate(**{age_name: age_expr})
    if not age_group:
        return table
    by_col: dict[str, list[tuple[str, tuple[int | float, int | float]]]] = defaultdict(list)
    for item in age_group:
        if len(item) == 2:
            col_name, (low, high) = item
            label = f"{low}-{high}"
        else:
            col_name, label, (low, high) = item
        by_col[col_name].append((label, (low, high)))
    for col_name, ranges in by_col.items():
        branches = []
        for label, (low, high) in ranges:
            if high == float("inf") or (isinstance(high, (int, float)) and high >= 1e10):
                cond = table[age_name] >= low
            else:
                cond = (table[age_name] >= low) & (table[age_name] <= high)
            branches.append((cond, ibis.literal(label)))
        expr = ibis.cases(*branches, else_=ibis.literal(missing_age_group_value))
        table = table.mutate(**{col_name: expr})
    return table


def _add_sex_column_demographics(
    table: Any,
    *,
    sex_name: str,
    missing_sex_value: str,
) -> Table:
    """Add sex column from gender_concept_id (8507 Male, 8532 Female)."""
    sex_expr = ibis.cases(
        (table.gender_concept_id == 8507, ibis.literal("Male")),
        (table.gender_concept_id == 8532, ibis.literal("Female")),
        else_=ibis.literal(missing_sex_value),
    )
    return table.mutate(**{sex_name: sex_expr})


def _drop_person_derived_columns(table: Any) -> Table:
    """Drop year_of_birth, month_of_birth, day_of_birth, gender_concept_id from table."""
    for c in ["year_of_birth", "month_of_birth", "day_of_birth", "gender_concept_id"]:
        if c in table.columns:
            table = table.drop(c)
    return table


def _add_demographics_query(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str,
    age: bool,
    age_name: str,
    age_missing_month: int,
    age_missing_day: int,
    age_impose_month: bool,
    age_impose_day: bool,
    age_group: list[tuple[str, tuple[int | float, int | float]]] | None,
    missing_age_group_value: str,
    sex: bool,
    sex_name: str,
    missing_sex_value: str,
    prior_observation: bool,
    prior_observation_name: str,
    prior_observation_type: str,
    future_observation: bool,
    future_observation_name: str,
    future_observation_type: str,
    date_of_birth: bool,
    date_of_birth_name: str,
) -> Table:
    """Add demographics by composing observation period join, person join, and column helpers."""
    person_id_col = _person_id_column(table)
    if index_date not in table.columns:
        logger.error("index_date %r is not a column of the table", index_date)
        raise ValueError(f"index_date {index_date!r} must be a column of the table.")

    need_obs = prior_observation or future_observation
    if need_obs:
        logger.debug("Adding prior/future observation (index_date=%s)", index_date)
        table = _join_observation_period_demographics(
            table,
            cdm,
            person_id_col=person_id_col,
            index_date=index_date,
            prior_observation=prior_observation,
            prior_observation_name=prior_observation_name,
            prior_observation_type=prior_observation_type,
            future_observation=future_observation,
            future_observation_name=future_observation_name,
            future_observation_type=future_observation_type,
        )

    need_person = age or bool(age_group) or sex or date_of_birth
    if need_person:
        logger.debug("Adding person-derived demographics (age=%s, sex=%s, date_of_birth=%s)", age, sex, date_of_birth)
        table = _join_person_for_demographics(table, cdm, person_id_col)
        if date_of_birth:
            table = _add_date_of_birth_column(
                table,
                date_of_birth_name=date_of_birth_name,
                age_missing_month=age_missing_month,
                age_missing_day=age_missing_day,
                age_impose_month=age_impose_month,
                age_impose_day=age_impose_day,
            )
        if age or age_group:
            table = _add_age_and_age_groups(
                table,
                index_date=index_date,
                age_name=age_name,
                age_group=age_group,
                missing_age_group_value=missing_age_group_value,
            )
        if sex:
            table = _add_sex_column_demographics(
                table,
                sex_name=sex_name,
                missing_sex_value=missing_sex_value,
            )
        table = _drop_person_derived_columns(table)

    return table


def add_age(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    age_name: str = "age",
    age_group: list[tuple[str, tuple[int | float, int | float]]] | None = None,
    age_missing_month: int = 1,
    age_missing_day: int = 1,
    age_impose_month: bool = False,
    age_impose_day: bool = False,
    missing_age_group_value: str = "None",
) -> Table:
    """Add age at index_date (and optional age groups).

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date column.
    cdm : Cdm
        CDM reference (person).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    age_name : str, optional
        Output column name (default "age").
    age_group : list or None, optional
        Optional age groups (col_name, (low, high)) or (col_name, label, (low, high)).
    age_missing_month, age_missing_day : int, optional
        Default month/day when missing.
    age_impose_month, age_impose_day : bool, optional
        Force default month/day.
    missing_age_group_value : str, optional
        Value for missing/unknown group.

    Returns
    -------
    ibis.expr.types.Table
        Table with age (and optional age_group) columns.
    """
    return add_demographics(
        table,
        cdm,
        index_date=index_date,
        age=True,
        age_name=age_name,
        age_group=age_group,
        age_missing_month=age_missing_month,
        age_missing_day=age_missing_day,
        age_impose_month=age_impose_month,
        age_impose_day=age_impose_day,
        missing_age_group_value=missing_age_group_value,
        sex=False,
        prior_observation=False,
        future_observation=False,
        date_of_birth=False,
    )


def add_sex(
    table: Any,
    cdm: Cdm,
    *,
    sex_name: str = "sex",
    missing_sex_value: str = "None",
) -> Table:
    """Add sex from person (gender_concept_id: 8507 Male, 8532 Female).

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id.
    cdm : Cdm
        CDM reference (person).
    sex_name : str, optional
        Output column name (default "sex").
    missing_sex_value : str, optional
        Value for unknown gender (default "None").

    Returns
    -------
    ibis.expr.types.Table
        Table with sex column.
    """
    return add_demographics(
        table,
        cdm,
        age=False,
        sex=True,
        sex_name=sex_name,
        missing_sex_value=missing_sex_value,
        prior_observation=False,
        future_observation=False,
        date_of_birth=False,
    )


def add_prior_observation(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    prior_observation_name: str = "prior_observation",
    prior_observation_type: str = "days",
) -> Table:
    """Add days (or date) of prior observation in the current observation period at index_date.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference (observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    prior_observation_name : str, optional
        Output column name (default "prior_observation").
    prior_observation_type : str, optional
        "days" or "date" (default "days").

    Returns
    -------
    ibis.expr.types.Table
        Table with prior_observation column.
    """
    return add_demographics(
        table,
        cdm,
        index_date=index_date,
        age=False,
        sex=False,
        prior_observation=True,
        prior_observation_name=prior_observation_name,
        prior_observation_type=prior_observation_type,
        future_observation=False,
        date_of_birth=False,
    )


def add_future_observation(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    future_observation_name: str = "future_observation",
    future_observation_type: str = "days",
) -> Table:
    """Add days (or date) of future observation from index_date to end of observation period.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference (observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    future_observation_name : str, optional
        Output column name (default "future_observation").
    future_observation_type : str, optional
        "days" or "date" (default "days").

    Returns
    -------
    ibis.expr.types.Table
        Table with future_observation column.
    """
    return add_demographics(
        table,
        cdm,
        index_date=index_date,
        age=False,
        sex=False,
        prior_observation=False,
        future_observation=True,
        future_observation_name=future_observation_name,
        future_observation_type=future_observation_type,
        date_of_birth=False,
    )


def add_date_of_birth(
    table: Any,
    cdm: Cdm,
    *,
    date_of_birth_name: str = "date_of_birth",
    missing_day: int = 1,
    missing_month: int = 1,
    impose_day: bool = False,
    impose_month: bool = False,
) -> Table:
    """Add date of birth from person (with optional impose day/month).

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id.
    cdm : Cdm
        CDM reference (person).
    date_of_birth_name : str, optional
        Output column name (default "date_of_birth").
    missing_day, missing_month : int, optional
        Default day/month when missing (default 1).
    impose_day, impose_month : bool, optional
        Force default day/month (default False).

    Returns
    -------
    ibis.expr.types.Table
        Table with date_of_birth column.
    """
    return add_demographics(
        table,
        cdm,
        age=False,
        sex=False,
        prior_observation=False,
        future_observation=False,
        date_of_birth=True,
        date_of_birth_name=date_of_birth_name,
        age_missing_day=missing_day,
        age_missing_month=missing_month,
        age_impose_day=impose_day,
        age_impose_month=impose_month,
    )


# --- In observation ---

def add_in_observation(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    window: tuple[int | float, int | float] = (0, 0),
    complete_interval: bool = False,
    name_style: str = "in_observation",
) -> Table:
    """Add column(s) indicating whether index_date is within observation and (optionally) within window.

    window (0, 0) means just "in observation"; otherwise name_style may use {window_name}.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference (observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    window : tuple, optional
        (low, high) days from index; (0, 0) means only "in observation".
    complete_interval : bool, optional
        If True, require full window inside observation (default False).
    name_style : str, optional
        Output column name (default "in_observation").

    Returns
    -------
    ibis.expr.types.Table
        Table with in_observation (0/1) column.
    """
    person_id_col = _person_id_column(table)
    if index_date not in table.columns:
        logger.error("index_date %r must be a column of the table for add_in_observation", index_date)
        raise ValueError(f"index_date {index_date!r} must be a column of the table.")
    logger.debug("Adding in_observation (index_date=%s, window=%s)", index_date, window)
    obs = cdm["observation_period"]
    op_sel = obs.select(
        obs.person_id.name(person_id_col),
        obs.observation_period_start_date.name("_op_start"),
        obs.observation_period_end_date.name("_op_end"),
    )
    distinct = table.select(person_id_col, index_date).distinct()
    join_obs = distinct.join(op_sel, person_id_col, how="inner")
    join_obs = join_obs.filter(
        (join_obs["_op_start"] <= join_obs[index_date])
        & (join_obs[index_date] <= join_obs["_op_end"])
    )
    # Days from index to op_start and to op_end
    join_obs = join_obs.mutate(
        _days_to_start=datediff(join_obs[index_date], join_obs["_op_start"], "day"),
        _days_to_end=datediff(join_obs[index_date], join_obs["_op_end"], "day"),
    )
    low, high = window
    if (low, high) == (0, 0):
        in_obs = ibis.literal(1)
    else:
        if complete_interval:
            in_obs = ibis.if_else(
                (join_obs["_days_to_start"] <= low) & (join_obs["_days_to_end"] >= high),
                ibis.literal(1),
                ibis.literal(0),
            )
        else:
            in_obs = ibis.if_else(
                (join_obs["_days_to_start"] <= high) & (join_obs["_days_to_end"] >= low),
                ibis.literal(1),
                ibis.literal(0),
            )
    join_obs = join_obs.mutate(**{name_style: in_obs}).drop(["_op_start", "_op_end", "_days_to_start", "_days_to_end"])
    out = table.left_join(join_obs, [person_id_col, index_date])
    out = out.mutate(**{name_style: ibis.coalesce(out[name_style], ibis.literal(0))})
    return out


# --- Death ---

def add_death_date(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: tuple[int | float, int | float] = (0, float("inf")),
    death_date_name: str = "date_of_death",
) -> Table:
    """Add date of death within window (only within same observation period as index_date).

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference (death, observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    censor_date : str or None, optional
        Optional censor date column name.
    window : tuple, optional
        (low, high) days from index_date (default (0, inf)).
    death_date_name : str, optional
        Output column name (default "date_of_death").

    Returns
    -------
    ibis.expr.types.Table
        Table with date_of_death column (null if no death in window).
    """
    return _add_death(table, cdm, value="date", index_date=index_date, censor_date=censor_date,
                      window=window, death_name=death_date_name)


def add_death_days(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: tuple[int | float, int | float] = (0, float("inf")),
    death_days_name: str = "days_to_death",
) -> Table:
    """Add days to death within window.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference (death, observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    censor_date : str or None, optional
        Optional censor date column name.
    window : tuple, optional
        (low, high) days from index_date (default (0, inf)).
    death_days_name : str, optional
        Output column name (default "days_to_death").

    Returns
    -------
    ibis.expr.types.Table
        Table with days_to_death column (null if no death in window).
    """
    return _add_death(table, cdm, value="days", index_date=index_date, censor_date=censor_date,
                      window=window, death_name=death_days_name)


def add_death_flag(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: tuple[int | float, int | float] = (0, float("inf")),
    death_flag_name: str = "death",
) -> Table:
    """Add flag for death within window (1/0).

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference (death, observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    censor_date : str or None, optional
        Optional censor date column name.
    window : tuple, optional
        (low, high) days from index_date (default (0, inf)).
    death_flag_name : str, optional
        Output column name (default "death").

    Returns
    -------
    ibis.expr.types.Table
        Table with death column (1 if death in window, 0 otherwise).
    """
    return _add_death(table, cdm, value="flag", index_date=index_date, censor_date=censor_date,
                      window=window, death_name=death_flag_name)


def _add_death(
    table: Any,
    cdm: Cdm,
    *,
    value: str,
    index_date: str,
    censor_date: str | None,
    window: tuple[int | float, int | float],
    death_name: str,
) -> Table:
    """Internal: add death date, days, or flag within window. value is "date", "days", or "flag"."""
    if "death" not in cdm._tables:
        logger.error("CDM must contain 'death' table for add_death_*.")
        raise ValueError("CDM must contain 'death' table.")
    logger.debug("Adding death %s (index_date=%s, window=%s)", value, index_date, window)
    person_id_col = _person_id_column(table)
    death_tbl = cdm["death"]
    death_date_col = "death_date"

    distinct = table.select(person_id_col, index_date).distinct()
    if censor_date and censor_date in table.columns:
        distinct = distinct.select(person_id_col, index_date, censor_date)

    # Restrict to same observation period
    obs = cdm["observation_period"]
    op_sel = obs.select(
        obs.person_id.name(person_id_col),
        obs.observation_period_start_date.name("_op_start"),
        obs.observation_period_end_date.name("_op_end"),
    )
    distinct = distinct.join(op_sel, person_id_col, how="inner")
    distinct = distinct.filter(
        (distinct["_op_start"] <= distinct[index_date])
        & (distinct[index_date] <= distinct["_op_end"])
    )

    death_sel = death_tbl.select(
        death_tbl.person_id.name(person_id_col),
        death_tbl[death_date_col],
    )
    merged = distinct.join(death_sel, person_id_col, how="left")
    merged = merged.filter(
        merged[death_date_col].notnull()
        & (merged[death_date_col] >= merged[index_date] + ibis.interval(days=int(window[0])))
    )
    if window[1] != float("inf"):
        merged = merged.filter(merged[death_date_col] <= merged[index_date] + ibis.interval(days=int(window[1])))

    if value == "flag":
        agg = merged.select(person_id_col, index_date).distinct().mutate(**{death_name: ibis.literal(1)})
        return table.left_join(agg, [person_id_col, index_date]).mutate(
            **{death_name: ibis.coalesce(ibis._[death_name], ibis.literal(0))}
        )
    if value == "date":
        first_death = merged.group_by([person_id_col, index_date]).aggregate(
            **{death_name: merged[death_date_col].min()}
        )
        return table.left_join(first_death, [person_id_col, index_date])
    # days
    merged = merged.mutate(**{death_name: datediff(merged[index_date], merged[death_date_col], "day")})
    first_days = merged.group_by([person_id_col, index_date]).aggregate(
        **{death_name: merged[death_name].min()}
    )
    return table.left_join(first_days, [person_id_col, index_date])


# --- Observation period id ---

def add_observation_period_id(
    table: Any,
    cdm: Cdm,
    *,
    index_date: str = "cohort_start_date",
    name_observation_period_id: str = "observation_period_id",
) -> Table:
    """Add the observation_period_id (ordinal within person) for the observation period containing index_date.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference (observation_period).
    index_date : str, optional
        Column name for index date (default "cohort_start_date").
    name_observation_period_id : str, optional
        Output column name (default "observation_period_id").

    Returns
    -------
    ibis.expr.types.Table
        Table with observation_period_id column.
    """
    person_id_col = _person_id_column(table)
    obs = cdm["observation_period"]
    op_sel = obs.select(
        obs.person_id.name(person_id_col),
        obs.observation_period_id.name(name_observation_period_id),
        obs.observation_period_start_date.name("_op_start"),
        obs.observation_period_end_date.name("_op_end"),
    )
    distinct = table.select(person_id_col, index_date).distinct()
    join_obs = distinct.join(op_sel, person_id_col, how="inner")
    join_obs = join_obs.filter(
        (join_obs["_op_start"] <= join_obs[index_date])
        & (join_obs[index_date] <= join_obs["_op_end"])
    )
    join_obs = join_obs.drop(["_op_start", "_op_end"])
    return table.left_join(
        join_obs.select(person_id_col, index_date, name_observation_period_id),
        [person_id_col, index_date],
    )


# --- Add cohort name, CDM name, concept name ---

def add_cohort_name(cohort: Any, cohort_set: Any) -> Table:
    """Left join cohort_set to add cohort_name by cohort_definition_id.

    Parameters
    ----------
    cohort : Any
        Ibis table with cohort_definition_id.
    cohort_set : Any
        Ibis table with cohort_definition_id, cohort_name.

    Returns
    -------
    ibis.expr.types.Table
        Cohort with cohort_name column added.
    """
    if "cohort_definition_id" not in cohort.columns:
        logger.error("Cohort table must have cohort_definition_id for add_cohort_name.")
        raise ValueError("Cohort table must have cohort_definition_id.")
    logger.debug("Adding cohort_name from cohort_set.")
    set_cols = [c for c in cohort_set.columns if c in ("cohort_definition_id", "cohort_name")]
    if "cohort_name" not in set_cols:
        set_cols.append("cohort_name")
    if "cohort_definition_id" not in set_cols:
        set_cols.insert(0, "cohort_definition_id")
    cohort_set_sel = cohort_set.select(set_cols)
    return cohort.left_join(cohort_set_sel, "cohort_definition_id")


def add_cdm_name(table: Any, cdm: Cdm) -> Table:
    """Add a column with the CDM name.

    Parameters
    ----------
    table : Any
        Ibis table.
    cdm : Cdm
        CDM reference.

    Returns
    -------
    ibis.expr.types.Table
        Table with cdm_name column (literal cdm.name).
    """
    return table.mutate(cdm_name=ibis.literal(cdm.name))


def add_concept_name(
    table: Any,
    cdm: Cdm,
    *,
    column: str | list[str] | None = None,
    name_style: str = "{column}_name",
) -> Table:
    """Add concept_name for concept_id column(s). If column is None, all columns ending with _concept_id are used.

    Parameters
    ----------
    table : Any
        Ibis table with one or more *_concept_id columns.
    cdm : Cdm
        CDM reference (concept table).
    column : str or list[str] or None, optional
        Concept ID column(s) to add names for; None = all *_concept_id.
    name_style : str, optional
        Format for name column (default "{column}_name", e.g. condition_name).

    Returns
    -------
    ibis.expr.types.Table
        Table with *_name column(s) added.
    """
    if column is None:
        column = [c for c in table.columns if c.endswith("_concept_id")]
    if isinstance(column, str):
        column = [column]
    concept = cdm["concept"]
    for col in column:
        if col not in table.columns:
            continue
        name_col = name_style.replace("{column}", col.replace("_concept_id", ""))
        concept_sel = concept.select(concept.concept_id.name(col), concept.concept_name.name(name_col))
        table = table.left_join(concept_sel, col)
    return table


# --- Filter cohort id / filter in observation ---

def filter_cohort_id(cohort: Any, cohort_id: int | list[int] | None) -> Any:
    """Filter cohort to rows with cohort_definition_id in cohort_id. If cohort_id is None, return cohort unchanged.

    Parameters
    ----------
    cohort : Any
        Ibis table or DataFrame with cohort_definition_id.
    cohort_id : int or list[int] or None
        cohort_definition_id(s) to keep; None returns cohort unchanged.

    Returns
    -------
    Any
        Filtered table or DataFrame.
    """
    if cohort_id is None:
        return cohort
    if isinstance(cohort_id, int):
        cohort_id = [cohort_id]
    if "cohort_definition_id" not in cohort.columns:
        logger.error("Table must have cohort_definition_id for filter_cohort_id.")
        raise ValueError("Table must have cohort_definition_id.")
    logger.debug("Filtering cohort to cohort_definition_id in %s", cohort_id)
    return cohort.filter(ibis._["cohort_definition_id"].isin(cohort_id))


def filter_in_observation(table: Any, cdm: Cdm, *, index_date: str) -> Table:
    """Keep only rows where index_date falls within an observation period.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date column.
    cdm : Cdm
        CDM reference (observation_period).
    index_date : str
        Column name for index date.

    Returns
    -------
    ibis.expr.types.Table
        Table with only rows in observation.
    """
    person_id_col = _person_id_column(table)
    obs = cdm["observation_period"]
    op_sel = obs.select(
        obs.person_id.name(person_id_col),
        obs.observation_period_start_date.name("_op_start"),
        obs.observation_period_end_date.name("_op_end"),
    )
    t = table.join(op_sel, person_id_col, how="inner")
    t = t.filter((t["_op_start"] <= t[index_date]) & (t[index_date] <= t["_op_end"]))
    return t.drop(["_op_start", "_op_end"])


# --- Add categories ---

def add_categories(
    table: Any,
    variable: str,
    categories: (
        dict[str, tuple[int | float, int | float] | list[tuple[int | float, int | float]]]
        | dict[str, dict[str, tuple[int | float, int | float]]]
    ),
    *,
    missing_category_value: str = "None",
    include_lower_bound: bool = True,
    include_upper_bound: bool = True,
) -> Table:
    """Categorise a numeric (or date) variable into named groups.

    categories: e.g. {"age_group": [(0, 39), (40, 79), (80, 150)]} (single label per column)
    or {"age_group": {"0-39": (0, 39), "40-79": (40, 79)}} (label per range).

    Parameters
    ----------
    table : Any
        Ibis table with variable column.
    variable : str
        Column name to categorise (numeric or date).
    categories : dict
        Maps output column name -> list of (low, high) or dict of label -> (low, high).
    missing_category_value : str, optional
        Value for out-of-range (default "None").
    include_lower_bound, include_upper_bound : bool, optional
        Whether bounds are inclusive (default True).

    Returns
    -------
    ibis.expr.types.Table
        Table with new category column(s).
    """
    if variable not in table.columns:
        logger.error("Variable %r must be a column of the table for add_categories", variable)
        raise ValueError(f"variable {variable!r} must be a column of the table.")

    logger.debug("Adding categories for variable=%s: %s", variable, list(categories.keys()))
    for cat_name, bounds in categories.items():
        if isinstance(bounds, dict):
            items = list(bounds.items())
        elif isinstance(bounds, (list, tuple)) and len(bounds) == 2 and isinstance(bounds[0], (int, float)):
            items = [(cat_name, bounds)]
        else:
            items = [(f"{b[0]}-{b[1]}", (b[0], b[1])) for b in bounds]
        cases = []
        for label, (low, high) in items:
            if high == float("inf") or (isinstance(high, (int, float)) and high >= 1e10):
                cond = table[variable] >= low if include_lower_bound else table[variable] > low
            elif low == float("-inf") or (isinstance(low, (int, float)) and low <= -1e10):
                cond = table[variable] <= high if include_upper_bound else table[variable] < high
            else:
                c1 = table[variable] >= low if include_lower_bound else table[variable] > low
                c2 = table[variable] <= high if include_upper_bound else table[variable] < high
                cond = c1 & c2
            cases.append((cond, label))
        default = ibis.literal(missing_category_value)
        branches = [(cond, ibis.literal(label)) for cond, label in cases]
        expr = ibis.cases(*branches, else_=default)
        table = table.mutate(**{cat_name: expr})
    return table


# ---------------------------------------------------------------------------
# Intersect helpers: window naming, nameStyle resolution, core engine
# ---------------------------------------------------------------------------

# Type alias for windows: a single tuple or list of tuples, or a dict of name→tuple
WindowSpec = (
    tuple[int | float, int | float]
    | list[tuple[int | float, int | float]]
    | dict[str, tuple[int | float, int | float]]
)
# Type alias for concept sets: dict mapping concept name → list of concept IDs
ConceptSet = dict[str, list[int]]


def _normalise_windows(
    window: WindowSpec,
) -> list[tuple[str, tuple[int | float, int | float]]]:
    """Normalise window spec into list of (window_name, (low, high))."""
    if isinstance(window, dict):
        return [(k, v) for k, v in window.items()]
    if isinstance(window, tuple) and len(window) == 2 and isinstance(window[0], (int, float)):
        return [(_window_name(window), window)]
    if isinstance(window, list):
        return [(_window_name(w), w) for w in window]
    raise ValueError(f"Invalid window spec: {window!r}")


def _window_name(window: tuple[int | float, int | float]) -> str:
    """Generate a window name from (low, high), e.g. (0, inf)→'0_to_inf', (-365, -1)→'m365_to_m1'."""
    def _fmt(v: int | float) -> str:
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "minf"
        iv = int(v)
        return f"m{abs(iv)}" if iv < 0 else str(iv)
    return f"{_fmt(window[0])}_to_{_fmt(window[1])}"


def _resolve_name_style(
    name_style: str,
    *,
    table_name: str = "",
    cohort_name: str = "",
    concept_name: str = "",
    window_name: str = "",
    id_name: str = "",
    field: str = "",
    value: str = "",
) -> str:
    """Resolve nameStyle placeholders like {table_name}, {cohort_name}, {window_name}, etc."""
    return (
        name_style
        .replace("{table_name}", table_name)
        .replace("{cohort_name}", cohort_name)
        .replace("{concept_name}", concept_name)
        .replace("{window_name}", window_name)
        .replace("{id_name}", id_name)
        .replace("{field}", field)
        .replace("{value}", value)
    )


def _filter_window(
    merged: Any,
    index_date: str,
    window: tuple[int | float, int | float],
    target_start: str = "_start",
    target_end: str | None = "_end",
) -> Any:
    """Filter merged table to records overlapping the time window relative to index_date.

    A record overlaps if its interval [target_start, target_end] intersects
    [index_date + low days, index_date + high days].
    """
    low, high = window
    # Lower bound: target_end >= index_date + low (or target_start if no end)
    end_col = target_end if (target_end and target_end in merged.columns) else target_start
    if low != float("-inf"):
        merged = merged.filter(merged[end_col] >= merged[index_date] + ibis.interval(days=int(low)))
    # Upper bound: target_start <= index_date + high
    if high != float("inf"):
        merged = merged.filter(merged[target_start] <= merged[index_date] + ibis.interval(days=int(high)))
    return merged


def _apply_censor_date(merged: Any, censor_date: str | None) -> Any:
    """If censor_date column exists, filter to records where _start <= censor_date."""
    if censor_date and censor_date in merged.columns:
        merged = merged.filter(merged["_start"] <= merged[censor_date])
    return merged


def _apply_in_observation(
    merged: Any,
    cdm: Cdm,
    person_id_col: str,
    index_date: str,
    in_observation: bool,
) -> Any:
    """If in_observation, filter to records within observation period."""
    if not in_observation:
        return merged
    obs = cdm["observation_period"]
    op_sel = obs.select(
        obs.person_id.name(person_id_col),
        obs.observation_period_start_date.name("_op_start"),
        obs.observation_period_end_date.name("_op_end"),
    )
    merged = merged.join(op_sel, person_id_col, how="inner")
    merged = merged.filter(
        (merged["_op_start"] <= merged[index_date])
        & (merged[index_date] <= merged["_op_end"])
    )
    keep = [c for c in merged.columns if c not in ("_op_start", "_op_end")]
    return merged.select(keep)


def _clean_join(table: Any, agg: Any, join_keys: list[str], new_col: str) -> Any:
    """Left join and clean up: keep original columns + new column, drop _right suffixes."""
    out = table.left_join(agg, join_keys)
    # Keep original table columns + the new column; drop _right join artifacts
    keep = list(table.columns) + [new_col]
    return out.select([c for c in out.columns if c in keep])


def _aggregate_intersect(
    table: Any,
    merged: Any,
    person_id_col: str,
    index_date: str,
    value: str,
    col_name: str,
    order: str = "first",
) -> Table:
    """Aggregate intersect results back onto the original table."""
    join_keys = [person_id_col, index_date]

    if value == "flag":
        has_any = merged.select(*join_keys).distinct().mutate(**{col_name: ibis.literal(1)})
        out = _clean_join(table, has_any, join_keys, col_name)
        return out.mutate(**{col_name: ibis.coalesce(out[col_name], ibis.literal(0))})
    if value == "count":
        cnt = merged.group_by(join_keys).aggregate(
            **{col_name: merged["_start"].count()}
        )
        out = _clean_join(table, cnt, join_keys, col_name)
        return out.mutate(**{col_name: ibis.coalesce(out[col_name], ibis.literal(0))})
    if value == "date":
        agg_fn = merged["_start"].min() if order == "first" else merged["_start"].max()
        agg = merged.group_by(join_keys).aggregate(**{col_name: agg_fn})
        return _clean_join(table, agg, join_keys, col_name)
    if value == "days":
        merged = merged.mutate(_days=datediff(merged[index_date], merged["_start"], "day"))
        agg_fn = merged["_days"].min() if order == "first" else merged["_days"].max()
        agg = merged.group_by(join_keys).aggregate(**{col_name: agg_fn})
        return _clean_join(table, agg, join_keys, col_name)
    # field value — _value column must exist
    order_col = ibis.asc(merged["_start"]) if order == "first" else ibis.desc(merged["_start"])
    ranked = merged.mutate(
        _rn=ibis.row_number().over(
            ibis.window(group_by=join_keys, order_by=order_col)
        )
    )
    first_row = ranked.filter(ranked["_rn"] == 0).select(*join_keys, ranked["_value"].name(col_name))
    return _clean_join(table, first_row, join_keys, col_name)


# ---------------------------------------------------------------------------
# Table intersect
# ---------------------------------------------------------------------------

def add_table_intersect_flag(
    table: Any,
    cdm: Cdm,
    table_name: str,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_start_date: str | None = None,
    target_end_date: str | None = None,
    in_observation: bool = True,
    name_style: str = "{table_name}_{window_name}",
) -> Table:
    """Add flag (1/0) for whether the person has a record in table_name within each window.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference.
    table_name : str
        OMOP table name (e.g. "condition_occurrence").
    index_date : str
        Column for index date (default "cohort_start_date").
    censor_date : str or None
        Optional censor date column name.
    window : WindowSpec
        Single tuple, list of tuples, or dict of name→tuple.
    target_start_date, target_end_date : str or None
        Override target date columns (default: from FIELD_TABLES_COLUMNS).
    in_observation : bool
        Restrict to records in observation period (default True).
    name_style : str
        Column name template with {table_name}, {window_name}.

    Returns
    -------
    ibis.expr.types.Table
    """
    return _add_table_intersect(
        table, cdm, table_name, value="flag", index_date=index_date,
        censor_date=censor_date, window=window,
        target_start_date=target_start_date, target_end_date=target_end_date,
        in_observation=in_observation, name_style=name_style,
    )


def add_table_intersect_count(
    table: Any,
    cdm: Cdm,
    table_name: str,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_start_date: str | None = None,
    target_end_date: str | None = None,
    in_observation: bool = True,
    name_style: str = "{table_name}_{window_name}",
) -> Table:
    """Add count of records in table_name within each window."""
    return _add_table_intersect(
        table, cdm, table_name, value="count", index_date=index_date,
        censor_date=censor_date, window=window,
        target_start_date=target_start_date, target_end_date=target_end_date,
        in_observation=in_observation, name_style=name_style,
    )


def add_table_intersect_date(
    table: Any,
    cdm: Cdm,
    table_name: str,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_date: str | None = None,
    in_observation: bool = True,
    order: str = "first",
    name_style: str = "{table_name}_{window_name}",
) -> Table:
    """Add date of first/last record in table_name within each window."""
    return _add_table_intersect(
        table, cdm, table_name, value="date", index_date=index_date,
        censor_date=censor_date, window=window,
        target_start_date=target_date, target_end_date=None,
        in_observation=in_observation, order=order, name_style=name_style,
    )


def add_table_intersect_days(
    table: Any,
    cdm: Cdm,
    table_name: str,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_date: str | None = None,
    in_observation: bool = True,
    order: str = "first",
    name_style: str = "{table_name}_{window_name}",
) -> Table:
    """Add days from index_date to first/last record in table_name within each window."""
    return _add_table_intersect(
        table, cdm, table_name, value="days", index_date=index_date,
        censor_date=censor_date, window=window,
        target_start_date=target_date, target_end_date=None,
        in_observation=in_observation, order=order, name_style=name_style,
    )


def add_table_intersect_field(
    table: Any,
    cdm: Cdm,
    table_name: str,
    field: str,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_date: str | None = None,
    in_observation: bool = True,
    order: str = "first",
    name_style: str = "{table_name}_{field}_{window_name}",
) -> Table:
    """Add a value column from the first/last record in table_name within each window."""
    return _add_table_intersect(
        table, cdm, table_name, value=field, index_date=index_date,
        censor_date=censor_date, window=window,
        target_start_date=target_date, target_end_date=None,
        in_observation=in_observation, order=order, name_style=name_style,
        is_field=True,
    )


def _add_table_intersect(
    table: Any,
    cdm: Cdm,
    table_name: str,
    *,
    value: str,
    index_date: str,
    censor_date: str | None,
    window: WindowSpec,
    target_start_date: str | None,
    target_end_date: str | None,
    in_observation: bool,
    order: str = "first",
    name_style: str,
    is_field: bool = False,
) -> Table:
    """Internal: add table intersect for one or more windows."""
    if table_name not in cdm._tables:
        raise ValueError(f"CDM must contain table {table_name!r}.")
    person_id_col = _person_id_column(table)
    target = cdm[table_name]
    t_start = target_start_date or start_date_column(table_name)
    t_end = target_end_date or end_date_column(table_name) or t_start
    target_person = "person_id" if "person_id" in target.columns else _person_id_column(target)

    select_cols = [
        target[target_person].name(person_id_col),
        target[t_start].name("_start"),
    ]
    if t_end != t_start:
        select_cols.append(target[t_end].name("_end"))
    if is_field and value in target.columns:
        select_cols.append(target[value].name("_value"))
    target_sel = target.select(select_cols)

    windows = _normalise_windows(window)
    for wname, wbounds in windows:
        col_name = _resolve_name_style(
            name_style, table_name=table_name, window_name=wname,
            field=value if is_field else "",
        )
        distinct = table.select(
            *([person_id_col, index_date] + ([censor_date] if censor_date and censor_date in table.columns else []))
        ).distinct()
        merged = distinct.join(target_sel, person_id_col, how="left")
        merged = merged.filter(merged["_start"].notnull())
        merged = _apply_censor_date(merged, censor_date)
        merged = _filter_window(
            merged, index_date, wbounds,
            target_start="_start",
            target_end="_end" if "_end" in merged.columns else None,
        )
        merged = _apply_in_observation(merged, cdm, person_id_col, index_date, in_observation)
        vtype = "field" if is_field else value
        table = _aggregate_intersect(table, merged, person_id_col, index_date, vtype, col_name, order)
    return table


# ---------------------------------------------------------------------------
# Cohort intersect
# ---------------------------------------------------------------------------

def add_cohort_intersect_flag(
    table: Any,
    cdm: Cdm,
    target_cohort_table: str,
    *,
    target_cohort_id: int | list[int] | None = None,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    target_start_date: str = "cohort_start_date",
    target_end_date: str = "cohort_end_date",
    window: WindowSpec = [(0, float("inf"))],
    name_style: str = "{cohort_name}_{window_name}",
) -> Table:
    """Add flag (1/0) for overlap with cohort(s) in target_cohort_table within each window.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference.
    target_cohort_table : str
        Name of the cohort table in the CDM.
    target_cohort_id : int, list[int], or None
        Cohort IDs to check; None = all.
    index_date : str
        Index date column (default "cohort_start_date").
    censor_date : str or None
        Optional censor date column.
    target_start_date, target_end_date : str
        Date columns in the target cohort table.
    window : WindowSpec
        Time window(s).
    name_style : str
        Column name template with {cohort_name}, {window_name}.
    """
    return _add_cohort_intersect(
        table, cdm, target_cohort_table, value="flag",
        target_cohort_id=target_cohort_id, index_date=index_date,
        censor_date=censor_date, target_start_date=target_start_date,
        target_end_date=target_end_date, window=window, name_style=name_style,
    )


def add_cohort_intersect_count(
    table: Any,
    cdm: Cdm,
    target_cohort_table: str,
    *,
    target_cohort_id: int | list[int] | None = None,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    target_start_date: str = "cohort_start_date",
    target_end_date: str = "cohort_end_date",
    window: WindowSpec = [(0, float("inf"))],
    name_style: str = "{cohort_name}_{window_name}",
) -> Table:
    """Add count of cohort entries within each window."""
    return _add_cohort_intersect(
        table, cdm, target_cohort_table, value="count",
        target_cohort_id=target_cohort_id, index_date=index_date,
        censor_date=censor_date, target_start_date=target_start_date,
        target_end_date=target_end_date, window=window, name_style=name_style,
    )


def add_cohort_intersect_date(
    table: Any,
    cdm: Cdm,
    target_cohort_table: str,
    *,
    target_cohort_id: int | list[int] | None = None,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    target_date: str = "cohort_start_date",
    order: str = "first",
    window: WindowSpec = [(0, float("inf"))],
    name_style: str = "{cohort_name}_{window_name}",
) -> Table:
    """Add first/last cohort date within each window."""
    return _add_cohort_intersect(
        table, cdm, target_cohort_table, value="date",
        target_cohort_id=target_cohort_id, index_date=index_date,
        censor_date=censor_date, target_start_date=target_date,
        target_end_date=target_date, window=window,
        order=order, name_style=name_style,
    )


def add_cohort_intersect_days(
    table: Any,
    cdm: Cdm,
    target_cohort_table: str,
    *,
    target_cohort_id: int | list[int] | None = None,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    target_date: str = "cohort_start_date",
    order: str = "first",
    window: WindowSpec = [(0, float("inf"))],
    name_style: str = "{cohort_name}_{window_name}",
) -> Table:
    """Add days to first/last cohort entry within each window."""
    return _add_cohort_intersect(
        table, cdm, target_cohort_table, value="days",
        target_cohort_id=target_cohort_id, index_date=index_date,
        censor_date=censor_date, target_start_date=target_date,
        target_end_date=target_date, window=window,
        order=order, name_style=name_style,
    )


def add_cohort_intersect_field(
    table: Any,
    cdm: Cdm,
    target_cohort_table: str,
    field: str,
    *,
    target_cohort_id: int | list[int] | None = None,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    target_date: str = "cohort_start_date",
    order: str = "first",
    window: WindowSpec = [(0, float("inf"))],
    name_style: str = "{cohort_name}_{field}_{window_name}",
) -> Table:
    """Add a field value from the first/last cohort record within each window."""
    return _add_cohort_intersect(
        table, cdm, target_cohort_table, value=field,
        target_cohort_id=target_cohort_id, index_date=index_date,
        censor_date=censor_date, target_start_date=target_date,
        target_end_date=target_date, window=window,
        order=order, name_style=name_style, is_field=True,
    )


def _get_cohort_names(cdm: Cdm, cohort_table: str) -> dict[int, str]:
    """Look up cohort names from the cohort_set table if available."""
    set_table_name = f"{cohort_table}_set"
    if set_table_name in cdm._tables:
        from cdmconnector.cdm import collect
        cs = collect(cdm[set_table_name].select("cohort_definition_id", "cohort_name"))
        return dict(zip(cs["cohort_definition_id"].tolist(), cs["cohort_name"].tolist()))
    return {}


def _add_cohort_intersect(
    table: Any,
    cdm: Cdm,
    cohort_table: str,
    *,
    value: str,
    target_cohort_id: int | list[int] | None,
    index_date: str,
    censor_date: str | None,
    target_start_date: str,
    target_end_date: str,
    window: WindowSpec,
    order: str = "first",
    name_style: str,
    is_field: bool = False,
) -> Table:
    """Internal: add cohort intersect for one or more cohort IDs and windows."""
    if cohort_table not in cdm._tables:
        raise ValueError(f"CDM must contain cohort table {cohort_table!r}.")
    person_id_col = _person_id_column(table)
    cohort_tbl = cdm[cohort_table]
    subj_col = "subject_id" if "subject_id" in cohort_tbl.columns else "person_id"

    # Resolve cohort IDs and names
    if target_cohort_id is not None:
        ids = [target_cohort_id] if isinstance(target_cohort_id, int) else target_cohort_id
    else:
        # Get all distinct cohort IDs
        if "cohort_definition_id" in cohort_tbl.columns:
            from cdmconnector.cdm import collect
            ids_df = collect(cohort_tbl.select("cohort_definition_id").distinct())
            ids = sorted(ids_df["cohort_definition_id"].tolist())
        else:
            ids = [1]

    cohort_names = _get_cohort_names(cdm, cohort_table)
    windows = _normalise_windows(window)

    for cid in ids:
        cname = cohort_names.get(cid, f"cohort_{cid}")
        # Normalise: lowercase, non-alphanumeric → underscore
        import re
        cname_norm = re.sub(r"[^a-z0-9]+", "_", cname.lower()).strip("_")

        base = cohort_tbl
        if "cohort_definition_id" in cohort_tbl.columns:
            base = cohort_tbl.filter(cohort_tbl.cohort_definition_id == cid)

        select_cols = [
            base[subj_col].name(person_id_col),
            base[target_start_date].name("_start"),
        ]
        if target_end_date != target_start_date and target_end_date in base.columns:
            select_cols.append(base[target_end_date].name("_end"))
        if is_field and value in base.columns:
            select_cols.append(base[value].name("_value"))
        cohort_sel = base.select(select_cols)

        for wname, wbounds in windows:
            col_name = _resolve_name_style(
                name_style, cohort_name=cname_norm, window_name=wname,
                field=value if is_field else "", id_name=cname_norm,
            )
            distinct = table.select(
                *([person_id_col, index_date] + ([censor_date] if censor_date and censor_date in table.columns else []))
            ).distinct()
            merged = distinct.join(cohort_sel, person_id_col, how="left")
            merged = merged.filter(merged["_start"].notnull())
            merged = _apply_censor_date(merged, censor_date)
            merged = _filter_window(
                merged, index_date, wbounds,
                target_start="_start",
                target_end="_end" if "_end" in merged.columns else None,
            )
            vtype = "field" if is_field else value
            table = _aggregate_intersect(table, merged, person_id_col, index_date, vtype, col_name, order)
    return table


# ---------------------------------------------------------------------------
# Concept intersect — uses concept_set dict with domain-based table lookup
# ---------------------------------------------------------------------------

def _build_concept_events_table(
    cdm: Cdm,
    concept_ids: list[int],
    person_id_col: str,
) -> Any:
    """Build a unified events table from all OMOP domain tables containing the given concept IDs.

    Returns an Ibis table with columns: person_id_col, _start, _end, _concept_id.
    """
    from cdmconnector.schemas import DOMAIN_TABLE_MAP, FIELD_TABLES_COLUMNS

    parts = []
    # Look up domains via the concept table
    if "concept" in cdm._tables:
        from cdmconnector.cdm import collect
        concept_tbl = cdm["concept"]
        concept_domain = collect(
            concept_tbl
            .filter(concept_tbl.concept_id.isin(concept_ids))
            .select("concept_id", "domain_id")
        )
        domains = set(concept_domain["domain_id"].str.lower().unique())
    else:
        # Fallback: check all domain tables
        domains = set(DOMAIN_TABLE_MAP.keys())

    for domain, tbl_name in DOMAIN_TABLE_MAP.items():
        if domain not in domains:
            continue
        if tbl_name not in cdm._tables:
            continue
        meta = FIELD_TABLES_COLUMNS.get(tbl_name)
        if not meta:
            continue
        concept_col = meta.get("standard_concept")
        start_col = meta["start_date"]
        end_col = meta.get("end_date") or start_col
        if not concept_col:
            continue

        tbl = cdm[tbl_name]
        if concept_col not in tbl.columns:
            continue

        filtered = tbl.filter(tbl[concept_col].isin(concept_ids))
        tbl_person = "person_id" if "person_id" in tbl.columns else _person_id_column(tbl)
        sel = filtered.select(
            filtered[tbl_person].name(person_id_col),
            filtered[start_col].name("_start"),
            filtered[end_col].name("_end") if end_col != start_col else filtered[start_col].name("_end"),
            filtered[concept_col].name("_concept_id"),
        )
        parts.append(sel)

    if not parts:
        raise ValueError(
            "No matching records found in any OMOP domain table for the given concept IDs."
        )
    result = parts[0]
    for p in parts[1:]:
        result = result.union(p)
    return result


def add_concept_intersect_flag(
    table: Any,
    cdm: Cdm,
    concept_set: ConceptSet,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_start_date: str = "event_start_date",
    target_end_date: str = "event_end_date",
    in_observation: bool = True,
    name_style: str = "{concept_name}_{window_name}",
) -> Table:
    """Add flag for presence of concepts within each window.

    Parameters
    ----------
    table : Any
        Ibis table with person_id or subject_id and index_date.
    cdm : Cdm
        CDM reference.
    concept_set : dict[str, list[int]]
        Named dict mapping concept names to lists of concept IDs.
        Example: {"acetaminophen": [1125315], "aspirin": [1112807]}
    index_date : str
        Index date column (default "cohort_start_date").
    censor_date : str or None
        Optional censor date column.
    window : WindowSpec
        Time window(s).
    target_start_date, target_end_date : str
        Standardised event date column names.
    in_observation : bool
        Restrict to records in observation period (default True).
    name_style : str
        Column name template with {concept_name}, {window_name}.
    """
    return _add_concept_intersect(
        table, cdm, concept_set, value="flag", index_date=index_date,
        censor_date=censor_date, window=window, in_observation=in_observation,
        name_style=name_style,
    )


def add_concept_intersect_count(
    table: Any,
    cdm: Cdm,
    concept_set: ConceptSet,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_start_date: str = "event_start_date",
    target_end_date: str = "event_end_date",
    in_observation: bool = True,
    name_style: str = "{concept_name}_{window_name}",
) -> Table:
    """Add count of concept occurrences within each window."""
    return _add_concept_intersect(
        table, cdm, concept_set, value="count", index_date=index_date,
        censor_date=censor_date, window=window, in_observation=in_observation,
        name_style=name_style,
    )


def add_concept_intersect_date(
    table: Any,
    cdm: Cdm,
    concept_set: ConceptSet,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_date: str = "event_start_date",
    order: str = "first",
    in_observation: bool = True,
    name_style: str = "{concept_name}_{window_name}",
) -> Table:
    """Add first/last concept date within each window."""
    return _add_concept_intersect(
        table, cdm, concept_set, value="date", index_date=index_date,
        censor_date=censor_date, window=window, in_observation=in_observation,
        order=order, name_style=name_style,
    )


def add_concept_intersect_days(
    table: Any,
    cdm: Cdm,
    concept_set: ConceptSet,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_date: str = "event_start_date",
    order: str = "first",
    in_observation: bool = True,
    name_style: str = "{concept_name}_{window_name}",
) -> Table:
    """Add days to first/last concept record within each window."""
    return _add_concept_intersect(
        table, cdm, concept_set, value="days", index_date=index_date,
        censor_date=censor_date, window=window, in_observation=in_observation,
        order=order, name_style=name_style,
    )


def add_concept_intersect_field(
    table: Any,
    cdm: Cdm,
    concept_set: ConceptSet,
    field: str,
    *,
    index_date: str = "cohort_start_date",
    censor_date: str | None = None,
    window: WindowSpec = [(0, float("inf"))],
    target_date: str = "event_start_date",
    order: str = "first",
    in_observation: bool = True,
    name_style: str = "{field}_{concept_name}_{window_name}",
) -> Table:
    """Add a field value from the first/last concept record within each window.

    For concept intersect field, the field must be a column available in the
    OMOP domain table(s) that contain the concept IDs.
    """
    return _add_concept_intersect(
        table, cdm, concept_set, value=field, index_date=index_date,
        censor_date=censor_date, window=window, in_observation=in_observation,
        order=order, name_style=name_style, is_field=True,
    )


def _add_concept_intersect(
    table: Any,
    cdm: Cdm,
    concept_set: ConceptSet,
    *,
    value: str,
    index_date: str,
    censor_date: str | None,
    window: WindowSpec,
    in_observation: bool,
    order: str = "first",
    name_style: str,
    is_field: bool = False,
) -> Table:
    """Internal: add concept intersect for each concept set entry and window."""
    import re

    person_id_col = _person_id_column(table)
    windows = _normalise_windows(window)

    # Collect all concept IDs across all sets
    all_ids: list[int] = []
    for ids in concept_set.values():
        all_ids.extend(ids)
    all_ids = list(set(all_ids))

    # Build unified events table from OMOP domain tables
    events = _build_concept_events_table(cdm, all_ids, person_id_col)

    for cs_name, cs_ids in concept_set.items():
        cname_norm = re.sub(r"[^a-z0-9]+", "_", cs_name.lower()).strip("_")
        # Filter events to this concept set
        cs_events = events.filter(events["_concept_id"].isin(cs_ids))

        for wname, wbounds in windows:
            col_name = _resolve_name_style(
                name_style, concept_name=cname_norm, window_name=wname,
                field=value if is_field else "", id_name=cname_norm,
            )
            distinct = table.select(
                *([person_id_col, index_date] + ([censor_date] if censor_date and censor_date in table.columns else []))
            ).distinct()
            merged = distinct.join(cs_events, person_id_col, how="left")
            merged = merged.filter(merged["_start"].notnull())
            merged = _apply_censor_date(merged, censor_date)
            merged = _filter_window(
                merged, index_date, wbounds,
                target_start="_start",
                target_end="_end" if "_end" in merged.columns else None,
            )
            merged = _apply_in_observation(merged, cdm, person_id_col, index_date, in_observation)
            vtype = "field" if is_field else value
            table = _aggregate_intersect(table, merged, person_id_col, index_date, vtype, col_name, order)
    return table


# --- Summarised result / variable types / available estimates ---

def summarise_result(
    table: Any,
    *,
    group: list | None = None,
    include_overall_group: bool = False,
    strata: list | None = None,
    include_overall_strata: bool = True,
    variables: list[str] | None = None,
    estimates: list[str] | None = None,
    counts: bool = True,
    weights: str | None = None,
    cdm_name: str = "unknown",
) -> Any:
    """Summarise variables into a summarised_result-like structure.

    Produces group_name, group_level, strata_name, strata_level, variable_name, variable_level,
    estimate_name, estimate_type, estimate_value. Collects table to memory and aggregates.

    Parameters
    ----------
    table : Any
        Ibis table or DataFrame with person_id or subject_id and variables to summarise.
    group : list or None, optional
        Grouping column names.
    include_overall_group : bool, optional
        If True, include overall group level (default False).
    strata : list or None, optional
        Stratification column names.
    include_overall_strata : bool, optional
        If True, include overall strata (default True).
    variables : list[str] or None, optional
        Variables to summarise; None = all.
    estimates : list[str] or None, optional
        Estimate names to include; None = default set.
    counts : bool, optional
        Include number_records, number_subjects (default True).
    weights : str or None, optional
        Optional weight column name.
    cdm_name : str, optional
        CDM name for result rows (default "unknown").

    Returns
    -------
    pandas.DataFrame
        Summarised result format (result_id, cdm_name, group_*, strata_*, variable_*, estimate_*).
    """
    import pandas as pd

    group = group or []
    strata = strata or []
    if not isinstance(group, list):
        group = [group]
    if not isinstance(strata, list):
        strata = [strata]

    logger.info("Summarising result: group=%s, strata=%s, counts=%s", group, strata, counts)
    from cdmconnector.cdm import collect

    if hasattr(table, "schema") or hasattr(table, "op"):
        df = collect(table)
    else:
        df = pd.DataFrame(table) if table is not None else pd.DataFrame()

    if df is None or len(df) == 0:
        logger.debug("summarise_result: empty table, returning empty result frame")
        return pd.DataFrame(columns=[
            "group_name", "group_level", "strata_name", "strata_level",
            "variable_name", "variable_level", "estimate_name", "estimate_type", "estimate_value",
        ])

    results = []
    strata_combos = [tuple()] if include_overall_strata else []
    strata_combos += [tuple(s) for s in strata]
    group_combos = [tuple()] if include_overall_group else []
    group_combos += [tuple(g) for g in group]

    for gk in group_combos:
        for sk in strata_combos:
            sub = df
            for col in list(gk) + list(sk):
                if col in sub.columns:
                    sub = sub  # filter by strata/group if needed
            if counts:
                n_rec = len(sub)
                n_sub = sub["person_id"].nunique() if "person_id" in sub.columns else (sub["subject_id"].nunique() if "subject_id" in sub.columns else n_rec)
                results.append({
                    "group_name": "overall", "group_level": "overall",
                    "strata_name": "overall", "strata_level": "overall",
                    "variable_name": "number_records", "variable_level": None,
                    "estimate_name": "count", "estimate_type": "integer", "estimate_value": str(n_rec),
                })
                results.append({
                    "group_name": "overall", "group_level": "overall",
                    "strata_name": "overall", "strata_level": "overall",
                    "variable_name": "number_subjects", "variable_level": None,
                    "estimate_name": "count", "estimate_type": "integer", "estimate_value": str(n_sub),
                })
    out = pd.DataFrame(results) if results else pd.DataFrame(columns=[
        "group_name", "group_level", "strata_name", "strata_level",
        "variable_name", "variable_level", "estimate_name", "estimate_type", "estimate_value",
    ])
    out["result_id"] = 1
    out["cdm_name"] = cdm_name
    out["additional_name"] = "overall"
    out["additional_level"] = "overall"
    return out


def variable_types(table: Any) -> Any:
    """Return a DataFrame with variable_name and variable_type (integer, numeric, date, categorical, logical).

    For Ibis tables, infers from schema without executing; for DataFrames uses dtypes.

    Parameters
    ----------
    table : Any
        Ibis table or pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        Columns: variable_name, variable_type.
    """
    import pandas as pd

    if hasattr(table, "schema") and not hasattr(table, "iloc"):
        # Ibis: infer from schema (no execution)
        logger.debug("variable_types: inferring from Ibis schema (no execution)")
        sch = table.schema()
        d = []
        for name in sch.names:
            dtype = sch[name]
            dtype_str = str(dtype).lower()
            if "int" in dtype_str:
                vt = "integer"
            elif "float" in dtype_str or "decimal" in dtype_str:
                vt = "numeric"
            elif "date" in dtype_str or "timestamp" in dtype_str:
                vt = "date"
            elif "bool" in dtype_str:
                vt = "logical"
            else:
                vt = "categorical"
            d.append({"variable_name": name, "variable_type": vt})
        return pd.DataFrame(d)

    from cdmconnector.cdm import collect

    if hasattr(table, "limit"):
        df = collect(table.limit(1))
    else:
        df = pd.DataFrame(table).head(1) if table is not None else pd.DataFrame()

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["variable_name", "variable_type"])

    d = []
    for col in df.columns:
        s = df[col].dtype
        if pd.api.types.is_integer_dtype(s):
            vt = "integer"
        elif pd.api.types.is_float_dtype(s) or pd.api.types.is_numeric_dtype(s):
            vt = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            vt = "date"
        elif pd.api.types.is_bool_dtype(s):
            vt = "logical"
        else:
            vt = "categorical"
        d.append({"variable_name": col, "variable_type": vt})
    return pd.DataFrame(d)


def available_estimates(
    variable_type: str | list[str] | None = None,
    full_quantiles: bool = False,
) -> Any:
    """Return DataFrame of estimate_name, estimate_description, estimate_type per variable_type.

    Parameters
    ----------
    variable_type : str or list[str] or None, optional
        Filter by variable_type(s): integer, numeric, date, categorical, logical; None = all.
    full_quantiles : bool, optional
        If True, include full quantile set (default False).

    Returns
    -------
    pandas.DataFrame
        Columns: estimate_name, estimate_description, variable_type.
    """
    import pandas as pd

    _formats = [
        ("mean", "mean of the variable of interest.", "numeric"),
        ("sd", "standard deviation of the variable of interest.", "numeric"),
        ("median", "median of the variable of interest.", "numeric"),
        ("min", "minimum of the variable of interest.", "numeric"),
        ("max", "maximum of the variable of interest.", "numeric"),
        ("sum", "sum of all the values for the variable of interest.", "numeric"),
        ("count_missing", "number of missing values.", "integer"),
        ("percentage_missing", "percentage of missing values", "percentage"),
        ("count", "number of times that each category is observed.", "integer"),
        ("percentage", "percentage of individuals with that category.", "percentage"),
        ("count_0", "count number of 1.", "integer"),
        ("percentage_0", "percentage of occurrences of 0 (NA are excluded).", "percentage"),
        ("count_person", "distinct counts of person_id.", "integer"),
        ("percentage_person", "percentage of distinct counts of person_id.", "percentage"),
        ("count_subject", "distinct counts of subject_id.", "integer"),
        ("percentage_subject", "percentage of distinct counts of subject_id.", "percentage"),
    ]
    df = pd.DataFrame(_formats, columns=["estimate_name", "estimate_description", "variable_type"])
    if variable_type is not None:
        vt = [variable_type] if isinstance(variable_type, str) else variable_type
        df = df[df["variable_type"].isin(vt)]
    return df


# --- Mock (optional: use eunomia or minimal mock) ---

def mock_patient_profiles(
    number_individuals: int = 100,
    seed: int | None = 1,
    *,
    source: str = "duckdb",
) -> Cdm:
    """Create a minimal mock CDM for testing PatientProfiles (person, observation_period, cohort-like table).

    For full OMOP mock use cdmconnector.eunomia (e.g. require_eunomia + cdm_from_con).

    Parameters
    ----------
    number_individuals : int, optional
        Number of persons (default 100).
    seed : int or None, optional
        Random seed (default 1).
    source : str, optional
        Backend ("duckdb"); default "duckdb".

    Returns
    -------
    Cdm
        CDM reference with person, observation_period, cohort1 tables.
    """
    import random
    from datetime import date, timedelta

    import pandas as pd

    logger.info("Creating mock patient profiles: number_individuals=%s, seed=%s, source=%s", number_individuals, seed, source)
    if seed is not None:
        random.seed(seed)
    n = number_individuals
    start = date(2010, 1, 1)
    person_ids = list(range(1, n + 1))
    gender = [8507 if random.random() < 0.5 else 8532 for _ in range(n)]
    yob = [random.randint(1950, 2000) for _ in range(n)]
    mob = [random.randint(1, 12) for _ in range(n)]
    dob = [random.randint(1, 28) for _ in range(n)]

    person_df = pd.DataFrame({
        "person_id": person_ids,
        "gender_concept_id": gender,
        "year_of_birth": yob,
        "month_of_birth": mob,
        "day_of_birth": dob,
    })
    obs_start = [start + timedelta(days=random.randint(0, 1000)) for _ in range(n)]
    obs_end = [s + timedelta(days=random.randint(365, 3000)) for s in obs_start]
    observation_period_df = pd.DataFrame({
        "observation_period_id": range(1, n + 1),
        "person_id": person_ids,
        "observation_period_start_date": obs_start,
        "observation_period_end_date": obs_end,
        "period_type_concept_id": [32817] * n,
    })
    cohort_start = [obs_start[i] + timedelta(days=random.randint(0, 500)) for i in range(n)]
    cohort_end = [min(c + timedelta(days=random.randint(0, 365)), obs_end[i]) for i, c in enumerate(cohort_start)]
    cohort_df = pd.DataFrame({
        "cohort_definition_id": [1] * n,
        "subject_id": person_ids,
        "cohort_start_date": cohort_start,
        "cohort_end_date": cohort_end,
    })

    con = ibis.duckdb.connect()
    con.create_table("person", person_df, overwrite=True)
    con.create_table("observation_period", observation_period_df, overwrite=True)
    con.create_table("cohort1", cohort_df, overwrite=True)
    tables = {
        "person": con.table("person"),
        "observation_period": con.table("observation_period"),
        "cohort1": con.table("cohort1"),
    }
    from cdmconnector.cdm import cdm_from_tables
    return cdm_from_tables(tables, "mock_patient_profiles")
