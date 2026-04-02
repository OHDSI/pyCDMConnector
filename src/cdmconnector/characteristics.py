# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Cohort characteristics: summarise_characteristics, summarise_cohort_count, table_characteristics.

Port of R CohortCharacteristics: summarise characteristics of cohorts in a cohort table
(counts, demographics: age, sex, prior/future observation, cohort dates) and format as tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# Default settings for summarised result
_RESULT_TYPE_CHARACTERISTICS = "summarise_characteristics"
_RESULT_TYPE_COHORT_COUNT = "summarise_cohort_count"
_PACKAGE_NAME = "cdmconnector"


def _cohort_to_df(cohort: Any) -> pd.DataFrame:
    """Materialize cohort to a pandas DataFrame (via collect only).

    Parameters
    ----------
    cohort : Any
        Ibis table, DataFrame, or array-like.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    from cdmconnector.cdm import collect

    if hasattr(cohort, "schema") or hasattr(cohort, "op"):
        out = collect(cohort)
    elif isinstance(cohort, pd.DataFrame):
        out = cohort.copy()
    else:
        out = pd.DataFrame(cohort)
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out)
    return out


def _table_to_df(tbl: Any) -> pd.DataFrame | None:
    """Materialize an Ibis table or DataFrame; return None if tbl is None (via collect only).

    Parameters
    ----------
    tbl : Any
        Ibis table, DataFrame, or None.

    Returns
    -------
    pandas.DataFrame or None
    """
    if tbl is None:
        return None
    import pandas as pd

    from cdmconnector.cdm import collect

    if hasattr(tbl, "schema") or hasattr(tbl, "op"):
        return collect(tbl)
    if isinstance(tbl, pd.DataFrame):
        return tbl.copy()
    return pd.DataFrame(tbl)


def _get_cdm_name(cdm: Any) -> str:
    """Get CDM name from Cdm object or return default.

    Parameters
    ----------
    cdm : Any
        Cdm object with .name or None.

    Returns
    -------
    str
        cdm.name or "unknown".
    """
    if cdm is None:
        return "unknown"
    name = getattr(cdm, "name", None)
    if name is not None:
        return str(name)
    return "unknown"


def _get_cohort_names(cohort: Any, cdm: Any, table_name: str | None) -> dict[int, str]:
    """Resolve cohort_definition_id -> cohort_name from cohort_set or cdm.

    Parameters
    ----------
    cohort : Any
        Cohort table or wrapper with optional cohort_set attribute.
    cdm : Any
        CDM reference (for {table_name}_set lookup).
    table_name : str or None
        Cohort table name (e.g. "cohort") for set name.

    Returns
    -------
    dict[int, str]
        cohort_definition_id -> cohort_name.
    """

    names: dict[int, str] = {}
    # From cohort wrapper (e.g. cohort_set attribute)
    cohort_set = getattr(cohort, "cohort_set", None)
    if cohort_set is not None:
        cs_df = _table_to_df(cohort_set)
        if cs_df is not None and "cohort_definition_id" in cs_df.columns and "cohort_name" in cs_df.columns:
            for _, row in cs_df.iterrows():
                names[int(row["cohort_definition_id"])] = str(row["cohort_name"])
            return names
    # From cdm: infer cohort table name and try cdm[name_set]
    if cdm is not None and table_name:
        set_name = f"{table_name}_set"
        if hasattr(cdm, "tables") and set_name in getattr(cdm, "tables", []):
            cs = cdm[set_name]
            cs_df = _table_to_df(cs)
            if cs_df is not None and "cohort_definition_id" in cs_df.columns and "cohort_name" in cs_df.columns:
                for _, row in cs_df.iterrows():
                    names[int(row["cohort_definition_id"])] = str(row["cohort_name"])
    if not names:
        # Fallback: use cohort_definition_id as name
        return {}
    return names


def _add_demographics_to_df(
    cohort_df: pd.DataFrame,
    person_df: pd.DataFrame,
    obs_period_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join cohort to person and observation_period; add age, sex, prior_obs_days, future_obs_days.

    Parameters
    ----------
    cohort_df : pandas.DataFrame
        Cohort with subject_id, cohort_start_date, cohort_end_date.
    person_df : pandas.DataFrame
        Person with person_id, year_of_birth, gender_concept_id.
    obs_period_df : pandas.DataFrame
        Observation period with person_id, start/end dates.

    Returns
    -------
    pandas.DataFrame
        cohort_df with age, sex, prior_observation, future_observation added.
    """
    import pandas as pd

    person_df = person_df.copy()
    obs_period_df = obs_period_df.copy()
    cohort_df = cohort_df.copy()

    # Ensure date columns
    for col in ("cohort_start_date", "cohort_end_date"):
        if col in cohort_df.columns and cohort_df[col].dtype == object:
            cohort_df[col] = pd.to_datetime(cohort_df[col], errors="coerce").dt.date
    if "birth_datetime" in person_df.columns:
        person_df["_birth_date"] = pd.to_datetime(person_df["birth_datetime"], errors="coerce").dt.date
    elif "year_of_birth" in person_df.columns:
        person_df["_birth_date"] = pd.to_datetime(
            person_df["year_of_birth"].astype(str) + "-07-01", errors="coerce"
        ).dt.date
    else:
        return cohort_df

    for col in ("observation_period_start_date", "observation_period_end_date"):
        if col in obs_period_df.columns and obs_period_df[col].dtype == object:
            obs_period_df[col] = pd.to_datetime(obs_period_df[col], errors="coerce").dt.date

    # Join person: subject_id = person_id
    cohort_df = cohort_df.merge(
        person_df[["person_id", "_birth_date"]].rename(columns={"person_id": "subject_id"}),
        on="subject_id",
        how="left",
    )
    # Gender: map concept_id to sex (simplified: 8532 F, 8507 M, else Unknown)
    if "gender_concept_id" in person_df.columns:
        gender = person_df[["person_id", "gender_concept_id"]].rename(columns={"person_id": "subject_id"})
        cohort_df = cohort_df.merge(gender, on="subject_id", how="left")
        cohort_df["sex"] = cohort_df["gender_concept_id"].map({8532: "Female", 8507: "Male"}).fillna("Unknown")
    else:
        cohort_df["sex"] = "Unknown"

    # Age at cohort_start (years)
    cohort_df["_start"] = pd.to_datetime(cohort_df["cohort_start_date"], errors="coerce")
    cohort_df["_birth"] = pd.to_datetime(cohort_df["_birth_date"], errors="coerce")
    cohort_df["age"] = ((cohort_df["_start"] - cohort_df["_birth"]).dt.days / 365.25).round(1)
    cohort_df.drop(columns=["_start", "_birth", "_birth_date"], inplace=True, errors="ignore")

    # Observation period: one row per person; keep period that contains cohort_start_date
    obs_period_df = obs_period_df.rename(columns={"person_id": "subject_id"})
    for col in ("observation_period_start_date", "observation_period_end_date"):
        if obs_period_df[col].dtype == object:
            obs_period_df[col] = pd.to_datetime(obs_period_df[col], errors="coerce")
    cohort_df = cohort_df.merge(
        obs_period_df[
            ["subject_id", "observation_period_start_date", "observation_period_end_date"]
        ],
        on="subject_id",
        how="left",
    )
    # Keep only the observation period that contains cohort_start_date (avoid duplicates)
    cohort_start = pd.to_datetime(cohort_df["cohort_start_date"], errors="coerce")
    mask = (
        (cohort_df["observation_period_start_date"] <= cohort_start)
        & (cohort_df["observation_period_end_date"] >= cohort_start)
    )
    cohort_df = cohort_df[mask | cohort_df["observation_period_start_date"].isna()].drop_duplicates(
        subset=["cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"],
        keep="first",
    )
    cohort_df["observation_period_start_date"] = pd.to_datetime(cohort_df["observation_period_start_date"], errors="coerce")
    cohort_df["observation_period_end_date"] = pd.to_datetime(cohort_df["observation_period_end_date"], errors="coerce")
    cohort_df["_start"] = pd.to_datetime(cohort_df["cohort_start_date"], errors="coerce")
    cohort_df["_end"] = pd.to_datetime(cohort_df["cohort_end_date"], errors="coerce")
    cohort_df["prior_observation"] = (cohort_df["_start"] - cohort_df["observation_period_start_date"]).dt.days
    cohort_df["future_observation"] = (cohort_df["observation_period_end_date"] - cohort_df["_end"]).dt.days
    cohort_df.drop(
        columns=["_start", "_end", "observation_period_start_date", "observation_period_end_date"],
        inplace=True,
        errors="ignore",
    )
    if "gender_concept_id" in cohort_df.columns:
        cohort_df.drop(columns=["gender_concept_id"], inplace=True, errors="ignore")
    return cohort_df


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    """Compute min, q25, median, q75, max, mean, sd for a numeric series.

    Parameters
    ----------
    series : pandas.Series
        Numeric column.

    Returns
    -------
    dict[str, float]
        Keys: min, q25, median, q75, max, mean, sd.
    """

    s = series.dropna()
    if len(s) == 0:
        return {"min": 0, "q25": 0, "median": 0, "q75": 0, "max": 0, "mean": 0, "sd": 0}
    q = s.quantile([0.25, 0.5, 0.75])
    return {
        "min": float(s.min()),
        "q25": float(q.iloc[0]),
        "median": float(q.iloc[1]),
        "q75": float(q.iloc[2]),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "sd": float(s.std()) if len(s) > 1 else 0.0,
    }


def _build_results_rows(
    cohort_df: pd.DataFrame,
    cohort_names: dict[int, str],
    cdm_name: str,
    counts: bool,
    demographics: bool,
    strata_cols: list[str],
    result_id: int = 1,
) -> pd.DataFrame:
    """Build summarised result rows from cohort DataFrame (with optional demographics columns).

    Parameters
    ----------
    cohort_df : pandas.DataFrame
        Cohort (and optionally demographics) data.
    cohort_names : dict[int, str]
        cohort_definition_id -> cohort_name.
    cdm_name : str
        CDM name for result rows.
    counts : bool
        Include number subjects/records rows.
    demographics : bool
        Include demographic summary rows.
    strata_cols : list[str]
        Stratification column names.
    result_id : int, optional
        result_id value for rows.

    Returns
    -------
    pandas.DataFrame
        Rows in summarised_result format.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    group_name = "cohort_name"
    strata_name = "overall" if not strata_cols else "&&&".join(strata_cols)
    cohort_ids = cohort_df["cohort_definition_id"].unique()

    for cid in cohort_ids:
        group_level = cohort_names.get(cid, str(cid))
        sub = cohort_df[cohort_df["cohort_definition_id"] == cid]

        if strata_cols:
            for stratum, sub2 in sub.groupby(strata_cols):
                stratum_level = stratum if isinstance(stratum, str) else "&&&".join(str(s) for s in stratum)
                _append_count_rows(rows, result_id, cdm_name, group_name, group_level, strata_name, stratum_level, sub2, counts)
                if demographics:
                    _append_demographic_rows(rows, result_id, cdm_name, group_name, group_level, strata_name, stratum_level, sub2)
        else:
            stratum_level = "overall"
            _append_count_rows(rows, result_id, cdm_name, group_name, group_level, strata_name, stratum_level, sub, counts)
            if demographics:
                _append_demographic_rows(rows, result_id, cdm_name, group_name, group_level, strata_name, stratum_level, sub)

    if not rows:
        return _empty_results_df()
    return pd.DataFrame(rows)


def _append_count_rows(
    rows: list[dict[str, Any]],
    result_id: int,
    cdm_name: str,
    group_name: str,
    group_level: str,
    strata_name: str,
    stratum_level: str,
    sub: pd.DataFrame,
    counts: bool,
) -> None:
    """Append number subjects and number records rows to rows list. Modifies rows in place."""
    if not counts:
        return
    n_records = len(sub)
    n_subjects = sub["subject_id"].nunique() if "subject_id" in sub.columns else n_records
    rows.append({
        "result_id": result_id,
        "cdm_name": cdm_name,
        "group_name": group_name,
        "group_level": group_level,
        "strata_name": strata_name,
        "strata_level": stratum_level,
        "variable_name": "Number subjects",
        "variable_level": "",
        "estimate_name": "count",
        "estimate_type": "integer",
        "estimate_value": str(n_subjects),
        "additional_name": "overall",
        "additional_level": "overall",
    })
    rows.append({
        "result_id": result_id,
        "cdm_name": cdm_name,
        "group_name": group_name,
        "group_level": group_level,
        "strata_name": strata_name,
        "strata_level": stratum_level,
        "variable_name": "Number records",
        "variable_level": "",
        "estimate_name": "count",
        "estimate_type": "integer",
        "estimate_value": str(n_records),
        "additional_name": "overall",
        "additional_level": "overall",
    })


def _append_demographic_rows(
    rows: list[dict[str, Any]],
    result_id: int,
    cdm_name: str,
    group_name: str,
    group_level: str,
    strata_name: str,
    stratum_level: str,
    sub: pd.DataFrame,
) -> None:
    """Append demographic summary rows (dates, age, sex, etc.) to rows list. Modifies rows in place."""
    import pandas as pd

    # Date variables: cohort_start_date, cohort_end_date
    for col, var_name in (("cohort_start_date", "Cohort start date"), ("cohort_end_date", "Cohort end date")):
        if col not in sub.columns:
            continue
        s = pd.to_datetime(sub[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        for est_name, val in (("min", s.min()), ("q25", s.quantile(0.25)), ("median", s.median()), ("q75", s.quantile(0.75)), ("max", s.max())):
            rows.append({
                "result_id": result_id,
                "cdm_name": cdm_name,
                "group_name": group_name,
                "group_level": group_level,
                "strata_name": strata_name,
                "strata_level": stratum_level,
                "variable_name": var_name,
                "variable_level": "",
                "estimate_name": est_name,
                "estimate_type": "date",
                "estimate_value": str(val.date()) if hasattr(val, "date") else str(val),
                "additional_name": "overall",
                "additional_level": "overall",
            })

    # Numeric: age, prior_observation, future_observation
    for col, var_name in (
        ("age", "Age"),
        ("prior_observation", "Prior observation"),
        ("future_observation", "Future observation"),
    ):
        if col not in sub.columns:
            continue
        s = sub[col].dropna()
        if len(s) == 0:
            continue
        summary = _numeric_summary(s)
        for est_name in ("min", "q25", "median", "q75", "max", "mean", "sd"):
            rows.append({
                "result_id": result_id,
                "cdm_name": cdm_name,
                "group_name": group_name,
                "group_level": group_level,
                "strata_name": strata_name,
                "strata_level": stratum_level,
                "variable_name": var_name,
                "variable_level": "",
                "estimate_name": est_name,
                "estimate_type": "numeric",
                "estimate_value": str(summary[est_name]),
                "additional_name": "overall",
                "additional_level": "overall",
            })

    # Categorical: sex
    if "sex" in sub.columns:
        sex_counts = sub["sex"].value_counts()
        n_total = len(sub)
        for level, count in sex_counts.items():
            pct = (100.0 * count / n_total) if n_total else 0
            rows.append({
                "result_id": result_id,
                "cdm_name": cdm_name,
                "group_name": group_name,
                "group_level": group_level,
                "strata_name": strata_name,
                "strata_level": stratum_level,
                "variable_name": "Sex",
                "variable_level": str(level),
                "estimate_name": "count",
                "estimate_type": "integer",
                "estimate_value": str(int(count)),
                "additional_name": "overall",
                "additional_level": "overall",
            })
            rows.append({
                "result_id": result_id,
                "cdm_name": cdm_name,
                "group_name": group_name,
                "group_level": group_level,
                "strata_name": strata_name,
                "strata_level": stratum_level,
                "variable_name": "Sex",
                "variable_level": str(level),
                "estimate_name": "percentage",
                "estimate_type": "percentage",
                "estimate_value": f"{pct:.2f}",
                "additional_name": "overall",
                "additional_level": "overall",
            })


def _empty_results_df() -> pd.DataFrame:
    """Return empty DataFrame with summarised result column names."""
    import pandas as pd

    return pd.DataFrame(columns=[
        "result_id", "cdm_name", "group_name", "group_level",
        "strata_name", "strata_level", "variable_name", "variable_level",
        "estimate_name", "estimate_type", "estimate_value",
        "additional_name", "additional_level",
    ])


def _empty_settings_df(result_type: str, table_name: str = "temp") -> pd.DataFrame:
    """Return one-row settings DataFrame (result_id, package_name, result_type, table_name)."""
    import pandas as pd

    try:
        from importlib.metadata import version as _pkg_version
        package_version = _pkg_version("cdmconnector")
    except Exception:
        package_version = "0.1.0"
    return pd.DataFrame([{
        "result_id": 1,
        "package_name": _PACKAGE_NAME,
        "package_version": package_version,
        "result_type": result_type,
        "table_name": table_name,
    }])


_SUMMARISED_RESULT_COLUMNS: tuple[str, ...] = (
    "result_id", "cdm_name", "group_name", "group_level",
    "strata_name", "strata_level", "variable_name", "variable_level",
    "estimate_name", "estimate_type", "estimate_value",
    "additional_name", "additional_level",
)

_SETTINGS_REQUIRED_COLUMNS: tuple[str, ...] = (
    "result_id", "result_type", "package_name", "package_version",
)

_ESTIMATE_TYPE_CHOICES: tuple[str, ...] = (
    "numeric", "integer", "date", "character", "proportion", "percentage", "logical",
)


@dataclass
class SummarisedResult:
    """Container for a summarised result (results table + settings table).

    Mirrors the omopgenerics ``summarised_result`` class. Provides methods for
    filtering, splitting, tidying, suppressing, and combining results.
    """

    results: pd.DataFrame
    settings_table: pd.DataFrame  # renamed to avoid clash with settings() method

    def __init__(self, results: Any = None, settings: Any = None) -> None:
        import pandas as pd

        if results is None:
            self.results = _empty_results_df()
        elif not isinstance(results, pd.DataFrame):
            self.results = pd.DataFrame(results)
        else:
            self.results = results

        if settings is None:
            self.settings_table = pd.DataFrame(columns=list(_SETTINGS_REQUIRED_COLUMNS))
        elif not isinstance(settings, pd.DataFrame):
            self.settings_table = pd.DataFrame(settings)
        else:
            self.settings_table = settings

    @property
    def settings(self) -> pd.DataFrame:
        """Return the settings table."""
        return self.settings_table

    def filter_settings(self, **kwargs: Any) -> SummarisedResult:
        """Filter results by settings column values.

        Parameters
        ----------
        **kwargs
            Column name / value pairs to filter the settings table by.
            Values can be a single value or a list of values.

        Returns
        -------
        SummarisedResult
            Filtered result.
        """
        s = self.settings_table.copy()
        for col, val in kwargs.items():
            if col not in s.columns:
                continue
            vals = [val] if not isinstance(val, (list, tuple, set)) else list(val)
            s = s[s[col].isin(vals)]
        result_ids = s["result_id"].tolist()
        filtered = self.results[self.results["result_id"].isin(result_ids)].copy()
        return SummarisedResult(results=filtered, settings=s)

    def filter_group(self, **kwargs: Any) -> SummarisedResult:
        """Filter results by group_name/group_level pairs.

        Parameters
        ----------
        **kwargs
            Pairs where key is the group_name value and value is the group_level
            value(s) to keep.

        Returns
        -------
        SummarisedResult
        """
        return self._filter_dimension("group", **kwargs)

    def filter_strata(self, **kwargs: Any) -> SummarisedResult:
        """Filter results by strata_name/strata_level pairs.

        Parameters
        ----------
        **kwargs
            Pairs where key is the strata_name value and value is the strata_level
            value(s) to keep.

        Returns
        -------
        SummarisedResult
        """
        return self._filter_dimension("strata", **kwargs)

    def filter_additional(self, **kwargs: Any) -> SummarisedResult:
        """Filter results by additional_name/additional_level pairs.

        Parameters
        ----------
        **kwargs
            Pairs where key is the additional_name value and value is the
            additional_level value(s) to keep.

        Returns
        -------
        SummarisedResult
        """
        return self._filter_dimension("additional", **kwargs)

    def _filter_dimension(self, dimension: str, **kwargs: Any) -> SummarisedResult:
        """Filter by a name/level dimension (group, strata, additional)."""
        name_col = f"{dimension}_name"
        level_col = f"{dimension}_level"
        mask = self.results[name_col].notna()  # start with all rows
        for name_val, level_val in kwargs.items():
            levels = [level_val] if not isinstance(level_val, (list, tuple, set)) else list(level_val)
            row_mask = (self.results[name_col] == name_val) & (
                self.results[level_col].isin(levels)
            )
            # For rows where group_name contains &&& (compound), check components
            compound_mask = self.results[name_col].str.contains("&&&", na=False)
            if compound_mask.any():
                # Handle compound name/level (e.g., "sex &&& age_group" / "Female &&& <40")
                for idx in self.results[compound_mask].index:
                    names = str(self.results.at[idx, name_col]).split(" &&& ")
                    lvls = str(self.results.at[idx, level_col]).split(" &&& ")
                    if name_val in names:
                        pos = names.index(name_val)
                        if pos < len(lvls) and lvls[pos] in levels:
                            row_mask.at[idx] = True
            mask = mask & row_mask
        filtered = self.results[mask].copy()
        return SummarisedResult(results=filtered, settings=self.settings_table.copy())

    def split_group(self) -> pd.DataFrame:
        """Split group_name/group_level into separate columns.

        Returns
        -------
        pandas.DataFrame
            Results with group_name/group_level replaced by individual columns.
        """
        return self._split_dimension("group")

    def split_strata(self) -> pd.DataFrame:
        """Split strata_name/strata_level into separate columns.

        Returns
        -------
        pandas.DataFrame
        """
        return self._split_dimension("strata")

    def split_additional(self) -> pd.DataFrame:
        """Split additional_name/additional_level into separate columns.

        Returns
        -------
        pandas.DataFrame
        """
        return self._split_dimension("additional")

    def split_all(self) -> pd.DataFrame:
        """Split all dimensions (group, strata, additional) into separate columns.

        Returns
        -------
        pandas.DataFrame
        """
        df = self._split_dimension("group")
        # Now split strata from the result
        df = _split_name_level_cols(df, "strata_name", "strata_level")
        df = _split_name_level_cols(df, "additional_name", "additional_level")
        return df

    def _split_dimension(self, dimension: str) -> pd.DataFrame:
        """Split a name/level dimension into individual columns."""
        name_col = f"{dimension}_name"
        level_col = f"{dimension}_level"
        return _split_name_level_cols(self.results.copy(), name_col, level_col)

    def tidy(self) -> pd.DataFrame:
        """Convert to tidy (wide) format.

        Splits all dimensions and pivots estimate_name into columns with
        values from estimate_value.

        Returns
        -------
        pandas.DataFrame
        """
        df = self.split_all()
        # Merge settings
        if not self.settings_table.empty and "result_id" in self.settings_table.columns:
            settings_cols = [
                c for c in self.settings_table.columns
                if c not in df.columns or c == "result_id"
            ]
            if len(settings_cols) > 1:
                df = df.merge(self.settings_table[settings_cols], on="result_id", how="left")
        # Pivot estimate_name → columns
        if "estimate_name" in df.columns and "estimate_value" in df.columns:
            idx_cols = [c for c in df.columns if c not in ("estimate_name", "estimate_type", "estimate_value")]
            if idx_cols:
                try:
                    # Fill NA values in index cols to avoid pivot_table dropping them
                    for col in idx_cols:
                        df[col] = df[col].fillna("")
                    pivoted = df.pivot_table(
                        index=idx_cols,
                        columns="estimate_name",
                        values="estimate_value",
                        aggfunc="first",
                    ).reset_index()
                    pivoted.columns.name = None
                    return pivoted
                except Exception:
                    pass
        return df

    def suppress(self, min_cell_count: int = 5) -> SummarisedResult:
        """Suppress results below the minimum cell count.

        Replaces estimate_value with NA for numeric estimates where the
        count is below min_cell_count, and obscures the exact count.

        Parameters
        ----------
        min_cell_count : int
            Minimum cell count threshold (default 5).

        Returns
        -------
        SummarisedResult
        """
        import numpy as np

        df = self.results.copy()
        settings = self.settings_table.copy()

        if min_cell_count > 0:
            # Find count rows and identify which are below threshold
            count_mask = df["estimate_name"].isin(["count", "number_subjects", "number_records"])
            numeric_mask = df["estimate_type"].isin(["integer", "numeric"])
            suppress_mask = count_mask & numeric_mask

            for idx in df[suppress_mask].index:
                try:
                    val = float(df.at[idx, "estimate_value"])
                    if 0 < val < min_cell_count:
                        df.at[idx, "estimate_value"] = np.nan
                except (ValueError, TypeError):
                    pass

            # Also suppress related percentage/proportion rows
            pct_mask = df["estimate_type"].isin(["proportion", "percentage"])
            for idx in df[pct_mask].index:
                # Check if the parent count was suppressed
                same_group = (
                    (df["result_id"] == df.at[idx, "result_id"])
                    & (df["group_name"] == df.at[idx, "group_name"])
                    & (df["group_level"] == df.at[idx, "group_level"])
                    & (df["strata_name"] == df.at[idx, "strata_name"])
                    & (df["strata_level"] == df.at[idx, "strata_level"])
                    & (df["variable_name"] == df.at[idx, "variable_name"])
                    & count_mask
                )
                parent = df[same_group]
                if not parent.empty and parent["estimate_value"].isna().any():
                    df.at[idx, "estimate_value"] = np.nan

        # Record suppression in settings
        if "min_cell_count" not in settings.columns:
            settings["min_cell_count"] = str(min_cell_count)
        else:
            settings["min_cell_count"] = str(min_cell_count)

        return SummarisedResult(results=df, settings=settings)

    def add_settings(self, **kwargs: str) -> SummarisedResult:
        """Add or update columns in the settings table.

        Parameters
        ----------
        **kwargs : str
            Column name / value pairs to add to settings.

        Returns
        -------
        SummarisedResult
        """
        settings = self.settings_table.copy()
        for col, val in kwargs.items():
            settings[col] = str(val)
        return SummarisedResult(results=self.results.copy(), settings=settings)

    def group_columns(self) -> list[str]:
        """Return unique group_name values (excluding 'overall').

        Returns
        -------
        list[str]
        """
        return _unique_dimension_names(self.results, "group_name")

    def strata_columns(self) -> list[str]:
        """Return unique strata_name values (excluding 'overall').

        Returns
        -------
        list[str]
        """
        return _unique_dimension_names(self.results, "strata_name")

    def additional_columns(self) -> list[str]:
        """Return unique additional_name values (excluding 'overall').

        Returns
        -------
        list[str]
        """
        return _unique_dimension_names(self.results, "additional_name")

    def settings_columns(self) -> list[str]:
        """Return non-standard settings column names.

        Returns
        -------
        list[str]
        """
        return [
            c for c in self.settings_table.columns
            if c not in _SETTINGS_REQUIRED_COLUMNS
        ]

    def unite_group(self) -> SummarisedResult:
        """Recombine individual group columns back into group_name/group_level.

        Reverses the effect of split_group. Finds columns that were produced
        by splitting and re-merges them into the ``group_name`` /
        ``group_level`` compound format.

        Returns
        -------
        SummarisedResult
        """
        df = self.results.copy()
        df = _unite_dimension_cols(df, "group")
        return SummarisedResult(results=df, settings=self.settings_table.copy())

    def unite_strata(self) -> SummarisedResult:
        """Recombine individual strata columns back into strata_name/strata_level.

        Returns
        -------
        SummarisedResult
        """
        df = self.results.copy()
        df = _unite_dimension_cols(df, "strata")
        return SummarisedResult(results=df, settings=self.settings_table.copy())

    def unite_additional(self) -> SummarisedResult:
        """Recombine individual additional columns back into additional_name/additional_level.

        Returns
        -------
        SummarisedResult
        """
        df = self.results.copy()
        df = _unite_dimension_cols(df, "additional")
        return SummarisedResult(results=df, settings=self.settings_table.copy())

    def combine_strata(self, *columns: str) -> SummarisedResult:
        """Combine multiple strata columns into a single compound strata.

        Parameters
        ----------
        *columns : str
            Column names to combine into strata.

        Returns
        -------
        SummarisedResult
        """
        if not columns:
            return SummarisedResult(results=self.results.copy(), settings=self.settings_table.copy())

        df = self.results.copy()
        existing_cols = [c for c in columns if c in df.columns]
        if not existing_cols:
            return SummarisedResult(results=self.results.copy(), settings=self.settings_table.copy())

        names = " &&& ".join(existing_cols)
        levels = []
        for _, row in df.iterrows():
            vals = [str(row.get(c, "overall")) for c in existing_cols]
            levels.append(" &&& ".join(vals))
        df["strata_name"] = names
        df["strata_level"] = levels
        return SummarisedResult(results=df, settings=self.settings_table.copy())

    def pivot_estimates(self) -> pd.DataFrame:
        """Pivot estimate_name into columns with estimate_value as values.

        Similar to tidy() but does not split dimensions or merge settings.

        Returns
        -------
        pandas.DataFrame
        """
        df = self.results.copy()
        if "estimate_name" not in df.columns or "estimate_value" not in df.columns:
            return df
        idx_cols = [
            c for c in df.columns
            if c not in ("estimate_name", "estimate_type", "estimate_value")
        ]
        if not idx_cols:
            return df
        for col in idx_cols:
            df[col] = df[col].fillna("")
        try:
            pivoted = df.pivot_table(
                index=idx_cols,
                columns="estimate_name",
                values="estimate_value",
                aggfunc="first",
            ).reset_index()
            pivoted.columns.name = None
            return pivoted
        except Exception:
            return df

    def tidy_columns(self) -> list[str]:
        """Return the column names that would appear after tidy().

        Returns
        -------
        list[str]
        """
        # Standard columns minus the ones that get pivoted/split
        base = ["result_id", "cdm_name", "variable_name", "variable_level"]
        base += self.group_columns()
        base += self.strata_columns()
        base += self.additional_columns()
        # Add settings columns
        base += [c for c in self.settings_table.columns if c != "result_id"]
        # Add estimate names
        if "estimate_name" in self.results.columns:
            base += sorted(self.results["estimate_name"].dropna().unique().tolist())
        return base

    def is_result_suppressed(self) -> bool:
        """Check if this result has been suppressed.

        Returns
        -------
        bool
            True if min_cell_count setting is > 0.
        """
        if "min_cell_count" in self.settings_table.columns:
            for val in self.settings_table["min_cell_count"]:
                try:
                    if int(val) > 0:
                        return True
                except (ValueError, TypeError):
                    pass
        return False

    def __repr__(self) -> str:
        n_rows = len(self.results)
        n_settings = len(self.settings_table)
        return f"SummarisedResult({n_rows} rows, {n_settings} settings)"

    def __len__(self) -> int:
        return len(self.results)


def bind_summarised_results(*results: SummarisedResult) -> SummarisedResult:
    """Combine multiple SummarisedResult objects.

    Re-assigns result_id values to avoid collisions.

    Parameters
    ----------
    *results : SummarisedResult
        Results to combine.

    Returns
    -------
    SummarisedResult
    """
    import pandas as pd

    if not results:
        return empty_summarised_result()

    all_results = []
    all_settings = []
    next_id = 1
    for sr in results:
        if not isinstance(sr, SummarisedResult):
            raise TypeError(f"Expected SummarisedResult, got {type(sr).__name__}")
        if sr.results.empty:
            continue
        # Remap result_ids
        id_map = {}
        for old_id in sr.results["result_id"].unique():
            id_map[old_id] = next_id
            next_id += 1
        r = sr.results.copy()
        r["result_id"] = r["result_id"].map(id_map)
        all_results.append(r)
        s = sr.settings_table.copy()
        s["result_id"] = s["result_id"].map(id_map)
        all_settings.append(s)

    if not all_results:
        return empty_summarised_result()

    return SummarisedResult(
        results=pd.concat(all_results, ignore_index=True),
        settings=pd.concat(all_settings, ignore_index=True),
    )


def empty_summarised_result(settings: Any = None) -> SummarisedResult:
    """Create an empty SummarisedResult.

    Parameters
    ----------
    settings : pandas.DataFrame or None
        Optional settings table.

    Returns
    -------
    SummarisedResult
    """
    return SummarisedResult(results=_empty_results_df(), settings=settings)


def result_columns() -> tuple[str, ...]:
    """Return the standard summarised_result column names.

    Returns
    -------
    tuple[str, ...]
    """
    return _SUMMARISED_RESULT_COLUMNS


def estimate_type_choices() -> tuple[str, ...]:
    """Return valid estimate_type values.

    Returns
    -------
    tuple[str, ...]
    """
    return _ESTIMATE_TYPE_CHOICES


def new_summarised_result(
    x: Any,
    settings: Any = None,
) -> SummarisedResult:
    """Construct a SummarisedResult from a DataFrame and optional settings.

    Parameters
    ----------
    x : pandas.DataFrame or dict
        Results data with standard summarised_result columns.
    settings : pandas.DataFrame, dict, or None
        Settings table.

    Returns
    -------
    SummarisedResult
    """
    return SummarisedResult(results=x, settings=settings)


def transform_to_summarised_result(
    x: Any,
    *,
    result_type: str = "custom",
    package_name: str = "cdmconnector",
) -> SummarisedResult:
    """Convert an arbitrary DataFrame to a SummarisedResult.

    Adds missing required columns with default values ("overall").

    Parameters
    ----------
    x : pandas.DataFrame
        Input data.
    result_type : str
        Result type for settings.
    package_name : str
        Package name for settings.

    Returns
    -------
    SummarisedResult
    """
    import pandas as pd

    df = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x.copy()

    # Add missing required columns
    defaults = {
        "result_id": 1,
        "cdm_name": "",
        "group_name": "overall",
        "group_level": "overall",
        "strata_name": "overall",
        "strata_level": "overall",
        "variable_name": "",
        "variable_level": None,
        "estimate_name": "",
        "estimate_type": "character",
        "estimate_value": "",
        "additional_name": "overall",
        "additional_level": "overall",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    try:
        from importlib.metadata import version as _pkg_version
        package_version = _pkg_version("cdmconnector")
    except Exception:
        package_version = "0.1.0"

    settings = pd.DataFrame([{
        "result_id": 1,
        "result_type": result_type,
        "package_name": package_name,
        "package_version": package_version,
    }])

    return SummarisedResult(results=df, settings=settings)


def result_package_version(result: SummarisedResult) -> dict[str, list[str]]:
    """Analyze package versions used in a SummarisedResult.

    Parameters
    ----------
    result : SummarisedResult

    Returns
    -------
    dict[str, list[str]]
        Mapping of package_name to list of versions found.
    """
    if not isinstance(result, SummarisedResult):
        raise TypeError(f"Expected SummarisedResult, got {type(result).__name__}")

    versions: dict[str, list[str]] = {}
    s = result.settings_table
    if "package_name" in s.columns and "package_version" in s.columns:
        for _, row in s.iterrows():
            pkg = str(row.get("package_name", ""))
            ver = str(row.get("package_version", ""))
            if pkg:
                versions.setdefault(pkg, [])
                if ver and ver not in versions[pkg]:
                    versions[pkg].append(ver)
    return versions


# ---------------------------------------------------------------------------
# Helpers for split / tidy / unite operations
# ---------------------------------------------------------------------------


def _split_name_level_cols(
    df: pd.DataFrame,
    name_col: str,
    level_col: str,
) -> pd.DataFrame:
    """Split compound name/level columns into individual columns."""
    if name_col not in df.columns or level_col not in df.columns:
        return df

    # Collect all unique individual names
    all_names: set[str] = set()
    for val in df[name_col].dropna().unique():
        for part in str(val).split(" &&& "):
            stripped = part.strip()
            if stripped and stripped != "overall":
                all_names.add(stripped)

    if not all_names:
        return df.drop(columns=[name_col, level_col], errors="ignore")

    # Create new columns
    for col_name in sorted(all_names):
        df[col_name] = None

    for idx in df.index:
        names_str = str(df.at[idx, name_col]) if df.at[idx, name_col] is not None else "overall"
        levels_str = str(df.at[idx, level_col]) if df.at[idx, level_col] is not None else "overall"
        names = [n.strip() for n in names_str.split(" &&& ")]
        levels = [lv.strip() for lv in levels_str.split(" &&& ")]
        for n, lv in zip(names, levels, strict=False):
            if n in all_names:
                df.at[idx, n] = lv

    return df.drop(columns=[name_col, level_col], errors="ignore")


def _unique_dimension_names(df: pd.DataFrame, col: str) -> list[str]:
    """Extract unique individual names from a compound name column."""
    if col not in df.columns:
        return []
    names: set[str] = set()
    for val in df[col].dropna().unique():
        for part in str(val).split(" &&& "):
            stripped = part.strip()
            if stripped and stripped != "overall":
                names.add(stripped)
    return sorted(names)


def _unite_dimension_cols(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Recombine individual columns back into name/level compound format.

    Looks for columns that don't belong to the standard result columns
    and aren't part of other dimensions, and merges them back into the
    ``{dimension}_name`` / ``{dimension}_level`` format.
    """
    name_col = f"{dimension}_name"
    level_col = f"{dimension}_level"

    # If the compound columns already exist and have non-trivial values, return as-is
    if name_col in df.columns and level_col in df.columns:
        non_overall = df[name_col].dropna().unique()
        if any(str(v) != "overall" for v in non_overall):
            return df

    # Find columns that might be dimension values (not standard result cols)
    standard_cols = set(_SUMMARISED_RESULT_COLUMNS)
    # Also exclude other dimension pairs
    for dim in ("group", "strata", "additional"):
        standard_cols.add(f"{dim}_name")
        standard_cols.add(f"{dim}_level")

    candidate_cols = [c for c in df.columns if c not in standard_cols]
    if not candidate_cols:
        if name_col not in df.columns:
            df[name_col] = "overall"
        if level_col not in df.columns:
            df[level_col] = "overall"
        return df

    # Build compound name/level from candidate columns
    for idx in df.index:
        parts_name = []
        parts_level = []
        for col in candidate_cols:
            val = df.at[idx, col]
            if val is not None and str(val) and str(val) != "None":
                parts_name.append(col)
                parts_level.append(str(val))
        if parts_name:
            df.at[idx, name_col] = " &&& ".join(parts_name)
            df.at[idx, level_col] = " &&& ".join(parts_level)
        else:
            df.at[idx, name_col] = "overall"
            df.at[idx, level_col] = "overall"

    # Drop the candidate columns
    df = df.drop(columns=candidate_cols, errors="ignore")
    return df


def summarise_characteristics(
    cohort: Any,
    cdm: Any = None,
    *,
    cohort_id: int | list[int] | None = None,
    strata: list[str] | None = None,
    counts: bool = True,
    demographics: bool = True,
    table_name: str | None = None,
) -> SummarisedResult:
    """
    Summarise characteristics of cohorts in a cohort table.

    Produces counts (number of subjects, number of records) and optionally
    demographics (age, sex, prior/future observation, cohort start/end dates)
    per cohort_definition_id, in summarised_result format.

    Parameters
    ----------
    cohort : Ibis table, DataFrame, or cohort_table-like with cohort_definition_id,
        subject_id, cohort_start_date, cohort_end_date. May have cohort_set
        attribute for cohort names.
    cdm : Optional Cdm. If provided and demographics is True, person and
        observation_period are used to add age, sex, prior_observation,
        future_observation. cdm.name is used for cdm_name in results.
    cohort_id : Optional cohort_definition_id(s) to include; if None, all.
    strata : Optional list of column names to stratify by (columns must exist
        on cohort or be added by demographics).
    counts : If True, include number of subjects and number of records.
    demographics : If True, include demographics (requires cdm with person
        and observation_period).
    table_name : Optional cohort table name (e.g. for settings and resolving
        cohort_set from cdm).

    Returns
    -------
    SummarisedResult
        .results : DataFrame with result_id, cdm_name, group_name, group_level,
            strata_name, strata_level, variable_name, variable_level,
            estimate_name, estimate_type, estimate_value, additional_name, additional_level.
        .settings : DataFrame with result_id, package_name, package_version,
            result_type, table_name.
    """
    import pandas as pd

    from cdmconnector.exceptions import CohortError

    tbl_name = table_name or getattr(cohort, "table_name", None) or "temp"
    strata_cols = list(strata) if strata else []

    cohort_df = _cohort_to_df(cohort)
    required = {"cohort_definition_id", "subject_id", "cohort_start_date", "cohort_end_date"}
    missing = required - set(cohort_df.columns)
    if missing:
        raise CohortError(f"Cohort table must have columns: {required}. Missing: {missing}.")

    if cohort_id is not None:
        ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
        cohort_df = cohort_df[cohort_df["cohort_definition_id"].isin(ids)]
    if cohort_df.empty:
        cohort_names = _get_cohort_names(cohort, cdm, tbl_name)
        if cohort_id is not None:
            ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
            for cid in ids:
                cohort_names.setdefault(cid, str(cid))
        results = _build_results_rows(
            pd.DataFrame(columns=cohort_df.columns),
            cohort_names,
            _get_cdm_name(cdm),
            counts,
            False,
            strata_cols,
        )
        settings_df = _empty_settings_df(_RESULT_TYPE_CHARACTERISTICS, tbl_name)
        return SummarisedResult(results=results, settings=settings_df)

    if demographics and cdm is not None:
        person_df = _table_to_df(getattr(cdm, "person", None))
        obs_df = _table_to_df(getattr(cdm, "observation_period", None))
        if person_df is not None and obs_df is not None:
            cohort_df = _add_demographics_to_df(cohort_df, person_df, obs_df)
        else:
            demographics = False

    cohort_names = _get_cohort_names(cohort, cdm, tbl_name)
    for cid in cohort_df["cohort_definition_id"].unique():
        cohort_names.setdefault(int(cid), str(cid))

    results = _build_results_rows(
        cohort_df,
        cohort_names,
        _get_cdm_name(cdm),
        counts,
        demographics,
        strata_cols,
    )
    settings_df = _empty_settings_df(_RESULT_TYPE_CHARACTERISTICS, tbl_name)
    return SummarisedResult(results=results, settings=settings_df)


def summarise_cohort_count(
    cohort: Any,
    cdm: Any = None,
    *,
    cohort_id: int | list[int] | None = None,
    strata: list[str] | None = None,
    table_name: str | None = None,
) -> SummarisedResult:
    """
    Summarise counts for cohorts in a cohort table.

    Same as summarise_characteristics with counts=True and demographics=False.
    Returns a summarised result with number of subjects and number of records
    per cohort (and per strata if strata is given).

    Parameters
    ----------
    cohort : Cohort table (Ibis table or DataFrame with cohort columns).
    cdm : Optional Cdm (for cdm_name and cohort_set).
    cohort_id : Optional cohort_definition_id(s) to include.
    strata : Optional list of column names to stratify by.
    table_name : Optional cohort table name.

    Returns
    -------
    SummarisedResult
        Results and settings in summarised_result format (result_type
        summarise_cohort_count).
    """
    res = summarise_characteristics(
        cohort,
        cdm,
        cohort_id=cohort_id,
        strata=strata,
        counts=True,
        demographics=False,
        table_name=table_name,
    )
    res.settings["result_type"] = _RESULT_TYPE_COHORT_COUNT
    return res


def table_characteristics(
    result: SummarisedResult | Any,
    *,
    header: list[str] | None = None,
    group_column: list[str] | None = None,
    hide: list[str] | None = None,
    estimate_format: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Format a summarise_characteristics (or summarise_cohort_count) result into a table.

    Pivots estimate_name/estimate_value so each estimate is a column, and
    optionally selects/orders columns for display.

    Parameters
    ----------
    result : SummarisedResult (with .results and .settings) or DataFrame of results.
    header : Columns to show first (e.g. ["cdm_name", "group_level"]). Default
        ["cdm_name", "group_level", "strata_level", "variable_name", "variable_level"].
    group_column : Extra grouping columns to keep.
    hide : Column names to drop from output.
    estimate_format : Optional mapping from estimate_name to display format
        (e.g. {"count": "N", "percentage": "%"}). Applied to column names after pivot.

    Returns
    -------
    pandas.DataFrame
        Formatted table with one row per (group, strata, variable) and columns
        for each estimate.
    """
    import pandas as pd

    if hasattr(result, "results"):
        df = result.results.copy()
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
    else:
        df = pd.DataFrame(result)

    if df.empty:
        return df

    # Drop density-style estimates for display
    if "estimate_name" in df.columns:
        df = df[~df["estimate_name"].str.contains(r"^density_[xy]", regex=True, na=False)]

    # Pivot: index = group/strata/variable, columns = estimate_name, values = estimate_value
    id_cols = [c for c in ["result_id", "cdm_name", "group_name", "group_level", "strata_name", "strata_level", "variable_name", "variable_level"] if c in df.columns]
    id_cols = [c for c in id_cols if df[c].nunique() > 0 or c in ("group_level", "variable_name", "variable_level")]
    if "estimate_name" not in df.columns or "estimate_value" not in df.columns:
        return df
    pivoted = df.pivot_table(
        index=id_cols,
        columns="estimate_name",
        values="estimate_value",
        aggfunc="first",
    ).reset_index()

    header = header or ["cdm_name", "group_level", "strata_level", "variable_name", "variable_level"]
    group_column = group_column or []
    hide = hide or []
    keep = [c for c in header + group_column if c in pivoted.columns and c not in hide]
    other = [c for c in pivoted.columns if c not in keep and c not in hide]
    reorder = [c for c in keep if c in pivoted.columns] + [c for c in other if c in pivoted.columns]
    pivoted = pivoted[[c for c in reorder if c in pivoted.columns]]

    if estimate_format:
        pivoted = pivoted.rename(columns=estimate_format)
    return pivoted


def settings_columns() -> tuple[str, ...]:
    """Return standard settings column names for summarised results.

    Returns
    -------
    tuple[str, ...]
        result_id, package_name, package_version, result_type, table_name.
    """
    return ("result_id", "package_name", "package_version", "result_type", "table_name")


def group_columns() -> tuple[str, ...]:
    """Return standard group column names for summarised results.

    Returns
    -------
    tuple[str, ...]
        group_name, group_level.
    """
    return ("group_name", "group_level")


def strata_columns() -> tuple[str, ...]:
    """Return standard strata column names for summarised results.

    Returns
    -------
    tuple[str, ...]
        strata_name, strata_level.
    """
    return ("strata_name", "strata_level")


def additional_columns() -> tuple[str, ...]:
    """Return standard additional column names for summarised results.

    Returns
    -------
    tuple[str, ...]
        additional_name, additional_level.
    """
    return ("additional_name", "additional_level")


# ---------------------------------------------------------------------------
# Result type constants for new summarise functions
# ---------------------------------------------------------------------------

_RESULT_TYPE_COHORT_ATTRITION = "summarise_cohort_attrition"
_RESULT_TYPE_COHORT_TIMING = "summarise_cohort_timing"
_RESULT_TYPE_COHORT_OVERLAP = "summarise_cohort_overlap"
_RESULT_TYPE_LARGE_SCALE = "summarise_large_scale_characteristics"


# ---------------------------------------------------------------------------
# summarise_cohort_attrition
# ---------------------------------------------------------------------------


def summarise_cohort_attrition(
    cohort: Any,
    cdm: Any = None,
    *,
    cohort_id: int | list[int] | None = None,
) -> SummarisedResult:
    """Summarise attrition for cohorts.

    Reads cohort_attrition metadata and formats it as a SummarisedResult with
    number_records, number_subjects, excluded_records, excluded_subjects per
    attrition step.

    Parameters
    ----------
    cohort : Cohort table with cohort_attrition attribute.
    cdm : Optional Cdm for cdm_name.
    cohort_id : Optional cohort_definition_id(s) to include.

    Returns
    -------
    SummarisedResult
    """
    import pandas as pd

    from cdmconnector.cohorts import attrition as get_attrition

    cdm_name = _get_cdm_name(cdm)
    cohort_names = _get_cohort_names(cohort, cdm, None)

    attr_df = get_attrition(cohort)
    if isinstance(attr_df, pd.DataFrame):
        attr_df = attr_df.copy()
    else:
        attr_df = pd.DataFrame(attr_df)

    if cohort_id is not None:
        ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
        attr_df = attr_df[attr_df["cohort_definition_id"].isin(ids)]

    if attr_df.empty:
        return SummarisedResult(
            results=_empty_results_df(),
            settings=_empty_settings_df(_RESULT_TYPE_COHORT_ATTRITION),
        )

    # Ensure cohort names for all IDs
    for cid in attr_df["cohort_definition_id"].unique():
        cohort_names.setdefault(int(cid), str(cid))

    rows: list[dict[str, Any]] = []
    result_id = 1

    for cid in sorted(attr_df["cohort_definition_id"].unique()):
        group_level = cohort_names.get(int(cid), str(cid))
        sub = attr_df[attr_df["cohort_definition_id"] == cid].sort_values("reason_id")

        for _, row in sub.iterrows():
            reason_id = int(row.get("reason_id", 0))
            reason = str(row.get("reason", ""))
            for var_name, est_col in (
                ("Number records", "number_records"),
                ("Number subjects", "number_subjects"),
                ("Excluded records", "excluded_records"),
                ("Excluded subjects", "excluded_subjects"),
            ):
                val = row.get(est_col, 0)
                rows.append({
                    "result_id": result_id,
                    "cdm_name": cdm_name,
                    "group_name": "cohort_name",
                    "group_level": group_level,
                    "strata_name": "overall",
                    "strata_level": "overall",
                    "variable_name": var_name,
                    "variable_level": reason,
                    "estimate_name": "count",
                    "estimate_type": "integer",
                    "estimate_value": str(int(val)) if pd.notna(val) else "0",
                    "additional_name": "reason_id",
                    "additional_level": str(reason_id),
                })

    results_df = pd.DataFrame(rows) if rows else _empty_results_df()
    settings_df = _empty_settings_df(_RESULT_TYPE_COHORT_ATTRITION)
    return SummarisedResult(results=results_df, settings=settings_df)


# ---------------------------------------------------------------------------
# summarise_cohort_timing
# ---------------------------------------------------------------------------


def summarise_cohort_timing(
    cohort: Any,
    cdm: Any = None,
    *,
    cohort_id: int | list[int] | None = None,
    strata: list[str] | None = None,
    restrict_to_first_entry: bool = True,
    estimates: tuple[str, ...] | list[str] = ("min", "q25", "median", "q75", "max"),
) -> SummarisedResult:
    """Summarise timing between cohort entries for individuals in multiple cohorts.

    Self-joins the cohort on subject_id to compute days between entry dates
    for each pair of cohorts.

    Parameters
    ----------
    cohort : Cohort table.
    cdm : Optional Cdm for cdm_name.
    cohort_id : Optional cohort_definition_id(s) to include.
    strata : Optional stratification columns.
    restrict_to_first_entry : If True, keep only earliest entry per subject/cohort.
    estimates : Summary statistics to compute.

    Returns
    -------
    SummarisedResult
    """
    import pandas as pd

    cdm_name = _get_cdm_name(cdm)
    cohort_names = _get_cohort_names(cohort, cdm, None)
    strata_cols = list(strata) if strata else []

    cohort_df = _cohort_to_df(cohort)
    required = {"cohort_definition_id", "subject_id", "cohort_start_date"}
    missing = required - set(cohort_df.columns)
    if missing:
        from cdmconnector.exceptions import CohortError
        raise CohortError(f"Cohort table missing columns: {missing}")

    if cohort_id is not None:
        ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
        cohort_df = cohort_df[cohort_df["cohort_definition_id"].isin(ids)]

    if cohort_df.empty or cohort_df["cohort_definition_id"].nunique() < 2:
        return SummarisedResult(
            results=_empty_results_df(),
            settings=_empty_settings_df(_RESULT_TYPE_COHORT_TIMING),
        )

    # Ensure cohort names
    for cid in cohort_df["cohort_definition_id"].unique():
        cohort_names.setdefault(int(cid), str(cid))

    # Add cohort_name column
    cohort_df = cohort_df.copy()
    cohort_df["cohort_name"] = cohort_df["cohort_definition_id"].map(
        lambda x: cohort_names.get(int(x), str(x))
    )
    cohort_df["cohort_start_date"] = pd.to_datetime(cohort_df["cohort_start_date"], errors="coerce")

    if restrict_to_first_entry:
        cohort_df = cohort_df.sort_values("cohort_start_date").drop_duplicates(
            subset=["cohort_definition_id", "subject_id"], keep="first"
        )

    # Self-join on subject_id
    ref = cohort_df[["subject_id", "cohort_name", "cohort_start_date"] + strata_cols].rename(
        columns={"cohort_name": "cohort_name_reference", "cohort_start_date": "start_reference"}
    )
    comp = cohort_df[["subject_id", "cohort_name", "cohort_start_date"]].rename(
        columns={"cohort_name": "cohort_name_comparator", "cohort_start_date": "start_comparator"}
    )
    joined = ref.merge(comp, on="subject_id", how="inner")

    # Remove self-comparisons
    joined = joined[joined["cohort_name_reference"] != joined["cohort_name_comparator"]]
    if joined.empty:
        return SummarisedResult(
            results=_empty_results_df(),
            settings=_empty_settings_df(_RESULT_TYPE_COHORT_TIMING),
        )

    joined["days_between"] = (joined["start_comparator"] - joined["start_reference"]).dt.days

    rows: list[dict[str, Any]] = []
    result_id = 1
    strata_name = "overall" if not strata_cols else "&&&".join(strata_cols)

    # Get unique cohort pairs
    pairs = joined[["cohort_name_reference", "cohort_name_comparator"]].drop_duplicates()

    for _, pair in pairs.iterrows():
        ref_name = pair["cohort_name_reference"]
        comp_name = pair["cohort_name_comparator"]
        group_name = "cohort_name_reference &&& cohort_name_comparator"
        group_level = f"{ref_name} &&& {comp_name}"

        pair_data = joined[
            (joined["cohort_name_reference"] == ref_name) &
            (joined["cohort_name_comparator"] == comp_name)
        ]

        def _add_timing_rows(
            sub: pd.DataFrame,
            s_name: str,
            s_level: str,
        ) -> None:
            days = sub["days_between"].dropna()
            if len(days) == 0:
                return
            summary = _numeric_summary(days)
            for est in estimates:
                if est == "density":
                    continue
                if est in summary:
                    rows.append({
                        "result_id": result_id,
                        "cdm_name": cdm_name,
                        "group_name": group_name,
                        "group_level": group_level,
                        "strata_name": s_name,
                        "strata_level": s_level,
                        "variable_name": "days_between_cohort_entries",
                        "variable_level": "",
                        "estimate_name": est,
                        "estimate_type": "numeric",
                        "estimate_value": str(summary[est]),
                        "additional_name": "overall",
                        "additional_level": "overall",
                    })

        if strata_cols:
            for stratum, sub in pair_data.groupby(strata_cols):
                s_level = stratum if isinstance(stratum, str) else "&&&".join(str(s) for s in stratum)
                _add_timing_rows(sub, strata_name, s_level)
            # Also overall
            _add_timing_rows(pair_data, "overall", "overall")
        else:
            _add_timing_rows(pair_data, "overall", "overall")

    results_df = pd.DataFrame(rows) if rows else _empty_results_df()
    settings_df = _empty_settings_df(_RESULT_TYPE_COHORT_TIMING)
    settings_df["restrict_to_first_entry"] = str(restrict_to_first_entry)
    return SummarisedResult(results=results_df, settings=settings_df)


# ---------------------------------------------------------------------------
# summarise_cohort_overlap
# ---------------------------------------------------------------------------


def summarise_cohort_overlap(
    cohort: Any,
    cdm: Any = None,
    *,
    cohort_id: int | list[int] | None = None,
    strata: list[str] | None = None,
) -> SummarisedResult:
    """Summarise overlap between cohorts.

    For each pair of cohorts, counts subjects that are only in the reference
    cohort, only in the comparator cohort, or in both.

    Parameters
    ----------
    cohort : Cohort table.
    cdm : Optional Cdm for cdm_name.
    cohort_id : Optional cohort_definition_id(s) to include.
    strata : Optional stratification columns.

    Returns
    -------
    SummarisedResult
    """
    import pandas as pd

    cdm_name = _get_cdm_name(cdm)
    cohort_names = _get_cohort_names(cohort, cdm, None)
    strata_cols = list(strata) if strata else []

    cohort_df = _cohort_to_df(cohort)
    required = {"cohort_definition_id", "subject_id"}
    missing = required - set(cohort_df.columns)
    if missing:
        from cdmconnector.exceptions import CohortError
        raise CohortError(f"Cohort table missing columns: {missing}")

    if cohort_id is not None:
        ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
        cohort_df = cohort_df[cohort_df["cohort_definition_id"].isin(ids)]

    if cohort_df.empty or cohort_df["cohort_definition_id"].nunique() < 2:
        return SummarisedResult(
            results=_empty_results_df(),
            settings=_empty_settings_df(_RESULT_TYPE_COHORT_OVERLAP),
        )

    for cid in cohort_df["cohort_definition_id"].unique():
        cohort_names.setdefault(int(cid), str(cid))

    cohort_df = cohort_df.copy()
    cohort_df["cohort_name"] = cohort_df["cohort_definition_id"].map(
        lambda x: cohort_names.get(int(x), str(x))
    )

    # Get distinct subjects per cohort
    distinct_subj = cohort_df[["cohort_name", "subject_id"] + strata_cols].drop_duplicates()

    rows: list[dict[str, Any]] = []
    result_id = 1
    strata_name = "overall" if not strata_cols else "&&&".join(strata_cols)
    all_cohort_names = sorted(distinct_subj["cohort_name"].unique())

    def _add_overlap_rows(
        ref_name: str,
        comp_name: str,
        ref_subj: set,
        comp_subj: set,
        s_name: str,
        s_level: str,
    ) -> None:
        both = ref_subj & comp_subj
        ref_only = ref_subj - comp_subj
        comp_only = comp_subj - ref_subj
        total = len(ref_subj | comp_subj)
        group_name = "cohort_name_reference &&& cohort_name_comparator"
        group_level = f"{ref_name} &&& {comp_name}"

        for var_name, count in (
            ("Only in reference cohort", len(ref_only)),
            ("In both cohorts", len(both)),
            ("Only in comparator cohort", len(comp_only)),
        ):
            pct = (100.0 * count / total) if total > 0 else 0.0
            rows.append({
                "result_id": result_id,
                "cdm_name": cdm_name,
                "group_name": group_name,
                "group_level": group_level,
                "strata_name": s_name,
                "strata_level": s_level,
                "variable_name": var_name,
                "variable_level": "subjects",
                "estimate_name": "count",
                "estimate_type": "integer",
                "estimate_value": str(count),
                "additional_name": "overall",
                "additional_level": "overall",
            })
            rows.append({
                "result_id": result_id,
                "cdm_name": cdm_name,
                "group_name": group_name,
                "group_level": group_level,
                "strata_name": s_name,
                "strata_level": s_level,
                "variable_name": var_name,
                "variable_level": "subjects",
                "estimate_name": "percentage",
                "estimate_type": "percentage",
                "estimate_value": f"{pct:.2f}",
                "additional_name": "overall",
                "additional_level": "overall",
            })

    for i, ref_name in enumerate(all_cohort_names):
        for comp_name in all_cohort_names[i + 1:]:
            if strata_cols:
                # Per stratum
                for stratum_vals, grp in distinct_subj.groupby(strata_cols):
                    s_level = stratum_vals if isinstance(stratum_vals, str) else "&&&".join(str(s) for s in stratum_vals)
                    ref_subj = set(grp[grp["cohort_name"] == ref_name]["subject_id"])
                    comp_subj = set(grp[grp["cohort_name"] == comp_name]["subject_id"])
                    _add_overlap_rows(ref_name, comp_name, ref_subj, comp_subj, strata_name, s_level)
                # Overall
                ref_subj = set(distinct_subj[distinct_subj["cohort_name"] == ref_name]["subject_id"])
                comp_subj = set(distinct_subj[distinct_subj["cohort_name"] == comp_name]["subject_id"])
                _add_overlap_rows(ref_name, comp_name, ref_subj, comp_subj, "overall", "overall")
            else:
                ref_subj = set(distinct_subj[distinct_subj["cohort_name"] == ref_name]["subject_id"])
                comp_subj = set(distinct_subj[distinct_subj["cohort_name"] == comp_name]["subject_id"])
                _add_overlap_rows(ref_name, comp_name, ref_subj, comp_subj, "overall", "overall")

    results_df = pd.DataFrame(rows) if rows else _empty_results_df()
    settings_df = _empty_settings_df(_RESULT_TYPE_COHORT_OVERLAP)
    return SummarisedResult(results=results_df, settings=settings_df)


# ---------------------------------------------------------------------------
# summarise_large_scale_characteristics
# ---------------------------------------------------------------------------

# Mapping from OMOP domain tables to their concept_id and date columns
_DOMAIN_TABLE_MAP: dict[str, dict[str, str]] = {
    "condition_occurrence": {
        "concept_id": "condition_concept_id",
        "date": "condition_start_date",
    },
    "drug_exposure": {
        "concept_id": "drug_concept_id",
        "date": "drug_exposure_start_date",
    },
    "procedure_occurrence": {
        "concept_id": "procedure_concept_id",
        "date": "procedure_date",
    },
    "measurement": {
        "concept_id": "measurement_concept_id",
        "date": "measurement_date",
    },
    "observation": {
        "concept_id": "observation_concept_id",
        "date": "observation_date",
    },
    "device_exposure": {
        "concept_id": "device_concept_id",
        "date": "device_exposure_start_date",
    },
    "visit_occurrence": {
        "concept_id": "visit_concept_id",
        "date": "visit_start_date",
    },
}


def _window_label(window: tuple[float, float] | list[float]) -> str:
    """Create a human-readable label for a time window."""
    lo, hi = window[0], window[1]
    lo_str = "-Inf" if lo == float("-inf") else str(int(lo))
    hi_str = "Inf" if hi == float("inf") else str(int(hi))
    return f"{lo_str} to {hi_str}"


def summarise_large_scale_characteristics(
    cohort: Any,
    cdm: Any,
    *,
    cohort_id: int | list[int] | None = None,
    strata: list[str] | None = None,
    window: list[tuple[float, float] | list[float]] | None = None,
    event_in_window: list[str] | None = None,
    episode_in_window: list[str] | None = None,
    index_date: str = "cohort_start_date",
    minimum_frequency: float = 0.005,
    excluded_codes: tuple[int, ...] | list[int] = (0,),
) -> SummarisedResult:
    """Summarise large-scale characteristics for cohorts.

    For each clinical domain table, counts concepts occurring in specified time
    windows relative to the index date. Reports counts and percentages for
    concepts above the minimum frequency threshold.

    Parameters
    ----------
    cohort : Cohort table.
    cdm : Cdm reference with clinical tables.
    cohort_id : Optional cohort_definition_id(s) to include.
    strata : Optional stratification columns.
    window : Time windows as (lower, upper) day offsets. Default spans from
        -Inf to Inf with standard breaks.
    event_in_window : Domain tables to analyze (e.g. ["condition_occurrence"]).
        If None, defaults to all available domain tables.
    episode_in_window : Not yet implemented.
    index_date : Column name for the reference date. Default "cohort_start_date".
    minimum_frequency : Minimum proportion to keep a concept. Default 0.005.
    excluded_codes : Concept IDs to exclude. Default (0,).

    Returns
    -------
    SummarisedResult
    """
    import pandas as pd

    if window is None:
        window = [
            (float("-inf"), -366),
            (-365, -31),
            (-30, -1),
            (0, 0),
            (1, 30),
            (31, 365),
            (366, float("inf")),
        ]

    cdm_name = _get_cdm_name(cdm)
    cohort_names = _get_cohort_names(cohort, cdm, None)
    strata_cols = list(strata) if strata else []

    cohort_df = _cohort_to_df(cohort)
    required = {"cohort_definition_id", "subject_id", index_date}
    missing = required - set(cohort_df.columns)
    if missing:
        from cdmconnector.exceptions import CohortError
        raise CohortError(f"Cohort table missing columns: {missing}")

    if cohort_id is not None:
        ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
        cohort_df = cohort_df[cohort_df["cohort_definition_id"].isin(ids)]

    if cohort_df.empty:
        return SummarisedResult(
            results=_empty_results_df(),
            settings=_empty_settings_df(_RESULT_TYPE_LARGE_SCALE),
        )

    for cid in cohort_df["cohort_definition_id"].unique():
        cohort_names.setdefault(int(cid), str(cid))

    cohort_df = cohort_df.copy()
    cohort_df["cohort_name"] = cohort_df["cohort_definition_id"].map(
        lambda x: cohort_names.get(int(x), str(x))
    )
    cohort_df[index_date] = pd.to_datetime(cohort_df[index_date], errors="coerce")

    # Determine which tables to analyze
    if event_in_window is not None:
        tables_to_check = [t for t in event_in_window if t in _DOMAIN_TABLE_MAP]
    else:
        tables_to_check = list(_DOMAIN_TABLE_MAP.keys())

    # Get concept table for name lookups
    concept_df = _table_to_df(getattr(cdm, "concept", None))
    concept_lookup: dict[int, str] = {}
    if concept_df is not None and "concept_id" in concept_df.columns and "concept_name" in concept_df.columns:
        for _, row in concept_df.iterrows():
            concept_lookup[int(row["concept_id"])] = str(row["concept_name"])

    rows: list[dict[str, Any]] = []
    result_id = 1

    for table_name in tables_to_check:
        domain_tbl = getattr(cdm, table_name, None)
        if domain_tbl is None:
            continue

        info = _DOMAIN_TABLE_MAP[table_name]
        domain_df = _table_to_df(domain_tbl)
        if domain_df is None or domain_df.empty:
            continue

        concept_col = info["concept_id"]
        date_col = info["date"]
        if concept_col not in domain_df.columns or date_col not in domain_df.columns:
            continue

        domain_df = domain_df[["person_id", concept_col, date_col]].copy()
        domain_df = domain_df.rename(columns={"person_id": "subject_id"})
        domain_df[date_col] = pd.to_datetime(domain_df[date_col], errors="coerce")

        # Join to cohort
        merged = cohort_df.merge(domain_df, on="subject_id", how="inner")
        if merged.empty:
            continue

        # Compute days offset
        merged["_days_offset"] = (merged[date_col] - merged[index_date]).dt.days

        for win in window:
            lo, hi = win[0], win[1]
            win_label = _window_label(win)

            # Filter to window
            if lo == float("-inf"):
                mask = merged["_days_offset"] <= hi
            elif hi == float("inf"):
                mask = merged["_days_offset"] >= lo
            else:
                mask = (merged["_days_offset"] >= lo) & (merged["_days_offset"] <= hi)

            win_data = merged[mask]
            if win_data.empty:
                continue

            # Exclude codes
            if excluded_codes:
                win_data = win_data[~win_data[concept_col].isin(excluded_codes)]

            if win_data.empty:
                continue

            # Count per cohort
            for c_name in win_data["cohort_name"].unique():
                cohort_sub = win_data[win_data["cohort_name"] == c_name]
                n_subjects = cohort_df[cohort_df["cohort_name"] == c_name]["subject_id"].nunique()

                concept_counts = (
                    cohort_sub.groupby(concept_col)["subject_id"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"subject_id": "count"})
                )

                for _, crow in concept_counts.iterrows():
                    cid_val = int(crow[concept_col])
                    count = int(crow["count"])
                    pct = (100.0 * count / n_subjects) if n_subjects > 0 else 0.0

                    if (count / n_subjects if n_subjects > 0 else 0) < minimum_frequency:
                        continue

                    c_name_lookup = concept_lookup.get(cid_val, str(cid_val))

                    for est_name, est_type, est_val in (
                        ("count", "integer", str(count)),
                        ("percentage", "percentage", f"{pct:.4f}"),
                    ):
                        rows.append({
                            "result_id": result_id,
                            "cdm_name": cdm_name,
                            "group_name": "cohort_name",
                            "group_level": c_name,
                            "strata_name": "overall",
                            "strata_level": "overall",
                            "variable_name": c_name_lookup,
                            "variable_level": win_label,
                            "estimate_name": est_name,
                            "estimate_type": est_type,
                            "estimate_value": est_val,
                            "additional_name": "concept_id",
                            "additional_level": str(cid_val),
                        })

        result_id += 1

    results_df = pd.DataFrame(rows) if rows else _empty_results_df()
    settings_df = _empty_settings_df(_RESULT_TYPE_LARGE_SCALE)
    if rows:
        # Update settings for each result_id
        import pandas as pd
        all_settings = []
        for rid in results_df["result_id"].unique():
            s = _empty_settings_df(_RESULT_TYPE_LARGE_SCALE)
            s["result_id"] = rid
            all_settings.append(s)
        settings_df = pd.concat(all_settings, ignore_index=True)

    return SummarisedResult(results=results_df, settings=settings_df)


# ---------------------------------------------------------------------------
# Table formatting functions
# ---------------------------------------------------------------------------


def table_cohort_count(
    result: SummarisedResult | Any,
    *,
    header: list[str] | None = None,
    group_column: list[str] | None = None,
    hide: list[str] | None = None,
) -> pd.DataFrame:
    """Format a summarise_cohort_count result into a table.

    Parameters
    ----------
    result : SummarisedResult or DataFrame.
    header : Columns for header. Default ["cohort_name"].
    group_column : Extra grouping columns.
    hide : Columns to hide.

    Returns
    -------
    pandas.DataFrame
    """
    header = header or ["cdm_name", "group_level"]
    return table_characteristics(
        result, header=header, group_column=group_column, hide=hide
    )


def table_cohort_attrition(
    result: SummarisedResult | Any,
    *,
    header: list[str] | None = None,
    group_column: list[str] | None = None,
    hide: list[str] | None = None,
) -> pd.DataFrame:
    """Format a summarise_cohort_attrition result into a table.

    Parameters
    ----------
    result : SummarisedResult or DataFrame.
    header : Columns for header.
    group_column : Extra grouping columns. Default ["cdm_name", "group_level"].
    hide : Columns to hide.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    if hasattr(result, "results"):
        df = result.results.copy()
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
    else:
        df = pd.DataFrame(result)

    if df.empty:
        return df

    header = header or ["group_level", "variable_level", "variable_name"]
    group_column = group_column or ["cdm_name"]
    hide = hide or ["result_id", "strata_name", "strata_level", "estimate_type",
                     "additional_name", "additional_level"]

    # Pivot estimate_name -> columns
    id_cols = [c for c in df.columns if c not in ("estimate_name", "estimate_type", "estimate_value")]
    pivoted = df.pivot_table(
        index=id_cols,
        columns="estimate_name",
        values="estimate_value",
        aggfunc="first",
    ).reset_index()
    pivoted.columns.name = None

    # Reorder
    keep = [c for c in header + group_column if c in pivoted.columns and c not in hide]
    other = [c for c in pivoted.columns if c not in keep and c not in hide]
    reorder = keep + other
    pivoted = pivoted[[c for c in reorder if c in pivoted.columns]]
    return pivoted


def table_cohort_timing(
    result: SummarisedResult | Any,
    *,
    time_scale: str = "days",
    unique_combinations: bool = True,
    header: list[str] | None = None,
    group_column: list[str] | None = None,
    hide: list[str] | None = None,
) -> pd.DataFrame:
    """Format a summarise_cohort_timing result into a table.

    Parameters
    ----------
    result : SummarisedResult or DataFrame.
    time_scale : "days" or "years".
    unique_combinations : If True, show only unique cohort pairs.
    header : Columns for header.
    group_column : Extra grouping columns.
    hide : Columns to hide.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    if hasattr(result, "results"):
        df = result.results.copy()
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
    else:
        df = pd.DataFrame(result)

    if df.empty:
        return df

    # Convert days to years if requested
    if time_scale == "years" and "estimate_value" in df.columns:
        numeric_mask = df["estimate_type"] == "numeric"
        for idx in df[numeric_mask].index:
            try:
                val = float(df.at[idx, "estimate_value"])
                df.at[idx, "estimate_value"] = f"{val / 365.25:.2f}"
            except (ValueError, TypeError):
                pass

    if unique_combinations:
        df = _get_unique_combinations(df)

    return table_characteristics(
        df, header=header, group_column=group_column, hide=hide
    )


def table_cohort_overlap(
    result: SummarisedResult | Any,
    *,
    unique_combinations: bool = True,
    header: list[str] | None = None,
    group_column: list[str] | None = None,
    hide: list[str] | None = None,
) -> pd.DataFrame:
    """Format a summarise_cohort_overlap result into a table.

    Parameters
    ----------
    result : SummarisedResult or DataFrame.
    unique_combinations : If True, show only unique cohort pairs.
    header : Columns for header.
    group_column : Extra grouping columns.
    hide : Columns to hide.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    if hasattr(result, "results"):
        df = result.results.copy()
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
    else:
        df = pd.DataFrame(result)

    if df.empty:
        return df

    if unique_combinations:
        df = _get_unique_combinations(df)

    return table_characteristics(
        df, header=header, group_column=group_column, hide=hide
    )


def table_large_scale_characteristics(
    result: SummarisedResult | Any,
    *,
    top_concepts: int | None = None,
    header: list[str] | None = None,
    group_column: list[str] | None = None,
    hide: list[str] | None = None,
) -> pd.DataFrame:
    """Format a summarise_large_scale_characteristics result into a table.

    Parameters
    ----------
    result : SummarisedResult or DataFrame.
    top_concepts : If set, limit to top N concepts by frequency.
    header : Columns for header.
    group_column : Extra grouping columns.
    hide : Columns to hide.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    if hasattr(result, "results"):
        df = result.results.copy()
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
    else:
        df = pd.DataFrame(result)

    if df.empty:
        return df

    if top_concepts is not None and top_concepts > 0:
        # Keep top N concepts by maximum percentage
        pct_df = df[df["estimate_name"] == "percentage"].copy()
        if not pct_df.empty:
            pct_df["_pct"] = pd.to_numeric(pct_df["estimate_value"], errors="coerce")
            top = pct_df.nlargest(top_concepts, "_pct")["variable_name"].unique()
            df = df[df["variable_name"].isin(top)]

    header = header or ["cdm_name", "group_level", "variable_name", "variable_level"]
    return table_characteristics(
        df, header=header, group_column=group_column, hide=hide
    )


def _get_unique_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to unique cohort pair combinations (A-B but not B-A)."""
    if "group_level" not in df.columns:
        return df
    seen: set[tuple[str, str]] = set()
    keep_mask = []
    for _, row in df.iterrows():
        gl = str(row["group_level"])
        parts = gl.split(" &&& ")
        if len(parts) == 2:
            key = tuple(sorted(parts))
            if key not in seen:
                seen.add(key)
                keep_mask.append(True)
            else:
                # Keep if this is the canonical order
                keep_mask.append(key == tuple(parts))
        else:
            keep_mask.append(True)
    return df[keep_mask].copy()
