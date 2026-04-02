# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""
visOmopResults port: format and visualise OMOP summarised results.

Formats DataFrames with summarised_result-style columns (estimate_name,
estimate_type, estimate_value, group/strata/variable columns) into
publication-ready tables.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import altair as alt
import pandas as pd

try:
    from great_tables import GT
except ImportError:
    GT = None  # type: ignore[misc, assignment]

# Default table types (dataframe, html, gt via great_tables)
TABLE_TYPES = ("dataframe", "html", "gt")
DEFAULT_TABLE_TYPE = "gt"
TABLE_STYLES = ("default", "darwin")
DEFAULT_STYLE = "default"

# Standard summarised_result columns
RESULT_COLUMNS = (
    "result_id",
    "cdm_name",
    "group_name",
    "group_level",
    "strata_name",
    "strata_level",
    "variable_name",
    "variable_level",
    "estimate_name",
    "estimate_type",
    "estimate_value",
    "additional_name",
    "additional_level",
)
ESTIMATE_TYPE_CHOICES = ("integer", "numeric", "percentage", "proportion", "character", "logical", "date")
NA_STR = "\u2013"  # en-dash, same as R


# ---------------------------------------------------------------------------
# Options and metadata
# ---------------------------------------------------------------------------


def default_table_options(user_options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Default table formatting options (mirrors visOmopResults defaultTableOptions).

    Parameters
    ----------
    user_options : dict or None, optional
        Overrides for default options (decimals, decimal_mark, title, etc.).

    Returns
    -------
    dict[str, Any]
        Options dict (decimals, decimal_mark, big_mark, na, title, etc.).
    """
    opts = {
        "decimals": {"integer": 0, "numeric": 2, "percentage": 1, "proportion": 3},
        "decimal_mark": ".",
        "big_mark": ",",
        "keep_not_formatted": True,
        "use_format_order": True,
        "delim": "\n",
        "include_header_name": True,
        "include_header_key": True,
        "na": NA_STR,
        "title": None,
        "subtitle": None,
        "caption": None,
        "group_as_column": False,
        "group_order": None,
        "merge": "all_columns",
    }
    if user_options:
        for k, v in user_options.items():
            if k in opts:
                opts[k] = v
    return opts


def table_options() -> dict[str, Any]:
    """Return default table options for vis_omop_table / vis_table.

    Returns
    -------
    dict[str, Any]
        Same as default_table_options(None).
    """
    return default_table_options(None)


def table_style() -> tuple[str, ...]:
    """Pre-defined table style names.

    Returns
    -------
    tuple[str, ...]
        ("default", "darwin").
    """
    return TABLE_STYLES


def table_type() -> tuple[str, ...]:
    """Supported table output types.

    Returns
    -------
    tuple[str, ...]
        ("dataframe", "html", "gt").
    """
    return TABLE_TYPES


def _group_columns(df: pd.DataFrame) -> list[str]:
    """Columns that form group (name/level pairs).

    Parameters
    ----------
    df : pandas.DataFrame
        Must have group_name, group_level.

    Returns
    -------
    list[str]
        ["group_name", "group_level"] or [].
    """
    out = []
    if "group_name" in df.columns and "group_level" in df.columns:
        out = ["group_name", "group_level"]
    return out


def _strata_columns(df: pd.DataFrame) -> list[str]:
    """Columns that form strata.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have strata_name, strata_level.

    Returns
    -------
    list[str]
        ["strata_name", "strata_level"] or [].
    """
    out = []
    if "strata_name" in df.columns and "strata_level" in df.columns:
        out = ["strata_name", "strata_level"]
    return out


def _additional_columns(df: pd.DataFrame) -> list[str]:
    """Additional name/level columns beyond group/strata.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have additional_name, additional_level.

    Returns
    -------
    list[str]
        ["additional_name", "additional_level"] or [].
    """
    out = []
    if "additional_name" in df.columns and "additional_level" in df.columns:
        out = ["additional_name", "additional_level"]
    return out


def table_columns(result: pd.DataFrame) -> list[str]:
    """Column names that can be used in table header/group for a summarised result.

    Parameters
    ----------
    result : pandas.DataFrame
        Summarised result with standard columns.

    Returns
    -------
    list[str]
        Ordered list of columns present (cdm_name, group_*, strata_*, variable_*, estimate_name, additional_*).
    """
    cols = (
        ["cdm_name"]
        + _group_columns(result)
        + _strata_columns(result)
        + ["variable_name", "variable_level", "estimate_name"]
        + _additional_columns(result)
    )
    return [c for c in cols if c in result.columns]


# ---------------------------------------------------------------------------
# Format estimate value
# ---------------------------------------------------------------------------


def _validate_decimals(
    result: pd.DataFrame,
    decimals: dict[str, int] | int | None,
) -> dict[str, int]:
    """Normalize decimals to a dict keyed by estimate_type (or int for all).

    Parameters
    ----------
    result : pandas.DataFrame
        Unused; kept for API compatibility.
    decimals : dict, int, or None
        Map estimate_type -> decimal places, or single int for all.

    Returns
    -------
    dict[str, int]
        estimate_type -> decimal places.
    """
    if decimals is None:
        return {}
    if isinstance(decimals, int):
        return {k: decimals for k in ESTIMATE_TYPE_CHOICES if k not in ("logical", "date")}
    return dict(decimals)


def format_estimate_value(
    result: pd.DataFrame,
    decimals: dict[str, int] | int | None = None,
    decimal_mark: str = ".",
    big_mark: str = ",",
) -> pd.DataFrame:
    """
    Format the estimate_value column by decimal places and number formatting.

    Parameters
    ----------
    result : DataFrame
        Must contain estimate_name, estimate_type, estimate_value.
    decimals : dict or int, optional
        Map estimate_type or estimate_name -> number of decimals.
        If int, same decimals for all. Keys can be integer, numeric, percentage, proportion.
    decimal_mark : str
        Decimal separator.
    big_mark : str
        Thousands separator (e.g. ",").

    Returns
    -------
    DataFrame
        Copy of result with estimate_value formatted as strings.
    """
    needed = ["estimate_name", "estimate_type", "estimate_value"]
    for c in needed:
        if c not in result.columns:
            raise ValueError(f"result must contain columns: {needed}")
    dec = _validate_decimals(result, decimals)
    if not dec:
        dec = {"integer": 0, "numeric": 2, "percentage": 1, "proportion": 3}
    out = result.copy()
    vals = out["estimate_value"].astype(str)
    mask_suppressed = vals.eq("-") | vals.str.contains("<", na=False) | vals.isna()
    # By estimate_type first, then by estimate_name
    for name, n in dec.items():
        if name in out["estimate_type"].values:
            sel = (out["estimate_type"] == name) & ~mask_suppressed
        elif name in out["estimate_name"].values:
            sel = (out["estimate_name"] == name) & ~mask_suppressed
        else:
            continue
        try:
            num = pd.to_numeric(out.loc[sel, "estimate_value"], errors="coerce")
            dec_places = n  # bind loop variable for lambda
            formatted = num.round(dec_places).apply(
                lambda x, _n=dec_places: f"{x:,.{_n}f}".replace(",", big_mark).replace(
                    ".", decimal_mark
                )
                if pd.notna(x) else ""
            )
            out.loc[sel, "estimate_value"] = formatted
        except Exception:
            pass
    # Any remaining numeric that wasn't in dec: format with 2 decimals
    return out


def format_min_cell_count(result: pd.DataFrame, settings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replace suppressed count placeholders with min_cell_count from settings.

    If settings has result_id and min_cell_count, values like '<5' are shown
    using the actual threshold from settings.

    Parameters
    ----------
    result : pandas.DataFrame
        Summarised result with result_id, estimate_name, estimate_value.
    settings : pandas.DataFrame or None, optional
        Settings with result_id, min_cell_count (optional).

    Returns
    -------
    pandas.DataFrame
        result with estimate_value updated for suppressed counts (or unchanged if no settings).
    """
    if settings is None or "min_cell_count" not in settings.columns:
        return result
    out = result.merge(
        settings[["result_id", "min_cell_count"]].drop_duplicates(),
        on="result_id",
        how="left",
    )
    if "min_cell_count" not in out.columns:
        return result
    mc = out["min_cell_count"].astype(str).str.replace(r"\.0$", "", regex=True)
    repl = "<" + mc
    out["estimate_value"] = out["estimate_value"].where(
        ~(out["estimate_value"].eq("-") & out["estimate_name"].str.contains("count", na=False)),
        other=repl,
    )
    return out.drop(columns=["min_cell_count"], errors="ignore")


# ---------------------------------------------------------------------------
# Format estimate name (combine multiple estimates into one label)
# ---------------------------------------------------------------------------

# Pattern to find <estimate_name> in format string
_ESTIMATE_PLACEHOLDER = re.compile(r"<([^>]+)>")


def format_estimate_name(
    result: pd.DataFrame,
    estimate_name: dict[str, str] | None = None,
    keep_not_formatted: bool = True,
    use_format_order: bool = True,
) -> pd.DataFrame:
    """
    Combine estimate_name and estimate_value into new labels.

    estimate_name maps display label -> format string with <estimate_name> placeholders,
    e.g. {"N (%)": "<count> (<percentage>)", "N": "<count>"}.

    Parameters
    ----------
    result : DataFrame
        Must contain estimate_name, estimate_value (and estimate_type).
    estimate_name : dict, optional
        Display name -> format template using <estimate_name> placeholders.
    keep_not_formatted : bool
        If False, drop rows that didn't match any template.
    use_format_order : bool
        Sort output by the order of keys in estimate_name.

    Returns
    -------
    DataFrame
    """
    if not estimate_name:
        return result.copy()
    needed = ["estimate_name", "estimate_value"]
    for c in needed:
        if c not in result.columns:
            raise ValueError(f"result must contain columns: {needed}")

    id_cols = [c for c in result.columns if c not in ("estimate_name", "estimate_type", "estimate_value")]
    order_map = {label: i for i, label in enumerate(estimate_name)}
    combined_rows: list[pd.DataFrame] = []
    drop_index: set[int] = set()

    for display_name, template in estimate_name.items():
        placeholders = _ESTIMATE_PLACEHOLDER.findall(template)
        if not placeholders:
            continue
        keys = list(dict.fromkeys(placeholders))
        sub = result[result["estimate_name"].isin(keys)]
        if sub.empty:
            continue
        # Groups that have all keys (one row per key)
        grp = sub.groupby(id_cols, dropna=False)
        _keys = keys  # bind loop variable for lambda
        complete_groups = grp.filter(lambda g, _k=_keys: set(_k) <= set(g["estimate_name"].unique()))
        if complete_groups.empty:
            continue
        # Fill NaN in id_cols to prevent pivot_table from dropping rows
        fill_map = {c: "__NA__" for c in id_cols if complete_groups[c].isna().any()}
        pivot_input = complete_groups.fillna(fill_map) if fill_map else complete_groups
        wide = pivot_input.pivot_table(
            index=id_cols,
            columns="estimate_name",
            values="estimate_value",
            aggfunc="first",
        ).reset_index()
        if fill_map:
            for c in fill_map:
                wide[c] = wide[c].replace("__NA__", pd.NA)
        if not all(k in wide.columns for k in keys):
            continue

        def fill_template(row: pd.Series, _tmpl: str = template, _keys: list = keys) -> str:
            """Replace <estimate_name> placeholders in template with row values."""
            s = _tmpl
            for k in _keys:
                v = row.get(k, pd.NA)
                if pd.isna(v) or (isinstance(v, str) and v == "-"):
                    return "-"
                s = s.replace(f"<{k}>", str(v))
            return s

        wide["estimate_value"] = wide.apply(fill_template, axis=1)
        wide["estimate_name"] = display_name
        wide["estimate_type"] = "character"
        combined_rows.append(wide)
        # Mark original rows to drop: (id_cols, estimate_name in keys) that appear in wide
        merge_keys = wide[id_cols].drop_duplicates()
        for _, mrow in merge_keys.iterrows():
            match = pd.Series(True, index=result.index)
            for c in id_cols:
                match = match & (result[c] == mrow[c])
            match = match & result["estimate_name"].isin(keys)
            drop_index.update(result.index[match].tolist())

    out = result.drop(index=drop_index, errors="ignore")
    if combined_rows:
        out = pd.concat([out, pd.concat(combined_rows, ignore_index=True)], ignore_index=True)
    if use_format_order and order_map:
        out["_order"] = out["estimate_name"].map(lambda x: order_map.get(x, len(order_map)))
        out = out.sort_values("_order").drop(columns=["_order"])
    if not keep_not_formatted and estimate_name:
        out = out[out["estimate_name"].isin(estimate_name)]
    return out


# ---------------------------------------------------------------------------
# Format header (pivot estimate_value into columns for table headers)
# ---------------------------------------------------------------------------


def format_header(
    result: pd.DataFrame,
    header: list[str],
    delim: str = "\n",
    include_header_name: bool = True,
    include_header_key: bool = True,
) -> pd.DataFrame:
    """
    Pivot result so header columns become column headers; estimate_value becomes cells.

    Parameters
    ----------
    result : DataFrame
        Must contain estimate_value. Header columns must be in result.
    header : list of str
        Column names to build headers from (order = hierarchy).
    delim : str
        Delimiter in generated header labels.
    include_header_name : bool
        Include the key name (e.g. "Study strata") in the header text.
    include_header_key : bool
        Include [header], [header_level] etc. in the label (for downstream styling).

    Returns
    -------
    DataFrame
        Pivoted table with new column names from header levels.
    """
    if "estimate_value" not in result.columns:
        raise ValueError("result must contain column estimate_value")
    orig = result.columns.tolist()
    header = [h for h in header if h in result.columns]
    if not header:
        if len(header) == 0 and result.columns.tolist() == orig:
            return result.copy()
        return result.rename(columns={"estimate_value": delim.join(header) if header else "Estimate"})

    detail = result[header + ["estimate_value"]].drop_duplicates()
    detail = detail.assign(_col=[f"column{i:03d}" for i in range(len(detail))])
    merged = result.merge(detail, on=header + ["estimate_value"], how="inner")
    rest = merged.drop(columns=header + ["estimate_value"]).drop_duplicates()
    piv = merged.pivot_table(
        index=[c for c in rest.columns if c != "_col"],
        columns="_col",
        values="estimate_value",
        aggfunc="first",
    ).reset_index()
    col_detail = detail.drop(columns=["estimate_value"]).drop_duplicates()
    new_names = []
    for _, row in col_detail.iterrows():
        parts = []
        for h in header:
            v = row.get(h, "")
            if pd.notna(v) and v != "":
                if include_header_key:
                    parts.append(f"[header_level]{v}" if include_header_name else f"[header_level]{v}")
                else:
                    parts.append(str(v))
        new_names.append((row["_col"], (delim.join(parts).rstrip(delim) if parts else row["_col"])))
    rename = dict(new_names)
    piv = piv.rename(columns=rename)
    reorder = [c for c in piv.columns if c not in orig] + [c for c in orig if c in piv.columns]
    return piv[[c for c in reorder if c in piv.columns]]


# ---------------------------------------------------------------------------
# Tidy summarised result (split group/strata/additional; pivot optional)
# ---------------------------------------------------------------------------


def tidy_summarised_result(
    result: pd.DataFrame,
    settings_column: list[str] | None = None,
    pivot_estimates_by: str | None = "estimate_name",
) -> pd.DataFrame:
    """
    Tidy a summarised result: keep long form; optionally pivot estimate_value by estimate_name.

    Parameters
    ----------
    result : DataFrame
        Summarised result (standard columns).
    settings_column : list, optional
        Settings columns to join (from settings table); not implemented here.
    pivot_estimates_by : str or None
        If "estimate_name", pivot estimate_value so each estimate_name becomes a column.

    Returns
    -------
    DataFrame
    """
    out = result.copy()
    if pivot_estimates_by and pivot_estimates_by in out.columns and pivot_estimates_by != "estimate_value":
        idv = [c for c in out.columns if c not in ("estimate_value", "estimate_type", pivot_estimates_by)]
        out = out.pivot_table(
            index=idv,
            columns=pivot_estimates_by,
            values="estimate_value",
            aggfunc="first",
        ).reset_index()
    return out


# ---------------------------------------------------------------------------
# Table output (dataframe / HTML)
# ---------------------------------------------------------------------------


def empty_table(
    type: str | None = None,
    style: str | None = None,
) -> pd.DataFrame | Any:
    """Return an empty formatted table (DataFrame or GT object).

    Parameters
    ----------
    type : str or None, optional
        "dataframe", "html", or "gt"; default from DEFAULT_TABLE_TYPE.
    style : str or None, optional
        Unused; kept for API compatibility.

    Returns
    -------
    pandas.DataFrame or Any
        Empty DataFrame or GT object (when type='gt' and great_tables available).
    """
    _ = style
    t = type or DEFAULT_TABLE_TYPE
    if t not in TABLE_TYPES:
        t = DEFAULT_TABLE_TYPE
    if t == "gt" and GT is not None:
        return GT(pd.DataFrame())
    return pd.DataFrame()


def format_table(
    x: pd.DataFrame,
    type: str | None = None,
    delim: str = "\n",
    na: str | None = NA_STR,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    group_column: list[str] | None = None,
    group_as_column: bool = False,
    group_order: list[str] | None = None,
    merge: str | list[str] | None = "all_columns",
) -> pd.DataFrame | str | Any:
    """
    Format a DataFrame as a table (dataframe, HTML string, or great_tables GT object).

    Parameters
    ----------
    x : DataFrame
    type : 'dataframe' | 'html' | 'gt'
    delim, na, title, subtitle, caption : optional
    group_column, group_as_column, group_order, merge : optional grouping

    Returns
    -------
    DataFrame, HTML str, or GT object (when type='gt')
    """
    _ = merge
    t = type or DEFAULT_TABLE_TYPE
    if t not in TABLE_TYPES:
        t = DEFAULT_TABLE_TYPE
    if x.empty:
        if t == "gt" and GT is not None:
            return GT(pd.DataFrame())
        if t == "dataframe":
            return pd.DataFrame()
        return "<table><tbody></tbody></table>"
    if na is not None:
        x = x.fillna(na)
    if t == "html":
        html = x.to_html(index=False)
        if caption:
            html = f"<caption>{caption}</caption>\n{html}"
        return html
    if t == "gt" and GT is not None:
        return _format_table_gt(
            x,
            delim=delim,
            na=na,
            title=title,
            subtitle=subtitle,
            caption=caption,
            group_column=group_column,
            group_as_column=group_as_column,
            group_order=group_order,
        )
    return x


def _format_table_gt(
    x: pd.DataFrame,
    delim: str = "\n",
    na: str | None = NA_STR,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    group_column: list[str] | None = None,
    group_as_column: bool = False,
    group_order: list[str] | None = None,
) -> Any:
    """Build a great_tables GT object from a DataFrame.

    Parameters
    ----------
    x : pandas.DataFrame
        Table data.
    delim : str, optional
        Delimiter for spanner labels in column names.
    na : str or None, optional
        String for missing values.
    title, subtitle, caption : str or None, optional
        Table header/source note.
    group_column : list[str] or None, optional
        Columns to use as row groups.
    group_as_column : bool, optional
        If True, show group as column.
    group_order : list[str] or None, optional
        Order for group levels.

    Returns
    -------
    Any
        great_tables GT object.
    """
    df = x.copy()
    groupname_col = None
    if group_column and not group_as_column:
        if len(group_column) == 1 and group_column[0] in df.columns:
            groupname_col = group_column[0]
            if group_order:
                df[groupname_col] = pd.Categorical(
                    df[groupname_col].astype(str), categories=group_order, ordered=True
                )
                df = df.sort_values(groupname_col)
        elif all(c in df.columns for c in group_column):
            groupname_col = "_gt_group_"
            df[groupname_col] = df[group_column].astype(str).agg(" | ".join, axis=1)
            df = df.drop(columns=group_column)
            if group_order:
                df[groupname_col] = pd.Categorical(
                    df[groupname_col], categories=group_order, ordered=True
                )
                df = df.sort_values(groupname_col)
    gt = GT(df, groupname_col=groupname_col, auto_align=True)
    if na is not None:
        gt = gt.sub_missing(missing_text=na)
    # Spanner from delimiter in column names (e.g. [header_level]value)
    cols_with_delim = [c for c in df.columns if delim in str(c)]
    if cols_with_delim:
        gt = gt.tab_spanner_delim(delim=delim, columns=cols_with_delim)
    if title is not None or subtitle is not None:
        gt = gt.tab_header(title=title or "", subtitle=subtitle or "")
    if caption is not None:
        gt = gt.tab_source_note(source_note=caption)
    return gt


def _rename_internal(col: str, rename: dict[str, str]) -> str:
    """Return rename[col] if col in rename else col. Parameters: col (str), rename (dict). Returns: str."""
    return rename.get(col, col)


def vis_table(
    result: pd.DataFrame,
    estimate_name: dict[str, str] | None = None,
    header: list[str] | None = None,
    group_column: list[str] | None = None,
    rename: dict[str, str] | None = None,
    type: str | None = None,
    hide: list[str] | None = None,
    style: str | None = None,
    options: dict[str, Any] | None = None,
) -> pd.DataFrame | str | Any:
    """
    Format a table (summarised_result-like DataFrame) into a display table.

    Parameters
    ----------
    result : DataFrame
        Must contain estimate_value; estimate_name, estimate_type optional for formatting.
    estimate_name : dict, optional
        Display name -> format template, e.g. {"N (%)": "<count> (<percentage>)"}.
    header : list of str, optional
        Columns to use as header (pivoted into columns).
    group_column : list of str, optional
        Columns to group by (e.g. strata_level).
    rename : dict, optional
        Column renames for display (e.g. {"cdm_name": "Database name"}).
    type : 'dataframe' | 'html' | 'gt'
    hide : list of str
        Columns to drop.
    style : str
        Style name (currently only affects future backends).
    options : dict
        Override default_table_options().

    Returns
    -------
    DataFrame, HTML str, or great_tables GT object (when type='gt')
    """
    _ = style
    header = header or []
    hide = list(hide) if hide else []
    rename = dict(rename) if rename else {}
    opts = default_table_options(options or {})

    if result.empty:
        return empty_table(type=type or DEFAULT_TABLE_TYPE)

    out = result.copy()
    if "estimate_value" in out.columns and "estimate_name" in out.columns and "estimate_type" in out.columns:
        dec = opts.get("decimals", {})
        if isinstance(dec, dict):
            out = format_estimate_value(
                out,
                decimals=dec,
                decimal_mark=opts.get("decimal_mark", "."),
                big_mark=opts.get("big_mark", ","),
            )
    if estimate_name:
        out = format_estimate_name(
            out,
            estimate_name=estimate_name,
            keep_not_formatted=opts.get("keep_not_formatted", True),
            use_format_order=opts.get("use_format_order", True),
        )
    hide = hide + ["result_id", "estimate_type"] if "result_id" in out.columns else hide
    hide = list(dict.fromkeys(hide))
    for c in hide:
        if c in out.columns:
            out = out.drop(columns=[c])
    for old_name, new_name in rename.items():
        if old_name in out.columns and old_name != "estimate_value":
            out = out.rename(columns={old_name: new_name})
    if opts.get("na") and out.notna().any().any():
        out = out.fillna(opts["na"])
    if header:
        out = format_header(
            out,
            header=header,
            delim=opts.get("delim", "\n"),
            include_header_name=opts.get("include_header_name", True),
            include_header_key=opts.get("include_header_key", True),
        )
    return format_table(
        out,
        type=type or DEFAULT_TABLE_TYPE,
        delim=opts.get("delim", "\n"),
        na=opts.get("na"),
        title=opts.get("title"),
        subtitle=opts.get("subtitle"),
        caption=opts.get("caption"),
        group_column=group_column,
        group_as_column=opts.get("group_as_column"),
        group_order=opts.get("group_order"),
        merge=opts.get("merge"),
    )


def vis_omop_table(
    result: pd.DataFrame,
    estimate_name: dict[str, str] | None = None,
    header: list[str] | None = None,
    settings_column: list[str] | None = None,
    group_column: list[str] | None = None,
    rename: dict[str, str] | None = None,
    type: str | None = None,
    hide: list[str] | None = None,
    column_order: list[str] | None = None,
    factor: dict[str, list[str]] | None = None,
    style: str | None = None,
    show_min_cell_count: bool = True,
    options: dict[str, Any] | None = None,
) -> pd.DataFrame | str:
    """
    Format a summarised_result DataFrame into a display table (visOmopTable port).

    Parameters
    ----------
    result : DataFrame
        Summarised result with standard columns (result_id, cdm_name, group_*, strata_*,
        variable_name, variable_level, estimate_name, estimate_type, estimate_value, etc.).
    estimate_name : dict, optional
        E.g. {"N (%)": "<count> (<percentage>)", "N": "<count>"}.
    header : list of str, optional
        Columns to use as table header.
    settings_column : list, optional
        Settings columns to include (if result has settings merged).
    group_column : list, optional
        E.g. [strata_columns(result)].
    rename : dict, optional
        Display names for columns.
    type : 'dataframe' | 'html' | 'gt'
    hide : list of str
        Columns to drop (default includes result_id, estimate_type).
    column_order : list of str, optional
        Final column order.
    factor : dict, optional
        Column -> ordered list of levels (for categorical order).
    style : str, optional
    show_min_cell_count : bool
        If True and settings has min_cell_count, show suppression threshold.
    options : dict, optional

    Returns
    -------
    DataFrame, HTML str, or great_tables GT object (when type='gt')
    """
    _ = settings_column, factor
    header = header or []
    hide = list(hide) if hide else []
    rename = dict(rename) if rename else {}
    if "cdm_name" not in rename:
        rename["cdm_name"] = "CDM name"
    opts = default_table_options(options or {})

    out = result.copy()
    if show_min_cell_count and "result_id" in out.columns:
        # format_min_cell_count would need settings; skip if no settings
        pass
    out = tidy_summarised_result(out, settings_column=settings_column, pivot_estimates_by=None)
    if column_order:
        order = [c for c in column_order if c in out.columns]
        if order:
            rest = [c for c in out.columns if c not in order]
            out = out[order + rest]
    return vis_table(
        result=out,
        estimate_name=estimate_name,
        header=header,
        group_column=group_column,
        rename=rename,
        type=type,
        hide=hide,
        style=style,
        options=opts,
    )


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def mock_summarised_result(seed: int = 1) -> pd.DataFrame:
    """Create a mock summarised_result DataFrame for examples (mirrors visOmopResults mockSummarisedResult).

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible mock data (default 1).

    Returns
    -------
    pandas.DataFrame
        Mock result with result_id, cdm_name, group_*, strata_*, variable_*, estimate_name, estimate_type, estimate_value.
    """
    import random

    rng = random.Random(seed)
    n = 9
    strata_name = ["overall"] + ["age_group &&& sex"] * 4 + ["sex"] * 2 + ["age_group"] * 2
    strata_level = [
        "overall", "<40 &&& Male", ">=40 &&& Male", "<40 &&& Female", ">=40 &&& Female",
        "Male", "Female", "<40", ">=40",
    ]
    rows = []
    rid = 1
    for cohort in ("cohort1", "cohort2"):
        for i in range(n):
            rows.append({
                "result_id": rid,
                "cdm_name": "mock",
                "group_name": "cohort_name",
                "group_level": cohort,
                "strata_name": strata_name[i],
                "strata_level": strata_level[i],
                "variable_name": "number subjects",
                "variable_level": None,
                "estimate_name": "count",
                "estimate_type": "integer",
                "estimate_value": str(round(10_000_000 * rng.random())),
                "additional_name": "overall",
                "additional_level": "overall",
            })
        for i in range(n):
            rows.append({
                "result_id": rid,
                "cdm_name": "mock",
                "group_name": "cohort_name",
                "group_level": cohort,
                "strata_name": strata_name[i],
                "strata_level": strata_level[i],
                "variable_name": "age",
                "variable_level": None,
                "estimate_name": "mean",
                "estimate_type": "numeric",
                "estimate_value": str(round(100 * rng.random(), 2)),
                "additional_name": "overall",
                "additional_level": "overall",
            })
        for i in range(n):
            rows.append({
                "result_id": rid,
                "cdm_name": "mock",
                "group_name": "cohort_name",
                "group_level": cohort,
                "strata_name": strata_name[i],
                "strata_level": strata_level[i],
                "variable_name": "age",
                "variable_level": None,
                "estimate_name": "sd",
                "estimate_type": "numeric",
                "estimate_value": str(round(10 * rng.random(), 2)),
                "additional_name": "overall",
                "additional_level": "overall",
            })
        for i in range(n):
            rows.append({
                "result_id": rid,
                "cdm_name": "mock",
                "group_name": "cohort_name",
                "group_level": cohort,
                "strata_name": strata_name[i],
                "strata_level": strata_level[i],
                "variable_name": "Medications",
                "variable_level": "Amoxicillin",
                "estimate_name": "count",
                "estimate_type": "integer",
                "estimate_value": str(round(100_000 * rng.random())),
                "additional_name": "overall",
                "additional_level": "overall",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------

PLOT_TYPES = ("altair",)
DEFAULT_PLOT_TYPE = "altair"


def plot_type() -> tuple[str, ...]:
    """Supported plot output types.

    Returns
    -------
    tuple[str, ...]
        ("altair",).
    """
    return PLOT_TYPES


def plot_columns(result: pd.DataFrame) -> list[str]:
    """Column names available for use in plot aesthetics for a summarised result.

    Includes tidy columns (split group/strata/additional name-level pairs) plus
    estimate names as potential y-axis values.

    Parameters
    ----------
    result : pandas.DataFrame
        Summarised result with standard columns.

    Returns
    -------
    list[str]
        Column names usable in plot x, y, facet, colour arguments.
    """
    tidy = _tidy_result_for_plot(result)
    return list(tidy.columns)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def customise_text(
    x: list[str] | str,
    custom: dict[str, str] | None = None,
    keep: list[str] | None = None,
) -> list[str] | str:
    """Style text: replace underscores with spaces and convert to sentence case.

    Parameters
    ----------
    x : str or list[str]
        Text to style.
    custom : dict, optional
        Specific overrides mapping original -> replacement text.
    keep : list[str], optional
        Values to keep unchanged.

    Returns
    -------
    str or list[str]
        Styled text.
    """
    scalar = isinstance(x, str)
    items = [x] if scalar else list(x)
    out = []
    for item in items:
        if keep and item in keep:
            out.append(item)
        elif custom and item in custom:
            out.append(custom[item])
        else:
            styled = item.replace("_", " ").replace("&&&", "and")
            styled = styled[:1].upper() + styled[1:] if styled else styled
            out.append(styled)
    return out[0] if scalar else out


def _style_label(x: str | list[str] | None) -> str:
    """Convert column name(s) to human-readable label for axis/legend.

    Parameters
    ----------
    x : str, list[str], or None
        Column name(s).

    Returns
    -------
    str
        Styled label or empty string.
    """
    if not x:
        return ""
    if isinstance(x, str):
        x = [x]
    styled = customise_text(x)
    if isinstance(styled, str):
        return styled
    return ", ".join(styled)


# ---------------------------------------------------------------------------
# Internal helpers for plots
# ---------------------------------------------------------------------------


def _split_name_level(
    df: pd.DataFrame,
    name_col: str,
    level_col: str,
) -> pd.DataFrame:
    """Split paired name/level columns (e.g. group_name/group_level) into separate columns.

    Handles '&&&'-delimited compound keys like 'age_group &&& sex' / '<40 &&& Male'.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    name_col : str
        Column with field names (e.g. 'group_name').
    level_col : str
        Column with field values (e.g. 'group_level').

    Returns
    -------
    pandas.DataFrame
        DataFrame with name_col/level_col replaced by individual columns.
    """
    if name_col not in df.columns or level_col not in df.columns:
        return df
    out = df.copy()
    unique_names = out[name_col].dropna().unique()
    for name_val in unique_names:
        if name_val == "overall":
            continue
        parts = [p.strip() for p in name_val.split("&&&")]
        mask = out[name_col] == name_val
        levels = out.loc[mask, level_col].astype(str)
        if len(parts) == 1:
            out.loc[mask, parts[0]] = levels.values
        else:
            split_levels = levels.str.split("&&&", expand=True)
            for i, part in enumerate(parts):
                if i < split_levels.shape[1]:
                    out.loc[mask, part] = split_levels[i].str.strip().values
    out = out.drop(columns=[name_col, level_col], errors="ignore")
    return out


def _tidy_result_for_plot(result: pd.DataFrame) -> pd.DataFrame:
    """Tidy a summarised_result for plotting: split name/level pairs, pivot estimates.

    Parameters
    ----------
    result : pandas.DataFrame
        Summarised result with standard columns.

    Returns
    -------
    pandas.DataFrame
        Tidied DataFrame with individual columns and pivoted estimate values.
    """
    out = result.copy()
    out = _split_name_level(out, "group_name", "group_level")
    out = _split_name_level(out, "strata_name", "strata_level")
    out = _split_name_level(out, "additional_name", "additional_level")
    # Drop columns that are entirely 'overall' after split
    for col in list(out.columns):
        if col not in result.columns and out[col].dropna().unique().tolist() == ["overall"]:
            out = out.drop(columns=[col])
    # Drop result_id
    out = out.drop(columns=["result_id"], errors="ignore")
    # Pivot estimate_name -> columns with estimate_value
    if "estimate_name" in out.columns and "estimate_value" in out.columns:
        id_cols = [
            c for c in out.columns if c not in ("estimate_name", "estimate_type", "estimate_value")
        ]
        # Fill NaN in id_cols to prevent pivot_table from dropping rows
        fill_map = {c: "__NA__" for c in id_cols if out[c].isna().any()}
        if fill_map:
            out = out.fillna(fill_map)
        pivoted = out.pivot_table(
            index=id_cols,
            columns="estimate_name",
            values="estimate_value",
            aggfunc="first",
        ).reset_index()
        pivoted.columns.name = None
        # Restore NaN
        if fill_map:
            for c in fill_map:
                pivoted[c] = pivoted[c].replace("__NA__", pd.NA)
        return pivoted
    return out


def _check_in_data(result: pd.DataFrame, columns: list[str]) -> None:
    """Validate that all requested columns are present in the (tidy) data.

    Parameters
    ----------
    result : pandas.DataFrame
        Tidied result DataFrame.
    columns : list[str]
        Column names that must be present.

    Raises
    ------
    ValueError
        If any column is missing.
    """
    missing = [c for c in columns if c and c not in result.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in data. "
            f"Available: {list(result.columns)}"
        )


def _prepare_data_for_plot(
    result: pd.DataFrame,
    needed_cols: list[str],
) -> pd.DataFrame:
    """Tidy the result, validate columns, and coerce numeric columns.

    Parameters
    ----------
    result : pandas.DataFrame
        Raw summarised result.
    needed_cols : list[str]
        All columns referenced in plot aesthetics.

    Returns
    -------
    pandas.DataFrame
        Tidied, validated DataFrame ready for Altair.
    """
    tidy = _tidy_result_for_plot(result)
    cols = [c for c in needed_cols if c]
    _check_in_data(tidy, cols)
    # Coerce numeric-looking columns
    for col in cols:
        if col in tidy.columns:
            import contextlib

            with contextlib.suppress(ValueError, TypeError):
                tidy[col] = pd.to_numeric(tidy[col], errors="ignore")
    return tidy


# ---------------------------------------------------------------------------
# Plotting functions (Altair-based, port of visOmopResults R package)
# ---------------------------------------------------------------------------


def empty_plot(
    title: str = "No data to plot",
    subtitle: str = "",
) -> alt.Chart:
    """Return an empty Altair chart with a title message.

    Parameters
    ----------
    title : str
        Title for the empty plot.
    subtitle : str
        Subtitle for the empty plot.

    Returns
    -------
    alt.Chart
        Empty chart with title.
    """
    chart = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_text(fontSize=14, color="gray")
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            text=alt.value(title),
        )
        .properties(
            title=alt.Title(text=title, subtitle=subtitle) if subtitle else title,
            width=400,
            height=300,
        )
    )
    return chart


def scatter_plot(
    result: pd.DataFrame,
    x: str,
    y: str,
    line: bool = False,
    point: bool = True,
    ribbon: bool = False,
    ymin: str | None = None,
    ymax: str | None = None,
    facet: str | list[str] | None = None,
    colour: str | None = None,
    group: str | None = None,
) -> alt.Chart:
    """Create a scatter/line plot from a summarised_result DataFrame.

    Parameters
    ----------
    result : pandas.DataFrame
        Summarised result with standard columns.
    x : str
        Column for x-axis (can be a column name or estimate name).
    y : str
        Column for y-axis (can be a column name or estimate name).
    line : bool
        Whether to draw lines connecting points.
    point : bool
        Whether to draw points.
    ribbon : bool
        Whether to draw a ribbon (requires ymin and ymax).
    ymin : str or None
        Column for ribbon/error bar lower bound.
    ymax : str or None
        Column for ribbon/error bar upper bound.
    facet : str, list[str], or None
        Column(s) to facet by.
    colour : str or None
        Column to map to colour.
    group : str or None
        Column to group by (defaults to colour).

    Returns
    -------
    alt.Chart
        Altair chart object.
    """
    if result.empty:
        warnings.warn("result object is empty, returning empty plot.", stacklevel=2)
        return empty_plot()

    if group is None:
        group = colour

    needed = [x, y, ymin, ymax, colour, group]
    if facet:
        needed.extend(facet if isinstance(facet, list) else [facet])
    needed = [c for c in needed if c]

    data = _prepare_data_for_plot(result, list(dict.fromkeys(needed)))

    # Build encoding
    encode_kwargs: dict[str, Any] = {
        "x": alt.X(f"{x}:N", title=_style_label(x)),
        "y": alt.Y(f"{y}:Q", title=_style_label(y)),
    }
    if colour:
        encode_kwargs["color"] = alt.Color(f"{colour}:N", title=_style_label(colour))

    layers: list[alt.Chart] = []
    base = alt.Chart(data)

    # Ribbon (area band)
    if ribbon and ymin and ymax:
        ribbon_layer = base.mark_area(opacity=0.3).encode(
            **{k: v for k, v in encode_kwargs.items() if k != "y"},
            y=alt.Y(f"{ymin}:Q"),
            y2=alt.Y2(f"{ymax}:Q"),
        )
        layers.append(ribbon_layer)

    # Error bars
    if ymin and ymax and not ribbon:
        err = base.mark_rule().encode(
            **{k: v for k, v in encode_kwargs.items() if k != "y"},
            y=alt.Y(f"{ymin}:Q"),
            y2=alt.Y2(f"{ymax}:Q"),
        )
        layers.append(err)

    # Line
    if line:
        line_layer = base.mark_line().encode(**encode_kwargs)
        layers.append(line_layer)

    # Points
    if point:
        point_layer = base.mark_circle(size=60).encode(**encode_kwargs)
        layers.append(point_layer)

    if not layers:
        point_layer = base.mark_circle(size=60).encode(**encode_kwargs)
        layers.append(point_layer)

    chart = alt.layer(*layers)

    # Facet
    chart = _apply_facet(chart, facet)

    return chart.properties(width=400, height=300)


def bar_plot(
    result: pd.DataFrame,
    x: str,
    y: str,
    position: str = "dodge",
    facet: str | list[str] | None = None,
    colour: str | None = None,
) -> alt.Chart:
    """Create a bar plot from a summarised_result DataFrame.

    Parameters
    ----------
    result : pandas.DataFrame
        Summarised result with standard columns.
    x : str
        Column for x-axis.
    y : str
        Column for y-axis (height of bars).
    position : str
        Bar position: 'dodge' (side-by-side) or 'stack'.
    facet : str, list[str], or None
        Column(s) to facet by.
    colour : str or None
        Column to map to colour/fill.

    Returns
    -------
    alt.Chart
        Altair chart object.
    """
    if position not in ("dodge", "stack"):
        raise ValueError(f"position must be 'dodge' or 'stack', got '{position}'")

    if result.empty:
        warnings.warn("result object is empty, returning empty plot.", stacklevel=2)
        return empty_plot()

    needed = [x, y, colour]
    if facet:
        needed.extend(facet if isinstance(facet, list) else [facet])
    needed = [c for c in needed if c]

    data = _prepare_data_for_plot(result, list(dict.fromkeys(needed)))

    encode_kwargs: dict[str, Any] = {
        "x": alt.X(f"{x}:N", title=_style_label(x)),
        "y": alt.Y(f"{y}:Q", title=_style_label(y)),
    }

    if colour:
        encode_kwargs["color"] = alt.Color(f"{colour}:N", title=_style_label(colour))
        if position == "dodge":
            encode_kwargs["xOffset"] = alt.XOffset(f"{colour}:N")

    if position == "stack":
        pass  # Altair stacks by default

    chart = alt.Chart(data).mark_bar().encode(**encode_kwargs)
    chart = _apply_facet(chart, facet)

    return chart.properties(width=400, height=300)


def box_plot(
    result: pd.DataFrame,
    x: str,
    lower: str = "q25",
    middle: str = "median",
    upper: str = "q75",
    ymin: str = "min",
    ymax: str = "max",
    facet: str | list[str] | None = None,
    colour: str | None = None,
) -> alt.Chart:
    """Create a box plot from pre-computed summary statistics.

    Unlike standard box plots that compute statistics from raw data, this function
    expects pre-computed statistics (q25, median, q75, min, max) as columns, matching
    the R visOmopResults boxPlot API.

    Parameters
    ----------
    result : pandas.DataFrame
        Summarised result with standard columns. Must contain columns matching
        lower, middle, upper, ymin, ymax (by estimate_name or column name).
    x : str
        Column for x-axis categories.
    lower : str
        Column name for lower quartile (Q1/q25).
    middle : str
        Column name for median.
    upper : str
        Column name for upper quartile (Q3/q75).
    ymin : str
        Column name for whisker minimum.
    ymax : str
        Column name for whisker maximum.
    facet : str, list[str], or None
        Column(s) to facet by.
    colour : str or None
        Column to map to colour.

    Returns
    -------
    alt.Chart
        Altair chart object.
    """
    if result.empty:
        warnings.warn("result object is empty, returning empty plot.", stacklevel=2)
        return empty_plot()

    needed = [x, lower, middle, upper, ymin, ymax, colour]
    if facet:
        needed.extend(facet if isinstance(facet, list) else [facet])
    needed = [c for c in needed if c]

    data = _prepare_data_for_plot(result, list(dict.fromkeys(needed)))

    color_encode = {}
    if colour:
        color_encode["color"] = alt.Color(f"{colour}:N", title=_style_label(colour))

    # Whiskers (min to max)
    whisker = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x=alt.X(f"{x}:N", title=_style_label(x)),
            y=alt.Y(f"{ymin}:Q", title=""),
            y2=alt.Y2(f"{ymax}:Q"),
            **color_encode,
        )
    )

    # Box (lower to upper)
    box = (
        alt.Chart(data)
        .mark_bar(size=20, opacity=0.7)
        .encode(
            x=alt.X(f"{x}:N"),
            y=alt.Y(f"{lower}:Q"),
            y2=alt.Y2(f"{upper}:Q"),
            **color_encode,
        )
    )

    # Median tick
    median_tick = (
        alt.Chart(data)
        .mark_tick(color="black", size=20, thickness=2)
        .encode(
            x=alt.X(f"{x}:N"),
            y=alt.Y(f"{middle}:Q"),
        )
    )

    chart = alt.layer(whisker, box, median_tick)
    chart = _apply_facet(chart, facet)

    return chart.properties(width=400, height=300)


def _apply_facet(
    chart: alt.Chart | alt.LayerChart,
    facet: str | list[str] | None,
) -> alt.Chart | alt.LayerChart | alt.FacetChart:
    """Apply faceting to an Altair chart.

    Parameters
    ----------
    chart : alt.Chart or alt.LayerChart
        Chart to facet.
    facet : str, list[str], or None
        Column(s) to facet by.

    Returns
    -------
    alt.Chart, alt.LayerChart, or alt.FacetChart
        Faceted chart or original if no facet.
    """
    if not facet:
        return chart
    if isinstance(facet, str):
        return chart.facet(facet=f"{facet}:N", columns=3)
    if len(facet) == 1:
        return chart.facet(facet=f"{facet[0]}:N", columns=3)
    if len(facet) == 2:
        return chart.facet(row=f"{facet[0]}:N", column=f"{facet[1]}:N")
    # More than 2: combine into one
    return chart.facet(facet=f"{facet[0]}:N", columns=3)
