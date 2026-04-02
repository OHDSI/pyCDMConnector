# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Plot functions for CohortCharacteristics results.

Port of R CohortCharacteristics plot functions. Uses matplotlib for rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd

    from cdmconnector.characteristics import SummarisedResult


def _extract_df(result: SummarisedResult | Any) -> pd.DataFrame:
    """Extract DataFrame from SummarisedResult or plain DataFrame."""
    import pandas as pd

    if hasattr(result, "results"):
        return result.results.copy()
    if isinstance(result, pd.DataFrame):
        return result.copy()
    return pd.DataFrame(result)


def _split_group_level(df: pd.DataFrame) -> pd.DataFrame:
    """Split compound group_level into separate columns for plotting."""
    if "group_name" not in df.columns or "group_level" not in df.columns:
        return df
    df = df.copy()
    for idx in df.index:
        gn = str(df.at[idx, "group_name"])
        gl = str(df.at[idx, "group_level"])
        if " &&& " in gn:
            names = gn.split(" &&& ")
            levels = gl.split(" &&& ")
            for n, l in zip(names, levels):
                df.at[idx, n] = l
    return df


def _apply_facet(fig: Any, ax: Any, df: pd.DataFrame, facet: list[str] | str | None) -> None:
    """Add facet title if facet columns exist."""
    if facet is None:
        return
    if isinstance(facet, str):
        facet = [facet]
    for col in facet:
        if col in df.columns:
            vals = df[col].unique()
            if len(vals) == 1:
                ax.set_title(f"{col}: {vals[0]}", fontsize=10)


def plot_characteristics(
    result: SummarisedResult | Any,
    *,
    plot_type: str = "barplot",
    facet: list[str] | str | None = None,
    colour: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot characteristics from a summarise_characteristics result.

    Parameters
    ----------
    result : SummarisedResult or DataFrame.
    plot_type : "barplot", "scatterplot", "boxplot", or "densityplot".
    facet : Column(s) to facet by.
    colour : Column to colour by.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = _extract_df(result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    df = _split_group_level(df)

    # Filter to numeric estimates for plotting
    numeric_df = df[df["estimate_type"].isin(["numeric", "integer", "percentage"])].copy()
    numeric_df["_val"] = pd.to_numeric(numeric_df["estimate_value"], errors="coerce")

    if numeric_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric data to plot", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "barplot":
        _plot_bar(ax, numeric_df, colour)
    elif plot_type == "scatterplot":
        _plot_scatter(ax, numeric_df, colour)
    elif plot_type == "boxplot":
        _plot_box(ax, numeric_df)
    elif plot_type == "densityplot":
        _plot_density(ax, numeric_df, colour)
    else:
        ax.text(0.5, 0.5, f"Unknown plot_type: {plot_type}", ha="center", va="center")

    _apply_facet(fig, ax, numeric_df, facet)
    fig.tight_layout()
    return fig


def _plot_bar(ax: Any, df: pd.DataFrame, colour: str | None) -> None:
    """Bar plot of estimate values by variable."""
    groups = df.groupby(["variable_name", "estimate_name"])
    labels = []
    values = []
    for (vn, en), grp in groups:
        labels.append(f"{vn}\n({en})")
        values.append(grp["_val"].mean())

    colors = None
    if colour and colour in df.columns:
        # Use colour column for grouping
        pass  # Simple bar for now

    ax.barh(range(len(labels)), values)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Value")


def _plot_scatter(ax: Any, df: pd.DataFrame, colour: str | None) -> None:
    """Scatter plot of estimate values."""
    ax.scatter(range(len(df)), df["_val"], alpha=0.6, s=20)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Value")


def _plot_box(ax: Any, df: pd.DataFrame) -> None:
    """Box plot by variable_name using summary statistics."""
    variables = df["variable_name"].unique()
    data = []
    labels = []
    for vn in variables:
        sub = df[df["variable_name"] == vn]
        vals = sub["_val"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(vn)
    if data:
        ax.boxplot(data, labels=labels, vert=True)
        ax.tick_params(axis="x", rotation=45)


def _plot_density(ax: Any, df: pd.DataFrame, colour: str | None) -> None:
    """Kernel density plot by variable_name."""
    variables = df["variable_name"].unique()
    for vn in variables:
        vals = df[df["variable_name"] == vn]["_val"].dropna()
        if len(vals) > 1:
            vals.plot.kde(ax=ax, label=vn)
    ax.legend(fontsize=8)
    ax.set_xlabel("Value")


def plot_cohort_count(
    result: SummarisedResult | Any,
    *,
    facet: list[str] | str | None = None,
    colour: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot cohort counts as a bar chart.

    Parameters
    ----------
    result : SummarisedResult from summarise_cohort_count.
    facet : Column(s) to facet by.
    colour : Column to colour by.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = _extract_df(result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Filter to count estimates
    counts = df[df["estimate_name"] == "count"].copy()
    counts["_val"] = pd.to_numeric(counts["estimate_value"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 6))

    if counts.empty:
        ax.text(0.5, 0.5, "No count data", ha="center", va="center")
        return fig

    # Group by cohort (group_level) and variable_name
    grouped = counts.groupby(["group_level", "variable_name"])["_val"].sum().unstack(fill_value=0)
    grouped.plot(kind="bar", ax=ax)
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Count")
    ax.set_title("Cohort Counts")
    ax.legend(title="Variable", fontsize=8)
    ax.tick_params(axis="x", rotation=45)

    _apply_facet(fig, ax, counts, facet)
    fig.tight_layout()
    return fig


def plot_cohort_attrition(
    result: SummarisedResult | Any,
    *,
    show: tuple[str, ...] | list[str] = ("subjects", "records"),
) -> matplotlib.figure.Figure:
    """Plot cohort attrition as a flow diagram.

    Parameters
    ----------
    result : SummarisedResult from summarise_cohort_attrition.
    show : Which metrics to display ("subjects" and/or "records").

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd

    df = _extract_df(result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attrition data", ha="center", va="center")
        return fig

    # Get unique cohorts
    cohorts = df["group_level"].unique()
    n_cohorts = len(cohorts)
    fig, axes = plt.subplots(1, n_cohorts, figsize=(6 * n_cohorts, 8), squeeze=False)

    for col_idx, cohort_name in enumerate(cohorts):
        ax = axes[0, col_idx]
        ax.set_xlim(0, 10)
        ax.axis("off")
        ax.set_title(cohort_name, fontsize=12, fontweight="bold")

        cohort_data = df[df["group_level"] == cohort_name].copy()

        # Get unique reasons (attrition steps) in order
        reasons = cohort_data.sort_values("additional_level")["variable_level"].unique()

        # Build step data
        steps = []
        for reason in reasons:
            step_data = cohort_data[cohort_data["variable_level"] == reason]
            step = {"reason": reason}
            for _, row in step_data.iterrows():
                vn = row["variable_name"]
                val = row["estimate_value"]
                if "subjects" in show and vn == "Number subjects":
                    step["n_subjects"] = val
                if "records" in show and vn == "Number records":
                    step["n_records"] = val
                if "subjects" in show and vn == "Excluded subjects":
                    step["excl_subjects"] = val
                if "records" in show and vn == "Excluded records":
                    step["excl_records"] = val
            steps.append(step)

        # Draw flow diagram
        y_pos = 0.95
        box_height = 0.08
        box_width = 0.7
        spacing = 0.04

        for i, step in enumerate(steps):
            # Count box
            label_parts = []
            if "n_subjects" in step:
                label_parts.append(f"Subjects: {step['n_subjects']}")
            if "n_records" in step:
                label_parts.append(f"Records: {step['n_records']}")
            label = "\n".join(label_parts) if label_parts else "N/A"

            rect = mpatches.FancyBboxPatch(
                (0.15, y_pos - box_height), box_width, box_height,
                boxstyle="round,pad=0.01",
                facecolor="white", edgecolor="black", linewidth=1.5,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)
            ax.text(
                0.5, y_pos - box_height / 2, label,
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes,
            )

            if i < len(steps) - 1:
                # Arrow
                arrow_y = y_pos - box_height - spacing / 2
                ax.annotate(
                    "", xy=(0.5, arrow_y - spacing),
                    xytext=(0.5, arrow_y),
                    arrowprops={"arrowstyle": "->", "lw": 1.5},
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                )

                # Reason + exclusion box (to the right)
                reason_label = step["reason"]
                excl_parts = []
                if "excl_subjects" in step:
                    excl_parts.append(f"Excl. subjects: {step['excl_subjects']}")
                if "excl_records" in step:
                    excl_parts.append(f"Excl. records: {step['excl_records']}")

                if excl_parts:
                    next_step = steps[i + 1] if i + 1 < len(steps) else None
                    excl_label = f"{reason_label}\n" + "\n".join(excl_parts)
                    ax.text(
                        0.95, arrow_y - spacing / 2, excl_label,
                        ha="right", va="center", fontsize=7,
                        color="gray", style="italic",
                        transform=ax.transAxes,
                    )

                y_pos = arrow_y - spacing - spacing

    fig.tight_layout()
    return fig


def plot_cohort_timing(
    result: SummarisedResult | Any,
    *,
    plot_type: str = "boxplot",
    time_scale: str = "days",
    unique_combinations: bool = True,
    facet: list[str] | str | None = None,
    colour: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot timing between cohort entries.

    Parameters
    ----------
    result : SummarisedResult from summarise_cohort_timing.
    plot_type : "boxplot" or "densityplot".
    time_scale : "days" or "years".
    unique_combinations : If True, show only unique cohort pairs.
    facet : Column(s) to facet by.
    colour : Column to colour by.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from cdmconnector.characteristics import _get_unique_combinations

    df = _extract_df(result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center")
        return fig

    if unique_combinations:
        df = _get_unique_combinations(df)

    # Filter to numeric timing estimates
    numeric_df = df[df["estimate_type"] == "numeric"].copy()
    numeric_df["_val"] = pd.to_numeric(numeric_df["estimate_value"], errors="coerce")

    if time_scale == "years":
        numeric_df["_val"] = numeric_df["_val"] / 365.25

    fig, ax = plt.subplots(figsize=(10, 6))

    if numeric_df.empty:
        ax.text(0.5, 0.5, "No numeric timing data", ha="center", va="center")
        return fig

    unit = "Years" if time_scale == "years" else "Days"

    if plot_type == "boxplot":
        # Reconstruct box stats from summary
        pairs = numeric_df["group_level"].unique()
        box_data = []
        labels = []
        for pair in pairs:
            pair_df = numeric_df[numeric_df["group_level"] == pair]
            stats = {}
            for _, row in pair_df.iterrows():
                stats[row["estimate_name"]] = row["_val"]
            if "median" in stats:
                box_data.append({
                    "med": stats.get("median", 0),
                    "q1": stats.get("q25", 0),
                    "q3": stats.get("q75", 0),
                    "whislo": stats.get("min", 0),
                    "whishi": stats.get("max", 0),
                    "fliers": [],
                })
                labels.append(pair.replace(" &&& ", "\nvs\n"))

        if box_data:
            ax.bxp(box_data, showfliers=False)
            ax.set_xticklabels(labels, fontsize=8)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel(f"{unit} between cohort entries")
    elif plot_type == "densityplot":
        ax.text(0.5, 0.5, "Density plot requires raw data (not yet supported)",
                ha="center", va="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, f"Unknown plot_type: {plot_type}", ha="center", va="center")

    _apply_facet(fig, ax, numeric_df, facet)
    fig.tight_layout()
    return fig


def plot_cohort_overlap(
    result: SummarisedResult | Any,
    *,
    unique_combinations: bool = True,
    facet: list[str] | str | None = None,
    colour: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot cohort overlap as a stacked bar chart.

    Parameters
    ----------
    result : SummarisedResult from summarise_cohort_overlap.
    unique_combinations : If True, show only unique cohort pairs.
    facet : Column(s) to facet by.
    colour : Column to colour by.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from cdmconnector.characteristics import _get_unique_combinations

    df = _extract_df(result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No overlap data", ha="center", va="center")
        return fig

    if unique_combinations:
        df = _get_unique_combinations(df)

    # Use percentage estimates
    pct_df = df[df["estimate_name"] == "percentage"].copy()
    pct_df["_val"] = pd.to_numeric(pct_df["estimate_value"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 6))

    if pct_df.empty:
        ax.text(0.5, 0.5, "No percentage data", ha="center", va="center")
        return fig

    pairs = pct_df["group_level"].unique()
    categories = ["Only in reference cohort", "In both cohorts", "Only in comparator cohort"]
    colors_map = {"Only in reference cohort": "#4e79a7", "In both cohorts": "#59a14f",
                  "Only in comparator cohort": "#e15759"}

    x = np.arange(len(pairs))
    width = 0.6
    bottoms = np.zeros(len(pairs))

    for cat in categories:
        vals = []
        for pair in pairs:
            v = pct_df[(pct_df["group_level"] == pair) & (pct_df["variable_name"] == cat)]["_val"]
            vals.append(float(v.iloc[0]) if len(v) > 0 else 0)
        ax.bar(x, vals, width, bottom=bottoms, label=cat, color=colors_map.get(cat, "gray"))
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" &&& ", "\nvs\n") for p in pairs], fontsize=8)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Cohort Overlap")
    ax.legend(fontsize=8, loc="upper right")

    _apply_facet(fig, ax, pct_df, facet)
    fig.tight_layout()
    return fig


def plot_large_scale_characteristics(
    result: SummarisedResult | Any,
    *,
    facet: list[str] | str | None = None,
    colour: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot large-scale characteristics as a scatter plot of concept frequencies.

    Parameters
    ----------
    result : SummarisedResult from summarise_large_scale_characteristics.
    facet : Column(s) to facet by.
    colour : Column to colour by. Default "variable_level" (window).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = _extract_df(result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Use percentage estimates
    pct_df = df[df["estimate_name"] == "percentage"].copy()
    pct_df["_val"] = pd.to_numeric(pct_df["estimate_value"], errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 8))

    if pct_df.empty:
        ax.text(0.5, 0.5, "No percentage data", ha="center", va="center")
        return fig

    colour_col = colour or "variable_level"
    if colour_col in pct_df.columns:
        for label, grp in pct_df.groupby(colour_col):
            ax.scatter(range(len(grp)), grp["_val"], label=str(label), alpha=0.6, s=15)
        ax.legend(fontsize=7, title=colour_col, loc="upper right")
    else:
        ax.scatter(range(len(pct_df)), pct_df["_val"], alpha=0.6, s=15)

    ax.set_xlabel("Concept (index)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Large-Scale Characteristics")

    _apply_facet(fig, ax, pct_df, facet)
    fig.tight_layout()
    return fig


def plot_compared_large_scale_characteristics(
    result: SummarisedResult | Any,
    *,
    colour: str | None = None,
    reference: str | None = None,
    facet: list[str] | str | None = None,
) -> matplotlib.figure.Figure:
    """Compare large-scale characteristics across groups as a scatter plot.

    Plots frequency in reference group (x-axis) vs frequency in comparator
    group (y-axis) for each concept.

    Parameters
    ----------
    result : SummarisedResult from summarise_large_scale_characteristics.
    colour : Column to colour by.
    reference : Reference group level for comparison.
    facet : Column(s) to facet by.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    df = _extract_df(result)
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    pct_df = df[df["estimate_name"] == "percentage"].copy()
    pct_df["_val"] = pd.to_numeric(pct_df["estimate_value"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 8))

    if pct_df.empty:
        ax.text(0.5, 0.5, "No percentage data", ha="center", va="center")
        return fig

    groups = pct_df["group_level"].unique()
    if reference is None:
        reference = groups[0] if len(groups) > 0 else None

    if reference is None or len(groups) < 2:
        ax.text(0.5, 0.5, "Need at least 2 groups to compare", ha="center", va="center")
        return fig

    ref_data = pct_df[pct_df["group_level"] == reference]

    for comp_name in groups:
        if comp_name == reference:
            continue
        comp_data = pct_df[pct_df["group_level"] == comp_name]

        # Match on variable_name + variable_level
        merged = ref_data.merge(
            comp_data,
            on=["variable_name", "variable_level"],
            suffixes=("_ref", "_comp"),
            how="outer",
        )
        merged["_val_ref"] = merged["_val_ref"].fillna(0)
        merged["_val_comp"] = merged["_val_comp"].fillna(0)

        ax.scatter(merged["_val_ref"], merged["_val_comp"], alpha=0.5, s=15, label=comp_name)

    # Diagonal line
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel(f"Percentage - {reference}")
    ax.set_ylabel("Percentage - comparator")
    ax.set_title("Compared Large-Scale Characteristics")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    _apply_facet(fig, ax, pct_df, facet)
    fig.tight_layout()
    return fig
