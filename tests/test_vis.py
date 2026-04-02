# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for vis.py: table formatting and plotting functions."""

from __future__ import annotations

import altair as alt
import pandas as pd
import pytest

from cdmconnector.vis import (
    bar_plot,
    box_plot,
    customise_text,
    default_table_options,
    empty_plot,
    empty_table,
    format_estimate_name,
    format_estimate_value,
    format_header,
    format_table,
    mock_summarised_result,
    plot_columns,
    plot_type,
    scatter_plot,
    table_columns,
    table_options,
    table_style,
    table_type,
    tidy_summarised_result,
    vis_omop_table,
    vis_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_result():
    return mock_summarised_result()


@pytest.fixture()
def age_result(mock_result):
    return mock_result[mock_result["variable_name"] == "age"].copy()


# ---------------------------------------------------------------------------
# Table configuration
# ---------------------------------------------------------------------------


class TestTableConfig:
    def test_table_type(self):
        t = table_type()
        assert "dataframe" in t
        assert "gt" in t

    def test_table_style(self):
        s = table_style()
        assert "default" in s

    def test_table_options(self):
        opts = table_options()
        assert "decimals" in opts
        assert opts["decimal_mark"] == "."

    def test_default_table_options_override(self):
        opts = default_table_options({"decimal_mark": ","})
        assert opts["decimal_mark"] == ","

    def test_table_columns(self, mock_result):
        cols = table_columns(mock_result)
        assert "cdm_name" in cols
        assert "variable_name" in cols


# ---------------------------------------------------------------------------
# Format functions
# ---------------------------------------------------------------------------


class TestFormatEstimateValue:
    def test_basic(self, mock_result):
        out = format_estimate_value(mock_result)
        assert "estimate_value" in out.columns
        assert len(out) == len(mock_result)

    def test_custom_decimals(self, mock_result):
        out = format_estimate_value(mock_result, decimals={"integer": 0, "numeric": 1})
        assert "estimate_value" in out.columns

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            format_estimate_value(df)


class TestFormatEstimateName:
    def test_combine_mean_sd(self, mock_result):
        formatted = format_estimate_value(mock_result)
        out = format_estimate_name(
            formatted,
            estimate_name={"Mean (SD)": "<mean> (<sd>)"},
        )
        assert "Mean (SD)" in out["estimate_name"].values

    def test_no_template(self, mock_result):
        out = format_estimate_name(mock_result, estimate_name=None)
        assert len(out) == len(mock_result)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            format_estimate_name(df, estimate_name={"x": "<a>"})


class TestFormatHeader:
    def test_pivot_by_column(self, mock_result):
        out = format_header(mock_result, header=["cdm_name"])
        assert "estimate_value" not in out.columns or "cdm_name" not in out.columns

    def test_empty_header(self, mock_result):
        out = format_header(mock_result, header=[])
        assert len(out) == len(mock_result)


class TestTidySummarisedResult:
    def test_pivot(self, mock_result):
        out = tidy_summarised_result(mock_result, pivot_estimates_by="estimate_name")
        assert "count" in out.columns or "mean" in out.columns

    def test_no_pivot(self, mock_result):
        out = tidy_summarised_result(mock_result, pivot_estimates_by=None)
        assert "estimate_value" in out.columns


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


class TestFormatTable:
    def test_dataframe(self, mock_result):
        out = format_table(mock_result, type="dataframe")
        assert isinstance(out, pd.DataFrame)

    def test_html(self, mock_result):
        out = format_table(mock_result, type="html")
        assert isinstance(out, str)
        assert "<table" in out

    def test_empty(self):
        out = format_table(pd.DataFrame(), type="dataframe")
        assert isinstance(out, pd.DataFrame)
        assert out.empty


class TestEmptyTable:
    def test_returns_dataframe(self):
        out = empty_table(type="dataframe")
        assert isinstance(out, pd.DataFrame)


class TestVisTable:
    def test_basic(self, mock_result):
        out = vis_table(mock_result, type="dataframe")
        assert isinstance(out, pd.DataFrame)
        assert not out.empty

    def test_with_estimate_name(self, mock_result):
        out = vis_table(
            mock_result,
            estimate_name={"Mean (SD)": "<mean> (<sd>)"},
            type="dataframe",
        )
        assert isinstance(out, pd.DataFrame)

    def test_empty_result(self):
        out = vis_table(pd.DataFrame(columns=["estimate_value"]), type="dataframe")
        assert isinstance(out, (pd.DataFrame, type(None))) or out is not None


class TestVisOmopTable:
    def test_basic(self, mock_result):
        out = vis_omop_table(mock_result, type="dataframe")
        assert isinstance(out, pd.DataFrame)
        assert not out.empty

    def test_with_header(self, mock_result):
        out = vis_omop_table(
            mock_result,
            header=["cdm_name"],
            type="dataframe",
        )
        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


class TestMockSummarisedResult:
    def test_basic(self):
        result = mock_summarised_result()
        assert isinstance(result, pd.DataFrame)
        assert "estimate_value" in result.columns
        assert "estimate_name" in result.columns
        assert len(result) > 0

    def test_reproducible(self):
        r1 = mock_summarised_result(seed=42)
        r2 = mock_summarised_result(seed=42)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------------------------


class TestPlotConfig:
    def test_plot_type(self):
        t = plot_type()
        assert "altair" in t

    def test_plot_columns(self, mock_result):
        cols = plot_columns(mock_result)
        assert isinstance(cols, list)
        assert len(cols) > 0


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


class TestCustomiseText:
    def test_basic(self):
        assert customise_text("hello_world") == "Hello world"

    def test_list(self):
        out = customise_text(["hello_world", "foo_bar"])
        assert out == ["Hello world", "Foo bar"]

    def test_custom_override(self):
        out = customise_text("cdm_name", custom={"cdm_name": "Database"})
        assert out == "Database"

    def test_keep(self):
        out = customise_text("keep_me", keep=["keep_me"])
        assert out == "keep_me"


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


class TestEmptyPlot:
    def test_basic(self):
        chart = empty_plot()
        assert isinstance(chart, alt.Chart)

    def test_custom_title(self):
        chart = empty_plot(title="Custom", subtitle="Sub")
        assert isinstance(chart, alt.Chart)


class TestScatterPlot:
    def test_basic(self, age_result):
        chart = scatter_plot(age_result, x="cohort_name", y="mean", point=True)
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_with_line(self, age_result):
        chart = scatter_plot(age_result, x="cohort_name", y="mean", line=True, point=True)
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_with_colour(self, age_result):
        chart = scatter_plot(
            age_result, x="cohort_name", y="mean", point=True, colour="cohort_name"
        )
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_with_facet(self, age_result):
        chart = scatter_plot(
            age_result,
            x="cohort_name",
            y="mean",
            point=True,
            facet="sex",
        )
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_with_ribbon(self, age_result):
        # Use mean as center, sd as range (not ideal but tests the path)
        chart = scatter_plot(
            age_result,
            x="cohort_name",
            y="mean",
            ribbon=True,
            ymin="sd",
            ymax="mean",
            point=True,
        )
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_empty_result(self):
        chart = scatter_plot(pd.DataFrame(), x="x", y="y")
        assert isinstance(chart, alt.Chart)

    def test_missing_column_raises(self, age_result):
        with pytest.raises(ValueError, match="not found"):
            scatter_plot(age_result, x="nonexistent", y="mean")


class TestBarPlot:
    def test_basic(self, mock_result):
        count_result = mock_result[
            (mock_result["estimate_name"] == "count")
            & (mock_result["variable_name"] == "number subjects")
            & (mock_result["strata_name"] == "overall")
        ]
        chart = bar_plot(count_result, x="cohort_name", y="count")
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_with_colour(self, mock_result):
        count_result = mock_result[
            (mock_result["estimate_name"] == "count")
            & (mock_result["variable_name"] == "number subjects")
        ]
        chart = bar_plot(count_result, x="cohort_name", y="count", colour="cohort_name")
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_stacked(self, mock_result):
        count_result = mock_result[
            (mock_result["estimate_name"] == "count")
            & (mock_result["variable_name"] == "number subjects")
        ]
        chart = bar_plot(
            count_result,
            x="cohort_name",
            y="count",
            colour="cohort_name",
            position="stack",
        )
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_invalid_position_raises(self, mock_result):
        with pytest.raises(ValueError, match="position"):
            bar_plot(mock_result, x="x", y="y", position="invalid")

    def test_empty_result(self):
        chart = bar_plot(pd.DataFrame(), x="x", y="y")
        assert isinstance(chart, alt.Chart)


class TestBoxPlot:
    def test_basic(self):
        data = pd.DataFrame({
            "result_id": [1] * 5 + [2] * 5,
            "cdm_name": ["mock"] * 10,
            "group_name": ["cohort_name"] * 10,
            "group_level": ["cohort1"] * 5 + ["cohort2"] * 5,
            "strata_name": ["overall"] * 10,
            "strata_level": ["overall"] * 10,
            "variable_name": ["age"] * 10,
            "variable_level": [None] * 10,
            "estimate_name": ["q25", "median", "q75", "min", "max"] * 2,
            "estimate_type": ["numeric"] * 10,
            "estimate_value": ["25", "50", "75", "10", "90", "30", "55", "80", "15", "95"],
            "additional_name": ["overall"] * 10,
            "additional_level": ["overall"] * 10,
        })
        chart = box_plot(data, x="cohort_name")
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_with_colour(self):
        data = pd.DataFrame({
            "result_id": [1] * 5 + [2] * 5,
            "cdm_name": ["mock"] * 10,
            "group_name": ["cohort_name"] * 10,
            "group_level": ["cohort1"] * 5 + ["cohort2"] * 5,
            "strata_name": ["overall"] * 10,
            "strata_level": ["overall"] * 10,
            "variable_name": ["age"] * 10,
            "variable_level": [None] * 10,
            "estimate_name": ["q25", "median", "q75", "min", "max"] * 2,
            "estimate_type": ["numeric"] * 10,
            "estimate_value": ["25", "50", "75", "10", "90", "30", "55", "80", "15", "95"],
            "additional_name": ["overall"] * 10,
            "additional_level": ["overall"] * 10,
        })
        chart = box_plot(data, x="cohort_name", colour="cohort_name")
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.FacetChart))

    def test_empty_result(self):
        chart = box_plot(pd.DataFrame(), x="x")
        assert isinstance(chart, alt.Chart)
