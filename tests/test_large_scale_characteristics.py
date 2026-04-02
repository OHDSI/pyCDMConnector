# Copyright 2026 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for CohortCharacteristics large-scale characteristics helpers."""

from __future__ import annotations

import pandas as pd
import pytest

import cdmconnector as cc
from cdmconnector.characteristics import (
    SummarisedResult,
    summarise_large_scale_characteristics,
    table_large_scale_characteristics,
)
from cdmconnector.plots import (
    plot_compared_large_scale_characteristics,
    plot_large_scale_characteristics,
)


@pytest.fixture()
def large_scale_test_data():
    """Minimal CDM + cohort inputs for large-scale characteristics tests."""
    concept = pd.DataFrame(
        {
            "concept_id": [0, 101, 202],
            "concept_name": ["No matching concept", "Hypertension", "Aspirin use"],
        }
    )
    condition_occurrence = pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3],
            "condition_concept_id": [101, 0, 101, 202, 202],
            "condition_start_date": pd.to_datetime(
                ["2020-01-05", "2020-01-07", "2020-01-10", "2020-01-20", "2019-12-20"]
            ),
        }
    )
    cdm = cc.cdm_from_tables(
        {
            "concept": concept,
            "condition_occurrence": condition_occurrence,
        },
        cdm_name="test_cdm",
    )
    cohort = pd.DataFrame(
        {
            "cohort_definition_id": [1, 1, 2, 2],
            "subject_id": [1, 2, 2, 3],
            "cohort_start_date": pd.to_datetime(["2020-01-01"] * 4),
            "cohort_end_date": pd.to_datetime(["2020-06-01"] * 4),
        }
    )
    return cdm, cohort


def test_summarise_large_scale_characteristics_counts_and_percentages(large_scale_test_data):
    """Large-scale characteristics reports count/percentage rows with concept names."""
    cdm, cohort = large_scale_test_data

    result = summarise_large_scale_characteristics(
        cohort,
        cdm,
        window=[(0, 30)],
        event_in_window=["condition_occurrence"],
        minimum_frequency=0.0,
    )

    assert isinstance(result, SummarisedResult)
    assert not result.results.empty
    assert set(result.results["estimate_name"]) == {"count", "percentage"}
    assert "Hypertension" in result.results["variable_name"].values
    assert "Aspirin use" in result.results["variable_name"].values
    assert "0" not in result.results["additional_level"].values
    assert set(result.results["variable_level"]) == {"0 to 30"}

    hypertension_count = result.results[
        (result.results["group_level"] == "1")
        & (result.results["variable_name"] == "Hypertension")
        & (result.results["estimate_name"] == "count")
    ]["estimate_value"].iloc[0]
    hypertension_pct = result.results[
        (result.results["group_level"] == "1")
        & (result.results["variable_name"] == "Hypertension")
        & (result.results["estimate_name"] == "percentage")
    ]["estimate_value"].iloc[0]

    assert hypertension_count == "2"
    assert hypertension_pct == "100.0000"


def test_summarise_large_scale_characteristics_empty_cohort_returns_empty(large_scale_test_data):
    """Empty cohorts return an empty-style summarised result."""
    cdm, _ = large_scale_test_data
    empty = pd.DataFrame(
        columns=[
            "cohort_definition_id",
            "subject_id",
            "cohort_start_date",
            "cohort_end_date",
        ]
    )

    result = summarise_large_scale_characteristics(empty, cdm)

    assert isinstance(result, SummarisedResult)
    assert result.results.empty
    assert not result.settings.empty
    assert "summarise_large_scale_characteristics" in result.settings["result_type"].values


def test_table_large_scale_characteristics_top_concepts_filters(large_scale_test_data):
    """table_large_scale_characteristics can keep only the top concepts."""
    cdm, cohort = large_scale_test_data
    result = summarise_large_scale_characteristics(
        cohort,
        cdm,
        window=[(0, 30)],
        event_in_window=["condition_occurrence"],
        minimum_frequency=0.0,
    )

    table = table_large_scale_characteristics(result, top_concepts=1)

    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert set(table["variable_name"]) == {"Hypertension"}


def test_large_scale_characteristics_plot_helpers_return_figures(large_scale_test_data):
    """Large-scale plot helpers return matplotlib figures."""
    matplotlib = pytest.importorskip("matplotlib")

    cdm, cohort = large_scale_test_data
    result = summarise_large_scale_characteristics(
        cohort,
        cdm,
        window=[(0, 30)],
        event_in_window=["condition_occurrence"],
        minimum_frequency=0.0,
    )

    fig1 = plot_large_scale_characteristics(result)
    fig2 = plot_compared_large_scale_characteristics(result, reference="1")

    assert isinstance(fig1, matplotlib.figure.Figure)
    assert isinstance(fig2, matplotlib.figure.Figure)
