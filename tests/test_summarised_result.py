# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for SummarisedResult methods and related functions."""

import pandas as pd
import pytest

from cdmconnector.characteristics import (
    SummarisedResult,
    bind_summarised_results,
    empty_summarised_result,
    estimate_type_choices,
    result_columns,
)


@pytest.fixture
def sample_result():
    """Create a sample SummarisedResult for testing."""
    results = pd.DataFrame({
        "result_id": [1, 1, 1, 1, 2, 2],
        "cdm_name": ["test"] * 6,
        "group_name": ["cohort_name"] * 4 + ["cohort_name"] * 2,
        "group_level": ["diabetes"] * 4 + ["hypertension"] * 2,
        "strata_name": ["overall", "overall", "sex", "sex", "overall", "overall"],
        "strata_level": ["overall", "overall", "Female", "Male", "overall", "overall"],
        "variable_name": ["number_subjects", "number_records", "number_subjects", "number_subjects",
                          "number_subjects", "number_records"],
        "variable_level": [None] * 6,
        "estimate_name": ["count", "count", "count", "count", "count", "count"],
        "estimate_type": ["integer"] * 6,
        "estimate_value": ["100", "150", "60", "40", "80", "90"],
        "additional_name": ["overall"] * 6,
        "additional_level": ["overall"] * 6,
    })
    settings = pd.DataFrame({
        "result_id": [1, 2],
        "result_type": ["summarise_characteristics", "summarise_characteristics"],
        "package_name": ["cdmconnector", "cdmconnector"],
        "package_version": ["0.1.0", "0.1.0"],
        "table_name": ["cohort", "cohort"],
    })
    return SummarisedResult(results=results, settings=settings)


class TestSummarisedResult:
    def test_construction(self, sample_result):
        assert len(sample_result) == 6
        assert len(sample_result.settings) == 2

    def test_empty_construction(self):
        sr = SummarisedResult()
        assert len(sr) == 0

    def test_filter_settings(self, sample_result):
        filtered = sample_result.filter_settings(result_id=1)
        assert len(filtered) == 4

    def test_filter_group(self, sample_result):
        filtered = sample_result.filter_group(cohort_name="diabetes")
        assert len(filtered) == 4

    def test_filter_strata(self, sample_result):
        filtered = sample_result.filter_strata(sex="Female")
        assert len(filtered) == 1

    def test_split_group(self, sample_result):
        df = sample_result.split_group()
        assert "cohort_name" in df.columns
        assert "group_name" not in df.columns

    def test_split_strata(self, sample_result):
        df = sample_result.split_strata()
        assert "sex" in df.columns or "strata_name" not in df.columns

    def test_split_all(self, sample_result):
        df = sample_result.split_all()
        assert "group_name" not in df.columns
        assert "strata_name" not in df.columns

    def test_tidy(self, sample_result):
        df = sample_result.tidy()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_suppress(self, sample_result):
        suppressed = sample_result.suppress(min_cell_count=50)
        assert isinstance(suppressed, SummarisedResult)
        # Values below 50 should be suppressed
        vals = suppressed.results["estimate_value"]
        # The count=40 should be NA now
        assert vals.isna().any()

    def test_add_settings(self, sample_result):
        updated = sample_result.add_settings(study_name="test_study")
        assert "study_name" in updated.settings.columns

    def test_group_columns(self, sample_result):
        cols = sample_result.group_columns()
        assert "cohort_name" in cols

    def test_strata_columns(self, sample_result):
        cols = sample_result.strata_columns()
        assert "sex" in cols

    def test_settings_columns(self, sample_result):
        cols = sample_result.settings_columns()
        assert "table_name" in cols

    def test_repr(self, sample_result):
        assert "SummarisedResult" in repr(sample_result)


class TestBindSummarisedResults:
    def test_bind_two(self, sample_result):
        combined = bind_summarised_results(sample_result, sample_result)
        assert len(combined) == 12

    def test_bind_empty(self):
        result = bind_summarised_results()
        assert len(result) == 0

    def test_bind_type_error(self, sample_result):
        with pytest.raises(TypeError):
            bind_summarised_results(sample_result, "not a result")


class TestStandaloneFunctions:
    def test_empty_summarised_result(self):
        sr = empty_summarised_result()
        assert len(sr) == 0
        assert isinstance(sr, SummarisedResult)

    def test_result_columns(self):
        cols = result_columns()
        assert "result_id" in cols
        assert "estimate_value" in cols
        assert len(cols) == 13

    def test_estimate_type_choices(self):
        choices = estimate_type_choices()
        assert "numeric" in choices
        assert "integer" in choices
        assert "character" in choices
