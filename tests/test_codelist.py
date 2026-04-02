# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for codelist.py: Codelist, CodelistWithDetails, ConceptSetExpression."""

import pandas as pd
import pytest

from cdmconnector.codelist import (
    Codelist,
    CodelistWithDetails,
    ConceptSetExpression,
    empty_codelist,
    empty_codelist_with_details,
    empty_concept_set_expression,
    new_codelist,
    new_codelist_with_details,
    new_concept_set_expression,
)


# ---------------------------------------------------------------------------
# Codelist
# ---------------------------------------------------------------------------


class TestCodelist:
    def test_basic_construction(self):
        cl = Codelist({"diabetes": [201826, 443238], "hypertension": [316866]})
        assert len(cl) == 2
        assert "diabetes" in cl
        assert cl["diabetes"] == [201826, 443238]

    def test_sorted_and_deduplicated(self):
        cl = Codelist({"a": [3, 1, 2, 1]})
        assert cl["a"] == [1, 2, 3]

    def test_alphabetical_ordering(self):
        cl = Codelist({"zebra": [1], "alpha": [2], "middle": [3]})
        assert list(cl) == ["alpha", "middle", "zebra"]

    def test_to_dataframe(self):
        cl = Codelist({"a": [1, 2], "b": [3]})
        df = cl.to_dataframe()
        assert list(df.columns) == ["codelist_name", "concept_id"]
        assert len(df) == 3

    def test_bind(self):
        cl1 = Codelist({"a": [1, 2]})
        cl2 = Codelist({"a": [3], "b": [4]})
        combined = cl1.bind(cl2)
        assert sorted(combined["a"]) == [1, 2, 3]
        assert combined["b"] == [4]

    def test_equality(self):
        cl1 = Codelist({"a": [1, 2]})
        cl2 = Codelist({"a": [2, 1]})
        assert cl1 == cl2

    def test_new_codelist(self):
        cl = new_codelist({"test": [100, 200]})
        assert isinstance(cl, Codelist)
        assert cl["test"] == [100, 200]

    def test_empty_codelist(self):
        cl = empty_codelist()
        assert len(cl) == 0
        assert cl.to_dataframe().empty

    def test_invalid_types(self):
        with pytest.raises(TypeError):
            Codelist("not a dict")
        with pytest.raises(TypeError):
            Codelist({1: [1, 2]})  # non-string key
        with pytest.raises(TypeError):
            Codelist({"a": "not a list"})

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NA/NaN"):
            Codelist({"a": [1, float("nan")]})

    def test_repr(self):
        cl = Codelist({"a": [1], "b": [2]})
        assert "Codelist(2 codelists" in repr(cl)

    def test_keys_values_items(self):
        cl = Codelist({"a": [1], "b": [2]})
        assert cl.keys() == ["a", "b"]
        assert len(cl.values()) == 2
        assert len(cl.items()) == 2


# ---------------------------------------------------------------------------
# CodelistWithDetails
# ---------------------------------------------------------------------------


class TestCodelistWithDetails:
    def test_basic_construction(self):
        data = {
            "diabetes": pd.DataFrame({"concept_id": [201826], "concept_name": ["DM"]}),
        }
        cl = CodelistWithDetails(data)
        assert len(cl) == 1
        assert "diabetes" in cl
        assert "concept_id" in cl["diabetes"].columns

    def test_to_dataframe(self):
        data = {
            "a": pd.DataFrame({"concept_id": [1, 2], "name": ["x", "y"]}),
        }
        cl = CodelistWithDetails(data)
        df = cl.to_dataframe()
        assert "codelist_name" in df.columns
        assert len(df) == 2

    def test_bind(self):
        cl1 = CodelistWithDetails({"a": pd.DataFrame({"concept_id": [1]})})
        cl2 = CodelistWithDetails({"b": pd.DataFrame({"concept_id": [2]})})
        combined = cl1.bind(cl2)
        assert len(combined) == 2

    def test_missing_concept_id(self):
        with pytest.raises(ValueError, match="concept_id"):
            CodelistWithDetails({"a": pd.DataFrame({"name": ["x"]})})

    def test_new_and_empty(self):
        cl = new_codelist_with_details({"a": pd.DataFrame({"concept_id": [1]})})
        assert isinstance(cl, CodelistWithDetails)
        empty = empty_codelist_with_details()
        assert len(empty) == 0


# ---------------------------------------------------------------------------
# ConceptSetExpression
# ---------------------------------------------------------------------------


class TestConceptSetExpression:
    def test_basic_construction(self):
        data = {
            "diabetes": pd.DataFrame({
                "concept_id": [201826],
                "excluded": [False],
                "descendants": [True],
                "mapped": [False],
            }),
        }
        cse = ConceptSetExpression(data)
        assert len(cse) == 1
        assert cse["diabetes"]["descendants"].iloc[0] == True  # noqa: E712

    def test_defaults_added(self):
        data = {"test": pd.DataFrame({"concept_id": [100]})}
        cse = ConceptSetExpression(data)
        df = cse["test"]
        assert "excluded" in df.columns
        assert "descendants" in df.columns
        assert "mapped" in df.columns
        assert df["excluded"].iloc[0] == False  # noqa: E712

    def test_to_dataframe(self):
        data = {"a": pd.DataFrame({"concept_id": [1, 2]})}
        cse = ConceptSetExpression(data)
        df = cse.to_dataframe()
        assert "concept_set_name" in df.columns
        assert set(df.columns) >= {"concept_set_name", "concept_id", "excluded", "descendants", "mapped"}

    def test_bind(self):
        cse1 = ConceptSetExpression({"a": pd.DataFrame({"concept_id": [1]})})
        cse2 = ConceptSetExpression({"b": pd.DataFrame({"concept_id": [2]})})
        combined = cse1.bind(cse2)
        assert len(combined) == 2

    def test_new_and_empty(self):
        cse = new_concept_set_expression({"a": pd.DataFrame({"concept_id": [1]})})
        assert isinstance(cse, ConceptSetExpression)
        empty = empty_concept_set_expression()
        assert len(empty) == 0

    def test_missing_concept_id(self):
        with pytest.raises(ValueError, match="concept_id"):
            ConceptSetExpression({"a": pd.DataFrame({"name": ["x"]})})
