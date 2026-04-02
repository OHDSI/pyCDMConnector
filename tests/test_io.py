# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for io.py: import/export for SummarisedResult, Codelist, ConceptSetExpression."""

import json

import pandas as pd
import pytest

from cdmconnector.characteristics import SummarisedResult
from cdmconnector.codelist import Codelist, ConceptSetExpression
from cdmconnector.io import (
    export_codelist,
    export_concept_set_expression,
    export_summarised_result,
    import_codelist,
    import_concept_set_expression,
    import_summarised_result,
)


@pytest.fixture
def sample_result():
    results = pd.DataFrame({
        "result_id": [1],
        "cdm_name": ["test"],
        "group_name": ["overall"],
        "group_level": ["overall"],
        "strata_name": ["overall"],
        "strata_level": ["overall"],
        "variable_name": ["count"],
        "variable_level": [None],
        "estimate_name": ["count"],
        "estimate_type": ["integer"],
        "estimate_value": ["100"],
        "additional_name": ["overall"],
        "additional_level": ["overall"],
    })
    settings = pd.DataFrame({
        "result_id": [1],
        "result_type": ["test"],
        "package_name": ["cdmconnector"],
        "package_version": ["0.1.0"],
    })
    return SummarisedResult(results=results, settings=settings)


class TestSummarisedResultIO:
    def test_roundtrip(self, sample_result, tmp_path):
        path = tmp_path / "result.csv"
        export_summarised_result(sample_result, path)
        assert path.exists()
        assert (tmp_path / "result_settings.csv").exists()

        loaded = import_summarised_result(path)
        assert isinstance(loaded, SummarisedResult)
        assert len(loaded) == 1
        assert len(loaded.settings) == 1

    def test_export_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            export_summarised_result("not a result", tmp_path / "test.csv")

    def test_import_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_summarised_result(tmp_path / "nonexistent.csv")


class TestCodelistIO:
    def test_roundtrip(self, tmp_path):
        cl = Codelist({"diabetes": [201826, 443238], "hypertension": [316866]})
        path = tmp_path / "codelist.json"
        export_codelist(cl, path)
        assert path.exists()

        loaded = import_codelist(path)
        assert isinstance(loaded, Codelist)
        assert loaded == cl

    def test_export_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            export_codelist("not a codelist", tmp_path / "test.json")

    def test_import_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_codelist(tmp_path / "nonexistent.json")


class TestConceptSetExpressionIO:
    def test_roundtrip(self, tmp_path):
        cse = ConceptSetExpression({
            "diabetes": pd.DataFrame({
                "concept_id": [201826],
                "excluded": [False],
                "descendants": [True],
                "mapped": [False],
            }),
        })
        outdir = tmp_path / "concept_sets"
        export_concept_set_expression(cse, outdir)
        assert outdir.exists()
        assert (outdir / "diabetes.json").exists()

        loaded = import_concept_set_expression(outdir)
        assert isinstance(loaded, ConceptSetExpression)
        assert "diabetes" in loaded
        assert loaded["diabetes"]["concept_id"].iloc[0] == 201826
        assert loaded["diabetes"]["descendants"].iloc[0] == True  # noqa: E712

    def test_single_file_import(self, tmp_path):
        data = {
            "items": [{
                "concept": {"CONCEPT_ID": 100},
                "isExcluded": False,
                "includeDescendants": True,
                "includeMapped": False,
            }]
        }
        path = tmp_path / "test.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = import_concept_set_expression(path)
        assert "test" in loaded
        assert loaded["test"]["concept_id"].iloc[0] == 100

    def test_export_type_error(self, tmp_path):
        with pytest.raises(TypeError):
            export_concept_set_expression("not a cse", tmp_path / "test")

    def test_import_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_concept_set_expression(tmp_path / "nonexistent")
