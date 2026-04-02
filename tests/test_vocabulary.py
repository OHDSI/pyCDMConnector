# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for cdmconnector.vocabulary (search_vocab / Hecate API)."""

import os
from unittest.mock import patch

import pandas as pd
import pytest
import requests

from cdmconnector.vocabulary import DEFAULT_HECATE_BASE_URL, search_vocab


# ---- Validation ----


def test_search_vocab_query_empty_raises():
    """search_vocab raises if query is empty or not a string."""
    with pytest.raises(ValueError, match="query.*non-empty string"):
        search_vocab("", base_url="https://api.example.com")
    with pytest.raises(ValueError, match="query.*non-empty string"):
        search_vocab("   ", base_url="https://api.example.com")
    with pytest.raises(ValueError, match="query.*non-empty string"):
        search_vocab(None, base_url="https://api.example.com")  # type: ignore[arg-type]


def test_search_vocab_limit_validation():
    """search_vocab raises if limit is not 1–150."""
    with pytest.raises(ValueError, match="limit.*1 and 150"):
        search_vocab("asthma", base_url="https://api.example.com", limit=0)
    with pytest.raises(ValueError, match="limit.*1 and 150"):
        search_vocab("asthma", base_url="https://api.example.com", limit=151)
    with pytest.raises(ValueError, match="limit.*1 and 150"):
        search_vocab("asthma", base_url="https://api.example.com", limit=20.5)  # type: ignore[arg-type]


def test_search_vocab_uses_public_default_base_url():
    """search_vocab uses the public Hecate URL when no base_url is configured."""
    with patch.dict(os.environ, {"HECATE_BASE_URL": ""}, clear=False):
        with patch("cdmconnector.vocabulary.requests.get") as mget:
            mget.return_value.status_code = 200
            mget.return_value.reason = "OK"
            mget.return_value.json.return_value = [
                {"concept_name": "x", "concepts": [{"concept_id": 1}]},
            ]
            mget.return_value.text = ""

            search_vocab("asthma")

    mget.assert_called_once()
    assert mget.call_args[0][0] == f"{DEFAULT_HECATE_BASE_URL}/search"


# ---- Mocked API success ----


def test_search_vocab_success_returns_flattened_dataframe():
    """search_vocab returns a flattened DataFrame on valid API response."""
    mock_response = [
        {
            "concept_name": "asthma",
            "concept_name_lower": "asthma",
            "score": 1.0,
            "concepts": [
                {
                    "concept_id": 317009,
                    "concept_name": "Asthma",
                    "domain_id": "Condition",
                    "vocabulary_id": "SNOMED",
                    "concept_class_id": "Clinical Finding",
                    "standard_concept": "S",
                    "concept_code": "195967001",
                    "invalid_reason": None,
                    "valid_start_date": "1970-01-01",
                    "valid_end_date": "2099-12-31",
                    "record_count": 100,
                },
            ],
        },
    ]

    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.return_value.status_code = 200
        mget.return_value.reason = "OK"
        mget.return_value.json.return_value = mock_response
        mget.return_value.text = ""

        df = search_vocab("asthma", base_url="https://api.example.com")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["concept_id"].iloc[0] == 317009
    assert df["concept_name"].iloc[0] == "Asthma"
    assert df["search_concept_name"].iloc[0] == "asthma"
    assert "concept_id" in df.columns and "concept_name" in df.columns


def test_search_vocab_success_multiple_groups_flattened():
    """search_vocab flattens multiple groups into one row per concept."""
    mock_response = [
        {
            "concept_name": "a",
            "concept_name_lower": "a",
            "score": 0.9,
            "concepts": [
                {"concept_id": 1, "concept_name": "A1", "domain_id": "Condition"},
            ],
        },
        {
            "concept_name": "b",
            "concept_name_lower": "b",
            "score": 0.8,
            "concepts": [
                {"concept_id": 2, "concept_name": "B1", "domain_id": "Condition"},
            ],
        },
    ]

    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.return_value.status_code = 200
        mget.return_value.reason = "OK"
        mget.return_value.json.return_value = mock_response
        mget.return_value.text = ""

        df = search_vocab("x", base_url="https://api.example.com")

    assert len(df) == 2
    assert list(df["concept_id"]) == [1, 2]
    assert list(df["concept_name"]) == ["A1", "B1"]


# ---- Mocked API errors ----


def test_search_vocab_http_error_returns_empty_dataframe():
    """search_vocab returns empty DataFrame on HTTP 4xx/5xx and warns."""
    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.return_value.status_code = 500
        mget.return_value.reason = "Internal Server Error"
        mget.return_value.json.return_value = {"error": "server error"}
        mget.return_value.text = ""

        with pytest.warns(RuntimeWarning, match="API returned HTTP"):
            df = search_vocab("asthma", base_url="https://api.example.com")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_search_vocab_request_exception_returns_empty_dataframe():
    """search_vocab returns empty DataFrame on request failure and warns."""
    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.side_effect = requests.RequestException("connection refused")

        with pytest.warns(RuntimeWarning, match="Request failed"):
            df = search_vocab("asthma", base_url="https://api.example.com")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_search_vocab_unexpected_shape_returns_empty_dataframe():
    """search_vocab returns empty DataFrame when response lacks 'concepts' and warns."""
    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.return_value.status_code = 200
        mget.return_value.reason = "OK"
        mget.return_value.json.return_value = [{"no_concepts_key": True}]
        mget.return_value.text = ""

        with pytest.warns(RuntimeWarning, match="concepts"):
            df = search_vocab("asthma", base_url="https://api.example.com")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_search_vocab_empty_response_returns_empty_dataframe():
    """search_vocab returns empty DataFrame for empty list response and warns."""
    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.return_value.status_code = 200
        mget.return_value.reason = "OK"
        mget.return_value.json.return_value = []
        mget.return_value.text = ""

        with pytest.warns(RuntimeWarning, match="Unexpected response format|empty"):
            df = search_vocab("asthma", base_url="https://api.example.com")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ---- Params and env ----


def test_search_vocab_passes_query_and_filters():
    """search_vocab sends q and optional filters in request params."""
    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.return_value.status_code = 200
        mget.return_value.reason = "OK"
        mget.return_value.json.return_value = [
            {"concept_name": "x", "concepts": [{"concept_id": 1, "concept_name": "X"}]},
        ]
        mget.return_value.text = ""

        search_vocab(
            "diabetes",
            vocabulary_id="SNOMED",
            standard_concept="S",
            domain_id="Condition",
            concept_class_id="Clinical Finding",
            limit=10,
            base_url="https://api.example.com",
        )

    mget.assert_called_once()
    call_kw = mget.call_args[1]
    assert call_kw["params"]["q"] == "diabetes"
    assert call_kw["params"]["vocabulary_id"] == "SNOMED"
    assert call_kw["params"]["standard_concept"] == "S"
    assert call_kw["params"]["domain_id"] == "Condition"
    assert call_kw["params"]["concept_class_id"] == "Clinical Finding"
    assert call_kw["params"]["limit"] == 10
    assert call_kw["timeout"] == 30.0  # default 30000 ms -> 30 s


def test_search_vocab_uses_env_base_url():
    """search_vocab uses HECATE_BASE_URL when base_url not passed."""
    with patch.dict(os.environ, {"HECATE_BASE_URL": "https://hecate.example.com"}, clear=False):
        with patch("cdmconnector.vocabulary.requests.get") as mget:
            mget.return_value.status_code = 200
            mget.return_value.reason = "OK"
            mget.return_value.json.return_value = [
                {"concept_name": "x", "concepts": [{"concept_id": 1}]},
            ]
            mget.return_value.text = ""

            search_vocab("test")

    mget.assert_called_once()
    assert mget.call_args[0][0] == "https://hecate.example.com/search"


def test_search_vocab_concept_id_and_record_count_numeric():
    """search_vocab coerces concept_id and record_count to numeric/Int64."""
    mock_response = [
        {
            "concept_name": "x",
            "concepts": [
                {
                    "concept_id": "317009",
                    "concept_name": "Asthma",
                    "record_count": "42",
                },
            ],
        },
    ]

    with patch("cdmconnector.vocabulary.requests.get") as mget:
        mget.return_value.status_code = 200
        mget.return_value.reason = "OK"
        mget.return_value.json.return_value = mock_response
        mget.return_value.text = ""

        df = search_vocab("x", base_url="https://api.example.com")

    assert df["concept_id"].dtype.name == "Int64"
    assert df["concept_id"].iloc[0] == 317009
    assert df["record_count"].iloc[0] == 42
