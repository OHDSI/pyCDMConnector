# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary search via Hecate API: search_vocab."""

from __future__ import annotations

import os
import time
import warnings
from collections import deque
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests


def search_vocab(
    query: str,
    vocabulary_id: Optional[str] = None,
    standard_concept: Optional[str] = None,
    domain_id: Optional[str] = None,
    concept_class_id: Optional[str] = None,
    limit: int = 20,
    base_url: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Search Hecate concepts and return results as a pandas DataFrame.

    Mirrors the behavior of the R `hecate_search()` wrapper:
      - GET {base_url}/search?q=...
      - Optional filters: vocabulary_id, standard_concept, domain_id, concept_class_id, limit
      - Rate limit: 100 calls / 60 seconds (process-local)
      - Returns a flattened DataFrame (one row per nested concept result)
      - On API errors or unexpected shapes: returns an empty DataFrame (instead of NULL)

    Parameters
    ----------
    query : str
        Search query (required, non-empty).
    vocabulary_id : str | None
        Optional vocabulary filter (comma-separated).
    standard_concept : str | None
        Optional standard concept flag (e.g. "S", "C").
    domain_id : str | None
        Optional domain filter (comma-separated).
    concept_class_id : str | None
        Optional concept class filter.
    limit : int
        Max results (default 20, max 150).
    base_url : str | None
        Hecate base URL. If None, uses env var HECATE_BASE_URL.
    timeout_ms : int | None
        Request timeout in ms. If None, uses env var HECATE_TIMEOUT_MS, else defaults to 30000.
    api_key : str | None
        Bearer token. If None, uses env var HECATE_API_KEY.

    Returns
    -------
    pandas.DataFrame
        Flattened results. Empty if error/no results/unexpected response.
    """

    # -----------------------------
    # Basic validation (like R)
    # -----------------------------
    if not isinstance(query, str) or not query or query.strip() == "":
        raise ValueError("`query` must be a non-empty string.")
    if not isinstance(limit, int) or limit < 1 or limit > 150:
        raise ValueError("`limit` must be an integer between 1 and 150.")

    # -----------------------------
    # Defaults (like app_config())
    # -----------------------------
    base_url = (base_url or os.getenv("HECATE_BASE_URL") or "").strip()
    if not base_url:
        raise ValueError(
            "Missing base_url. Pass base_url=... or set env var HECATE_BASE_URL."
        )
    base_url = base_url.rstrip("/")

    api_key = api_key if api_key is not None else os.getenv("HECATE_API_KEY", "")
    api_key = api_key.strip()

    if timeout_ms is None:
        env_timeout = os.getenv("HECATE_TIMEOUT_MS", "").strip()
        timeout_ms = int(env_timeout) if env_timeout.isdigit() else 30000

    # Basic sanity check without printing key
    if api_key and len(api_key) < 10:
        warnings.warn(
            "API key appears too short. Please check HECATE_API_KEY environment variable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # -----------------------------
    # Process-local rate limiter
    # (100 calls / 60 seconds)
    # -----------------------------
    if not hasattr(search_vocab, "_rl_call_times"):
        search_vocab._rl_call_times = deque()  # type: ignore[attr-defined]

    def _rate_limit(max_calls: int = 100, per_seconds: int = 60) -> None:
        now = time.time()
        call_times: deque = search_vocab._rl_call_times  # type: ignore[attr-defined]
        while call_times and call_times[0] <= now - per_seconds:
            call_times.popleft()
        if len(call_times) >= max_calls:
            wait_time = per_seconds - (now - call_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
            call_times.clear()
        call_times.append(time.time())

    # -----------------------------
    # Build request (hecate_request)
    # -----------------------------
    params: Dict[str, Any] = {
        "q": query,
        "vocabulary_id": vocabulary_id,
        "standard_concept": standard_concept,
        "domain_id": domain_id,
        "concept_class_id": concept_class_id,
        "limit": limit,
    }
    params = {k: v for k, v in params.items() if v is not None}

    url = f"{base_url}/search"
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # -----------------------------
    # Perform request (hecate_perform)
    # -----------------------------
    _rate_limit()

    try:
        resp = requests.get(
            url, params=params, headers=headers, timeout=timeout_ms / 1000.0
        )
    except requests.RequestException as e:
        warnings.warn(f"Request failed: {e}", RuntimeWarning, stacklevel=2)
        return pd.DataFrame()

    parsed: Union[list, dict, str, None]
    try:
        parsed = resp.json()
    except ValueError:
        parsed = resp.text

    if resp.status_code >= 400:
        warnings.warn(
            f"API returned HTTP {resp.status_code} {resp.reason}: {parsed}",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    # -----------------------------
    # Shape checks (like R)
    # -----------------------------
    if not isinstance(parsed, list) or len(parsed) == 0:
        warnings.warn(
            "Unexpected response format or empty response",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    first = parsed[0]
    if not isinstance(first, dict) or "concepts" not in first:
        warnings.warn(
            "Response does not match expected structure (missing 'concepts' field)",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    # -----------------------------
    # Flatten: one row per concept
    # -----------------------------
    rows: list[dict[str, Any]] = []

    for group in parsed:
        if not isinstance(group, dict):
            continue

        search_concept_name = group.get("concept_name")
        search_concept_name_lower = group.get("concept_name_lower")
        search_score = group.get("score")

        concepts = group.get("concepts") or []
        if not isinstance(concepts, list) or len(concepts) == 0:
            continue

        for concept in concepts:
            if not isinstance(concept, dict):
                continue

            row = {
                "search_concept_name": search_concept_name,
                "search_concept_name_lower": search_concept_name_lower,
                "search_score": search_score,
                "concept_id": concept.get("concept_id"),
                "concept_name": concept.get("concept_name"),
                "domain_id": concept.get("domain_id"),
                "vocabulary_id": concept.get("vocabulary_id"),
                "concept_class_id": concept.get("concept_class_id"),
                "standard_concept": concept.get("standard_concept"),
                "concept_code": concept.get("concept_code"),
                "invalid_reason": concept.get("invalid_reason"),
                "valid_start_date": concept.get("valid_start_date"),
                "valid_end_date": concept.get("valid_end_date"),
                "record_count": concept.get("record_count"),
            }
            rows.append(row)

    if not rows:
        warnings.warn(
            "No concepts found in response", RuntimeWarning, stacklevel=2
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "concept_id" in df.columns:
        df["concept_id"] = pd.to_numeric(df["concept_id"], errors="coerce").astype(
            "Int64"
        )
    if "record_count" in df.columns:
        df["record_count"] = pd.to_numeric(df["record_count"], errors="coerce")
        if df["record_count"].dropna().apply(lambda x: float(x).is_integer()).all():
            df["record_count"] = df["record_count"].astype("Int64")

    return df
