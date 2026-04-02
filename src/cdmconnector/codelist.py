# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Codelist, CodelistWithDetails, and ConceptSetExpression classes.

Python equivalents of the omopgenerics R classes for managing collections
of OMOP concept IDs used in phenotype definitions and cohort building.

Classes
-------
Codelist
    Named mapping of codelist names to sorted integer concept ID lists.
CodelistWithDetails
    Named mapping of codelist names to DataFrames with concept_id + detail columns.
ConceptSetExpression
    Named mapping of concept set names to DataFrames with concept_id,
    excluded, descendants, mapped columns.

Functions
---------
new_codelist
    Construct and validate a Codelist from a dict.
new_codelist_with_details
    Construct and validate a CodelistWithDetails from a dict of DataFrames.
new_concept_set_expression
    Construct and validate a ConceptSetExpression from a dict of DataFrames.
empty_codelist
    Return an empty Codelist.
empty_codelist_with_details
    Return an empty CodelistWithDetails.
empty_concept_set_expression
    Return an empty ConceptSetExpression.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class Codelist:
    """Named collection of concept ID lists.

    A Codelist maps codelist names (str) to sorted, deduplicated lists of
    integer concept IDs. Mirrors the omopgenerics ``codelist`` R class.

    Parameters
    ----------
    data : dict[str, list[int]]
        Mapping of codelist names to concept ID lists.

    Examples
    --------
    >>> cl = Codelist({"diabetes": [201826, 443238], "hypertension": [316866]})
    >>> cl["diabetes"]
    [201826, 443238]
    >>> cl.to_dataframe()
       codelist_name  concept_id
    0       diabetes      201826
    1       diabetes      443238
    2   hypertension      316866
    """

    def __init__(self, data: dict[str, list[int]]) -> None:
        _validate_codelist_data(data)
        self._data: dict[str, list[int]] = {
            k: sorted({int(v) for v in vals})
            for k, vals in sorted(data.items())
        }

    def __getitem__(self, key: str) -> list[int]:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Codelist):
            return NotImplemented
        return self._data == other._data

    def keys(self) -> list[str]:
        """Return codelist names."""
        return list(self._data.keys())

    def values(self) -> list[list[int]]:
        """Return concept ID lists."""
        return list(self._data.values())

    def items(self) -> list[tuple[str, list[int]]]:
        """Return (name, concept_ids) pairs."""
        return list(self._data.items())

    def to_dataframe(self) -> Any:
        """Convert to a DataFrame with columns codelist_name, concept_id.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        rows = []
        for name, ids in self._data.items():
            for cid in ids:
                rows.append({"codelist_name": name, "concept_id": cid})
        return pd.DataFrame(rows, columns=["codelist_name", "concept_id"])

    def bind(self, *others: Codelist) -> Codelist:
        """Combine this codelist with others, merging concept IDs per name.

        Parameters
        ----------
        *others : Codelist
            Additional codelists to merge.

        Returns
        -------
        Codelist
            Combined codelist.
        """
        merged: dict[str, list[int]] = {k: list(v) for k, v in self._data.items()}
        for other in others:
            if not isinstance(other, Codelist):
                raise TypeError(f"Expected Codelist, got {type(other).__name__}")
            for k, v in other._data.items():
                if k in merged:
                    merged[k] = list(set(merged[k]) | set(v))
                else:
                    merged[k] = list(v)
        return Codelist(merged)

    def __repr__(self) -> str:
        n = len(self._data)
        total = sum(len(v) for v in self._data.values())
        names = ", ".join(list(self._data.keys())[:5])
        if n > 5:
            names += ", ..."
        return f"Codelist({n} codelists, {total} concepts: {names})"


class CodelistWithDetails:
    """Named collection of concept ID DataFrames with additional detail columns.

    Each entry maps a codelist name to a DataFrame containing at minimum a
    ``concept_id`` column, plus any additional metadata columns (e.g.
    concept_name, vocabulary_id, domain_id).

    Parameters
    ----------
    data : dict[str, pandas.DataFrame]
        Mapping of codelist names to DataFrames with concept_id column.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        import pandas as pd

        _validate_codelist_with_details_data(data)
        self._data: dict[str, pd.DataFrame] = {}
        for k in sorted(data.keys()):
            df = pd.DataFrame(data[k]) if not isinstance(data[k], pd.DataFrame) else data[k].copy()
            df["concept_id"] = df["concept_id"].astype(int)
            # Drop columns that are all NA
            df = df.dropna(axis=1, how="all")
            df = df.sort_values("concept_id").drop_duplicates(subset=["concept_id"]).reset_index(
                drop=True
            )
            self._data[k] = df

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodelistWithDetails):
            return NotImplemented
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        return all(self._data[k].equals(other._data[k]) for k in self._data)

    def keys(self) -> list[str]:
        """Return codelist names."""
        return list(self._data.keys())

    def to_dataframe(self) -> Any:
        """Convert to a single DataFrame with codelist_name column prepended.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        frames = []
        for name, df in self._data.items():
            frame = df.copy()
            frame.insert(0, "codelist_name", name)
            frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["codelist_name", "concept_id"])
        return pd.concat(frames, ignore_index=True)

    def bind(self, *others: CodelistWithDetails) -> CodelistWithDetails:
        """Combine with other CodelistWithDetails objects.

        For duplicate names, DataFrames are concatenated and deduplicated by concept_id.

        Parameters
        ----------
        *others : CodelistWithDetails
            Additional objects to merge.

        Returns
        -------
        CodelistWithDetails
        """
        import pandas as pd

        merged: dict[str, pd.DataFrame] = {k: v.copy() for k, v in self._data.items()}
        for other in others:
            if not isinstance(other, CodelistWithDetails):
                raise TypeError(f"Expected CodelistWithDetails, got {type(other).__name__}")
            for k, v in other._data.items():
                if k in merged:
                    merged[k] = pd.concat([merged[k], v], ignore_index=True)
                else:
                    merged[k] = v.copy()
        return CodelistWithDetails(merged)

    def __repr__(self) -> str:
        n = len(self._data)
        names = ", ".join(list(self._data.keys())[:5])
        if n > 5:
            names += ", ..."
        return f"CodelistWithDetails({n} codelists: {names})"


class ConceptSetExpression:
    """Named collection of concept set definitions with inclusion/exclusion rules.

    Each entry maps a concept set name to a DataFrame with required columns:
    ``concept_id``, ``excluded``, ``descendants``, ``mapped``.

    Parameters
    ----------
    data : dict[str, pandas.DataFrame]
        Mapping of concept set names to DataFrames.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        import pandas as pd

        _validate_concept_set_expression_data(data)
        self._data: dict[str, pd.DataFrame] = {}
        for k in sorted(data.keys()):
            df = pd.DataFrame(data[k]) if not isinstance(data[k], pd.DataFrame) else data[k].copy()
            df["concept_id"] = df["concept_id"].astype(int)
            for col in ("excluded", "descendants", "mapped"):
                if col not in df.columns:
                    df[col] = False
                df[col] = df[col].astype(bool)
            df = df.sort_values("concept_id").drop_duplicates(subset=["concept_id"]).reset_index(
                drop=True
            )
            self._data[k] = df

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConceptSetExpression):
            return NotImplemented
        if set(self._data.keys()) != set(other._data.keys()):
            return False
        return all(self._data[k].equals(other._data[k]) for k in self._data)

    def keys(self) -> list[str]:
        """Return concept set names."""
        return list(self._data.keys())

    def to_dataframe(self) -> Any:
        """Convert to a single DataFrame with concept_set_name column.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        frames = []
        for name, df in self._data.items():
            frame = df.copy()
            frame.insert(0, "concept_set_name", name)
            frames.append(frame)
        if not frames:
            return pd.DataFrame(
                columns=["concept_set_name", "concept_id", "excluded", "descendants", "mapped"]
            )
        return pd.concat(frames, ignore_index=True)

    def bind(self, *others: ConceptSetExpression) -> ConceptSetExpression:
        """Combine with other ConceptSetExpression objects.

        Parameters
        ----------
        *others : ConceptSetExpression
            Additional objects to merge.

        Returns
        -------
        ConceptSetExpression
        """
        import pandas as pd

        merged: dict[str, pd.DataFrame] = {k: v.copy() for k, v in self._data.items()}
        for other in others:
            if not isinstance(other, ConceptSetExpression):
                raise TypeError(f"Expected ConceptSetExpression, got {type(other).__name__}")
            for k, v in other._data.items():
                if k in merged:
                    merged[k] = pd.concat([merged[k], v], ignore_index=True)
                else:
                    merged[k] = v.copy()
        return ConceptSetExpression(merged)

    def __repr__(self) -> str:
        n = len(self._data)
        names = ", ".join(list(self._data.keys())[:5])
        if n > 5:
            names += ", ..."
        return f"ConceptSetExpression({n} concept sets: {names})"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_codelist_data(data: dict[str, list[int]]) -> None:
    """Validate input for Codelist constructor."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")
    for k, v in data.items():
        if not isinstance(k, str):
            raise TypeError(f"Codelist names must be strings, got {type(k).__name__}")
        if not isinstance(v, (list, tuple, set)):
            raise TypeError(
                f"Codelist values must be list/tuple/set of integers, got {type(v).__name__} for '{k}'"
            )
        for item in v:
            if not isinstance(item, (int, float)):
                raise TypeError(
                    f"Concept IDs must be numeric, got {type(item).__name__} in '{k}'"
                )
            if item != item:  # NaN check
                raise ValueError(f"Concept IDs must not be NA/NaN in '{k}'")
            if isinstance(item, float) and item != int(item):
                raise ValueError(f"Concept IDs must be integers, got {item} in '{k}'")


def _validate_codelist_with_details_data(data: dict[str, Any]) -> None:
    """Validate input for CodelistWithDetails constructor."""
    import pandas as pd

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")
    for k, v in data.items():
        if not isinstance(k, str):
            raise TypeError(f"Codelist names must be strings, got {type(k).__name__}")
        df = pd.DataFrame(v) if not isinstance(v, pd.DataFrame) else v
        if "concept_id" not in df.columns:
            raise ValueError(f"Each codelist entry must have a 'concept_id' column, missing in '{k}'")
        if df["concept_id"].isna().any():
            raise ValueError(f"concept_id must not contain NA values in '{k}'")


def _validate_concept_set_expression_data(data: dict[str, Any]) -> None:
    """Validate input for ConceptSetExpression constructor."""
    import pandas as pd

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")
    for k, v in data.items():
        if not isinstance(k, str):
            raise TypeError(f"Concept set names must be strings, got {type(k).__name__}")
        df = pd.DataFrame(v) if not isinstance(v, pd.DataFrame) else v
        if "concept_id" not in df.columns:
            raise ValueError(
                f"Each concept set must have a 'concept_id' column, missing in '{k}'"
            )
        if df["concept_id"].isna().any():
            raise ValueError(f"concept_id must not contain NA values in '{k}'")


# ---------------------------------------------------------------------------
# Constructor / factory functions
# ---------------------------------------------------------------------------


def new_codelist(x: dict[str, list[int]]) -> Codelist:
    """Construct and validate a Codelist.

    Parameters
    ----------
    x : dict[str, list[int]]
        Mapping of codelist names to concept ID lists.

    Returns
    -------
    Codelist
    """
    return Codelist(x)


def new_codelist_with_details(x: dict[str, Any]) -> CodelistWithDetails:
    """Construct and validate a CodelistWithDetails.

    Parameters
    ----------
    x : dict[str, pandas.DataFrame]
        Mapping of codelist names to DataFrames with concept_id column.

    Returns
    -------
    CodelistWithDetails
    """
    return CodelistWithDetails(x)


def new_concept_set_expression(x: dict[str, Any]) -> ConceptSetExpression:
    """Construct and validate a ConceptSetExpression.

    Parameters
    ----------
    x : dict[str, pandas.DataFrame]
        Mapping of concept set names to DataFrames with concept_id column.

    Returns
    -------
    ConceptSetExpression
    """
    return ConceptSetExpression(x)


def empty_codelist() -> Codelist:
    """Return an empty Codelist.

    Returns
    -------
    Codelist
    """
    return Codelist({})


def empty_codelist_with_details() -> CodelistWithDetails:
    """Return an empty CodelistWithDetails.

    Returns
    -------
    CodelistWithDetails
    """
    return CodelistWithDetails({})


def empty_concept_set_expression() -> ConceptSetExpression:
    """Return an empty ConceptSetExpression.

    Returns
    -------
    ConceptSetExpression
    """
    return ConceptSetExpression({})
