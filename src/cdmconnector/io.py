# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Import/export functions for SummarisedResult, Codelist, and ConceptSetExpression.

Mirrors the omopgenerics import*/export* functions for serialization of
standardized analysis objects to CSV and JSON files.

Functions
---------
import_summarised_result
    Read a CSV file into a SummarisedResult.
export_summarised_result
    Write a SummarisedResult to a CSV file.
import_codelist
    Read a JSON file into a Codelist.
export_codelist
    Write a Codelist to a JSON file.
import_concept_set_expression
    Read a JSON directory/file into a ConceptSetExpression.
export_concept_set_expression
    Write a ConceptSetExpression to JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_summarised_result(
    result: Any,
    path: str | Path,
) -> Path:
    """Write a SummarisedResult to a CSV file.

    The settings table is stored as a separate CSV with ``_settings`` suffix.

    Parameters
    ----------
    result : SummarisedResult
        Result to export.
    path : str or Path
        Output CSV file path for the results table. Settings are written to
        ``<stem>_settings<suffix>``.

    Returns
    -------
    Path
        Path to the results CSV file.
    """
    from cdmconnector.characteristics import SummarisedResult

    if not isinstance(result, SummarisedResult):
        raise TypeError(f"Expected SummarisedResult, got {type(result).__name__}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    result.results.to_csv(p, index=False)

    settings_path = p.parent / f"{p.stem}_settings{p.suffix}"
    result.settings_table.to_csv(settings_path, index=False)

    return p


def import_summarised_result(path: str | Path) -> Any:
    """Read a SummarisedResult from a CSV file.

    Expects a results CSV at ``path`` and a settings CSV at
    ``<stem>_settings<suffix>`` in the same directory.

    Parameters
    ----------
    path : str or Path
        Path to the results CSV file.

    Returns
    -------
    SummarisedResult
    """
    import pandas as pd

    from cdmconnector.characteristics import SummarisedResult

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")

    results = pd.read_csv(p, dtype=str)

    settings_path = p.parent / f"{p.stem}_settings{p.suffix}"
    if settings_path.exists():
        settings = pd.read_csv(settings_path, dtype=str)
        if "result_id" in settings.columns:
            settings["result_id"] = settings["result_id"].astype(int)
    else:
        settings = None

    if "result_id" in results.columns:
        results["result_id"] = results["result_id"].astype(int)

    return SummarisedResult(results=results, settings=settings)


def export_codelist(codelist: Any, path: str | Path) -> Path:
    """Write a Codelist to a JSON file.

    Parameters
    ----------
    codelist : Codelist
        Codelist to export.
    path : str or Path
        Output JSON file path.

    Returns
    -------
    Path
        Path to the output file.
    """
    from cdmconnector.codelist import Codelist

    if not isinstance(codelist, Codelist):
        raise TypeError(f"Expected Codelist, got {type(codelist).__name__}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    data = dict(codelist.items())
    with open(p, "w") as f:
        json.dump(data, f, indent=2)

    return p


def import_codelist(path: str | Path) -> Any:
    """Read a Codelist from a JSON file.

    The JSON should be an object mapping codelist names to arrays of
    integer concept IDs.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    Codelist
    """
    from cdmconnector.codelist import Codelist

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Codelist file not found: {p}")

    with open(p) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Codelist JSON must be an object mapping names to concept ID arrays")

    return Codelist({k: [int(x) for x in v] for k, v in data.items()})


def export_concept_set_expression(
    expression: Any,
    path: str | Path,
) -> Path:
    """Write a ConceptSetExpression to a directory of JSON files.

    Each concept set is written as a separate JSON file in the output directory.

    Parameters
    ----------
    expression : ConceptSetExpression
        Concept set expression to export.
    path : str or Path
        Output directory path.

    Returns
    -------
    Path
        Path to the output directory.
    """
    from cdmconnector.codelist import ConceptSetExpression

    if not isinstance(expression, ConceptSetExpression):
        raise TypeError(f"Expected ConceptSetExpression, got {type(expression).__name__}")

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    for name, df in expression._data.items():
        items = []
        for _, row in df.iterrows():
            item: dict[str, Any] = {"concept": {"CONCEPT_ID": int(row["concept_id"])}}
            item["isExcluded"] = bool(row.get("excluded", False))
            item["includeDescendants"] = bool(row.get("descendants", False))
            item["includeMapped"] = bool(row.get("mapped", False))
            items.append(item)
        safe_name = name.replace("/", "_").replace("\\", "_")
        with open(p / f"{safe_name}.json", "w") as f:
            json.dump({"items": items}, f, indent=2)

    return p


def import_concept_set_expression(path: str | Path) -> Any:
    """Read a ConceptSetExpression from a JSON file or directory.

    Accepts either a single JSON file (with concept set name taken from filename)
    or a directory of JSON files. Each JSON follows the OHDSI/Atlas concept set
    expression format.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file or directory of JSON files.

    Returns
    -------
    ConceptSetExpression
    """
    import pandas as pd

    from cdmconnector.codelist import ConceptSetExpression

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Concept set expression path not found: {p}")

    files = list(p.glob("*.json")) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {p}")

    data: dict[str, pd.DataFrame] = {}
    for fp in files:
        name = fp.stem
        with open(fp) as f:
            raw = json.load(f)

        items = raw.get("items", raw) if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            items = [items]

        rows = []
        for item in items:
            concept = item.get("concept", item)
            concept_id = concept.get("CONCEPT_ID", concept.get("concept_id"))
            if concept_id is None:
                continue
            rows.append({
                "concept_id": int(concept_id),
                "excluded": bool(item.get("isExcluded", False)),
                "descendants": bool(item.get("includeDescendants", False)),
                "mapped": bool(item.get("includeMapped", False)),
            })

        data[name] = pd.DataFrame(
            rows,
            columns=["concept_id", "excluded", "descendants", "mapped"],
        )

    return ConceptSetExpression(data)
