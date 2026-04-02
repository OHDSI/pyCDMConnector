#!/usr/bin/env python3
"""Verify cdm.snapshot() returns a pandas DataFrame and show correct usage.

Run from repo root: python scripts/verify_snapshot_returns_dataframe.py

If you see Result or <class 'method'>, either:
  1. Call with parentheses: cdm.snapshot()  (not cdm.snapshot)
  2. Reinstall: pip install -e .  (from repo root)
"""
import sys

def main():
    import cdmconnector as cc
    from cdmconnector.cdm import cdm_from_tables
    import pandas as pd

    # Check we're using repo code
    print("cdmconnector location:", cc.__file__)
    if "site-packages" in cc.__file__ and "pyCDMConnector" not in cc.__file__:
        print("WARNING: Not using editable install. Run: pip install -e . from repo root")

    # Minimal CDM with required tables
    tables = {
        "person": pd.DataFrame({"person_id": [1], "year_of_birth": [1990], "gender_concept_id": [0], "race_concept_id": [0], "ethnicity_concept_id": [0]}),
        "observation_period": pd.DataFrame({"observation_period_id": [1], "person_id": [1], "observation_period_start_date": ["2000-01-01"], "observation_period_end_date": ["2023-12-31"], "period_type_concept_id": [0]}),
        "cdm_source": pd.DataFrame({"cdm_source_name": ["Test"], "cdm_source_abbreviation": ["TEST"], "source_description": [""], "source_documentation_reference": [""], "cdm_holder": [""], "cdm_release_date": ["2020-01-01"], "cdm_version": ["5.3"]}),
        "vocabulary": pd.DataFrame({"vocabulary_id": ["None"], "vocabulary_name": ["None"], "vocabulary_reference": [""], "vocabulary_version": ["v1"], "vocabulary_concept_id": [0]}),
    }
    cdm = cdm_from_tables(tables, cdm_name="Verify")

    # Correct: call with parentheses
    snap = cdm.snapshot()
    print("type(cdm.snapshot()):", type(snap))
    assert isinstance(snap, pd.DataFrame), f"Expected DataFrame, got {type(snap)}"
    print("cdm.snapshot() returns a pandas DataFrame (1 row):", snap.shape)
    print("Columns:", list(snap.columns)[:5], "...")
    print("OK: Use cdm.snapshot() with parentheses to get the DataFrame.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
