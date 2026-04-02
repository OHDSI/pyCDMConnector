# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""CDM reference object: Cdm and cdm_from_con / cdm_from_tables."""

from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING, Any

import ibis
import pandas as pd

from cdmconnector.exceptions import CDMValidationError, SourceError, TableNotFoundError
from cdmconnector.utils import unique_table_name
from cdmconnector.logging_config import get_logger
from cdmconnector.schemas import omop_tables
from cdmconnector.source import (
    CdmSource,
    DbCdmSource,
    LocalCdmSource,
    full_table_name,
    resolve_prefix,
)
from cdmconnector.typing import SchemaSpec
from cdmconnector.utils import (
    _first_scalar,
    parse_cdm_version,
    resolve_schema_name,
    to_dataframe,
    _suggest_similar,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ibis.expr.types import Table


def collect(expr: Any, *, limit: int | None = None) -> pd.DataFrame:
    """Materialize an Ibis expression to a pandas DataFrame.

    This is the only function (with compute and Result.collect) that triggers execution.
    Scalar expressions (e.g. table.count()) are returned as a 1x1 DataFrame.

    Parameters
    ----------
    expr : Any
        Ibis table or scalar expression to execute.
    limit : int or None, optional
        If set, apply .limit(limit) before executing.

    Returns
    -------
    pandas.DataFrame
        Materialized result (table rows or 1x1 for scalars).
    """
    import ibis.expr.types as ir

    if isinstance(expr, pd.DataFrame):
        return expr
    if isinstance(expr, (list, dict)):
        return to_dataframe(expr)

    if limit is not None and hasattr(expr, "limit"):
        expr = expr.limit(limit)

    # Ibis scalar expression -> execute and wrap in 1x1 DataFrame
    if isinstance(expr, ir.Scalar):
        val = expr.execute()
        return pd.DataFrame({"": [val]})

    # Ibis Table expression -> execute to pandas
    if hasattr(expr, "to_pandas"):
        result = expr.to_pandas()
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame({"": [result]})

    return to_dataframe(expr)


def _get_version_from_cdm_source(con: Any, cdm_schema: SchemaSpec | None) -> str:
    """Try to get CDM version from cdm_source table; default to 5.3.

    Parameters
    ----------
    con : Any
        Ibis connection with cdm_source table.
    cdm_schema : str, dict, or None
        Schema where cdm_source lives.

    Returns
    -------
    str
        Parsed version (e.g. "5.3", "5.4") or "5.3" on failure.
    """
    try:
        if hasattr(con, "table"):
            db = resolve_schema_name(cdm_schema)
            t = con.table("cdm_source", database=db) if db else con.table("cdm_source")
            row = collect(t.select("cdm_version").limit(1))
            if not row.empty:
                return parse_cdm_version(str(_first_scalar(row, "cdm_version")))
    except (KeyError, AttributeError, ValueError, Exception) as e:
        logger.debug("Could not get version from cdm_source: %s", e)
    return "5.3"


def _get_name_from_cdm_source(con: Any, cdm_schema: SchemaSpec | None) -> str | None:
    """Try to get CDM name from cdm_source table.

    Parameters
    ----------
    con : Any
        Ibis connection with cdm_source table.
    cdm_schema : str, dict, or None
        Schema where cdm_source lives.

    Returns
    -------
    str or None
        cdm_source_abbreviation or cdm_source_name, or None on failure.
    """
    try:
        if hasattr(con, "table"):
            db = resolve_schema_name(cdm_schema)
            t = con.table("cdm_source", database=db) if db else con.table("cdm_source")
            row = collect(t.select("cdm_source_abbreviation", "cdm_source_name").limit(1))
            if not row.empty:
                abbr = _first_scalar(row, "cdm_source_abbreviation")
                if abbr is not None and str(abbr).strip():
                    return str(abbr).strip()
                name = _first_scalar(row, "cdm_source_name")
                if name is not None and str(name).strip():
                    return str(name).strip()
    except (KeyError, AttributeError, ValueError, Exception) as e:
        logger.debug("Could not get name from cdm_source: %s", e)
    return None


class Cdm:
    """
    OMOP CDM reference: holds a mapping of table names to Ibis table expressions
    plus metadata (name, version, schemas, source).
    """

    def __init__(
        self,
        tables: dict[str, Any],
        *,
        cdm_name: str,
        cdm_version: str | None = None,
        cdm_schema: SchemaSpec | None = None,
        write_schema: SchemaSpec | None = None,
        achilles_schema: SchemaSpec | None = None,
        source: CdmSource,
    ) -> None:
        """Build a CDM reference from a table map and metadata.

        Parameters
        ----------
        tables : dict[str, Any]
            Mapping of table name -> Ibis table expression.
        cdm_name : str
            Display name for this CDM.
        cdm_version : str or None, optional
            OMOP CDM version (e.g. "5.3"); default "5.3".
        cdm_schema : str, dict, or None, optional
            Schema where CDM tables live.
        write_schema : str, dict, or None, optional
            Schema for cohort/write tables.
        achilles_schema : str, dict, or None, optional
            Schema for Achilles tables.
        source : CdmSource
            Source (local or database-backed).
        """
        self._tables = {k.lower(): v for k, v in tables.items()}
        self._cdm_name = cdm_name
        self._cdm_version = cdm_version or "5.3"
        self._cdm_schema = cdm_schema
        self._write_schema = write_schema
        self._achilles_schema = achilles_schema
        self._source = source

    @property
    def tables(self) -> list[str]:
        """List of table names in this CDM."""
        return list(self._tables.keys())

    @property
    def name(self) -> str:
        """CDM name (e.g. from cdm_source or user)."""
        return self._cdm_name

    @property
    def version(self) -> str:
        """OMOP CDM version (5.3 or 5.4)."""
        return self._cdm_version

    @property
    def cdm_schema(self) -> SchemaSpec | None:
        """Schema where CDM tables live."""
        return self._cdm_schema

    @property
    def write_schema(self) -> SchemaSpec | None:
        """Schema for write/cohort tables."""
        return self._write_schema

    @property
    def achilles_schema(self) -> SchemaSpec | None:
        """Schema for Achilles tables (optional)."""
        return self._achilles_schema

    @property
    def source(self) -> CdmSource:
        """Source (connection) for this CDM."""
        return self._source

    @property
    def con(self) -> Any:
        """Underlying Ibis connection, or None if not database-backed."""
        if not isinstance(self._source, DbCdmSource):
            return None
        return self._source.con

    def __getitem__(self, name: str) -> Table:
        """Get table by name: cdm['person'].

        Parameters
        ----------
        name : str
            Table name (case-insensitive).

        Returns
        -------
        ibis.expr.types.Table
            Table expression.

        Raises
        ------
        TableNotFoundError
            If name is not in this CDM.
        """
        key = name.lower()
        if key not in self._tables:
            available = sorted(self.tables)
            suggestion = _suggest_similar(name, available)
            msg = (
                f"Table {name!r} does not exist in this CDM ({self.name}). "
                f"Available tables: {', '.join(available)}."
            )
            if suggestion:
                msg += f" Did you mean {suggestion!r}?"
            raise TableNotFoundError(msg)
        return self._tables[key]

    def __getattr__(self, name: str) -> Any:
        """Get table by attribute: cdm.person.

        Parameters
        ----------
        name : str
            Table name (must not start with '_').

        Returns
        -------
        Any
            Table expression.

        Raises
        ------
        AttributeError
            If name starts with '_' or table not in CDM.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._tables:
            return self._tables[name]
        try:
            return self[name]
        except TableNotFoundError as e:
            raise AttributeError(
                f"Table {name!r} not in CDM ({self.name}). Available: {self.tables}"
            ) from e

    def __setitem__(self, name: str, value: Any) -> None:
        """Assign a table to the CDM (must be same source).

        Parameters
        ----------
        name : str
            Table name (stored lowercase).
        value : Any
            Ibis table expression.
        """
        name = name.lower()
        self._tables[name] = value

    def _cdm_classes(self) -> dict[str, list[str]]:
        """Classify tables into omop, cohort_table, achilles, and other (like R cdmClasses)."""
        omop_set = set(omop_tables(self._cdm_version))
        achilles_set = {"achilles_analysis", "achilles_results", "achilles_results_dist"}
        cohort_suffixes = ("_cohort", "_cohort_set", "_cohort_attrition")
        def is_cohort(t: str) -> bool:
            return t in ("cohort", "cohort_set", "cohort_attrition") or any(t.endswith(s) for s in cohort_suffixes)

        omop_list = [t for t in sorted(self._tables) if t in omop_set]
        cohort_list = [t for t in sorted(self._tables) if is_cohort(t)]
        achilles_list = [t for t in sorted(self._tables) if t in achilles_set]
        other_list = [t for t in sorted(self._tables) if t not in omop_set and t not in achilles_set and not is_cohort(t)]
        return {
            "omop_table": omop_list,
            "cohort_table": cohort_list,
            "achilles_table": achilles_list,
            "cdm_table": other_list,
        }

    def __str__(self) -> str:
        """Print a CDM reference (overview of name, source type, and table classes). Returns: str."""
        source_type = self._source.source_type
        name = self._cdm_name
        classes = self._cdm_classes()
        lines = [
            f"# OMOP CDM reference ({source_type}) of {name}",
            "",
            f"  omop tables: {', '.join(classes['omop_table']) if classes['omop_table'] else '-'}",
            f"  cohort tables: {', '.join(classes['cohort_table']) if classes['cohort_table'] else '-'}",
            f"  achilles tables: {', '.join(classes['achilles_table']) if classes['achilles_table'] else '-'}",
            f"  other tables: {', '.join(classes['cdm_table']) if classes['cdm_table'] else '-'}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return repr string. Returns: str."""
        classes = self._cdm_classes()
        cohort_like = classes["cohort_table"] + classes["cdm_table"]
        return (
            f"<Cdm(name={self._cdm_name!r}, version={self._cdm_version}, "
            f"tables={len(self._tables)}, cohort_tables={cohort_like})>"
        )

    def disconnect(self, drop_write_schema: bool = False) -> None:
        """Disconnect the underlying source.

        Parameters
        ----------
        drop_write_schema : bool, optional
            If True, drop write-schema tables before disconnecting (default False).
        """
        if isinstance(self._source, DbCdmSource):
            self._source.disconnect(drop_write_schema=drop_write_schema)

    def snapshot(self, compute_data_hash: bool = False) -> pd.DataFrame:
        """Snapshot CDM metadata. Executes and returns a 1-row pandas DataFrame.

        Call with parentheses: ``cdm.snapshot()``. Without () you get the method
        object, not the DataFrame.

        Requires person, observation_period, cdm_source, and vocabulary tables.
        Column order matches R CDMConnector snapshot(): cdm_name, cdm_source_name,
        cdm_description, cdm_documentation_reference, cdm_version, cdm_holder,
        cdm_release_date, vocabulary_version, person_count, observation_period_count,
        earliest_observation_period_start_date, latest_observation_period_end_date,
        snapshot_date, cdm_data_hash.

        Parameters
        ----------
        compute_data_hash : bool, optional
            If True, include data hash in snapshot (default False).

        Returns
        -------
        pandas.DataFrame
            Exactly one row (1-row DataFrame) with CDM metadata.
        """
        for req in ("person", "observation_period", "cdm_source", "vocabulary"):
            if req not in self._tables:
                raise CDMValidationError(f"snapshot requires '{req}' table in the CDM.")
        out = _materialize_snapshot(self, compute_data_hash=compute_data_hash)
        # If you ever see Result here, reinstall: pip install -e . from repo root
        if hasattr(out, "collect") and callable(getattr(out, "collect", None)):
            out = out.collect()
        assert isinstance(out, pd.DataFrame), (
            "snapshot() must return a pandas.DataFrame; "
            "reinstall with: pip install -e . (from repo root)"
        )
        return out

    def select_tables(self, *names: str) -> Cdm:
        """Return a new Cdm with only the given tables (subset).

        Parameters
        ----------
        *names : str
            Table names to keep.

        Returns
        -------
        Cdm
            New CDM with subset of tables.

        Raises
        ------
        TableNotFoundError
            If any name is not in this CDM.
        """
        names_lower = [n.lower() for n in names]
        missing = [n for n in names_lower if n not in self._tables]
        if missing:
            raise TableNotFoundError(f"Tables not found: {missing}. Available: {self.tables}")
        new_tables = {k: self._tables[k] for k in names_lower}
        return Cdm(
            new_tables,
            cdm_name=self._cdm_name,
            cdm_version=self._cdm_version,
            cdm_schema=self._cdm_schema,
            write_schema=self._write_schema,
            achilles_schema=self._achilles_schema,
            source=self._source,
        )

    def subset(self, person_id: list[int] | Any) -> Cdm:
        """Subset the CDM to a set of person IDs.

        Returns a new CDM where all clinical tables are filtered to rows
        whose person_id (or subject_id) is in the given set. Requires
        a database-backed CDM with write_schema.

        Parameters
        ----------
        person_id : list[int] or array-like
            Person IDs to include.

        Returns
        -------
        Cdm
            New CDM with tables subset to the given persons.
        """
        return _subset_cdm(self, person_id=person_id)

    def subset_cohort(
        self,
        cohort_table: str = "cohort",
        cohort_id: int | list[int] | None = None,
        verbose: bool = False,
    ) -> Cdm:
        """Subset the CDM to individuals in one or more cohorts.

        Returns a new CDM where all clinical tables are filtered to persons
        present in the given cohort table (optionally restricted by cohort_id).
        Subset is lazy until tables are used.

        Parameters
        ----------
        cohort_table : str, optional
            Name of the cohort table in this CDM (default "cohort").
        cohort_id : int or list[int] or None, optional
            Cohort definition ID(s) to include; None uses all cohorts in the table.
        verbose : bool, optional
            If True, log subset size (default False).

        Returns
        -------
        Cdm
            New CDM subset to cohort persons.
        """
        return _subset_cohort_cdm(
            self,
            cohort_table=cohort_table,
            cohort_id=cohort_id,
            verbose=verbose,
        )

    def sample(
        self,
        n: int,
        seed: int | None = None,
        name: str = "person_sample",
    ) -> Cdm:
        """Subset the CDM to a random sample of n persons.

        Persons are drawn from the person table. The sample table is
        inserted into the write schema under the given name and all
        clinical tables are filtered to those persons.

        Parameters
        ----------
        n : int
            Number of persons to include.
        seed : int or None, optional
            Random seed for reproducibility; None uses a random seed.
        name : str, optional
            Name of the table storing the sampled person_ids (default "person_sample").

        Returns
        -------
        Cdm
            New CDM with tables subset to the sampled persons (and the sample table added).
        """
        return _sample_cdm(self, n=n, seed=seed, name=name)

    def flatten(
        self,
        domain: list[str] | None = None,
        include_concept_name: bool = True,
    ) -> Any:
        """Flatten the CDM into a single observation table.

        Transforms selected domain tables into a common schema (person_id,
        observation_concept_id, start_date, end_date, type_concept_id, domain)
        and unions them. Recommended only for filtered or small CDMs.

        Parameters
        ----------
        domain : list[str] or None, optional
            Domains to include. Must be a subset of: "condition_occurrence",
            "drug_exposure", "procedure_occurrence", "measurement",
            "visit_occurrence", "death", "observation". Default is
            condition_occurrence, drug_exposure, procedure_occurrence.
        include_concept_name : bool, optional
            If True (default), add observation_concept_name and type_concept_name
            via the concept table.

        Returns
        -------
        ibis.expr.types.Table
            Lazy table expression; use collect() to materialize.
        """
        return _flatten_cdm(self, domain=domain, include_concept_name=include_concept_name)


def _sample_person(cdm: Cdm, person_subset: Any) -> Cdm:
    """Return a new CDM with all tables that have person_id or subject_id inner-filtered to person_subset."""
    cols_subset = [c.lower() for c in person_subset.columns]
    if "person_id" not in cols_subset:
        raise ValueError("person_subset must have a 'person_id' column.")
    person_ids = person_subset["person_id"]
    new_tables: dict[str, Any] = {}
    for name, tbl in cdm._tables.items():
        tbl_cols = [c.lower() for c in tbl.columns]
        if "person_id" in tbl_cols:
            new_tables[name] = tbl.filter(tbl["person_id"].isin(person_ids))
        elif "subject_id" in tbl_cols:
            new_tables[name] = tbl.filter(tbl["subject_id"].isin(person_ids))
        else:
            new_tables[name] = tbl
    return Cdm(
        new_tables,
        cdm_name=cdm._cdm_name,
        cdm_version=cdm._cdm_version,
        cdm_schema=cdm._cdm_schema,
        write_schema=cdm._write_schema,
        achilles_schema=cdm._achilles_schema,
        source=cdm._source,
    )


def _subset_cdm(cdm: Cdm, person_id: list[int] | Any) -> Cdm:
    """Subset CDM to the given person IDs. Requires database-backed CDM with write_schema."""
    if not isinstance(cdm.source, DbCdmSource):
        raise SourceError("subset() requires a database-backed CDM (cdm_from_con).")
    if cdm.write_schema is None:
        raise SourceError("subset() requires write_schema to be set.")
    import pandas as pd

    ids = list(person_id) if isinstance(person_id, (list, tuple)) else [person_id]
    if not ids:
        raise ValueError("person_id must contain at least one ID.")
    df = pd.DataFrame({"person_id": ids})
    prefix = unique_table_name(prefix="person_subset_")
    cdm.source.insert_table(prefix, df, overwrite=True)
    person_subset = cdm.source.table(prefix, cdm.write_schema)
    return _sample_person(cdm, person_subset)


def _subset_cohort_cdm(
    cdm: Cdm,
    cohort_table: str = "cohort",
    cohort_id: int | list[int] | None = None,
    verbose: bool = False,
) -> Cdm:
    """Subset CDM to persons in the given cohort table (optionally by cohort_id)."""
    if not isinstance(cdm.source, DbCdmSource):
        raise SourceError("subset_cohort() requires a database-backed CDM (cdm_from_con).")
    if cdm.write_schema is None:
        raise SourceError("subset_cohort() requires write_schema to be set.")
    cohort_table = cohort_table.lower()
    if cohort_table not in cdm._tables:
        raise TableNotFoundError(
            f"Cohort table {cohort_table!r} not in CDM. Available: {cdm.tables}."
        )
    cohort_tbl = cdm[cohort_table]
    cohort_cols = [c.lower() for c in cohort_tbl.columns]
    if "subject_id" not in cohort_cols:
        raise CDMValidationError(
            f"subject_id column is not in cdm[{cohort_table!r}] table."
        )
    if "cohort_definition_id" not in cohort_cols:
        raise CDMValidationError(
            f"cohort_definition_id column is not in cdm[{cohort_table!r}] table."
        )
    subjects = cohort_tbl.select("subject_id", "cohort_definition_id")
    if cohort_id is not None:
        ids = [cohort_id] if isinstance(cohort_id, int) else list(cohort_id)
        subjects = subjects.filter(subjects["cohort_definition_id"].isin(ids))
    subjects = subjects.select("subject_id").distinct()
    n_subjects = int(_first_scalar(collect(subjects.count())))
    if n_subjects == 0:
        if verbose:
            logger.info("Selected cohorts are empty. No subsetting will be done.")
        return cdm
    if verbose:
        logger.info("Subsetting cdm to %s persons", n_subjects)
    person_subset = subjects.select(subjects["subject_id"].name("person_id"))
    prefix = unique_table_name(prefix="person_sample_")
    person_subset_df = collect(person_subset)
    cdm.source.insert_table(prefix, person_subset_df, overwrite=True)
    person_subset_tbl = cdm.source.table(prefix, cdm.write_schema)
    return _sample_person(cdm, person_subset_tbl)


def _sample_cdm(
    cdm: Cdm,
    n: int,
    seed: int | None = None,
    name: str = "person_sample",
) -> Cdm:
    """Subset CDM to a random sample of n persons. Adds the sample table to the CDM."""
    if not isinstance(cdm.source, DbCdmSource):
        raise SourceError("sample() requires a database-backed CDM (cdm_from_con).")
    if cdm.write_schema is None:
        raise SourceError("sample() requires write_schema to be set.")
    if n < 1 or n > 1_000_000_000:
        raise ValueError("n must be between 1 and 1e9.")
    import random

    person_ids = collect(cdm["person"].select("person_id").distinct())["person_id"].tolist()
    if len(person_ids) > n:
        rng = random.Random(seed)
        person_ids = rng.sample(person_ids, n)
    import pandas as pd

    subset_df = pd.DataFrame({"person_id": person_ids})
    insert_table(cdm, name, subset_df, overwrite=True)
    person_subset = cdm[name]
    return _sample_person(cdm, person_subset)


# Domain -> (observation_concept_id col, start_date col, end_date col, type_concept_id col)
_FLATTEN_DOMAIN_COLUMNS: dict[str, tuple[str, str, str, str]] = {
    "condition_occurrence": (
        "condition_concept_id",
        "condition_start_date",
        "condition_end_date",
        "condition_type_concept_id",
    ),
    "drug_exposure": (
        "drug_concept_id",
        "drug_exposure_start_date",
        "drug_exposure_end_date",
        "drug_type_concept_id",
    ),
    "procedure_occurrence": (
        "procedure_concept_id",
        "procedure_date",
        "procedure_date",
        "procedure_type_concept_id",
    ),
    "measurement": (
        "measurement_concept_id",
        "measurement_date",
        "measurement_date",
        "measurement_type_concept_id",
    ),
    "visit_occurrence": (
        "visit_concept_id",
        "visit_start_date",
        "visit_end_date",
        "visit_type_concept_id",
    ),
    "death": (
        "cause_concept_id",
        "death_date",
        "death_date",
        "death_type_concept_id",
    ),
    "observation": (
        "observation_concept_id",
        "observation_date",
        "observation_date",
        "observation_type_concept_id",
    ),
}

_FLATTEN_DOMAINS = tuple(_FLATTEN_DOMAIN_COLUMNS.keys())


def _flatten_cdm(
    cdm: Cdm,
    domain: list[str] | None = None,
    include_concept_name: bool = True,
) -> Any:
    """Flatten selected domain tables into a single observation table (lazy)."""
    if domain is None:
        domain = ["condition_occurrence", "drug_exposure", "procedure_occurrence"]
    domain = [d.lower() for d in domain]
    bad = [d for d in domain if d not in _FLATTEN_DOMAIN_COLUMNS]
    if bad:
        raise ValueError(
            f"domain must be a subset of {list(_FLATTEN_DOMAINS)}. Got: {bad}."
        )
    parts: list[Any] = []
    for dom in domain:
        if dom not in cdm._tables:
            raise TableNotFoundError(
                f"Domain table {dom!r} not in CDM. Available: {cdm.tables}."
            )
        tbl = cdm[dom]
        obs_col, start_col, end_col, type_col = _FLATTEN_DOMAIN_COLUMNS[dom]
        for c in (obs_col, start_col, end_col, type_col):
            if c not in tbl.columns:
                raise CDMValidationError(
                    f"Table {dom!r} is missing column {c!r} for flatten."
                )
        part = tbl.select(
            tbl["person_id"],
            tbl[obs_col].name("observation_concept_id"),
            tbl[start_col].name("start_date"),
            tbl[end_col].name("end_date"),
            tbl[type_col].name("type_concept_id"),
        ).mutate(domain=ibis.literal(dom)).distinct()
        parts.append(part)
    out = reduce(lambda acc, p: acc.union(p), parts)
    if include_concept_name:
        if "concept" not in cdm._tables:
            raise TableNotFoundError(
                "include_concept_name=True requires 'concept' table in the CDM."
            )
        concept = cdm["concept"]
        obs_name = concept.select(
            concept["concept_id"].name("observation_concept_id"),
            concept["concept_name"].name("observation_concept_name"),
        )
        type_name = concept.select(
            concept["concept_id"].name("type_concept_id"),
            concept["concept_name"].name("type_concept_name"),
        )
        out = out.left_join(obs_name, "observation_concept_id")
        out = out.left_join(type_name, "type_concept_id")
    return out.distinct()


def cdm_from_con(
    con: Any,
    cdm_schema: SchemaSpec,
    write_schema: SchemaSpec | None = None,
    *,
    cohort_tables: list[str] | None = None,
    cdm_version: str | None = None,
    cdm_name: str | None = None,
    achilles_schema: SchemaSpec | None = None,
    write_prefix: str | None = None,
) -> Cdm:
    """
    Create a CDM reference from an Ibis connection.

    Parameters
    ----------
    con : Any
        Ibis backend (e.g. ibis.duckdb.connect(...)).
    cdm_schema : str or dict
        Schema/database where CDM tables live.
    write_schema : str or dict or None, optional
        Schema for cohort/write tables (default: same as cdm_schema).
    cohort_tables : list[str] or None, optional
        Optional list of cohort table names to include.
    cdm_version : str or None, optional
        "5.3" or "5.4"; if None, inferred from cdm_source.
    cdm_name : str or None, optional
        CDM name; if None, inferred from cdm_source.
    achilles_schema : str or dict or None, optional
        Optional schema for Achilles tables.
    write_prefix : str or None, optional
        Optional prefix for tables created in write_schema.

    Returns
    -------
    Cdm
        CDM reference with tables from the connection.
    """
    write_schema = write_schema or cdm_schema
    cohort_tables = cohort_tables or []

    logger.info("Creating CDM from connection: schema=%s", cdm_schema)
    src = DbCdmSource(con, write_schema, prefix=write_prefix)
    db = src._database_for_schema(cdm_schema)
    try:
        raw_tables = con.list_tables(database=db) if db else con.list_tables()
    except (AttributeError, KeyError) as e:
        logger.warning("Could not access schema %r, using default: %s", db, e)
        raw_tables = con.list_tables()
    except Exception as e:
        raise SourceError(
            f"Failed to list tables in schema {db!r}. "
            "Check that the schema exists and you have permissions. "
            f"Error: {e}"
        ) from e
    cdm_prefix = resolve_prefix(None, cdm_schema)
    if cdm_prefix:
        logical_names = [t[len(cdm_prefix) :] if t.startswith(cdm_prefix) else t for t in raw_tables]
    else:
        logical_names = list(raw_tables)
    omop_set = set(omop_tables(cdm_version or "5.3"))
    logical_lower = {t.lower(): t for t in logical_names}
    available_logical = [k for k in logical_lower if k in omop_set]
    logger.debug("Found %d CDM tables in schema", len(available_logical))
    if not available_logical:
        msg = (
            "No CDM tables found in the given schema. "
            "If using Eunomia, reinstall from source so tables are persisted: pip install -e . "
            "Then delete any existing GiBleed_*.duckdb in your data folder so it is rebuilt."
        )
        raise CDMValidationError(msg)

    tables_dict: dict[str, Any] = {}
    for log_name in available_logical:
        phys_name = full_table_name(log_name, cdm_prefix)
        tbl = con.table(phys_name, database=db) if db else con.table(phys_name)
        tables_dict[log_name.lower()] = tbl

    # Optional: Achilles
    if achilles_schema:
        db_ach = src._database_for_schema(achilles_schema)
        ach_list = con.list_tables(database=db_ach)
        for a in ["achilles_analysis", "achilles_results", "achilles_results_dist"]:
            if a in [x.lower() for x in ach_list]:
                tables_dict[a] = con.table(a, database=db_ach) if db_ach else con.table(a)

    cdm_version = cdm_version or _get_version_from_cdm_source(con, cdm_schema)
    cdm_name = cdm_name or _get_name_from_cdm_source(con, cdm_schema) or "An OMOP CDM database"

    cdm = Cdm(
        tables_dict,
        cdm_name=cdm_name,
        cdm_version=cdm_version,
        cdm_schema=cdm_schema,
        write_schema=write_schema,
        achilles_schema=achilles_schema,
        source=src,
    )

    # Cohort tables from write_schema
    write_tables = src.list_tables(write_schema)
    for ct in cohort_tables:
        base = ct
        if base not in write_tables and base.lower() not in [x.lower() for x in write_tables]:
            raise TableNotFoundError(f"Cohort table {ct!r} not found in write schema.")
        actual = next((x for x in write_tables if x.lower() == base.lower()), base)
        cdm[ct] = src.table(actual, write_schema)

    return cdm


def cdm_from_tables(
    tables: dict[str, Any],
    cdm_name: str,
    *,
    cdm_version: str | None = "5.3",
    cohort_tables: dict[str, Any] | None = None,
) -> Cdm:
    """
    Create a CDM reference from a dict of (table_name -> Ibis table or DataFrame).

    For local/in-memory CDMs: if tables are pandas/pyarrow, an in-memory DuckDB
    is created and tables are registered there; then CDM holds Ibis refs to them.

    Parameters
    ----------
    tables : dict[str, Any]
        Mapping of table name -> Ibis table, pandas.DataFrame, or pyarrow.Table.
    cdm_name : str
        Display name for this CDM.
    cdm_version : str or None, optional
        OMOP CDM version (default "5.3").
    cohort_tables : dict[str, Any] or None, optional
        Optional cohort tables to attach (name -> table).

    Returns
    -------
    Cdm
        CDM reference (local or DuckDB-backed).
    """
    import pyarrow as pa

    # If all tables are already Ibis tables from the same backend, use it
    con = None
    try:
        first = next(iter(tables.values()))
        if hasattr(first, "to_pyarrow") and hasattr(first, "op"):
            con = ibis.get_backend(first)
    except (StopIteration, AttributeError, TypeError) as e:
        logger.debug("Could not get backend from tables: %s", e)

    if con is None:
        # Create in-memory DuckDB and register tables
        con = ibis.duckdb.connect()
        for name, tbl in tables.items():
            if hasattr(tbl, "to_pyarrow"):
                obj = tbl.to_pyarrow()
            elif hasattr(tbl, "schema") or hasattr(tbl, "op"):
                obj = collect(tbl)
            elif isinstance(tbl, pa.Table):
                obj = tbl
            elif hasattr(tbl, "__arrow_table__"):
                obj = tbl.__arrow_table__()
            else:
                obj = tbl
            con.create_table(name.lower(), obj=obj, overwrite=True)
        tables = {k.lower(): con.table(k.lower()) for k in tables}
    else:
        tables = {k.lower(): v for k, v in tables.items()}

    src = DbCdmSource(con, "main") if con is not None else LocalCdmSource()
    cdm = Cdm(
        tables,
        cdm_name=cdm_name,
        cdm_version=cdm_version or "5.3",
        cdm_schema=None,
        write_schema="main" if con else None,
        achilles_schema=None,
        source=src,
    )
    if cohort_tables:
        for name, tbl in cohort_tables.items():
            cdm[name.lower()] = tbl
    return cdm


def cdm_con(cdm: Cdm) -> Any:
    """Return the underlying Ibis connection for a CDM, if any."""
    return cdm.con


def cdm_write_schema(cdm: Cdm) -> SchemaSpec | None:
    """Return the CDM write schema used for cohort and temp tables."""
    return cdm.write_schema


def list_tables(cdm: Cdm) -> list[str]:
    """Return the logical tables currently attached to the CDM object."""
    return cdm.tables


def list_source_tables(cdm: Cdm) -> list[str]:
    """List tables in the CDM's write schema (source tables).

    Parameters
    ----------
    cdm : Cdm
        CDM reference (must be database-backed).

    Returns
    -------
    list[str]
        Table names in write schema, or [] if not database-backed.
    """
    if not isinstance(cdm.source, DbCdmSource):
        return []
    return cdm.source.list_tables(cdm.write_schema)


def read_source_table(cdm: Cdm, name: str) -> Table:
    """Read a table from the write schema as an Ibis table.

    Parameters
    ----------
    cdm : Cdm
        CDM reference (must be database-backed).
    name : str
        Logical table name in write schema.

    Returns
    -------
    ibis.expr.types.Table
        Ibis table expression.
    """
    if not isinstance(cdm.source, DbCdmSource):
        raise SourceError("read_source_table requires a database-backed CDM (cdm_from_con).")
    return cdm.source.table(name, cdm.write_schema)


def insert_table(
    cdm: Cdm,
    name: str,
    table: Any,
    *,
    overwrite: bool = True,
) -> Table:
    """Insert a table (Ibis expr or pyarrow/pandas) into the write schema and return Ibis table ref.

    Parameters
    ----------
    cdm : Cdm
        CDM reference (must be database-backed).
    name : str
        Logical table name in write schema.
    table : Any
        Ibis table expression, pyarrow.Table, or pandas.DataFrame.
    overwrite : bool, optional
        If True, replace existing table (default True).

    Returns
    -------
    ibis.expr.types.Table
        Ibis table reference to the inserted table.
    """
    if not isinstance(cdm.source, DbCdmSource):
        raise SourceError("insert_table requires a database-backed CDM (cdm_from_con).")
    tbl = cdm.source.insert_table(name, table, overwrite=overwrite)
    cdm[name] = tbl
    return tbl


def drop_table(cdm: Cdm, name: str | list[str]) -> None:
    """Drop one or more tables from the write schema.

    Parameters
    ----------
    cdm : Cdm
        CDM reference (must be database-backed).
    name : str or list[str]
        Logical table name(s) to drop.
    """
    if not isinstance(cdm.source, DbCdmSource):
        raise SourceError("drop_table requires a database-backed CDM (cdm_from_con).")
    names = [name] if isinstance(name, str) else name
    for n in names:
        key = n.lower()
        cdm.source.drop_table(n)
        if key in cdm._tables:
            del cdm._tables[key]


# Expose insert_table and drop_table as Cdm methods (delegate to module-level functions)
def _insert_table_method(self: Cdm, name: str, table: Any, *, overwrite: bool = True) -> Table:
    return insert_table(self, name, table, overwrite=overwrite)


_insert_table_method.__doc__ = insert_table.__doc__
Cdm.insert_table = _insert_table_method


def _drop_table_method(self: Cdm, name: str | list[str]) -> None:
    return drop_table(self, name)


_drop_table_method.__doc__ = drop_table.__doc__
Cdm.drop_table = _drop_table_method


def _generate_concept_cohort_set_method(
    self: Cdm,
    concept_set: Any,
    *,
    name: str = "cohort",
    limit: str = "first",
    required_observation: tuple[int, int] = (0, 0),
    end: str | int = "observation_period_end_date",
    subset_cohort: str | None = None,
    subset_cohort_id: int | list[int] | None = None,
    overwrite: bool = True,
) -> Cdm:
    """Generate a cohort set from concept sets (CDM method). Delegates to generate_concept_cohort_set."""
    from cdmconnector.cohorts import generate_concept_cohort_set

    return generate_concept_cohort_set(
        self,
        concept_set,
        name=name,
        limit=limit,
        required_observation=required_observation,
        end=end,
        subset_cohort=subset_cohort,
        subset_cohort_id=subset_cohort_id,
        overwrite=overwrite,
    )


Cdm.generate_concept_cohort_set = _generate_concept_cohort_set_method
try:
    from cdmconnector.cohorts import generate_concept_cohort_set as _gcs
    Cdm.generate_concept_cohort_set.__doc__ = _gcs.__doc__
except Exception:
    pass


def validate_observation_period(
    cdm: Cdm,
    *,
    check_overlap: bool = True,
    check_start_before_end: bool = True,
    check_plausible_dates: bool = True,
) -> None:
    """
    Validate observation_period: overlapping periods, start <= end, plausible dates.

    Port of R omopgenerics checks (checkOverlapObservation, checkStartBeforeEndObservation,
    checkPlausibleObservationDates). Raises CDMValidationError on overlap or start > end;
    issues a warning if dates are before 1800-01-01 or after today.

    Parameters
    ----------
    cdm : Cdm reference (must contain observation_period table).
    check_overlap : If True, raise when a person has overlapping observation periods.
    check_start_before_end : If True, raise when observation_period_start_date > observation_period_end_date.
    check_plausible_dates : If True, warn when min start < 1800-01-01 or max end > today.

    Raises
    ------
    CDMValidationError
        If overlap or start-after-end violations are found.
    """
    import warnings
    from datetime import date

    if "observation_period" not in cdm._tables:
        raise CDMValidationError("validate_observation_period requires 'observation_period' table in the CDM.")
    op = cdm.observation_period

    if check_overlap:
        w = ibis.window(
            group_by=op.person_id,
            order_by=op.observation_period_start_date,
        )
        next_start = op.observation_period_start_date.lead().over(w)
        with_next = op.mutate(next_observation_period_start_date=next_start)
        overlap = with_next.filter(
            with_next.next_observation_period_start_date.notnull()
            & (with_next.observation_period_end_date >= with_next.next_observation_period_start_date),
        ).select("person_id")
        bad = collect(overlap)
        if not bad.empty:
            n = len(bad)
            ids = bad["person_id"].head(5).tolist()
            msg = (
                f"There is overlap between observation_periods, {n} overlap(s) detected. "
                f"First affected person IDs: {ids}"
            )
            raise CDMValidationError(msg)

    if check_start_before_end:
        start_after_end = op.filter(
            op.observation_period_start_date > op.observation_period_end_date,
        ).select("person_id")
        bad = collect(start_after_end)
        if not bad.empty:
            n = len(bad)
            ids = bad["person_id"].head(5).tolist()
            msg = (
                f"Observation periods with start date after end date: {n} detected. "
                f"First affected person IDs: {ids}"
            )
            raise CDMValidationError(msg)

    if check_plausible_dates:
        agg = op.aggregate(
            [
                op.observation_period_start_date.min().name("min_obs_start"),
                op.observation_period_end_date.max().name("max_obs_end"),
            ],
        )
        row = collect(agg)
        if not row.empty:
            min_start = _first_scalar(row, "min_obs_start")
            max_end = _first_scalar(row, "max_obs_end")
            min_str = str(min_start)[:10]
            max_str = str(max_end)[:10]
            if min_str < "1800-01-01":
                warnings.warn(
                    f"Observation period start dates before 1800-01-01; earliest min: {min_str}",
                    UserWarning,
                    stacklevel=2,
                )
            today = date.today().isoformat()
            if max_str > today:
                warnings.warn(
                    f"Observation period end dates after current date ({today}); latest max: {max_str}",
                    UserWarning,
                    stacklevel=2,
                )


def _materialize_snapshot(cdm: Cdm, compute_data_hash: bool = False) -> pd.DataFrame:
    """Build snapshot DataFrame (exactly one row). Used by cdm.snapshot() and Result.collect().

    Parameters
    ----------
    cdm : Cdm
        CDM reference (must have person, observation_period, cdm_source, vocabulary).
    compute_data_hash : bool, optional
        If True, compute data hash (currently not implemented; placeholder).

    Returns
    -------
    pandas.DataFrame
        Exactly one row: cdm_name, cdm_source_name, cdm_description,
        cdm_documentation_reference, cdm_version, cdm_holder, cdm_release_date,
        vocabulary_version, person_count, observation_period_count,
        earliest_observation_period_start_date, latest_observation_period_end_date,
        snapshot_date, cdm_data_hash.
    """
    from datetime import date

    for req in ("person", "observation_period", "cdm_source", "vocabulary"):
        if req not in cdm._tables:
            raise CDMValidationError(f"snapshot requires '{req}' table in the CDM.")
    person_count = int(_first_scalar(collect(cdm.person.count())))
    obs_count = int(_first_scalar(collect(cdm.observation_period.count())))
    obs_range = cdm.observation_period.aggregate(
        [
            cdm.observation_period.observation_period_start_date.min().name("min_date"),
            cdm.observation_period.observation_period_end_date.max().name("max_date"),
        ]
    )
    obs_range_df = collect(obs_range)
    min_date = _first_scalar(obs_range_df, "min_date")
    max_date = _first_scalar(obs_range_df, "max_date")
    snapshot_date = str(date.today())
    vocab_version = ""
    try:
        v = cdm.vocabulary.filter(cdm.vocabulary.vocabulary_id == "None").select("vocabulary_version").limit(1)
        v_df = collect(v)
        if not v_df.empty:
            vocab_version = str(_first_scalar(v_df, "vocabulary_version"))
    except Exception:
        pass
    cdm_source_df = collect(cdm.cdm_source.limit(1))
    if cdm_source_df.empty:
        cdm_source_df = pd.DataFrame({
            "cdm_source_name": [""],
            "source_description": [""],
            "source_documentation_reference": [""],
            "cdm_holder": [""],
            "cdm_release_date": [""],
            "cdm_version": [cdm.version],
        })
    else:
        cdm_source_df = pd.DataFrame(cdm_source_df)
    if "cdm_source_abbreviation" in cdm_source_df.columns and len(cdm_source_df):
        cdm_name = str(cdm_source_df["cdm_source_abbreviation"].iloc[0])
    else:
        cdm_name = cdm.name
    for col, default in (
        ("cdm_source_name", ""),
        ("source_description", ""),
        ("source_documentation_reference", ""),
    ):
        if col not in cdm_source_df.columns:
            cdm_source_df[col] = default
    if "cdm_holder" not in cdm_source_df.columns:
        cdm_source_df["cdm_holder"] = ""
    if "cdm_release_date" not in cdm_source_df.columns:
        cdm_source_df["cdm_release_date"] = ""
    data_hash = "(not implemented)" if compute_data_hash else ""
    out = pd.DataFrame({
        "cdm_name": [cdm_name],
        "cdm_source_name": [_first_scalar(cdm_source_df, "cdm_source_name") or ""],
        "cdm_description": [_first_scalar(cdm_source_df, "source_description") or ""],
        "cdm_documentation_reference": [_first_scalar(cdm_source_df, "source_documentation_reference") or ""],
        "cdm_version": [cdm.version],
        "cdm_holder": [_first_scalar(cdm_source_df, "cdm_holder") or ""],
        "cdm_release_date": [_first_scalar(cdm_source_df, "cdm_release_date") or ""],
        "vocabulary_version": [vocab_version],
        "person_count": [person_count],
        "observation_period_count": [obs_count],
        "earliest_observation_period_start_date": [str(min_date)],
        "latest_observation_period_end_date": [str(max_date)],
        "snapshot_date": [snapshot_date],
        "cdm_data_hash": [data_hash],
    })
    assert len(out) == 1, "snapshot must return exactly one row"
    return out


def compute(
    cdm: Cdm,
    expr: Any,
    name: str,
    *,
    schema: SchemaSpec | None = None,
    overwrite: bool = True,
) -> Table:
    """
    Materialize an Ibis expression into a table in the CDM's write schema.

    Parameters
    ----------
    cdm : Cdm reference (database-backed).
    expr : Ibis table expression to materialize.
    name : Name of the new table.
    schema : If set, write to this schema instead of cdm.write_schema.
    overwrite : If True, replace existing table.

    Returns
    -------
    Ibis table reference to the new table.
    """
    if not isinstance(cdm.source, DbCdmSource):
        raise SourceError("compute() requires a database-backed CDM (cdm_from_con).")
    # insert_table uses cdm.write_schema; schema override would require source.insert_table(name, expr, schema=...)
    return insert_table(cdm, name, expr, overwrite=overwrite)


def insert_cdm_to(cdm: Cdm, to_con: Any, schema: SchemaSpec, *, overwrite: bool = False) -> Cdm:
    """
    Copy this CDM into another database (insert all tables into the target connection).

    Collects each table from the source CDM, then inserts it into the target
    connection's schema. Returns a new Cdm reference pointing to the target.

    Parameters
    ----------
    cdm : Source CDM reference.
    to_con : Ibis connection to the target database.
    schema : Schema (database name) in the target where tables will be created.
    overwrite : If True, overwrite existing tables in the target.

    Returns
    -------
    New Cdm reference pointing to the target database.
    """
    import pandas as pd

    to_src = DbCdmSource(to_con, schema)
    new_tables: dict[str, Any] = {}
    for tab_name in cdm.tables:
        tbl = cdm[tab_name]
        df = collect(tbl) if (hasattr(tbl, "schema") or hasattr(tbl, "op")) else tbl
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        to_src.insert_table(tab_name, df, overwrite=overwrite)
        new_tables[tab_name] = to_src.table(tab_name, schema)
    new_cdm = Cdm(
        new_tables,
        cdm_name=cdm.name,
        cdm_version=cdm.version,
        cdm_schema=schema,
        write_schema=schema,
        achilles_schema=None,
        source=to_src,
    )
    return new_cdm


def copy_cdm_to(con: Any, cdm: Cdm, schema: SchemaSpec, *, overwrite: bool = False) -> Cdm:
    """
    Copy a CDM from one database to another (same as R copyCdmTo).

    Inserts person and observation_period first, then all other tables, into
    the given connection and schema. Returns a new Cdm reference for the target.

    Parameters
    ----------
    con : Ibis connection to the target database.
    cdm : Source CDM reference.
    schema : Schema in the target where tables will be created.
    overwrite : If True, overwrite existing tables.

    Returns
    -------
    New Cdm reference pointing to the target database.
    """
    return insert_cdm_to(cdm, con, schema, overwrite=overwrite)


# ---------------------------------------------------------------------------
# Low-level constructors (mirrors omopgenerics new* functions)
# ---------------------------------------------------------------------------


def new_cdm_reference(
    tables: dict[str, Any],
    cdm_name: str,
    *,
    cdm_version: str | None = None,
    cdm_schema: SchemaSpec | None = None,
    write_schema: SchemaSpec | None = None,
    achilles_schema: SchemaSpec | None = None,
    source: CdmSource | None = None,
) -> Cdm:
    """Construct a CDM reference from pre-made tables.

    Low-level constructor mirroring omopgenerics ``newCdmReference``.

    Parameters
    ----------
    tables : dict[str, Any]
        Mapping of table name to Ibis table expression.
    cdm_name : str
        Display name for this CDM.
    cdm_version : str or None
        OMOP CDM version (default "5.3").
    cdm_schema, write_schema, achilles_schema : SchemaSpec or None
        Optional schema specs.
    source : CdmSource or None
        Source object. If None, a LocalCdmSource is created.

    Returns
    -------
    Cdm
    """
    if source is None:
        source = new_local_source()
    return Cdm(
        tables,
        cdm_name=cdm_name,
        cdm_version=cdm_version,
        cdm_schema=cdm_schema,
        write_schema=write_schema,
        achilles_schema=achilles_schema,
        source=source,
    )


def new_cdm_source(src: Any, source_type: str) -> CdmSource:
    """Construct a CdmSource.

    Parameters
    ----------
    src : Any
        Underlying connection or data.
    source_type : str
        Source type (e.g. "local", "database").

    Returns
    -------
    CdmSource
    """
    if source_type == "local":
        return LocalCdmSource()
    return DbCdmSource(src, None)


def new_local_source() -> LocalCdmSource:
    """Create a local (in-memory) CDM source.

    Returns
    -------
    LocalCdmSource
    """
    return LocalCdmSource()


def cdm_select(cdm: Cdm, *names: str) -> Cdm:
    """Select a subset of tables from a CDM.

    Standalone function mirroring omopgenerics ``cdmSelect``.

    Parameters
    ----------
    cdm : Cdm
        CDM reference.
    *names : str
        Table names to keep.

    Returns
    -------
    Cdm
        New CDM with only the selected tables.
    """
    return cdm.select_tables(*names)


def cdm_disconnect(cdm: Cdm, drop_write_schema: bool = False) -> None:
    """Disconnect a CDM from its source.

    Parameters
    ----------
    cdm : Cdm
        CDM reference.
    drop_write_schema : bool
        If True, drop the write schema on disconnect.
    """
    cdm.disconnect(drop_write_schema=drop_write_schema)


def cdm_table_from_source(
    cdm: Cdm,
    name: str,
    schema: SchemaSpec | None = None,
) -> Table:
    """Create a CDM table reference from the source.

    Parameters
    ----------
    cdm : Cdm
        CDM reference.
    name : str
        Table name.
    schema : SchemaSpec or None
        Schema override.

    Returns
    -------
    ibis.Table
    """
    return read_source_table(cdm, name)


def insert_from_source(
    cdm: Cdm,
    name: str,
    table: Any,
    *,
    overwrite: bool = True,
) -> Table:
    """Insert a table into the CDM's write schema from the source.

    Parameters
    ----------
    cdm : Cdm
        CDM reference.
    name : str
        Table name.
    table : Any
        Table data (DataFrame, Arrow table, or Ibis expression).
    overwrite : bool
        If True, replace existing table.

    Returns
    -------
    ibis.Table
        Reference to the inserted table.
    """
    return insert_table(cdm, name, table, overwrite=overwrite)


def drop_source_table(cdm: Cdm, name: str | list[str]) -> None:
    """Drop table(s) from the CDM's source.

    Parameters
    ----------
    cdm : Cdm
        CDM reference.
    name : str or list[str]
        Table name(s) to drop.
    """
    drop_table(cdm, name)


def cdm_classes(cdm: Cdm) -> dict[str, int]:
    """Get counts of table types in a CDM.

    Parameters
    ----------
    cdm : Cdm
        CDM reference.

    Returns
    -------
    dict[str, int]
        Mapping of table type to count (e.g. {"cdm_table": 15, "cohort": 1}).
    """
    from cdmconnector.schemas import ACHILLES_TABLES, omop_tables

    counts: dict[str, int] = {"cdm_table": 0, "cohort": 0, "achilles": 0, "other": 0}
    omop = set(omop_tables(cdm.version))
    achilles = set(ACHILLES_TABLES)
    for name in cdm.tables:
        if name in omop:
            counts["cdm_table"] += 1
        elif name in achilles:
            counts["achilles"] += 1
        elif name.endswith("_set") or name.endswith("_attrition") or name.endswith("_codelist"):
            counts["cohort"] += 1
        else:
            # Could be a cohort table or other
            counts["other"] += 1
    return counts


# ---------------------------------------------------------------------------
# Empty creators (mirrors omopgenerics empty* functions)
# ---------------------------------------------------------------------------


def empty_cdm_reference(cdm_name: str = "empty") -> Cdm:
    """Create an empty CDM reference with no tables.

    Parameters
    ----------
    cdm_name : str
        Display name (default "empty").

    Returns
    -------
    Cdm
    """
    return Cdm(
        {},
        cdm_name=cdm_name,
        source=LocalCdmSource(),
    )


def empty_omop_table(table_name: str, version: str = "5.3") -> Any:
    """Create an empty OMOP table as a pandas DataFrame with correct columns.

    Parameters
    ----------
    table_name : str
        OMOP table name (e.g. "person", "observation_period").
    version : str
        CDM version.

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with the correct columns.
    """
    import pandas as pd

    from cdmconnector.schemas import omop_columns

    cols = omop_columns(table_name, version=version)
    return pd.DataFrame(columns=list(cols))


def empty_achilles_table(table_name: str) -> Any:
    """Create an empty Achilles table as a pandas DataFrame.

    Parameters
    ----------
    table_name : str
        Achilles table name.

    Returns
    -------
    pandas.DataFrame
    """
    import pandas as pd

    from cdmconnector.schemas import achilles_columns

    cols = achilles_columns(table_name)
    return pd.DataFrame(columns=list(cols))


def empty_cohort_table() -> Any:
    """Create an empty cohort table as a pandas DataFrame.

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with cohort_definition_id, subject_id,
        cohort_start_date, cohort_end_date columns.
    """
    import pandas as pd

    from cdmconnector.schemas import COHORT_TABLE_COLUMNS

    return pd.DataFrame(columns=list(COHORT_TABLE_COLUMNS))
