# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for CDMConnector tests.

Live database testing
---------------------
Mirrors the R CDMConnector test matrix (tests/testthat/setup.R).

* Set ``CDMCONNECTOR_TEST_DB`` to a comma-separated list of backends
  (default: ``"duckdb"``).  Example::

      CDMCONNECTOR_TEST_DB=duckdb,postgres pytest tests/ -m integration

* Each backend requires its own set of environment variables (see below).
  Backends whose env vars are missing are silently skipped.

Environment variables per backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**PostgreSQL** – ``CDM5_POSTGRESQL_DBNAME``, ``CDM5_POSTGRESQL_HOST``,
  ``CDM5_POSTGRESQL_USER``, ``CDM5_POSTGRESQL_PASSWORD``,
  ``CDM5_POSTGRESQL_CDM_SCHEMA``, ``CDM5_POSTGRESQL_SCRATCH_SCHEMA``

**Redshift** – ``CDM5_REDSHIFT_DBNAME``, ``CDM5_REDSHIFT_HOST``,
  ``CDM5_REDSHIFT_PORT``, ``CDM5_REDSHIFT_USER``, ``CDM5_REDSHIFT_PASSWORD``,
  ``CDM5_REDSHIFT_CDM_SCHEMA``, ``CDM5_REDSHIFT_SCRATCH_SCHEMA``

**SQL Server** – ``CDM5_SQL_SERVER_USER``, ``CDM5_SQL_SERVER_PASSWORD``,
  ``CDM5_SQL_SERVER_SERVER``, ``CDM5_SQL_SERVER_PORT``,
  ``CDM5_SQL_SERVER_CDM_DATABASE``, ``CDM5_SQL_SERVER_CDM_SCHEMA`` (catalog.schema),
  ``CDM5_SQL_SERVER_SCRATCH_SCHEMA`` (catalog.schema)

**Snowflake** – ``SNOWFLAKE_USER``, ``SNOWFLAKE_PASSWORD``,
  ``SNOWFLAKE_ACCOUNT``, ``SNOWFLAKE_DATABASE``, ``SNOWFLAKE_WAREHOUSE``,
  ``SNOWFLAKE_CDM_SCHEMA`` (database.schema), ``SNOWFLAKE_SCRATCH_SCHEMA``

**BigQuery** – ``BIGQUERY_PROJECT_ID``, ``BIGQUERY_CDM_SCHEMA``,
  ``BIGQUERY_SCRATCH_SCHEMA``

**Spark / Databricks** – ``DATABRICKS_HOST``, ``DATABRICKS_TOKEN``,
  ``DATABRICKS_HTTPPATH``, ``DATABRICKS_CDM_SCHEMA``, ``DATABRICKS_SCRATCH_SCHEMA``
"""

from __future__ import annotations

import datetime
import os
import tempfile

import ibis
import pyarrow as pa
import pytest

# ---------------------------------------------------------------------------
# Resolve which backends to test
# ---------------------------------------------------------------------------

# Databases supported in CI (superset – individual runs pick a subset).
CI_TEST_DBS = ("duckdb", "postgres", "redshift", "sqlserver", "snowflake", "bigquery", "spark")

# CDMCONNECTOR_TEST_DB may be a single value or comma-separated list.
_raw_dbs = os.environ.get("CDMCONNECTOR_TEST_DB", "duckdb")
DB_TO_TEST: list[str] = [d.strip().lower() for d in _raw_dbs.split(",") if d.strip()]

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def get_cdm_schema(dbms: str) -> str | dict | None:
    """Return CDM schema for *dbms*.

    For backends that use catalog.schema (SQL Server, Snowflake) returns a dict
    ``{"catalog": ..., "schema": ...}`` when the env var contains a dot.
    """
    dbms = dbms.lower()
    if dbms == "duckdb":
        return "main"

    env_map: dict[str, str] = {
        "postgres": "CDM5_POSTGRESQL_CDM_SCHEMA",
        "redshift": "CDM5_REDSHIFT_CDM_SCHEMA",
        "sqlserver": "CDM5_SQL_SERVER_CDM_SCHEMA",
        "snowflake": "SNOWFLAKE_CDM_SCHEMA",
        "bigquery": "BIGQUERY_CDM_SCHEMA",
        "spark": "DATABRICKS_CDM_SCHEMA",
    }
    raw = os.environ.get(env_map.get(dbms, ""), "")
    if not raw:
        return None

    # catalog.schema for sqlserver / snowflake
    if dbms in ("sqlserver", "snowflake") and "." in raw:
        parts = raw.split(".", 1)
        return {"catalog": parts[0], "schema": parts[1]}
    return raw


def get_write_schema(dbms: str, prefix: str | None = None) -> str | dict:
    """Return write/scratch schema for *dbms*.

    Always includes a ``prefix`` key for non-duckdb backends to isolate
    concurrent test runs (mirrors R CDMConnector behaviour).
    """
    dbms = dbms.lower()
    if dbms == "duckdb":
        return "main"

    env_map: dict[str, str] = {
        "postgres": "CDM5_POSTGRESQL_SCRATCH_SCHEMA",
        "redshift": "CDM5_REDSHIFT_SCRATCH_SCHEMA",
        "sqlserver": "CDM5_SQL_SERVER_SCRATCH_SCHEMA",
        "snowflake": "SNOWFLAKE_SCRATCH_SCHEMA",
        "bigquery": "BIGQUERY_SCRATCH_SCHEMA",
        "spark": "DATABRICKS_SCRATCH_SCHEMA",
    }
    raw = os.environ.get(env_map.get(dbms, ""), "")
    if not raw:
        return "main"

    if prefix is None:
        prefix = (
            f"temp{(int(datetime.datetime.now().timestamp() * 100) % 100000 + os.getpid()) % 100000}_"
        )

    # catalog.schema for sqlserver / snowflake
    if dbms in ("sqlserver", "snowflake") and "." in raw:
        parts = raw.split(".", 1)
        return {"catalog": parts[0], "schema": parts[1], "prefix": prefix}
    return {"schema": raw, "prefix": prefix}


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

# Env-var "gate" for each backend: if the gate var is empty the backend is
# skipped without attempting to connect.
_BACKEND_GATE: dict[str, str] = {
    "postgres": "CDM5_POSTGRESQL_DBNAME",
    "redshift": "CDM5_REDSHIFT_DBNAME",
    "sqlserver": "CDM5_SQL_SERVER_USER",
    "snowflake": "SNOWFLAKE_USER",
    "bigquery": "BIGQUERY_PROJECT_ID",
    "spark": "DATABRICKS_HTTPPATH",
}


def _connect_duckdb(eunomia: bool = False):
    """DuckDB: in-memory with minimal tables, or Eunomia."""
    import cdmconnector as cc

    if eunomia:
        folder = os.environ.get("EUNOMIA_DATA_FOLDER")
        if not folder:
            folder = tempfile.mkdtemp(prefix="cdmconnector_eunomia_")
            os.environ["EUNOMIA_DATA_FOLDER"] = folder
        path = cc.eunomia_dir("synpuf-1k", cdm_version="5.3")
        return ibis.duckdb.connect(path), "main", "main"

    con = ibis.duckdb.connect()
    d0, d1 = datetime.date(2000, 1, 1), datetime.date(2023, 12, 31)
    con.create_table(
        "person",
        obj=pa.table({
            "person_id": [1, 2, 3],
            "gender_concept_id": [8507, 8532, 0],
            "year_of_birth": [1990, 1985, 2000],
            "month_of_birth": [1, 6, 1],
            "day_of_birth": [1, 15, 1],
            "race_concept_id": [0, 0, 0],
            "ethnicity_concept_id": [0, 0, 0],
        }),
        overwrite=True,
    )
    con.create_table(
        "observation_period",
        obj=pa.table({
            "observation_period_id": [1, 2, 3],
            "person_id": [1, 2, 3],
            "observation_period_start_date": pa.array([d0, d0, d0], type=pa.date32()),
            "observation_period_end_date": pa.array([d1, d1, d1], type=pa.date32()),
            "period_type_concept_id": [0, 0, 0],
        }),
        overwrite=True,
    )
    return con, "main", "main"


def _connect_postgres():
    con = ibis.postgres.connect(
        host=os.environ.get("CDM5_POSTGRESQL_HOST", "localhost"),
        port=int(os.environ.get("CDM5_POSTGRESQL_PORT", "5432")),
        database=os.environ.get("CDM5_POSTGRESQL_DBNAME", ""),
        user=os.environ.get("CDM5_POSTGRESQL_USER"),
        password=os.environ.get("CDM5_POSTGRESQL_PASSWORD"),
    )
    return con, get_cdm_schema("postgres"), get_write_schema("postgres")


def _connect_redshift():
    """Redshift uses the Postgres wire protocol."""
    con = ibis.postgres.connect(
        host=os.environ.get("CDM5_REDSHIFT_HOST", "localhost"),
        port=int(os.environ.get("CDM5_REDSHIFT_PORT", "5439")),
        database=os.environ.get("CDM5_REDSHIFT_DBNAME", ""),
        user=os.environ.get("CDM5_REDSHIFT_USER"),
        password=os.environ.get("CDM5_REDSHIFT_PASSWORD"),
    )
    return con, get_cdm_schema("redshift"), get_write_schema("redshift")


def _connect_sqlserver():
    """SQL Server via ibis MSSQL backend."""
    con = ibis.mssql.connect(
        host=os.environ.get("CDM5_SQL_SERVER_SERVER", "localhost"),
        port=int(os.environ.get("CDM5_SQL_SERVER_PORT", "1433")),
        database=os.environ.get("CDM5_SQL_SERVER_CDM_DATABASE", ""),
        user=os.environ.get("CDM5_SQL_SERVER_USER"),
        password=os.environ.get("CDM5_SQL_SERVER_PASSWORD"),
    )
    return con, get_cdm_schema("sqlserver"), get_write_schema("sqlserver")


def _connect_snowflake():
    con = ibis.snowflake.connect(
        account=os.environ.get("SNOWFLAKE_ACCOUNT", ""),
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        database=os.environ.get("SNOWFLAKE_DATABASE", ""),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", ""),
    )
    return con, get_cdm_schema("snowflake"), get_write_schema("snowflake")


def _connect_bigquery():
    con = ibis.bigquery.connect(
        project_id=os.environ.get("BIGQUERY_PROJECT_ID", ""),
    )
    return con, get_cdm_schema("bigquery"), get_write_schema("bigquery")


def _connect_spark():
    """Databricks / Spark via the PySpark or Databricks Ibis backend."""
    con = ibis.databricks.connect(
        server_hostname=os.environ.get("DATABRICKS_HOST", ""),
        http_path=os.environ.get("DATABRICKS_HTTPPATH", ""),
        access_token=os.environ.get("DATABRICKS_TOKEN", ""),
    )
    return con, get_cdm_schema("spark"), get_write_schema("spark")


_CONNECTORS: dict[str, callable] = {
    "duckdb": lambda: _connect_duckdb(eunomia=False),
    "postgres": _connect_postgres,
    "redshift": _connect_redshift,
    "sqlserver": _connect_sqlserver,
    "snowflake": _connect_snowflake,
    "bigquery": _connect_bigquery,
    "spark": _connect_spark,
}


def get_connection_duckdb(eunomia: bool = False):
    """Public helper kept for backwards-compat with test_generate_cohort_set_live.py."""
    return _connect_duckdb(eunomia=eunomia)


def get_connection(dbms: str):
    """Return ``(con, cdm_schema, write_schema)`` for *dbms*, or ``None`` if unavailable."""
    dbms = dbms.lower()
    if dbms == "duckdb":
        return _connect_duckdb(eunomia=False)

    # Gate: skip if the required env var is missing.
    gate = _BACKEND_GATE.get(dbms)
    if gate and not os.environ.get(gate):
        return None

    connector = _CONNECTORS.get(dbms)
    if connector is None:
        return None
    try:
        return connector()
    except Exception as exc:
        print(f"[conftest] Could not connect to {dbms}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Determine which backends are actually available for parametrised tests
# ---------------------------------------------------------------------------


def _available_backends() -> list[str]:
    """Return the subset of DB_TO_TEST whose connection env vars are present."""
    available = []
    for db in DB_TO_TEST:
        if db == "duckdb":
            available.append(db)
            continue
        gate = _BACKEND_GATE.get(db)
        if gate and os.environ.get(gate):
            available.append(db)
        elif gate:
            print(f"[conftest] Skipping {db} – {gate} not set")
        else:
            print(f"[conftest] Unknown backend: {db}")
    return available


AVAILABLE_BACKENDS: list[str] = _available_backends()


# ---------------------------------------------------------------------------
# Fixtures — basic (non-live)
# ---------------------------------------------------------------------------


@pytest.fixture
def duckdb_con():
    """In-memory DuckDB connection (Ibis)."""
    con = ibis.duckdb.connect()
    yield con
    try:
        con.disconnect()
    except Exception:
        pass


@pytest.fixture
def minimal_cdm_tables():
    """Minimal OMOP tables (person, observation_period) as Arrow tables."""
    person = pa.table({
        "person_id": [1, 2, 3],
        "gender_concept_id": [8507, 8532, 0],  # Male, Female, unknown
        "year_of_birth": [1990, 1985, 2000],
        "month_of_birth": [1, 6, 1],
        "day_of_birth": [1, 15, 1],
        "race_concept_id": [0, 0, 0],
        "ethnicity_concept_id": [0, 0, 0],
    })
    d0 = datetime.date(2000, 1, 1)
    d1 = datetime.date(2023, 12, 31)
    observation_period = pa.table({
        "observation_period_id": [1, 2, 3],
        "person_id": [1, 2, 3],
        "observation_period_start_date": pa.array([d0, d0, d0], type=pa.date32()),
        "observation_period_end_date": pa.array([d1, d1, d1], type=pa.date32()),
        "period_type_concept_id": [0, 0, 0],
    })
    return {"person": person, "observation_period": observation_period}


@pytest.fixture
def cdm_from_tables_fixture(minimal_cdm_tables):
    """CDM built from cdm_from_tables with minimal tables."""
    import cdmconnector as cc

    return cc.cdm_from_tables(minimal_cdm_tables, cdm_name="test_cdm", cdm_version="5.3")


@pytest.fixture
def cdm_from_duckdb(duckdb_con, minimal_cdm_tables):
    """CDM built from Ibis DuckDB with minimal tables (person, observation_period)."""
    import cdmconnector as cc

    con = duckdb_con
    for name, tbl in minimal_cdm_tables.items():
        con.create_table(name, obj=tbl, overwrite=True)
    return cc.cdm_from_con(con, cdm_schema="main", write_schema="main", cdm_name="test_cdm")


# ---------------------------------------------------------------------------
# Fixtures — live database (single-backend, legacy)
# ---------------------------------------------------------------------------


@pytest.fixture
def live_db_connection():
    """Yield (con, cdm_schema, write_schema) for the *first* DB in DB_TO_TEST. Skip if unavailable."""
    dbms = DB_TO_TEST[0] if DB_TO_TEST else "duckdb"
    res = get_connection(dbms)
    if res is None:
        pytest.skip(f"Live DB {dbms!r} not available (check env vars).")
    con, cdm_schema, write_schema = res
    yield con, cdm_schema, write_schema
    try:
        con.disconnect()
    except Exception:
        pass


@pytest.fixture
def live_cdm(live_db_connection):
    """Yield a CDM for the live DB. Uses minimal tables for duckdb."""
    import cdmconnector as cc

    con, cdm_schema, write_schema = live_db_connection
    return cc.cdm_from_con(con, cdm_schema=cdm_schema, write_schema=write_schema, cdm_name="test_cdm")


# ---------------------------------------------------------------------------
# Fixtures — parametrised across all available backends
# ---------------------------------------------------------------------------


@pytest.fixture(params=AVAILABLE_BACKENDS, ids=lambda db: db)
def db_connection(request):
    """Parametrised fixture: yields ``(dbms, con, cdm_schema, write_schema)``
    for every backend listed in ``CDMCONNECTOR_TEST_DB`` whose env vars are present.

    Usage in tests::

        @pytest.mark.integration
        def test_something(db_connection):
            dbms, con, cdm_schema, write_schema = db_connection
            ...
    """
    dbms = request.param
    res = get_connection(dbms)
    if res is None:
        pytest.skip(f"{dbms} connection not available")
    con, cdm_schema, write_schema = res
    yield dbms, con, cdm_schema, write_schema
    try:
        con.disconnect()
    except Exception:
        pass


@pytest.fixture
def db_cdm(db_connection):
    """Parametrised CDM across all available backends.

    Usage::

        @pytest.mark.integration
        def test_person_table(db_cdm):
            dbms, cdm = db_cdm
            assert "person" in cdm.tables
    """
    import cdmconnector as cc

    dbms, con, cdm_schema, write_schema = db_connection
    cdm = cc.cdm_from_con(
        con, cdm_schema=cdm_schema, write_schema=write_schema, cdm_name="test_cdm"
    )
    yield dbms, cdm
