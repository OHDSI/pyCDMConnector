# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Tests for cdmconnector.eunomia (example dataset helpers).

Coverage for src/cdmconnector/eunomia.py is required to be >= 85%.
Run: pytest tests/test_eunomia.py tests/test_cdm.py -k eunomia --cov=src/cdmconnector --cov-report=term-missing
"""

import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cdmconnector.exceptions import EunomiaError
from cdmconnector.eunomia import (
    download_eunomia_data,
    eunomia_dir,
    eunomia_is_available,
    example_datasets,
    require_eunomia,
)


# ---- example_datasets ----


def test_example_datasets_returns_tuple():
    """example_datasets returns tuple of known names."""
    ds = example_datasets()
    assert isinstance(ds, tuple)
    assert "GiBleed" in ds
    assert "synpuf-1k" in ds
    assert "Synthea27NjParquet" in ds
    assert "empty_cdm" in ds


# ---- download_eunomia_data ----


def test_download_eunomia_invalid_cdm_version(tmp_path):
    """download_eunomia_data raises for cdm_version not 5.3 or 5.4."""
    with pytest.raises(EunomiaError, match="cdm_version must be"):
        download_eunomia_data("GiBleed", cdm_version="5.0", path=str(tmp_path))
    with pytest.raises(EunomiaError, match="cdm_version must be"):
        download_eunomia_data("GiBleed", cdm_version="6.0", path=str(tmp_path))


def test_download_eunomia_synthea27_only_54(tmp_path):
    """Synthea27NjParquet only available in CDM 5.4."""
    with pytest.raises(EunomiaError, match="Synthea27NjParquet is only available"):
        download_eunomia_data("Synthea27NjParquet", cdm_version="5.3", path=str(tmp_path))


def test_download_eunomia_54_restricted_datasets(tmp_path):
    """CDM 5.4 only allows synpuf-1k, Synthea27NjParquet, empty_cdm."""
    with pytest.raises(EunomiaError, match="is only available in CDM version 5.3"):
        download_eunomia_data("GiBleed", cdm_version="5.4", path=str(tmp_path))


def test_download_eunomia_path_required():
    """download_eunomia_data requires path or env (use 5.3 so path check runs before 5.4 restriction)."""
    with pytest.raises(EunomiaError, match="path must be set|EUNOMIA_DATA_FOLDER"):
        download_eunomia_data("GiBleed", cdm_version="5.3", path="")


def test_download_eunomia_unknown_dataset(tmp_path):
    """Unknown dataset name raises."""
    with pytest.raises(EunomiaError, match="Unknown dataset"):
        download_eunomia_data("NotADataset", path=str(tmp_path))


def test_download_eunomia_path_from_env(tmp_path):
    """download_eunomia_data uses EUNOMIA_DATA_FOLDER when path is None."""
    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        os.environ["EUNOMIA_DATA_FOLDER"] = str(tmp_path)
        with patch("cdmconnector.eunomia._download_with_progress") as mock_get:
            out = download_eunomia_data("GiBleed", cdm_version="5.3", path=None)
            assert out == str(tmp_path)
            mock_get.assert_called_once()
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)


def test_download_eunomia_existing_zip_no_overwrite(tmp_path):
    """When zip exists and overwrite is False, returns path without downloading."""
    (tmp_path / "GiBleed_5.3.zip").touch()
    with patch("cdmconnector.eunomia._download_with_progress") as mock_get:
        out = download_eunomia_data("GiBleed", cdm_version="5.3", path=str(tmp_path))
        assert out == str(tmp_path)
        mock_get.assert_not_called()


def test_download_eunomia_overwrite_downloads(tmp_path):
    """When overwrite=True, downloads even if zip exists."""
    (tmp_path / "GiBleed_5.3.zip").touch()
    with patch("cdmconnector.eunomia._download_with_progress") as mock_get:
        out = download_eunomia_data(
            "GiBleed", cdm_version="5.3", path=str(tmp_path), overwrite=True
        )
        assert out == str(tmp_path)
        mock_get.assert_called_once()


def test_download_eunomia_downloads_when_missing(tmp_path):
    """Downloads when zip is missing (mock _download_with_progress to write file)."""
    def _fake_download(url, dest, *, desc=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fake zip")

    with patch("cdmconnector.eunomia._download_with_progress", side_effect=_fake_download):
        out = download_eunomia_data("GiBleed", cdm_version="5.3", path=str(tmp_path))
        assert out == str(tmp_path)
        assert (tmp_path / "GiBleed_5.3.zip").exists()


def test_download_eunomia_path_accepts_path_object(tmp_path):
    """path can be a Path object."""
    with patch("cdmconnector.eunomia._download_with_progress"):
        out = download_eunomia_data("GiBleed", cdm_version="5.3", path=tmp_path)
        assert out == str(tmp_path)


def test_download_eunomia_blob_url_for_non_synthea27(tmp_path):
    """Uses blob URL for non-Synthea27NjParquet datasets."""
    with patch("cdmconnector.eunomia._download_with_progress") as mock_get:
        download_eunomia_data("synpuf-1k", cdm_version="5.4", path=str(tmp_path))
        call_args = mock_get.call_args
        assert "blob.core.windows.net" in call_args[0][0]
        assert call_args[0][1] == tmp_path / "synpuf-1k_5.4.zip"


def test_download_eunomia_github_url_for_synthea27(tmp_path):
    """Uses GitHub URL for Synthea27NjParquet."""
    with patch("cdmconnector.eunomia._download_with_progress") as mock_get:
        download_eunomia_data("Synthea27NjParquet", cdm_version="5.4", path=str(tmp_path))
        call_args = mock_get.call_args[0]
        assert "github.com" in call_args[0] and "Synthea27NjParquet" in call_args[0]


def test_download_eunomia_download_failure_raises(tmp_path):
    """Download failure raises EunomiaError."""
    with patch("cdmconnector.eunomia._download_with_progress", side_effect=OSError("network error")):
        with pytest.raises(EunomiaError, match="Download failed"):
            download_eunomia_data("GiBleed", cdm_version="5.3", path=str(tmp_path))


# ---- eunomia_dir ----


def test_eunomia_dir_requires_eunomia_data_folder_when_path_unset():
    """eunomia_dir raises when path is None and EUNOMIA_DATA_FOLDER is unset."""
    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        with pytest.raises(EunomiaError, match="EUNOMIA_DATA_FOLDER must be set"):
            eunomia_dir("GiBleed", cdm_version="5.3", path=None)
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old


def _make_eunomia_zip(zip_path: Path, dataset_name: str, parquet_at_root: bool = False):
    """Create a minimal zip with one parquet (for testing eunomia_dir / _build)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({"person_id": [1, 2], "year_of_birth": [1990, 1985]})
    buf = pa.BufferOutputStream()
    pq.write_table(tbl, buf)
    parquet_bytes = buf.getvalue().to_pybytes()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if parquet_at_root:
            zf.writestr("person.parquet", parquet_bytes)
        else:
            zf.writestr(f"{dataset_name}/person.parquet", parquet_bytes)


def test_eunomia_dir_with_existing_zip_builds_duckdb(tmp_path):
    """eunomia_dir with existing zip in cache builds DuckDB and returns path to a copy."""
    _make_eunomia_zip(tmp_path / "GiBleed_5.3.zip", "GiBleed")
    old = os.environ.get("EUNOMIA_DATA_FOLDER")
    os.environ["EUNOMIA_DATA_FOLDER"] = str(tmp_path)
    try:
        out = eunomia_dir("GiBleed", cdm_version="5.3")
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    assert Path(out).exists()
    assert Path(out).suffix == ".duckdb"
    assert (tmp_path / "GiBleed_5.3_py.duckdb").exists()


def test_eunomia_dir_database_file_arg(tmp_path):
    """eunomia_dir with database_file (copy destination) returns that path."""
    _make_eunomia_zip(tmp_path / "GiBleed_5.3.zip", "GiBleed")
    old = os.environ.get("EUNOMIA_DATA_FOLDER")
    os.environ["EUNOMIA_DATA_FOLDER"] = str(tmp_path)
    try:
        db_file = tmp_path / "my_copy.duckdb"
        out = eunomia_dir("GiBleed", cdm_version="5.3", database_file=str(db_file))
        assert out == str(db_file)
        assert db_file.exists()
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)


def test_eunomia_dir_uses_env_when_path_not_given(tmp_path):
    """eunomia_dir uses EUNOMIA_DATA_FOLDER when path is None."""
    _make_eunomia_zip(tmp_path / "GiBleed_5.3.zip", "GiBleed")
    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        os.environ["EUNOMIA_DATA_FOLDER"] = str(tmp_path)
        out = eunomia_dir("GiBleed", cdm_version="5.3", path=None)
        assert Path(out).exists()
        assert Path(out).suffix == ".duckdb"
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)


def test_eunomia_dir_downloads_if_zip_missing(tmp_path):
    """eunomia_dir downloads zip if not present in cache (mock _download_with_progress)."""
    old = os.environ.get("EUNOMIA_DATA_FOLDER")
    os.environ["EUNOMIA_DATA_FOLDER"] = str(tmp_path)
    try:
        with patch("cdmconnector.eunomia._download_with_progress") as mock_dl:
            def write_zip(url, dest, *, desc=None):
                _make_eunomia_zip(Path(dest), "GiBleed")

            mock_dl.side_effect = write_zip
            out = eunomia_dir("GiBleed", cdm_version="5.3")
        assert Path(out).exists()
        mock_dl.assert_called_once()
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)


def test_eunomia_dir_path_is_copy_destination(tmp_path):
    """eunomia_dir path parameter is where the DB copy is written."""
    _make_eunomia_zip(tmp_path / "GiBleed_5.3.zip", "GiBleed")
    old = os.environ.get("EUNOMIA_DATA_FOLDER")
    os.environ["EUNOMIA_DATA_FOLDER"] = str(tmp_path)
    try:
        copy_dest = tmp_path / "output" / "my.duckdb"
        out = eunomia_dir("GiBleed", cdm_version="5.3", path=str(copy_dest))
        assert out == str(copy_dest)
        assert copy_dest.exists()
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)


def test_build_duckdb_from_zip_parquet_at_root(tmp_path):
    """_build_duckdb_from_zip uses tmp as extract_dir when dataset subdir missing."""
    from cdmconnector.eunomia import _build_duckdb_from_zip

    _make_eunomia_zip(tmp_path / "x.zip", "NoSubdir", parquet_at_root=True)
    db_path = tmp_path / "out.duckdb"
    _build_duckdb_from_zip(tmp_path / "x.zip", db_path, "NoSubdir")
    assert db_path.exists()


def test_build_duckdb_from_zip_no_parquet_raises(tmp_path):
    """_build_duckdb_from_zip raises when zip has no parquet files."""
    from cdmconnector.eunomia import _build_duckdb_from_zip

    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as zf:
        zf.writestr("readme.txt", b"no parquet")
    with pytest.raises(EunomiaError, match="No parquet files"):
        _build_duckdb_from_zip(tmp_path / "empty.zip", tmp_path / "out.duckdb", "x")


# ---- eunomia_is_available ----


def test_eunomia_is_available_nonexistent_path():
    """eunomia_is_available returns False for path with no zip."""
    assert eunomia_is_available(path="/nonexistent/eunomia/folder") is False


def test_eunomia_is_available_with_zip(tmp_path):
    """eunomia_is_available returns True when zip exists."""
    (tmp_path / "GiBleed_5.3.zip").touch()
    assert eunomia_is_available("GiBleed", "5.3", path=str(tmp_path)) is True
    assert eunomia_is_available("GiBleed", "5.3", path=str(tmp_path / "other")) is False


def test_eunomia_is_available_path_empty_raises():
    """eunomia_is_available raises when path and env both empty."""
    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        with pytest.raises(EunomiaError, match="EUNOMIA_DATA_FOLDER"):
            eunomia_is_available(path="")
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old


def test_eunomia_is_available_invalid_cdm_version(tmp_path):
    """eunomia_is_available raises for cdm_version not 5.3 or 5.4."""
    with pytest.raises(EunomiaError, match="cdm_version must be"):
        eunomia_is_available("GiBleed", "5.0", path=str(tmp_path))


# ---- require_eunomia ----


def test_require_eunomia_unknown_dataset_raises():
    """require_eunomia raises for unknown dataset."""
    with pytest.raises(EunomiaError, match="Unknown dataset"):
        require_eunomia("NotADataset", path="/tmp")


def test_require_eunomia_invalid_cdm_version_raises(tmp_path):
    """require_eunomia raises for cdm_version not 5.3 or 5.4."""
    with pytest.raises(EunomiaError, match="cdm_version must be"):
        require_eunomia("GiBleed", cdm_version="5.0", path=str(tmp_path))


def test_require_eunomia_sets_env_when_unset(tmp_path):
    """require_eunomia sets EUNOMIA_DATA_FOLDER to temp dir when unset."""
    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        with patch("cdmconnector.eunomia._download_with_progress"):
            path = require_eunomia("GiBleed", path=None)
            assert path
            assert os.environ.get("EUNOMIA_DATA_FOLDER") == path
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)


def test_require_eunomia_with_path_uses_existing_zip(tmp_path):
    """require_eunomia with path and existing zip does not set env; returns path."""
    (tmp_path / "GiBleed_5.3.zip").touch()
    old = os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    try:
        out = require_eunomia("GiBleed", cdm_version="5.3", path=str(tmp_path))
        assert out == str(tmp_path)
        # Should not have set env since we passed path
        assert os.environ.get("EUNOMIA_DATA_FOLDER") != str(tmp_path) or old == str(tmp_path)
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old


def test_require_eunomia_downloads_when_zip_missing(tmp_path):
    """require_eunomia downloads when zip is missing (mock _download_with_progress to write file)."""
    def _fake_download(url, dest, *, desc=None):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"fake zip")

    with patch("cdmconnector.eunomia._download_with_progress", side_effect=_fake_download):
        out = require_eunomia("GiBleed", cdm_version="5.3", path=str(tmp_path))
        assert out == str(tmp_path)
        assert (tmp_path / "GiBleed_5.3.zip").exists()


def test_eunomia_accessible_and_queried_with_ibis(tmp_path):
    """Eunomia DuckDB from eunomia_dir can be connected and queried with Ibis via cdm_from_con."""
    import cdmconnector as cc
    import ibis

    _make_eunomia_zip(tmp_path / "GiBleed_5.3.zip", "GiBleed")
    old = os.environ.get("EUNOMIA_DATA_FOLDER")
    os.environ["EUNOMIA_DATA_FOLDER"] = str(tmp_path)
    try:
        path = cc.eunomia_dir("GiBleed", cdm_version="5.3")
    finally:
        if old is not None:
            os.environ["EUNOMIA_DATA_FOLDER"] = old
        else:
            os.environ.pop("EUNOMIA_DATA_FOLDER", None)
    con = ibis.duckdb.connect(path)
    cdm = cc.cdm_from_con(
        con,
        cdm_schema="main",
        write_schema="main",
        cdm_name="eunomia",
    )
    assert "person" in cdm.tables
    assert cc.cdm_tables(cdm)
    df = cc.collect(cdm.person.limit(2))
    assert len(df) <= 2
    con.disconnect()


@pytest.mark.integration
def test_eunomia_quick_start_requires_eunomia_data_folder():
    """
    Exact quick-start snippet: connect to Eunomia and build CDM.

    Requires EUNOMIA_DATA_FOLDER to be set. Uses cached data if present;
    downloads and builds on first run (needs internet then).

    If you see "No CDM tables found": reinstall from source (pip install -e .),
    delete any existing GiBleed_*.duckdb in your data folder so it is rebuilt,
    or use a separate folder for tests, e.g. EUNOMIA_DATA_FOLDER=/path/eunomia_data_py.
    """
    if not os.environ.get("EUNOMIA_DATA_FOLDER"):
        pytest.skip("EUNOMIA_DATA_FOLDER must be set for this test")

    import cdmconnector as cc
    import ibis

    from cdmconnector.exceptions import CDMValidationError

    # ------------------------------------------------------------
    # 1. Connect to Eunomia (DuckDB)
    # ------------------------------------------------------------

    path = cc.eunomia_dir("GiBleed", cdm_version="5.3")

    con = ibis.duckdb.connect(path)

    try:
        cdm = cc.cdm_from_con(
            con,
            cdm_schema="main",
            write_schema="main",
            cdm_name="eunomia",
        )
    except CDMValidationError as e:
        if "No CDM tables found" in str(e):
            pytest.skip(
                "Eunomia DuckDB has no CDM tables (reinstall from source: pip install -e ., "
                "then delete GiBleed_*.duckdb in EUNOMIA_DATA_FOLDER so it is rebuilt; "
                "or use a separate folder for tests, e.g. EUNOMIA_DATA_FOLDER=/path/eunomia_data_py)"
            )
        raise
    finally:
        try:
            con.disconnect()
        except Exception:
            pass

    print(cdm.tables)

    assert cdm.tables
    assert "person" in cdm.tables
    cdm.disconnect()
