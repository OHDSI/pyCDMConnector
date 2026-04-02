# Copyright 2025 DARWIN EU®
#
# This file is part of CDMConnector
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eunomia example dataset helpers: download_eunomia_data, eunomia_dir, example_datasets, eunomia_is_available, require_eunomia."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from tqdm import tqdm

from cdmconnector.exceptions import EunomiaError
from cdmconnector.logging_config import get_logger

logger = get_logger(__name__)

# Datasets available (from R CDMConnector exampleDatasets())
EXAMPLE_DATASETS = (
    "GiBleed",
    "synthea-allergies-10k",
    "synthea-anemia-10k",
    "synthea-breast_cancer-10k",
    "synthea-contraceptives-10k",
    "synthea-covid19-10k",
    "synthea-covid19-200k",
    "synthea-dermatitis-10k",
    "synthea-heart-10k",
    "synthea-hiv-10k",
    "synthea-lung_cancer-10k",
    "synthea-medications-10k",
    "synthea-metabolic_syndrome-10k",
    "synthea-opioid_addiction-10k",
    "synthea-rheumatoid_arthritis-10k",
    "synthea-snf-10k",
    "synthea-surgery-10k",
    "synthea-total_joint_replacement-10k",
    "synthea-veteran_prostate_cancer-10k",
    "synthea-veterans-10k",
    "synthea-weight_loss-10k",
    "synpuf-1k",
    "synpuf-110k",
    "empty_cdm",
    "Synthea27NjParquet",
)


def example_datasets() -> tuple[str, ...]:
    """Return the list of available Eunomia example dataset names.

    Returns
    -------
    tuple[str, ...]
        Dataset names (e.g. "GiBleed", "synthea-allergies-10k", ...).
    """
    return EXAMPLE_DATASETS


def download_eunomia_data(
    dataset_name: str = "GiBleed",
    cdm_version: str = "5.4",
    path: str | Path | None = None,
    overwrite: bool = False,
) -> str:
    """
    Download Eunomia data from CDMConnector blob storage (or GitHub for Synthea27NjParquet).

    Parameters
    ----------
    dataset_name : str, optional
        One of example_datasets() (default "GiBleed").
    cdm_version : str, optional
        "5.3" or "5.4" (default "5.4").
    path : str or Path or None, optional
        Directory to save the ZIP; default EUNOMIA_DATA_FOLDER env var.
    overwrite : bool, optional
        If True, overwrite existing ZIP (default False).

    Returns
    -------
    str
        Path to the directory where data is stored.
    """
    if dataset_name not in example_datasets():
        raise EunomiaError(f"Unknown dataset: {dataset_name}. Use one of example_datasets().")
    if cdm_version not in ("5.3", "5.4"):
        raise EunomiaError("cdm_version must be '5.3' or '5.4'.")
    # R CDMConnector: Synthea27NjParquet only in 5.4; 5.4 only synpuf-1k, Synthea27NjParquet, empty_cdm
    if dataset_name == "Synthea27NjParquet" and cdm_version == "5.3":
        raise EunomiaError("Synthea27NjParquet is only available in CDM version 5.4")
    if (
        cdm_version == "5.4"
        and dataset_name not in ("synpuf-1k", "Synthea27NjParquet", "empty_cdm")
    ):
        raise EunomiaError(f"{dataset_name} is only available in CDM version 5.3")
    if path is None:
        path = os.environ.get("EUNOMIA_DATA_FOLDER", "")
    if not path:
        raise EunomiaError(
            "path must be set or EUNOMIA_DATA_FOLDER environment variable. "
            "e.g. export EUNOMIA_DATA_FOLDER=/tmp/eunomia"
        )
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    zip_name = f"{dataset_name}_{cdm_version}.zip"
    zip_path = path / zip_name
    if zip_path.exists() and not overwrite:
        return str(path)
    if dataset_name == "Synthea27NjParquet":
        url = "https://github.com/OHDSI/EunomiaDatasets/raw/main/datasets/Synthea27NjParquet/Synthea27NjParquet_5.4.zip"
    else:
        url = f"https://cdmconnectordata.blob.core.windows.net/cdmconnector-example-data/{zip_name}"
    try:
        _download_with_progress(url, zip_path, desc=zip_name)
    except Exception as e:
        raise EunomiaError(f"Download failed: {e}") from e
    return str(path)


def _download_with_progress(
    url: str,
    dest: Path,
    *,
    desc: str | None = None,
    chunk_size: int = 65536,
) -> None:
    """Download url to dest with a progress bar (uses requests + tqdm)."""
    import requests

    desc = desc or dest.name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        with open(dest, "wb") as f:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                miniters=1,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def _corrupt_zip_error_message(
    cache_dir: Path,
    zip_name: str,
    db_name: str,
) -> str:
    """Error message asking the user to delete corrupt zip and duckdb in the data folder."""
    return (
        f"Corrupt or partial zip file. Please look in {cache_dir} (EUNOMIA_DATA_FOLDER) "
        f"and delete {zip_name} and {db_name}, then try again."
    )


def _ensure_valid_zip(
    zip_path: Path,
    *,
    dataset_name: str,
    cdm_version: str,
    cache_dir: Path,
) -> None:
    """If zip_path exists but is not a valid zip (corrupt/partial/HTML), raise with instructions."""
    if not zip_path.exists():
        download_eunomia_data(
            dataset_name=dataset_name, cdm_version=cdm_version, path=cache_dir
        )
        return
    db_name = f"{dataset_name}_{cdm_version}_py.duckdb"
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.testzip()
    except zipfile.BadZipFile:
        raise EunomiaError(
            _corrupt_zip_error_message(cache_dir, zip_path.name, db_name)
        ) from None


def eunomia_dir(
    dataset_name: str = "GiBleed",
    cdm_version: str = "5.3",
    path: str | Path | None = None,
    database_file: str | Path | None = None,
) -> str:
    """
    Return path to a DuckDB file containing the Eunomia dataset.

    EUNOMIA_DATA_FOLDER must be set; it is the cache location (zip and DuckDB).
    If the dataset is not yet available, downloads it. If the DuckDB file does not
    exist, creates it from the ZIP. Always returns a path to a *copy* of the DB.
    path (or database_file) is where to copy the DB; if None, a temp file is used.

    Parameters
    ----------
    dataset_name : str, optional
        One of example_datasets() (default "GiBleed").
    cdm_version : str, optional
        "5.3" or "5.4" (default "5.3").
    path : str or Path or None, optional
        Where to copy the DuckDB file (file or directory); if None, a temp file.
    database_file : str or Path or None, optional
        Alias for path (where to copy the DB). path takes precedence.

    Returns
    -------
    str
        Path to a copy of the DuckDB database file (cached DB is never returned).
    """
    import tempfile

    cache_dir = os.environ.get("EUNOMIA_DATA_FOLDER")
    if not cache_dir:
        raise EunomiaError(
            "EUNOMIA_DATA_FOLDER must be set. "
            "e.g. export EUNOMIA_DATA_FOLDER=/path/to/eunomia_data"
        )
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_folder_path = str(cache_dir.resolve())
    print("eunomia_dir: data folder =", data_folder_path)
    logger.info("eunomia_dir: data folder=%s", data_folder_path)

    zip_name = f"{dataset_name}_{cdm_version}.zip"
    zip_path = cache_dir / zip_name
    zip_exists = zip_path.exists()
    logger.info("eunomia_dir: zip %s exists=%s", zip_path.name, zip_exists)
    if not zip_exists:
        logger.info("eunomia_dir: downloading %s", zip_path.name)
        download_eunomia_data(
            dataset_name=dataset_name, cdm_version=cdm_version, path=cache_dir
        )

    db_name = f"{dataset_name}_{cdm_version}_py.duckdb"
    db_path = cache_dir / db_name
    db_exists = db_path.exists()
    logger.info("eunomia_dir: DuckDB %s exists=%s", db_path.name, db_exists)
    if not db_exists:
        # Ensure zip is valid (re-download if corrupt/partial/HTML from prior run)
        _ensure_valid_zip(zip_path, dataset_name=dataset_name, cdm_version=cdm_version, cache_dir=cache_dir)
        logger.info("eunomia_dir: building DuckDB from zip")
        _build_duckdb_from_zip(zip_path, db_path, dataset_name)

    import shutil
    out = path or database_file
    if out is None:
        out = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False).name
    else:
        out = Path(out)
        if out.suffix != ".duckdb" and (not out.exists() or out.is_dir()):
            out = out / db_name
        out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(db_path, out)
    logger.info("eunomia_dir: returning copy path=%s", out)
    return str(out)


def _build_duckdb_from_zip(zip_path: Path, db_path: Path, dataset_name: str) -> None:
    """Unzip parquet files and load into a new DuckDB database.

    Parameters
    ----------
    zip_path : Path
        Path to the Eunomia ZIP file.
    db_path : Path
        Path for the new DuckDB file.
    dataset_name : str
        Dataset name (used to locate parquet dir inside ZIP).
    """
    import shutil

    import ibis
    tmp = db_path.parent / "_eunomia_extract"
    tmp.mkdir(exist_ok=True)
    cache_dir = zip_path.parent
    zip_name = zip_path.name
    db_name = db_path.name
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)
        extract_dir = tmp / dataset_name
        if not extract_dir.exists():
            extract_dir = tmp
        parquet_files = sorted(extract_dir.glob("*.parquet"))
        if not parquet_files:
            raise EunomiaError(f"No parquet files in {extract_dir}")
        con = ibis.duckdb.connect(str(db_path))
        for p in parquet_files:
            name = p.stem.lower()
            tbl = con.read_parquet(str(p))
            con.create_table(name, obj=tbl.to_pyarrow(), overwrite=True)
        tables_in_db = con.list_tables()
        con.disconnect()
        if not tables_in_db:
            raise EunomiaError(
                "No tables were persisted to the DuckDB file. "
                "Reinstall from source: pip install -e ."
            )
    except zipfile.BadZipFile:
        raise EunomiaError(
            _corrupt_zip_error_message(cache_dir, zip_name, db_name)
        ) from None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def eunomia_is_available(
    dataset_name: str = "GiBleed",
    cdm_version: str = "5.3",
    path: str | Path | None = None,
) -> bool:
    """Return True if the Eunomia dataset ZIP is present in the data folder.

    Parameters
    ----------
    dataset_name : str, optional
        Dataset name (default "GiBleed").
    cdm_version : str, optional
        "5.3" or "5.4" (default "5.3").
    path : str or Path or None, optional
        Data folder; default EUNOMIA_DATA_FOLDER env.

    Returns
    -------
    bool
        True if {dataset_name}_{cdm_version}.zip exists in path.

    Raises
    ------
    EunomiaError
        If EUNOMIA_DATA_FOLDER is not set and path is not provided.
    """
    path = path or os.environ.get("EUNOMIA_DATA_FOLDER", "")
    if not path:
        raise EunomiaError(
            "Set the environment variable EUNOMIA_DATA_FOLDER to the eunomia cache location"
        )
    if cdm_version not in ("5.3", "5.4"):
        raise EunomiaError("cdm_version must be '5.3' or '5.4'.")
    return (Path(path) / f"{dataset_name}_{cdm_version}.zip").exists()


def require_eunomia(
    dataset_name: str = "GiBleed",
    cdm_version: str = "5.3",
    path: str | Path | None = None,
) -> str:
    """
    Ensure the Eunomia dataset is available; download if needed.

    Sets EUNOMIA_DATA_FOLDER to a temp directory if unset. Creates the folder
    if it does not exist. Downloads the dataset if not present (matches R
    requireEunomia / eunomiaDir behavior).

    Parameters
    ----------
    dataset_name : str, optional
        One of example_datasets() (default "GiBleed").
    cdm_version : str, optional
        "5.3" or "5.4" (default "5.3").
    path : str or Path or None, optional
        Override data folder (default: use EUNOMIA_DATA_FOLDER env).

    Returns
    -------
    str
        Path to the data folder.
    """
    if dataset_name not in example_datasets():
        raise EunomiaError(f"Unknown dataset: {dataset_name}. Use one of example_datasets().")
    if cdm_version not in ("5.3", "5.4"):
        raise EunomiaError("cdm_version must be '5.3' or '5.4'.")
    if path is None:
        path = os.environ.get("EUNOMIA_DATA_FOLDER", "")
    if not path:
        import tempfile
        path = tempfile.gettempdir()
        os.environ["EUNOMIA_DATA_FOLDER"] = path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not eunomia_is_available(dataset_name=dataset_name, cdm_version=cdm_version, path=str(path)):
        try:
            download_eunomia_data(dataset_name=dataset_name, cdm_version=cdm_version, path=str(path))
        except EunomiaError as exc:
            logger.warning(
                "Unable to download Eunomia dataset %s %s into %s: %s",
                dataset_name,
                cdm_version,
                path,
                exc,
            )
    return str(path)
