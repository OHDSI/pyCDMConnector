# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration for CDMConnector."""

from __future__ import annotations

import logging

logger = logging.getLogger("cdmconnector")


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name (e.g. __name__).

    Parameters
    ----------
    name : str
        Module name; if not already prefixed with "cdmconnector.", it is prefixed.

    Returns
    -------
    logging.Logger
        Logger instance for cdmconnector (or cdmconnector.{name}).
    """
    if name.startswith("cdmconnector."):
        return logging.getLogger(name)
    return logging.getLogger(f"cdmconnector.{name}")


def create_log_file(path: str | None = None) -> str:
    """Create a structured log file and attach a file handler to the cdmconnector logger.

    Parameters
    ----------
    path : str or None
        Path to the log file. If None, uses ``cdmconnector.log`` in the
        current directory.

    Returns
    -------
    str
        Path to the log file.
    """
    import datetime
    from pathlib import Path

    if path is None:
        path = "cdmconnector.log"

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    root_logger = logging.getLogger("cdmconnector")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    with open(log_path, "a") as f:
        f.write(f"\n--- CDMConnector log started at {datetime.datetime.now().isoformat()} ---\n")

    return str(log_path)


def log_message(message: str, level: str = "info") -> None:
    """Log a message at the specified level.

    Parameters
    ----------
    message : str
        Message to log.
    level : str
        Log level: "debug", "info", "warning", "error", or "critical".
    """
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)


def summarise_log_file(path: str) -> dict[str, int]:
    """Parse a cdmconnector log file and summarize by log level.

    Parameters
    ----------
    path : str
        Path to the log file.

    Returns
    -------
    dict[str, int]
        Mapping of log level to count of messages.
    """
    from pathlib import Path

    log_path = Path(path)
    if not log_path.exists():
        return {}

    counts: dict[str, int] = {
        "DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0,
    }
    with open(log_path) as f:
        for line in f:
            for lv in counts:
                if f"| {lv}" in line:
                    counts[lv] += 1
                    break

    return counts
