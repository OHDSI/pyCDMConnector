"""Regression tests for optional dependency imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_package_imports_without_altair() -> None:
    """Importing cdmconnector should not require Altair."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

    code = """
import builtins

real_import = builtins.__import__

def fake_import(name, *args, **kwargs):
    if name == "altair":
        raise ModuleNotFoundError("No module named 'altair'")
    return real_import(name, *args, **kwargs)

builtins.__import__ = fake_import

import cdmconnector as cc

assert callable(cc.cdm_from_con)
assert callable(cc.vis_table)
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
