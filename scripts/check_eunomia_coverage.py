#!/usr/bin/env python3
"""Exit 0 if src/cdmconnector/eunomia.py coverage >= 85%, else 1."""
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_eunomia.py",
        "tests/test_cdm.py",
        "-k",
        "eunomia",
        "--cov=src/cdmconnector",
        "--cov-report=term",
        "-q",
    ],
    capture_output=True,
    text=True,
    cwd=repo_root,
)
out = result.stdout + result.stderr
# Parse "src/cdmconnector/eunomia.py  98  0  ...  99%"
for line in out.splitlines():
    if "eunomia.py" in line:
        parts = line.split()
        for p in parts:
            if p.endswith("%"):
                pct = int(p.rstrip("%"))
                if pct >= 85:
                    print(f"eunomia coverage: {pct}% (>= 85%)")
                    sys.exit(0)
                print(f"eunomia coverage: {pct}% (required >= 85%)")
                sys.exit(1)
        break
else:
    print("Could not find eunomia coverage line")
    sys.exit(1)
