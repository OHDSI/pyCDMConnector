#!/usr/bin/env python3
"""Generate ibis_queries/*.py stubs for every queries/*.md (1:1)."""

import re
from pathlib import Path

QUERIES_DIR = Path(__file__).resolve().parent.parent / "queries"
IBS_DIR = Path(__file__).resolve().parent.parent / "ibis_queries"


def extract_sql(md_path: Path) -> str | None:
    content = md_path.read_text()
    m = re.search(r"```sql\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_title(md_path: Path) -> str:
    content = md_path.read_text()
    for line in content.splitlines():
        if line.startswith("# ") and ": " in line:
            return line[2:].strip()
    return md_path.stem


def stub_py_content(rel: Path, title: str, sql: str | None, md_path: Path) -> str:
    name = rel.stem
    sql_escaped = (sql or "# (no SQL block)").replace('"""', '\\"\\"\\"')
    return f'''# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0
# Translated from queries/{rel} to cdmconnector with Ibis.

"""{title} - Ibis implementation.

Original SQL (queries/{rel}):

```sql
{sql_escaped}
```
"""

from cdmconnector import collect


def run(cdm, **kwargs):
    """Execute the query. Returns Ibis expression; call collect(run(cdm)) to materialize."""
    # TODO: full Ibis translation of the SQL above
    # Placeholder: return person table limited to 0 rows so this module is runnable
    return cdm["person"].limit(0)
'''



def main():
    for md_path in sorted(QUERIES_DIR.rglob("*.md")):
        rel = md_path.relative_to(QUERIES_DIR)
        py_path = IBS_DIR / rel.with_suffix(".py")
        if py_path.exists():
            continue
        py_path.parent.mkdir(parents=True, exist_ok=True)
        title = extract_title(md_path)
        sql = extract_sql(md_path)
        content = stub_py_content(rel, title, sql, md_path)
        py_path.write_text(content, encoding="utf-8")
        print(f"Wrote {py_path.relative_to(IBS_DIR.parent)}")


if __name__ == "__main__":
    main()
