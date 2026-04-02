#!/usr/bin/env python3
"""Extract SQL and metadata from queries/*.md for reference when writing ibis_queries/*.py."""

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


def main():
    for md_path in sorted(QUERIES_DIR.rglob("*.md")):
        rel = md_path.relative_to(QUERIES_DIR)
        py_path = IBS_DIR / rel.with_suffix(".py")
        sql = extract_sql(md_path)
        title = extract_title(md_path)
        print(f"{rel} -> {py_path.relative_to(IBS_DIR.parent)}")
        if sql:
            print(f"  Title: {title}")
            print(f"  SQL lines: {len(sql.splitlines())}")


if __name__ == "__main__":
    main()
