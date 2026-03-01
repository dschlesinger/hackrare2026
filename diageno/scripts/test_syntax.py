#!/usr/bin/env python3
"""Test that Streamlit UI files can be parsed (syntax check)."""

import ast
import sys
from pathlib import Path

ui_dir = Path("/Users/bhargav/Desktop/hackrare2026/diageno/ui")
files = list(ui_dir.rglob("*.py"))

errors = []
for f in files:
    try:
        ast.parse(f.read_text())
        print(f"  OK: {f.relative_to(ui_dir)}")
    except SyntaxError as e:
        errors.append((f, e))
        print(f"  FAIL: {f.relative_to(ui_dir)}: {e}")

if errors:
    print(f"\n{len(errors)} syntax errors found!")
    sys.exit(1)
else:
    print(f"\nAll {len(files)} UI files parse OK")

# Also check all other Python files in diageno
diageno_dir = Path("/Users/bhargav/Desktop/hackrare2026/diageno")
all_py = [f for f in diageno_dir.rglob("*.py") if "data" not in str(f) and ".venv" not in str(f)]

print(f"\nChecking all {len(all_py)} Python files...")
py_errors = []
for f in all_py:
    try:
        ast.parse(f.read_text())
    except SyntaxError as e:
        py_errors.append((f, e))
        print(f"  SYNTAX ERROR: {f.relative_to(diageno_dir)}: {e}")

if py_errors:
    print(f"\n{len(py_errors)} total syntax errors!")
    sys.exit(1)
else:
    print(f"All {len(all_py)} Python files parse OK")
    print("\n=== SYNTAX CHECK PASSED ===")
