#!/usr/bin/env python3
"""Check the code stored in chapters/*/code/.

* every .py file is syntax-checked;
* a .py file whose first 5 lines contain the marker "# ci: run" is also executed
  (from its own folder, 5-minute timeout, headless matplotlib).

Use the marker only for fast scripts that need no downloads or GPU.
"""
import os
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
failed = []

for py in sorted(ROOT.glob("chapters/*/code/*.py")):
    rel = py.relative_to(ROOT)
    try:
        py_compile.compile(str(py), doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"SYNTAX ERROR {rel}: {exc}")
        failed.append(rel)
        continue
    head = "".join(py.read_text(encoding="utf-8").splitlines(True)[:5])
    if "# ci: run" in head:
        env = dict(os.environ, MPLBACKEND="Agg")
        res = subprocess.run([sys.executable, py.name], cwd=py.parent, env=env,
                             capture_output=True, text=True, timeout=300)
        status = "ok" if res.returncode == 0 else "FAILED"
        print(f"run {status}: {rel}")
        if res.returncode != 0:
            print(res.stderr[-800:])
            failed.append(rel)
    else:
        print(f"syntax ok: {rel}")

if failed:
    print(f"\n{len(failed)} file(s) failed")
    sys.exit(1)
print("\nall code checks passed")
