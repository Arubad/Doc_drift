"""Mine (function source, docstring) pairs from a Python code base.

By default the corpus is the local CPython standard library, which is large,
well documented, and available offline on every machine. Any directory of
Python files can be passed instead (e.g. a cloned project)."""
from __future__ import annotations

import ast
import os
import sysconfig
from dataclasses import dataclass, asdict

import pandas as pd

SKIP_DIRS = {"test", "tests", "idlelib", "lib2to3", "__pycache__",
             "turtledemo", "tkinter", "ensurepip", "venv"}


@dataclass
class FunctionRecord:
    uid: str
    project: str          # top-level package / module (used for group splits)
    file: str
    name: str
    lineno: int
    source: str           # full function source incl. def line and docstring
    docstring: str
    n_params: int
    n_lines: int


def _iter_py_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _project_of(path: str, root: str) -> str:
    """Top-level package name; third-party packages installed under
    dist-packages / site-packages are named by their own top-level package."""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if parts[0] in ("dist-packages", "site-packages") and len(parts) > 1:
        parts = parts[1:]
    return parts[0] if len(parts) > 1 else os.path.splitext(parts[0])[0]


def mine(root: str | None = None, min_doc_words: int = 8, min_body_lines: int = 3,
         max_lines: int = 80, limit_per_file: int = 40) -> pd.DataFrame:
    root = root or sysconfig.get_paths()["stdlib"]
    records = []
    for path in _iter_py_files(root):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                src = fh.read()
            tree = ast.parse(src)
        except Exception:
            continue
        lines = src.splitlines()
        n_file = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc or len(doc.split()) < min_doc_words:
                continue
            end = getattr(node, "end_lineno", None)
            if end is None:
                continue
            n_lines = end - node.lineno + 1
            if n_lines > max_lines:
                continue
            body_stmts = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
            if len(body_stmts) < min_body_lines - 1:
                continue
            seg = "\n".join(lines[node.lineno - 1:end])
            # dedent so every function starts at column 0
            try:
                import textwrap
                seg = textwrap.dedent(seg)
                ast.parse(seg)
            except Exception:
                continue
            params = node.args
            n_params = len(params.args) + len(params.kwonlyargs) + len(getattr(params, "posonlyargs", []))
            uid = f"{os.path.relpath(path, root)}::{node.name}::{node.lineno}"
            records.append(FunctionRecord(uid, _project_of(path, root), os.path.relpath(path, root),
                                          node.name, node.lineno, seg, doc, n_params, n_lines))
            n_file += 1
            if n_file >= limit_per_file:
                break
    df = pd.DataFrame([asdict(r) for r in records])
    return df
