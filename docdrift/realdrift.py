"""Mine *real* documentation drift from git history.

A **doc-fix pair** is a function, at some commit, whose docstring was edited
while its code (AST with the docstring removed) stayed byte-identical. The
pre-commit (code, old docstring) is the drifted / deficient version (label 1)
and the post-commit (code, new docstring) the corrected one (label 0). Because
the code is identical, a detector can only separate the pair by judging the
docstring against the code -- exactly the static drift task -- and the labels
are produced by developers, not by our mutation operators (no circularity
with the synthetic benchmark).

Filters (all independent of the detectors' features):
  * small commits only (<= MAX_FILES .py files, <= MAX_LINES changed lines);
  * a commit editing more than MAX_DOC_EDITS docstrings is a style sweep
    (e.g. numpydoc conversion) and is dropped entirely;
  * trivial edits are dropped: whitespace/punctuation/markup-only changes and
    single-token typo fixes (edit distance <= 2);
  * the same length limits as the synthetic corpus (docstring >= 8 words,
    3..80 lines).
"""
from __future__ import annotations

import ast
import re
import subprocess
import textwrap
import warnings
from dataclasses import dataclass, asdict

MAX_FILES, MAX_LINES, MAX_DOC_EDITS = 3, 60, 3
MIN_DOC_WORDS, MIN_LINES, MAX_FUNC_LINES = 8, 3, 80
SKIP_PATH = re.compile(r"(^|/)(tests?|testing|benchmarks?|asv_bench|doc|docs|examples?)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
_TOK = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class DocFixPair:
    repo: str
    project: str
    commit: str
    parent: str
    date: str
    message: str
    file: str
    qualname: str
    source_old: str     # ast.unparse(function) with the OLD docstring
    doc_old: str
    source_new: str     # ast.unparse(function) with the NEW docstring
    doc_new: str
    n_lines: int


# ----------------------------------------------------------------- git helpers
class Git:
    def __init__(self, path: str):
        self.path = path
        self._cat = subprocess.Popen(["git", "-C", path, "cat-file", "--batch"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def run(self, *args: str) -> str:
        return subprocess.run(["git", "-C", self.path, *args], check=True,
                              capture_output=True).stdout.decode("utf-8", "replace")

    def blob(self, rev: str, path: str) -> str | None:
        self._cat.stdin.write(f"{rev}:{path}\n".encode()); self._cat.stdin.flush()
        header = self._cat.stdout.readline().split()
        if len(header) < 3 or header[1] == b"missing":
            return None
        size = int(header[2])
        data = self._cat.stdout.read(size); self._cat.stdout.read(1)
        return data.decode("utf-8", "replace")

    def close(self):
        self._cat.stdin.close(); self._cat.wait(); self._cat.stdout.close()


def candidate_commits(git: Git, since: str = "2012-01-01"):
    """Yield (sha, parent, date, subject, [py files]) for small non-merge commits."""
    out = git.run("log", "--no-merges", f"--since={since}", "--numstat",
                  "--format=\x01%H %P\x02%cI\x02%s", "--", "*.py")
    for block in out.split("\x01")[1:]:
        head, _, rest = block.partition("\n")
        shas, date, subject = head.split("\x02", 2)
        parts = shas.split()
        if len(parts) != 2:          # root commit or merge
            continue
        files, lines = [], 0
        for ln in rest.strip().splitlines():
            cols = ln.split("\t")
            if len(cols) != 3 or not cols[2].endswith(".py") or cols[0] == "-":
                continue
            lines += int(cols[0]) + int(cols[1])
            if not SKIP_PATH.search(cols[2]) and "=>" not in cols[2]:
                files.append(cols[2])
        if files and len(files) <= MAX_FILES and lines <= MAX_LINES:
            yield parts[0], parts[1], date, subject, files


# ----------------------------------------------------------------- AST helpers
def _functions(src: str) -> dict[str, ast.AST]:
    """qualname -> FunctionDef for every (nested) function in a module."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # old code: invalid escape sequences etc.
            tree = ast.parse(src)
    except Exception:
        return {}
    out = {}

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}{ch.name}"
                if not isinstance(ch, ast.ClassDef):
                    out.setdefault(q, ch)
                visit(ch, q + ".")
    visit(tree, "")
    return out


def _code_key(fn) -> str:
    """AST dump of the function with its docstring removed (formatting-free)."""
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[])) + ast.dump(fn.args) + \
        "".join(ast.dump(d) for d in fn.decorator_list) + (ast.dump(fn.returns) if fn.returns else "")


def _render(src_lines: list[str], fn) -> str | None:
    # same extraction as docdrift.corpus: from the `def` line (no decorators), dedented
    seg = textwrap.dedent("\n".join(src_lines[fn.lineno - 1:fn.end_lineno]))
    try:
        return ast.unparse(ast.parse(seg))
    except Exception:
        return None


_DIRECTIVE = re.compile(r"^([ \t]*)\.\.[ \t]+(versionadded|versionchanged|deprecated)::.*(?:\n(?:\1[ \t]+\S.*|[ \t]*(?=\n)))*", re.M)
_ROLE = re.compile(r":\w+(?::\w+)?:(?=`)")          # :class:`X`  -> `X`
_LINK_TARGET = re.compile(r"`([^`<]*?)\s*<[^>`]*>`")  # `Text <target>` -> `Text`


def normalise_doc(doc: str) -> str:
    """Drop version-history directives and reST markup that carry no claim."""
    doc = _DIRECTIVE.sub("", doc)
    doc = _ROLE.sub("", doc)
    doc = _LINK_TARGET.sub(r"`\1`", doc)
    return doc.replace("'", "").replace("’", "")


def _norm_tokens(doc: str) -> list[str]:
    return _TOK.findall(normalise_doc(doc).lower())


# --- what an edit is about (analysis metadata; never used to pick labels) ---
_RETURN_W = {"return", "returns", "returned", "returning", "yield", "yields", "yielded", "generator",
             "iterator", "rtype"}
_RAISE_W = {"raise", "raises", "raised", "raising", "exception", "exceptions"}
_TYPE_W = {"list", "lists", "tuple", "tuples", "dict", "dicts", "dictionary", "str", "string", "strings",
           "int", "integer", "integers", "float", "floats", "bool", "boolean", "none", "bytes", "set",
           "array", "arrays", "ndarray", "callable", "iterable", "sequence", "mapping", "object", "type"}
_DEFAULT_W = {"default", "defaults", "optional"}
INTERFACE_CATEGORIES = ("param", "return", "raise", "type", "default")


def _changed_tokens(old: str, new: str) -> set[str]:
    from collections import Counter
    a, b = Counter(_norm_tokens(old)), Counter(_norm_tokens(new))
    return set((a - b) | (b - a))


def edit_categories(old: str, new: str, source: str) -> list[str]:
    """Which interface claims an old->new docstring edit touches."""
    ch = _changed_tokens(old, new)
    try:
        fn = ast.parse(source).body[0]
        a = fn.args
        params = {p.arg.lower() for p in getattr(a, "posonlyargs", []) + a.args + a.kwonlyargs
                  + [x for x in (a.vararg, a.kwarg) if x]} - {"self", "cls"}
    except Exception:
        params = set()
    cats = []
    if ch & params:
        cats.append("param")
    if ch & _RETURN_W:
        cats.append("return")
    if ch & _RAISE_W or any(t.endswith(("error", "exception", "warning")) for t in ch):
        cats.append("raise")
    if ch & _TYPE_W:
        cats.append("type")
    if ch & _DEFAULT_W or any(t.isdigit() for t in ch):
        cats.append("default")
    return cats or ["prose"]


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_trivial_edit(old: str, new: str) -> bool:
    """Whitespace / punctuation / markup-only, or a single small typo fix."""
    a, b = _norm_tokens(old), _norm_tokens(new)
    if a == b:
        return True
    if len(a) == len(b):
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        if len(diff) == 1 and _edit_distance(*diff[0]) <= 2:
            return True
    return False


# ----------------------------------------------------------------- mining
def mine_repo(path: str, repo: str, project: str, since: str = "2012-01-01", log=None):
    git = Git(path)
    pairs = []
    try:
        for sha, parent, date, subject, files in candidate_commits(git, since):
            commit_pairs = []
            n_doc_edits = 0
            for f in files:
                old_src, new_src = git.blob(parent, f), git.blob(sha, f)
                if not old_src or not new_src:
                    continue
                old_fns, new_fns = _functions(old_src), _functions(new_src)
                old_lines, new_lines = old_src.splitlines(), new_src.splitlines()
                for q, fn_new in new_fns.items():
                    fn_old = old_fns.get(q)
                    if fn_old is None:
                        continue
                    d_old, d_new = ast.get_docstring(fn_old), ast.get_docstring(fn_new)
                    if not d_old or not d_new or d_old == d_new:
                        continue
                    n_doc_edits += 1
                    if _code_key(fn_old) != _code_key(fn_new):
                        continue
                    if is_trivial_edit(d_old, d_new):
                        continue
                    n_lines = fn_new.end_lineno - fn_new.lineno + 1
                    if (len(d_old.split()) < MIN_DOC_WORDS or len(d_new.split()) < MIN_DOC_WORDS
                            or not MIN_LINES <= n_lines <= MAX_FUNC_LINES):
                        continue
                    s_old, s_new = _render(old_lines, fn_old), _render(new_lines, fn_new)
                    if not s_old or not s_new:
                        continue
                    commit_pairs.append(DocFixPair(repo, project, sha, parent, date, subject, f, q,
                                                   s_old, d_old, s_new, d_new, n_lines))
            if commit_pairs and n_doc_edits <= MAX_DOC_EDITS:
                pairs.extend(commit_pairs)
    finally:
        git.close()
    if log:
        log(f"{repo}: {len(pairs)} doc-fix pairs")
    return [asdict(p) for p in pairs]
