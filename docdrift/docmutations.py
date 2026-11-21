"""Doc-side drift: mutate the DOCSTRING, keep the code fixed.

Code-side operators (docdrift/mutations.py) change the code under an unchanged docstring.
Real drift is fixed by editing the docstring, so the realistic training signal is the
mirror image: a docstring that no longer matches unchanged code. Each operator takes a
function source and returns the source rendered with a mutated docstring, or None when it
does not apply. All outputs -- clean, mutated, benign -- go through `render`, which
re-inserts the cleaned docstring and calls ast.unparse, so formatting cannot leak the label.
"""
from __future__ import annotations

import ast
import re
import textwrap
import zlib

import numpy as np

from .features import _param_names, _exceptions_raised, _defaults

DOC_DRIFT = ["doc_rename_param", "doc_drop_param", "doc_return_yield", "doc_drop_raise",
             "doc_add_raise", "doc_change_number", "doc_swap_type", "doc_negate"]
DOC_BENIGN = ["doc_reflow", "doc_synonym", "doc_drop_section"]
DOC_INTERFACE = {"doc_rename_param", "doc_drop_param", "doc_return_yield", "doc_drop_raise",
                 "doc_add_raise", "doc_change_number", "doc_swap_type"}

_SECTION_HEADS = {"args", "arguments", "parameters", "params", "keyword arguments", "returns", "return",
                  "yields", "yield", "raises", "raise", "examples", "example", "see also", "notes",
                  "note", "references", "attributes", "warns", "other parameters"}
_DROPPABLE = {"examples", "example", "see also", "notes", "note", "references"}
_COMMON = {"a", "an", "the", "is", "of", "to", "in", "and", "or", "if", "it", "be", "on", "for", "as", "x",
           "y", "n", "i", "s", "f", "self", "cls", "data", "value", "name", "type", "default", "key"}
_TYPE_SWAPS = [("list", "tuple"), ("str", "bytes"), ("int", "float"), ("True", "False"),
               ("dict", "list"), ("string", "bytes"), ("integer", "float"), ("boolean", "string")]
_SYNONYMS = [("compute", "calculate"), ("computes", "calculates"), ("create", "construct"),
             ("creates", "constructs"), ("check", "verify"), ("checks", "verifies"),
             ("specified", "given"), ("provided", "supplied"), ("helper", "utility"),
             ("method", "routine"), ("instance", "object"), ("useful", "handy"),
             ("obtain", "get"), ("retrieve", "fetch"), ("retrieves", "fetches"),
             ("currently", "at present"), ("e.g.", "for example"), ("i.e.", "that is")]
_NEGATABLE = re.compile(r"\b(is|are|will|should|must|can|does|do|has|have)\b(?! not\b)")
_RAISE_TEMPLATES = {
    "rst": ":raises {E}: if ``{p}`` is not valid.",
    "google": "Raises:\n    {E}: If `{p}` is not supported.",
    "numpy": "Raises\n------\n{E}\n    If `{p}` is invalid.",
    "prose": "Raises {E} if {p} is invalid.",
}


# ----------------------------------------------------------------- rendering
def _fn(source: str):
    tree = ast.parse(source)
    return tree, tree.body[0]


def render(source: str, new_doc: str | None = None) -> str | None:
    """ast.unparse of the function with its docstring replaced by the CLEANED text
    (new_doc if given, else the original). Used for every benchmark row."""
    try:
        tree, fn = _fn(source)
    except Exception:
        return None
    doc = ast.get_docstring(fn)
    if doc is None:
        return None
    fn.body[0] = ast.Expr(ast.Constant(new_doc if new_doc is not None else doc))
    return ast.unparse(tree)


def _style(doc: str) -> str:
    if re.search(r":param\b|:raises?\b|:returns?:", doc):
        return "rst"
    if re.search(r"^\s*(Args|Arguments|Returns|Raises):\s*$", doc, re.M):
        return "google"
    if re.search(r"^\s*-{3,}\s*$", doc, re.M):
        return "numpy"
    return "prose"


def _word(w: str) -> re.Pattern:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(w)}(?![A-Za-z0-9_])")


def _case_like(src: str, tgt: str) -> str:
    return tgt[:1].upper() + tgt[1:] if src[:1].isupper() else tgt


# ----------------------------------------------------------------- block helpers
def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _remove_block(lines: list[str], i: int) -> list[str]:
    """Remove line i and its more-indented continuation lines."""
    base, j = _indent(lines[i]), i + 1
    while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > base):
        if not lines[j].strip() and (j + 1 >= len(lines) or _indent(lines[j + 1]) <= base):
            break
        j += 1
    return lines[:i] + lines[j:]


def _drop_empty_sections(lines: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(lines):
        head = lines[i].strip().rstrip(":").lower()
        underline = i + 1 < len(lines) and re.fullmatch(r"\s*-{3,}\s*", lines[i + 1])
        if head in _SECTION_HEADS:
            k = i + (2 if underline else 1)
            while k < len(lines) and not lines[k].strip():
                k += 1
            nxt = lines[k].strip().rstrip(":").lower() if k < len(lines) else None
            if nxt is None or nxt in _SECTION_HEADS:
                i = k
                continue
        out.append(lines[i]); i += 1
    return out


def _tidy(lines: list[str]) -> str:
    text = "\n".join(_drop_empty_sections(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ----------------------------------------------------------------- operators
def doc_rename_param(doc, fn, rng, pools):
    params = [p for p in _param_names(fn) if len(p) > 1 and p.lower() not in _COMMON and _word(p).search(doc)]
    if not params:
        return None
    p = params[rng.integers(len(params))]
    taken = set(_param_names(fn)) | set(re.findall(r"[A-Za-z_]\w*", doc))
    cands = [c for c in pools["params"] if c not in taken]
    if not cands:
        return None
    return _word(p).sub(cands[rng.integers(len(cands))], doc)


def doc_drop_param(doc, fn, rng, pools):
    lines = doc.splitlines()
    params = set(_param_names(fn))
    starts = [i for i, l in enumerate(lines)
              if (m := re.match(r"\s*(?::param\s+(?:[\w\[\], .]+\s+)?(\w+)\s*:|(\*{0,2}\w+)\s*(?:\(.*\))?\s*:(?!:)|(\w+)\s*--)", l))
              and (m.group(1) or m.group(2) or m.group(3) or "").lstrip("*") in params]
    if not starts:
        return None
    i = starts[rng.integers(len(starts))]
    lines = _remove_block(lines, i)
    # drop a matching ':type p:' line for rst
    return _tidy(lines)


def doc_return_yield(doc, fn, rng, pools):
    has_yield = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))
    if has_yield and re.search(r"\b[Yy]ield(s|ed|ing)?\b", doc):
        return re.sub(r"\b([Yy])ield(s|ed|ing)?\b",
                      lambda m: ("R" if m.group(1) == "Y" else "r") + "eturn" + {"s": "s", "ed": "ed", "ing": "ing"}.get(m.group(2) or "", ""), doc)
    if not has_yield and re.search(r"\b[Rr]eturn(s|ed|ing)?\b", doc):
        return re.sub(r"\b([Rr])eturn(s|ed|ing)?\b",
                      lambda m: ("Y" if m.group(1) == "R" else "y") + "ield" + {"s": "s", "ed": "ed", "ing": "ing"}.get(m.group(2) or "", ""), doc)
    return None


def doc_drop_raise(doc, fn, rng, pools):
    raised = _exceptions_raised(fn)
    lines = doc.splitlines()
    starts = [i for i, l in enumerate(lines)
              if re.match(r"\s*:raises?\b", l) or any(_word(e).search(l) for e in raised)]
    if not starts:
        return None
    i = starts[rng.integers(len(starts))]
    return _tidy(_remove_block(lines, i))


def doc_add_raise(doc, fn, rng, pools):
    raised = _exceptions_raised(fn)
    cands = [e for e in pools["exceptions"] if e not in raised and not _word(e).search(doc)]
    params = _param_names(fn)
    if not cands or not params:
        return None
    text = _RAISE_TEMPLATES[_style(doc)].format(E=cands[rng.integers(len(cands))], p=params[rng.integers(len(params))])
    return doc.rstrip() + "\n\n" + text


def doc_change_number(doc, fn, rng, pools):
    defaults = {str(d) for d in _defaults(fn)}
    nums = [m for m in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)", doc) if m.group() in defaults]
    if not nums:
        return None
    m = nums[rng.integers(len(nums))]
    v = float(m.group())
    new = v * 2 + 1 if v != 0 else 1
    new_s = str(int(new)) if "." not in m.group() else f"{new:g}"
    return doc[:m.start()] + new_s + doc[m.end():]


def doc_swap_type(doc, fn, rng, pools):
    opts = [(a, b) for a, b in _TYPE_SWAPS if _word(a).search(doc)] + \
           [(b, a) for a, b in _TYPE_SWAPS if _word(b).search(doc)]
    if not opts:
        return None
    a, b = opts[rng.integers(len(opts))]
    return _word(a).sub(b, doc)


def doc_negate(doc, fn, rng, pools):
    ms = list(_NEGATABLE.finditer(doc))
    if not ms:
        return None
    m = ms[rng.integers(len(ms))]
    return doc[:m.end()] + " not" + doc[m.end():]


# ---- benign (label 0): edits that keep every claim true
def doc_reflow(doc, fn, rng, pools):
    paras = doc.split("\n\n")
    changed = False
    for k, p in enumerate(paras):
        ls = p.splitlines()
        if len(ls) >= 1 and all(_indent(l) == 0 for l in ls) and not re.search(r"^\s*(>>>|[-*] |\.\. |:)", p, re.M) \
                and len(p) > 60 and p.strip().rstrip(":").lower() not in _SECTION_HEADS:
            width = int(rng.integers(45, 100))
            new = textwrap.fill(" ".join(l.strip() for l in ls), width=width)
            if new != p:
                paras[k], changed = new, True
    return "\n\n".join(paras) if changed else None


def _code_identifiers(fn) -> set[str]:
    ids = {fn.name}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            ids.add(n.id)
        elif isinstance(n, ast.Attribute):
            ids.add(n.attr)
        elif isinstance(n, ast.arg):
            ids.add(n.arg)
    return {i.lower() for i in ids}


def doc_synonym(doc, fn, rng, pools):
    """Swap 1-3 neutral words; never a word that is an identifier in the code, never in doctests."""
    code_ids = _code_identifiers(fn)
    prose = "\n".join(l for l in doc.splitlines() if not l.lstrip().startswith((">>>", "...")))
    hits = [(a, b) for a, b in _SYNONYMS if a.lower() not in code_ids and b.lower() not in code_ids and _word(a).search(prose)]
    hits += [(b, a) for a, b in _SYNONYMS if a.lower() not in code_ids and b.lower() not in code_ids and _word(b).search(prose)]
    if not hits:
        return None
    k = int(rng.integers(1, min(3, len(hits)) + 1))
    lines = doc.splitlines()
    for idx in rng.choice(len(hits), size=k, replace=False):
        a, b = hits[idx]
        lines = [l if l.lstrip().startswith((">>>", "...")) else _word(a).sub(lambda m: _case_like(m.group(), b), l)
                 for l in lines]
    return "\n".join(lines)


def doc_drop_section(doc, fn, rng, pools):
    lines = doc.splitlines()
    heads = [i for i, l in enumerate(lines) if l.strip().rstrip(":").lower() in _DROPPABLE]
    doctest = [i for i, l in enumerate(lines) if l.lstrip().startswith(">>>")]
    if heads:
        i = heads[rng.integers(len(heads))]
        j = i + 1
        while j < len(lines) and not (lines[j].strip().rstrip(":").lower() in _SECTION_HEADS and j > i + 1):
            j += 1
        new = lines[:i] + lines[j:]
    elif doctest:
        i = doctest[0]
        j = i
        while j < len(lines) and lines[j].strip():
            j += 1
        new = lines[:i] + lines[j:]
    else:
        return None
    out = _tidy(new)
    return out if len(out.split()) >= 8 and out != doc.strip() else None


OPS = {f.__name__: f for f in [doc_rename_param, doc_drop_param, doc_return_yield, doc_drop_raise,
                                 doc_add_raise, doc_change_number, doc_swap_type, doc_negate,
                                 doc_reflow, doc_synonym, doc_drop_section]}


def apply_doc_mutation(source: str, op: str, seed: int, pools: dict) -> tuple[str, str] | None:
    """Returns (rendered_source, new_docstring) or None if the operator does not apply."""
    try:
        _, fn = _fn(source)
    except Exception:
        return None
    doc = ast.get_docstring(fn)
    if not doc:
        return None
    rng = np.random.default_rng(seed + zlib.crc32(op.encode()) % 10_000)   # stable across runs
    try:
        new = OPS[op](doc, fn, rng, pools)
    except Exception:
        return None
    if not new or new.strip() == doc.strip() or len(new.split()) < 5:
        return None
    out = render(source, new)
    return (out, ast.get_docstring(_fn(out)[1])) if out else None


def build_pools(sources) -> dict:
    """Corpus-wide pools of parameter names and exception names for realistic replacements."""
    from collections import Counter
    params, excs = Counter(), Counter()
    for s in sources:
        try:
            _, fn = _fn(s)
        except Exception:
            continue
        params.update(p for p in _param_names(fn) if len(p) > 2 and p.lower() not in _COMMON)
        excs.update(_exceptions_raised(fn))
    return {"params": [p for p, c in params.most_common(400)],
            "exceptions": [e for e, c in excs.most_common(60) if e[0].isupper()]}
