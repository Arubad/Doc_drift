"""Structural-alignment features between a function's code and its docstring.

The intuition: a docstring makes claims about the *interface* (parameter
names, defaults, return / yield, raised exceptions). Each claim can be checked
cheaply against the AST. Features are counts of matched / unmatched claims,
plus lexical overlap statistics, so a linear model remains interpretable."""
from __future__ import annotations

import ast
import re
from functools import lru_cache

import numpy as np

FEATURE_NAMES = [
    "n_params", "n_doc_params_matched", "n_doc_params_unmatched", "n_sig_params_unmentioned",
    "frac_params_mentioned", "doc_mentions_return", "code_has_return_value", "doc_mentions_yield",
    "code_has_yield", "return_yield_mismatch", "doc_mentions_raise", "code_has_raise",
    "n_doc_exceptions_matched", "n_doc_exceptions_unmatched", "n_code_exceptions_undocumented",
    "n_doc_numbers_matched_defaults", "n_doc_numbers_unmatched_defaults", "n_defaults",
    "ident_jaccard", "ident_overlap_doc_in_code", "doc_len", "code_len", "n_if", "n_binop",
    "doc_mentions_none", "code_returns_none_literal", "doc_mentions_list", "doc_mentions_str",
    "doc_mentions_tuple", "ret_wrapped_list", "ret_wrapped_str", "ret_wrapped_tuple",
]

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?")
_EXC = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Warning|Exit|Interrupt))\b")
_PARAM_SECTION = re.compile(r"(?::param\s+(\w+)|:arg\s+(\w+)|^\s*(\w+)\s*(?:\(|:)\s*)", re.M)


def _func(tree):
    return tree.body[0]


def _param_names(fn):
    a = fn.args
    return [p.arg for p in (getattr(a, "posonlyargs", []) + a.args + a.kwonlyargs)
            if p.arg not in ("self", "cls")]


def _defaults(fn):
    vals = []
    for d in fn.args.defaults + fn.args.kw_defaults:
        if isinstance(d, ast.Constant) and isinstance(d.value, (int, float)) and not isinstance(d.value, bool):
            vals.append(d.value)
    return vals


def _exceptions_raised(fn):
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Raise) and n.exc is not None:
            e = n.exc
            if isinstance(e, ast.Call):
                e = e.func
            if isinstance(e, ast.Name):
                out.add(e.id)
            elif isinstance(e, ast.Attribute):
                out.add(e.attr)
    return out


@lru_cache(maxsize=400000)
def extract(source: str, docstring: str) -> np.ndarray:
    tree = ast.parse(source)
    fn = _func(tree)
    params = _param_names(fn)
    doc_words = set(_WORD.findall(docstring))
    doc_lower = docstring.lower()

    # identifiers used in code (excluding docstring)
    code_ids = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            code_ids.add(n.id)
        elif isinstance(n, ast.Attribute):
            code_ids.add(n.attr)
        elif isinstance(n, ast.arg):
            code_ids.add(n.arg)
    code_ids = {i for i in code_ids if len(i) > 1}

    matched = [p for p in params if p in doc_words]
    # words in docstring that *look like* parameter references (backticked or
    # in a :param: / Args: section) but are not in the signature
    doc_param_refs = set()
    for m in _PARAM_SECTION.finditer(docstring):
        for g in m.groups():
            if g:
                doc_param_refs.add(g)
    doc_param_refs |= set(re.findall(r"`([A-Za-z_]\w*)`", docstring))
    doc_param_refs |= set(re.findall(r"\*([A-Za-z_]\w*)\*", docstring))
    doc_unmatched = [w for w in doc_param_refs if w not in params and w not in code_ids
                     and w.lower() not in {"none", "true", "false", "return", "returns", "raises", "args", "arguments", "note", "example", "examples", "yields", "see", "also", "parameters"}]

    has_ret_val = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn))
    has_yield = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))
    doc_ret = int(bool(re.search(r"\breturn(s|ed|ing)?\b", doc_lower)))
    doc_yield = int(bool(re.search(r"\byield(s|ed|ing)?\b|\bgenerator\b|\biterat(or|es)\b", doc_lower)))
    ry_mismatch = int((doc_yield and has_ret_val and not has_yield) or (doc_ret and has_yield and not doc_yield))

    doc_exc = set(_EXC.findall(docstring))
    code_exc = _exceptions_raised(fn)
    doc_raise = int(bool(re.search(r"\brais(e|es|ed|ing)\b", doc_lower)) or bool(doc_exc))

    defaults = _defaults(fn)
    doc_nums = {float(x) for x in _NUM.findall(docstring)}
    nums_matched = sum(1 for d in defaults if float(d) in doc_nums)
    nums_unmatched = len(defaults) - nums_matched

    ret_wrapped = {"list": 0, "str": 0, "tuple": 0}
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) \
                and n.value.func.id in ret_wrapped:
            ret_wrapped[n.value.func.id] = 1
    returns_none_lit = int(any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and n.value.value is None
                               for n in ast.walk(fn)))

    inter = len(doc_words & code_ids)
    union = len(doc_words | code_ids) or 1
    feats = [
        len(params), len(matched), len(doc_unmatched), len(params) - len(matched),
        (len(matched) / len(params)) if params else 1.0, doc_ret, int(has_ret_val), doc_yield,
        int(has_yield), ry_mismatch, doc_raise, int(bool(code_exc)),
        len(doc_exc & code_exc), len(doc_exc - code_exc), len(code_exc - doc_exc),
        nums_matched, nums_unmatched, len(defaults),
        inter / union, inter / (len(doc_words) or 1), len(docstring.split()), len(source.splitlines()),
        sum(isinstance(n, ast.If) for n in ast.walk(fn)), sum(isinstance(n, ast.BinOp) for n in ast.walk(fn)),
        int("none" in doc_lower), returns_none_lit, int("list" in doc_lower), int("string" in doc_lower or " str" in doc_lower),
        int("tuple" in doc_lower), ret_wrapped["list"], ret_wrapped["str"], ret_wrapped["tuple"],
    ]
    return np.array(feats, dtype=float)


@lru_cache(maxsize=400000)
def code_tokens(source: str) -> str:
    """Code without its docstring, as a token string for TF-IDF."""
    try:
        tree = ast.parse(source)
        fn = tree.body[0]
        if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
            fn.body = fn.body[1:] or [ast.Pass()]
        src = ast.unparse(fn)
    except Exception:
        src = source
    return " ".join(_WORD.findall(src))
