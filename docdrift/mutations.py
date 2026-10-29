"""Mutation taxonomy for synthesising documentation drift.

Each mutation edits the *code* of a function and leaves the docstring intact,
producing an inconsistent (code, doc) pair. Mutations are grouped by the kind
of drift they emulate:

Surface-level (the docstring and signature disagree lexically)
  rename_param      a parameter is renamed everywhere in the function
  remove_param      a defaulted parameter is deleted (and its uses replaced
                    by the default)
  add_param         a new keyword parameter is appended and used
  change_default    the literal default of a parameter is changed

Behavioural, docstring-visible (the docstring describes a contract that the
code no longer honours, but nothing in the signature changes)
  add_raise         a guard that raises an undocumented exception is inserted
  remove_raise      a documented raise statement is removed
  return_to_yield   the function becomes a generator
  wrap_return       the returned value is wrapped (list / str / tuple)

Semantic, docstring-invisible (no lexical trace; the docstring's prose is
not violated in a way detectable without executing the code)
  negate_condition  an `if` condition is negated
  swap_binop        + <-> -, * <-> //, < <-> >=, == <-> !=

Benign edits (used as hard negatives: code changes, docstring still correct)
  rename_local      a *local* variable (not a parameter) is renamed
  reorder_imports   / add_pass / wrap_in_parens style no-ops
"""
from __future__ import annotations

import ast
import copy
import random
import re

SEMANTIC = {"negate_condition", "swap_binop"}
SURFACE = {"rename_param", "remove_param", "add_param", "change_default"}
BEHAVIOURAL = {"add_raise", "remove_raise", "return_to_yield", "wrap_return"}
BENIGN = {"rename_local", "rewrite_augassign"}
ALL_MUTATIONS = sorted(SURFACE | BEHAVIOURAL | SEMANTIC)

SYNONYMS = ["value", "item", "data", "obj", "arg", "elem", "entry", "val", "x", "payload"]


class _Rename(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old, self.new = old, new

    def visit_Name(self, node):
        if node.id == self.old:
            node.id = self.new
        return node

    def visit_arg(self, node):
        if node.arg == self.old:
            node.arg = self.new
        return node

    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == self.old:
            node.arg = self.new
        return node


def _func(tree):
    return tree.body[0]


def _params(fn):
    a = fn.args
    return [p for p in (getattr(a, "posonlyargs", []) + a.args + a.kwonlyargs) if p.arg not in ("self", "cls")]


def _strip_doc(fn):
    """Return (docstring_node, body_without_doc)."""
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        return fn.body[0], fn.body[1:]
    return None, fn.body


# ------------------------------- surface -----------------------------------
def rename_param(tree, rng):
    fn = _func(tree)
    ps = _params(fn)
    if not ps:
        return None
    old = rng.choice(ps).arg
    new = rng.choice([s for s in SYNONYMS if s != old])
    # don't rename to an existing name
    existing = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {q.arg for q in _params(fn)}
    if new in existing:
        new = new + "_"
    doc, body = _strip_doc(fn)
    fn.args = _Rename(old, new).visit(fn.args)
    fn.body = ([doc] if doc else []) + [_Rename(old, new).visit(s) for s in body]
    return tree


def remove_param(tree, rng):
    fn = _func(tree)
    a = fn.args
    n_pos = len(a.args)
    cand = [(i, p) for i, p in enumerate(a.args) if p.arg not in ("self", "cls")]
    # only params with a default (last len(defaults) of args)
    n_def = len(a.defaults)
    cand = [(i, p) for i, p in cand if i >= n_pos - n_def]
    if not cand:
        return None
    i, p = rng.choice(cand)
    default = a.defaults[i - (n_pos - n_def)]
    del a.defaults[i - (n_pos - n_def)]
    del a.args[i]

    class Sub(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == p.arg and isinstance(node.ctx, ast.Load):
                return copy.deepcopy(default)
            return node
    doc, body = _strip_doc(fn)
    fn.body = ([doc] if doc else []) + [Sub().visit(s) for s in body]
    return tree


def add_param(tree, rng):
    fn = _func(tree)
    name = rng.choice(["strict", "verbose", "limit", "encoding", "timeout", "flags"])
    existing = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {p.arg for p in _params(fn)}
    if name in existing:
        name += "_"
    fn.args.kwonlyargs.append(ast.arg(arg=name))
    fn.args.kw_defaults.append(ast.Constant(value=None))
    doc, body = _strip_doc(fn)
    guard = ast.parse(f"if {name} is not None:\n    {name} = {name}").body[0]
    fn.body = ([doc] if doc else []) + [guard] + body
    return tree


def change_default(tree, rng):
    fn = _func(tree)
    a = fn.args
    cands = [d for d in a.defaults + a.kw_defaults if isinstance(d, ast.Constant)]
    if not cands:
        return None
    d = rng.choice(cands)
    v = d.value
    if isinstance(v, bool):
        d.value = not v
    elif isinstance(v, int):
        d.value = v + rng.choice([1, 2, 10]) if v != 0 else rng.choice([1, 2, 10])
    elif isinstance(v, str):
        d.value = v + "_" if v else "x"
    elif v is None:
        d.value = rng.choice([0, "", False])
    else:
        return None
    return tree


# ------------------------------ behavioural --------------------------------
EXC = ["ValueError", "TypeError", "KeyError", "RuntimeError", "OSError"]


def add_raise(tree, rng):
    fn = _func(tree)
    ps = _params(fn)
    if not ps:
        return None
    p = rng.choice(ps)
    exc = rng.choice(EXC)
    doc, body = _strip_doc(fn)
    guard = ast.parse(f"if {p.arg} is None:\n    raise {exc}('{p.arg} must not be None')").body[0]
    fn.body = ([doc] if doc else []) + [guard] + body
    return tree


def remove_raise(tree, rng):
    fn = _func(tree)
    raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
    if not raises:
        return None
    target = rng.choice(raises)

    class Rm(ast.NodeTransformer):
        def visit_Raise(self, node):
            if node is target:
                return ast.Pass()
            return node
    Rm().visit(fn)
    return tree


def return_to_yield(tree, rng):
    fn = _func(tree)
    if any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn)):
        return None
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    if not rets:
        return None

    class R2Y(ast.NodeTransformer):
        def visit_Return(self, node):
            if node.value is None:
                return node
            return ast.Expr(value=ast.Yield(value=node.value))
    R2Y().visit(fn)
    return tree


def wrap_return(tree, rng):
    fn = _func(tree)
    rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    if not rets:
        return None
    wrapper = rng.choice(["list", "str", "tuple"])
    for r in rets:
        r.value = ast.Call(func=ast.Name(id=wrapper, ctx=ast.Load()), args=[r.value], keywords=[])
    return tree


# ------------------------------- semantic ----------------------------------
def negate_condition(tree, rng):
    fn = _func(tree)
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)]
    if not ifs:
        return None
    n = rng.choice(ifs)
    n.test = ast.UnaryOp(op=ast.Not(), operand=n.test)
    return tree


_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
         ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.Eq: ast.NotEq,
         ast.NotEq: ast.Eq}


def swap_binop(tree, rng):
    fn = _func(tree)
    cands = []
    for n in ast.walk(fn):
        if isinstance(n, ast.BinOp) and type(n.op) in _SWAP:
            cands.append(("bin", n, 0))
        if isinstance(n, ast.Compare):
            for i, op in enumerate(n.ops):
                if type(op) in _SWAP:
                    cands.append(("cmp", n, i))
    if not cands:
        return None
    kind, n, i = rng.choice(cands)
    if kind == "bin":
        n.op = _SWAP[type(n.op)]()
    else:
        n.ops[i] = _SWAP[type(n.ops[i])]()
    return tree


# -------------------------------- benign -----------------------------------
LOCAL_POOL = ["result", "tmp", "out", "cur", "buf", "acc", "node", "line", "key", "val", "count", "idx",
              "res", "rv", "ret", "chunk", "token", "entry"]


def rename_local(tree, rng):
    """Rename a *local* (non-parameter) variable to another plausible name
    that does not already occur in the function. Docstrings describe the
    interface, not locals, so the pair stays consistent."""
    fn = _func(tree)
    params = {p.arg for p in _params(fn)} | {"self", "cls"}
    stores = [n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
    locals_ = sorted({s for s in stores if s not in params})
    if not locals_:
        return None
    old = rng.choice(locals_)
    existing = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | params
    pool = [w for w in LOCAL_POOL if w not in existing]
    if not pool:
        return None
    new = rng.choice(pool)
    doc, body = _strip_doc(fn)
    fn.body = ([doc] if doc else []) + [_Rename(old, new).visit(s) for s in body]
    return tree


def rewrite_augassign(tree, rng):
    """`x += y` -> `x = x + y` (and the reverse): semantics-preserving."""
    fn = _func(tree)
    augs = [n for n in ast.walk(fn) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)]
    plain = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.BinOp)
             and isinstance(n.value.left, ast.Name) and n.value.left.id == n.targets[0].id
             and isinstance(n.value.op, (ast.Add, ast.Sub, ast.Mult))]
    cands = [("aug", n) for n in augs] + [("plain", n) for n in plain]
    if not cands:
        return None
    kind, n = rng.choice(cands)

    class Rw(ast.NodeTransformer):
        def visit_AugAssign(self, node):
            if node is n:
                return ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())],
                                  value=ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value))
            return node

        def visit_Assign(self, node):
            if node is n:
                return ast.AugAssign(target=ast.Name(id=node.targets[0].id, ctx=ast.Store()), op=node.value.op, right=node.value.right)
            return node
    Rw().visit(fn)
    return tree


def add_noop(tree, rng):
    """Insert a semantics-free statement after the docstring (kept as a
    *third*, deliberately artifact-laden benign edit to measure artifact
    learning: see Exp 7)."""
    fn = _func(tree)
    doc, body = _strip_doc(fn)
    noop = ast.parse(rng.choice(["_ = None", "pass", "_tmp = 0"])).body[0]
    fn.body = ([doc] if doc else []) + [noop] + body
    return tree


MUTATORS = {
    "rename_param": rename_param, "remove_param": remove_param, "add_param": add_param,
    "change_default": change_default, "add_raise": add_raise, "remove_raise": remove_raise,
    "return_to_yield": return_to_yield, "wrap_return": wrap_return,
    "negate_condition": negate_condition, "swap_binop": swap_binop,
    "rename_local": rename_local, "rewrite_augassign": rewrite_augassign, "add_noop": add_noop,
}


def apply_mutation(source: str, mutation: str, seed: int = 0) -> str | None:
    """Return mutated source or None if the mutation is not applicable."""
    rng = random.Random(seed)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    if not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    out = MUTATORS[mutation](copy.deepcopy(tree), rng)
    if out is None:
        return None
    try:
        ast.fix_missing_locations(out)
        new_src = ast.unparse(out)
        ast.parse(new_src)
    except Exception:
        return None
    if new_src.strip() == ast.unparse(tree).strip():
        return None
    return new_src


def kind_of(mutation: str) -> str:
    if mutation in SURFACE: return "surface"
    if mutation in BEHAVIOURAL: return "behavioural"
    if mutation in SEMANTIC: return "semantic"
    return "benign"
