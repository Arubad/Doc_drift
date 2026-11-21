"""Unit tests for doc-side mutation operators.  Run: python -m unittest discover -s tests"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from docdrift.docmutations import apply_doc_mutation, render, DOC_DRIFT, DOC_BENIGN  # noqa: E402

POOLS = {"params": ["timeout", "encoding", "verbose"], "exceptions": ["KeyError", "TypeError", "ValueError"]}

GOOGLE = '''def fetch(url, retries=3, verbose=False):
    """Fetch the resource at url and return the decoded body.

    The request is retried on failure. This function is useful for scripts.

    Args:
        url (str): The address to fetch.
        retries (int): Number of attempts, default 3.
        verbose (bool): Print progress if True.

    Returns:
        str: The decoded response body.

    Raises:
        ValueError: If url is empty.

    Examples:
        >>> fetch("http://x")
        'ok'
    """
    if not url:
        raise ValueError("empty")
    for _ in range(retries):
        body = get(url)
    return body.decode()
'''

GEN = '''def walk(root):
    """Yield every file path below root, one at a time, in sorted order of names."""
    for p in sorted(listdir(root)):
        yield p
'''


def doc_of(src):
    return ast.get_docstring(ast.parse(src).body[0])


def code_of(src):
    fn = ast.parse(src).body[0]
    fn.body = fn.body[1:]
    return ast.dump(fn)


class DocMutationTest(unittest.TestCase):
    def run_op(self, op, src=GOOGLE, seed=0):
        return apply_doc_mutation(src, op, seed, POOLS)

    def test_code_never_changes(self):
        for op in DOC_DRIFT + DOC_BENIGN:
            out = self.run_op(op)
            if out is not None:
                self.assertEqual(code_of(out[0]), code_of(GOOGLE), op)
                self.assertNotEqual(out[1].strip(), doc_of(GOOGLE).strip(), op)

    def test_render_is_stable(self):
        r = render(GOOGLE)
        self.assertEqual(render(r), r)          # idempotent: clean rows and mutated rows share format

    def test_rename_param(self):
        _, d = self.run_op("doc_rename_param")
        self.assertTrue(any(p in d for p in POOLS["params"]))
        self.assertTrue(("url" not in d) or ("retries" not in d) or ("verbose" not in d.split("Args")[1]))

    def test_drop_param(self):
        _, d = self.run_op("doc_drop_param")
        self.assertEqual(sum(f"{p} (" in d for p in ["url", "retries", "verbose"]), 2)
        self.assertIn("Args:", d)

    def test_return_yield_both_ways(self):
        _, d = self.run_op("doc_return_yield")
        self.assertIn("yield", d.lower()); self.assertIn("Yields:", d)
        _, d2 = self.run_op("doc_return_yield", GEN)
        self.assertTrue(d2.startswith("Return every file path"))

    def test_drop_and_add_raise(self):
        _, d = self.run_op("doc_drop_raise")
        self.assertNotIn("ValueError", d); self.assertNotIn("Raises:", d)   # empty section removed
        _, d = self.run_op("doc_add_raise")
        self.assertTrue("KeyError" in d or "TypeError" in d)
        self.assertEqual(d.count("Raises:"), 2)                              # google-style template

    def test_change_number_and_negate(self):
        _, d = self.run_op("doc_change_number")
        self.assertIn("default 7", d)
        _, d = self.run_op("doc_negate")
        self.assertIn(" not", d)

    def test_benign(self):
        _, d = self.run_op("doc_drop_section")
        self.assertNotIn(">>>", d); self.assertIn("Raises:", d)
        _, d = self.run_op("doc_synonym")
        self.assertIn("handy", d)                       # 'useful' -> 'handy'
        self.assertIn('fetch("http://x")', d)           # function name / doctest untouched
        self.assertNotIn("retrieve", d)                 # 'fetch' is a code identifier: never swapped


if __name__ == "__main__":
    unittest.main()
