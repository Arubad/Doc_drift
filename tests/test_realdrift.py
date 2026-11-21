"""Unit tests for real-drift mining on a throwaway git repository.

Run:  python -m unittest discover -s tests
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from docdrift.realdrift import mine_repo, is_trivial_edit  # noqa: E402

BASE = '''
def scale(values, factor=2):
    """Multiply every element of values by factor and return a new list of results."""
    out = []
    for v in values:
        out.append(v * factor)
    return out


def other(a, b):
    """Return the sum of a and b, computed in the simplest possible way here."""
    total = a + b
    return total
'''


def git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


def commit(repo, path, text, msg):
    full = os.path.join(repo, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(textwrap.dedent(text))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)


class MineRepoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = self.tmp.name
        git(self.repo, "init", "-q")
        commit(self.repo, "pkg/mod.py", BASE, "initial")

    def tearDown(self):
        self.tmp.cleanup()

    def mine(self):
        return mine_repo(self.repo, "t/repo", "pkg", since="2000-01-01")

    def test_doc_only_fix_is_a_pair(self):
        fixed = BASE.replace("and return a new list of results.",
                             "and return a new list; factor defaults to 2.")
        commit(self.repo, "pkg/mod.py", fixed, "docs: mention default factor")
        pairs = self.mine()
        self.assertEqual(len(pairs), 1)
        p = pairs[0]
        self.assertEqual(p["qualname"], "scale")
        self.assertIn("new list of results", p["doc_old"])
        self.assertIn("defaults to 2", p["doc_new"])
        self.assertIn("defaults to 2", p["source_new"])          # rendered with new docstring
        self.assertTrue(p["source_old"].startswith("def scale"))  # ast.unparse rendering

    def test_code_change_is_not_a_pair(self):
        changed = BASE.replace("and return a new list of results.", "and return a new list of products.") \
                      .replace("out.append(v * factor)", "out.append(factor * v)")
        commit(self.repo, "pkg/mod.py", changed, "refactor")
        self.assertEqual(self.mine(), [])

    def test_typo_fix_is_trivial(self):
        typo = BASE.replace("possible way here", "possibel way here")
        commit(self.repo, "pkg/mod.py", typo, "typo")
        self.assertEqual(self.mine(), [])  # possibel -> possible: dropped as trivial

    def test_test_files_ignored(self):
        commit(self.repo, "tests/test_mod.py", BASE, "add test")
        commit(self.repo, "tests/test_mod.py",
               BASE.replace("new list of results", "fresh list of values"), "doc in tests")
        self.assertEqual(self.mine(), [])

    def test_trivial_edit_helper(self):
        self.assertTrue(is_trivial_edit("Return the x.", "Return the ``x``."))
        self.assertTrue(is_trivial_edit("Retrun the value", "Return the value"))
        self.assertFalse(is_trivial_edit("Return a list.", "Yield items one by one."))
        # cases seen in real requests/click history
        self.assertTrue(is_trivial_edit("changing it's behavior here", "changing its behavior here"))
        self.assertTrue(is_trivial_edit("Constructs a PreparedRequest for transmission.",
                                        "Constructs a :class:`PreparedRequest <PreparedRequest>` for transmission."))
        self.assertTrue(is_trivial_edit("Format the item.\n\n.. versionchanged:: 8.3\n    Escape newlines.",
                                        "Format the item.\n\n.. versionchanged:: 8.4\n    Escape all newlines."))
        self.assertFalse(is_trivial_edit("Raises ValueError: need more than 1 value",
                                         "Raises ValueError: cannot encode objects that are not 2-tuples"))

    def test_edit_categories(self):
        from docdrift.realdrift import edit_categories
        src = "def put(url, data=None, **kwargs):\n    pass"
        self.assertIn("param", edit_categories("Send url.", "Send url.\n:param json: json data", src) +
                      edit_categories("Send url.", "Send url and data.", src))
        self.assertEqual(edit_categories("Return x.", "Yield x.", src), ["return"])
        self.assertIn("raise", edit_categories("Fails.", "Raises KeyError on missing key.", src))
        self.assertEqual(edit_categories("Does a thing.", "Does a nice thing.", src), ["prose"])


if __name__ == "__main__":
    unittest.main()
