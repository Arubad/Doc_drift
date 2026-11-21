"""Experiment 8 -- Mine real documentation drift (doc-fix pairs) from git history.

For each repository of the pinned corpus packages, find commits that edited a
function's docstring while leaving its code identical (see docdrift/realdrift.py).
Output: data/real_pairs.pkl (one row per pair) and results/exp8_* summaries.

Clones go to $DOCDRIFT_REPOS (default ./repos). Runs repositories in parallel.
"""
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from common import DATA, save_csv
from docdrift.realdrift import mine_repo

# GitHub repo -> top-level package (= `project` in the synthetic benchmark)
REPOS = {
    "psf/requests": "requests", "pallets/click": "click", "pallets/flask": "flask",
    "pallets/werkzeug": "werkzeug", "pallets/jinja": "jinja2", "Textualize/rich": "rich",
    "python-attrs/attrs": "attr", "pypa/packaging": "packaging", "encode/httpx": "httpx",
    "cpburnz/python-pathspec": "pathspec", "python-jsonschema/jsonschema": "jsonschema",
    "facelessuser/pymdown-extensions": "pymdownx", "networkx/networkx": "networkx",
    "Python-Markdown/markdown": "markdown", "pygments/pygments": "pygments",
    "sphinx-doc/sphinx": "sphinx", "pytest-dev/pytest": "_pytest", "tornadoweb/tornado": "tornado",
    "aio-libs/aiohttp": "aiohttp", "pydata/xarray": "xarray", "mwaskom/seaborn": "seaborn",
    "statsmodels/statsmodels": "statsmodels", "scikit-learn/scikit-learn": "sklearn",
    "matplotlib/matplotlib": "matplotlib", "pandas-dev/pandas": "pandas", "numpy/numpy": "numpy",
    "scipy/scipy": "scipy", "django/django": "django", "sqlalchemy/sqlalchemy": "sqlalchemy",
    "nltk/nltk": "nltk", "bokeh/bokeh": "bokeh", "pydantic/pydantic": "pydantic",
    "pypa/setuptools": "setuptools",
}
ROOT = os.environ.get("DOCDRIFT_REPOS", os.path.join(os.path.dirname(__file__), "..", "repos"))
SINCE = os.environ.get("DOCDRIFT_SINCE", "2012-01-01")


def clone_and_mine(repo: str, project: str):
    path = os.path.join(ROOT, repo.replace("/", "__"))
    if not os.path.isdir(path):
        subprocess.run(["git", "clone", "--quiet", "--no-tags", "--single-branch",
                        f"https://github.com/{repo}.git", path], check=True)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    pairs = mine_repo(path, repo, project, since=SINCE)
    return repo, head, pairs


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    rows, heads = [], []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("DOCDRIFT_JOBS", "4"))) as ex:
        futs = {ex.submit(clone_and_mine, r, p): r for r, p in REPOS.items()}
        for f in as_completed(futs):
            try:
                repo, head, pairs = f.result()
            except Exception as e:  # one bad clone must not kill the run
                print(f"[fail] {futs[f]}: {e}", flush=True)
                continue
            print(f"{repo}: {len(pairs)} pairs (HEAD {head[:10]})", flush=True)
            rows.extend(pairs); heads.append({"repo": repo, "head": head, "n_pairs": len(pairs)})

    df = pd.DataFrame(rows)
    df["pair_id"] = range(len(df))
    df.to_pickle(os.path.join(DATA, "real_pairs.pkl"))
    save_csv(pd.DataFrame(heads).sort_values("n_pairs", ascending=False), "exp8_repos.csv")
    print(f"\ntotal {len(df)} doc-fix pairs from {df.repo.nunique() if len(df) else 0} repos, "
          f"{df.commit.nunique() if len(df) else 0} commits")
