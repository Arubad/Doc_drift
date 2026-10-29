import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


def save_csv(df, name):
    p = os.path.join(RESULTS, name)
    df.to_csv(p, index=False)
    print(f"[saved] {os.path.relpath(p)}")


def save_json(obj, name):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_benchmark():
    return pd.read_pickle(os.path.join(DATA, "benchmark.pkl"))


def fig_path(name):
    return os.path.join(RESULTS, name)
