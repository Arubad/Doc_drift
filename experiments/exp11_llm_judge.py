"""Experiment 11 -- Zero-shot LLM judges on real doc-fix pairs (protocol Addendum B, H5/H5b).

Needs a GPU. Scores every real-pair row (old and new docstring) and 1,000 synthetic
code-side rows with P(No) / (P(Yes) + P(No)) for the question "does this docstring
accurately describe the code?". Deterministic (logits, no sampling).

Usage: python exp11_llm_judge.py MODEL_KEY      (MODEL_KEY in MODELS; resumable)
       DOCDRIFT_QUICK=1 ... scores 40 rows only (smoke test)
Outputs: data/exp11_scores_<key>.pkl, results/exp11_<key>.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, load_benchmark, save_csv

MODELS = {
    "qwen-1.5b": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", False),
    "qwen-7b": ("Qwen/Qwen2.5-Coder-7B-Instruct", True),       # 4-bit NF4 on a 16 GB T4
}
PROMPT = ("Here is a Python function with its docstring.\n\n```python\n{source}\n```\n\n"
          "Does the docstring accurately describe the function's code -- its parameters, return "
          "value, raised exceptions and behaviour? Answer with a single word: Yes or No.")
MAX_TOKENS, BATCH = 1536, 8
QUICK = bool(os.environ.get("DOCDRIFT_QUICK"))
B = int(os.environ.get("DOCDRIFT_BOOT", "1000"))

key = sys.argv[1]
name, four_bit = MODELS[key]
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = "left"
kw = {"device_map": "cuda:0"}
if four_bit:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16)
else:
    kw["torch_dtype"] = torch.float16
model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]


def build(source: str) -> str:
    ids = tok.encode(source, add_special_tokens=False)
    if len(ids) > MAX_TOKENS:                          # functions are <= 80 lines; rarely triggers
        source = tok.decode(ids[:MAX_TOKENS]) + "\n# ... (truncated)"
    msgs = [{"role": "user", "content": PROMPT.format(source=source)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score(sources, part_path=None) -> np.ndarray:
    """Batched scoring; if part_path is given, saves progress every 100 batches and resumes."""
    out, start = [], 0
    if part_path and os.path.exists(part_path):
        out = [np.load(part_path)]; start = len(out[0])
        print(f"  resuming at {start}/{len(sources)}", flush=True)
    for i in range(start, len(sources), BATCH):
        enc = tok([build(s) for s in sources[i:i + BATCH]], return_tensors="pt", padding=True).to("cuda:0")
        logits = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits, -1)
        py, pn = p[:, yes_ids].sum(-1), p[:, no_ids].sum(-1)
        out.append((pn / (py + pn + 1e-12)).cpu().numpy())
        if (i // BATCH) % 50 == 0:
            print(f"  {i + len(sources[i:i + BATCH])}/{len(sources)}", flush=True)
        if part_path and (i // BATCH) % 100 == 99:
            np.save(part_path, np.concatenate(out))
    return np.concatenate(out)


# ---- rows to score: real pairs (old then new) + fixed synthetic sample
real = pd.read_pickle(os.path.join(DATA, "real_pairs_clean.pkl")).reset_index(drop=True)
n = len(real)
real_src = list(real.source_old) + list(real.source_new)
bench = load_benchmark()
syn = pd.concat([bench[bench.label == 1].sample(500, random_state=0),
                 bench[bench.kind == "clean"].sample(500, random_state=0)])
if QUICK:
    real_src, syn = real_src[:20] + real_src[n:n + 20], syn.sample(40, random_state=0)

ckpt = os.path.join(DATA, f"exp11_scores_{key}{'_quick' if QUICK else ''}.pkl")
done = pd.read_pickle(ckpt) if os.path.exists(ckpt) else {}
if "synthetic" not in done:
    done["synthetic"] = score(list(syn.source)); pd.to_pickle(done, ckpt)
    from sklearn.metrics import roc_auc_score
    print(f"synthetic AUROC {roc_auc_score(syn.label, done['synthetic']):.3f}", flush=True)
if "real" not in done:
    done["real"] = score(real_src, ckpt.replace(".pkl", "_real_part.npy")); pd.to_pickle(done, ckpt)
if QUICK:
    print("quick scores:", np.round(done["real"], 3)); sys.exit(0)

# ---- metrics (same definitions as exp10)
from sklearn.metrics import roc_auc_score
s = done["real"]
wo, wn = real.doc_old.str.split().str.len(), real.doc_new.str.split().str.len()
real["length_matched"] = (wn - wo).abs() <= 0.10 * wo
SUBSETS = {"interface": real.interface.values, "interface-length-matched": (real.interface & real.length_matched).values,
           "all": np.ones(n, bool)}
corr = (s[:n] > s[n:]) + 0.5 * (s[:n] == s[n:])
L = np.r_[wo.values, wn.values].astype(float)
len_corr = (-L[:n] > -L[n:]) + 0.5 * (L[:n] == L[n:])
rng = np.random.default_rng(0)
rows = []
for sub, m in SUBSETS.items():
    idx = np.where(m)[0]
    cm = real.commit.values[idx]
    commits = np.unique(cm)
    by = {c: idx[cm == c] for c in commits}
    draws = [np.concatenate([by[c] for c in rng.choice(commits, len(commits))]) for _ in range(B)]
    acc_b = np.array([corr[d].mean() for d in draws])
    diff_b = np.array([corr[d].mean() - len_corr[d].mean() for d in draws])
    rows.append({"model": key, "subset": sub, "n_pairs": len(idx), "pair_acc": corr[idx].mean(),
                 "ci_lo": np.percentile(acc_b, 2.5), "ci_hi": np.percentile(acc_b, 97.5),
                 "length_baseline": len_corr[idx].mean(), "diff_vs_length": corr[idx].mean() - len_corr[idx].mean(),
                 "diff_ci_lo": np.percentile(diff_b, 2.5), "diff_ci_hi": np.percentile(diff_b, 97.5),
                 "real_auroc": roc_auc_score(np.r_[np.ones(len(idx)), np.zeros(len(idx))], np.r_[s[idx], s[n + idx]]),
                 "synthetic_auroc": roc_auc_score(syn.label, done["synthetic"])})
res = pd.DataFrame(rows)
save_csv(res, f"exp11_{key}.csv")
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
