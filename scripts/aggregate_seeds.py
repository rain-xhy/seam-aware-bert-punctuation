"""Aggregate the same experiment run with several training seeds.

    python -m scripts.aggregate_seeds --inputs results/E4_seed0.json results/E4_seed1.json \
        results/E4_seed2.json --metric micro_f1 --out results/E4_seeds.json

Each input is a result file written by run_methods.py (``methods`` -> name ->
``quality``) or run_e7.py (``configs`` -> name -> ``all``). For every method
the metric is collected across files and reported as mean, sample standard
deviation (n-1), min, max and n. With a single file the standard deviation is
reported as null, never as 0, so one seed cannot be mistaken for a tight
spread.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.runner import dump  # noqa: E402


def extract(data: dict, metric: str) -> Dict[str, float]:
    """name -> metric value for one result file."""
    if "methods" in data:
        block, key = data["methods"], "quality"
    elif "configs" in data:
        block, key = data["configs"], "all"
    else:
        raise ValueError("result file has neither 'methods' nor 'configs'")
    out = {}
    for name, entry in block.items():
        q = entry.get(key, {})
        if metric in q:
            out[name] = float(q[metric])
    return out


def aggregate(per_file: List[Dict[str, float]]) -> Dict[str, dict]:
    names = sorted({n for d in per_file for n in d})
    res = {}
    for n in names:
        vals = np.array([d[n] for d in per_file if n in d], dtype=np.float64)
        res[n] = {
            "n": int(vals.size), "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if vals.size > 1 else None,
            "min": float(vals.min()), "max": float(vals.max()), "values": vals.tolist(),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--metric", default="micro_f1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_file = []
    for p in args.inputs:
        with open(p, encoding="utf-8") as f:
            per_file.append(extract(json.load(f), args.metric))
    res = {"metric": args.metric, "inputs": args.inputs, "n_files": len(args.inputs),
           "methods": aggregate(per_file)}
    for n, r in res["methods"].items():
        s = "n/a (single run)" if r["std"] is None else f"{r['std']:.4f}"
        print(f"{n}: mean={r['mean']:.4f} std={s} n={r['n']}")
    dump(args.out, res)


if __name__ == "__main__":
    main()
