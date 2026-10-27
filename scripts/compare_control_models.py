"""Paired bootstrap (over articles) of each control model against T1_varlen, test split.

    python -m scripts.compare_control_models --method parallel --out results/control_bootstrap.json

CPU only. diff = F1(control) - F1(T1). CI is a 95% percentile interval.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.bootstrap import paired_bootstrap_test  # noqa: E402
from punct.runner import dump, load_split  # noqa: E402

PREDS = {
    "T1_varlen": "results/E4_test_preds.npz",
    "T3_noedge": "results/T3_noedge_test_preds.npz",
    "T5_no_class_weight": "results/T5_no_class_weight_test_preds.npz",
    "T2_center_only": "results/T2_center_only_test_preds.npz",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--methods", nargs="+", default=["parallel", "serial_teacher"])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--delta", type=float, default=0.002)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, gold = load_split(args.data_dir, "test")
    lengths = [len(g) for g in gold]
    total = sum(lengths)

    def split(flat):
        out, off = [], 0
        for n in lengths:
            out.append(flat[off:off + n]); off += n
        return out

    res = {"split": "test", "n_articles": len(gold), "n_boot": args.n_boot, "delta": args.delta, "comparisons": {}}
    for method in args.methods:
        base = np.load(PREDS["T1_varlen"])[method]
        assert base.shape[0] == total, "T1 prediction length mismatch"
        base = split(base)
        for name, path in PREDS.items():
            if name == "T1_varlen":
                continue
            flat = np.load(path)[method]
            assert flat.shape[0] == total, f"{name} prediction length mismatch"
            r = paired_bootstrap_test(gold, base, split(flat), delta=args.delta, n_bootstrap=args.n_boot, seed=0)
            r = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in r.items()
                 if not isinstance(v, (list, np.ndarray))}
            res["comparisons"][f"{name}_vs_T1/{method}"] = r
            print(f"{method:15s} {name:20s} diff={r['diff']:+.4f} CI=[{r['ci_lower']:+.4f},{r['ci_upper']:+.4f}] "
                  f"CI_excludes_0={r['ci_lower'] > 0 or r['ci_upper'] < 0}", flush=True)
    dump(args.out, res)


if __name__ == "__main__":
    main()
