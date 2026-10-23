"""Independent recomputation of the numbers in a run_methods result file.

Reads the saved predictions (npz) and the gold labels and recomputes micro
P/R/F1, macro-F1 and accuracy with its own small implementation (it does not
import punct.metrics), then compares them with the values stored in the
results JSON. Any difference above `--tol` is listed and the exit code is 1.

    python -m scripts.recompute --result results/E4_dev.json \
        --preds results/E4_dev_preds.npz --data_dir prepared_data_complete
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

NUM_LABELS = 8


def prf_from_scratch(gold: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Counts per class by direct boolean sums (no confusion matrix)."""
    tp_all = fp_all = fn_all = 0
    f1s = []
    for c in range(1, NUM_LABELS):
        g, p = gold == c, pred == c
        tp, fp, fn = int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum())
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    prec = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    rec = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    return {
        "micro_precision": prec,
        "micro_recall": rec,
        "micro_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "macro_f1": float(np.mean(f1s)),
        "accuracy": float((gold == pred).mean()),
    }


def compare(result: Dict, preds: Dict[str, np.ndarray], gold: np.ndarray, tol: float = 1e-9) -> Dict:
    """-> {"checked": n, "mismatches": [(method, key, stored, recomputed)]}"""
    checked, bad = 0, []
    for name, entry in result["methods"].items():
        if name not in preds:
            bad.append((name, "<missing predictions>", None, None))
            continue
        p = preds[name]
        if p.shape != gold.shape:
            bad.append((name, "<length>", int(gold.size), int(p.size)))
            continue
        mine = prf_from_scratch(gold, p)
        for key, val in mine.items():
            stored = entry["quality"][key]
            checked += 1
            if abs(stored - val) > tol:
                bad.append((name, key, stored, val))
    return {"checked": checked, "mismatches": bad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    with open(args.result, encoding="utf-8") as f:
        result = json.load(f)
    labels = np.load(os.path.join(args.data_dir, f"{result['split']}_articles_label.npy"), allow_pickle=True)
    gold = np.concatenate([np.asarray(x, dtype=np.int64) for x in labels[: result["n_articles"]]])
    npz = np.load(args.preds)
    preds = {k: npz[k] for k in npz.files}

    out = compare(result, preds, gold, args.tol)
    print(f"checked {out['checked']} numbers, {len(out['mismatches'])} mismatch(es)")
    for m in out["mismatches"]:
        print("  MISMATCH", m)
    return 1 if out["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
