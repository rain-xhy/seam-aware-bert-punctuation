"""The three pre-registered hypotheses (完善方案.md §6), tested with a paired
bootstrap over articles.

  H1  B is non-inferior to the teacher's 400/300 serial decoding, at lower TER.
  H2  At equal TER, C is better than B.
  H3  At equal budget, C (confidence-selected patches) is better than random patches.

Predictions come from .npz files (one concatenated label array per config
name, as written by run_methods.py / run_e7.py). delta is fixed BEFORE any
test-set run: it is chosen on the validation split from the width of the
bootstrap interval (see --delta_from_val), then passed explicitly here.

    python -m scripts.run_hypotheses --split dev --skip 500 \
        --serial_npz results/E4_dev_preds.npz --serial_name serial_teacher \
        --b_npz results/E4_dev_preds.npz --b_name parallel \
        --e7_npz results/E7_dev_preds.npz --delta 0.002 --out results/hypotheses_dev.json
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.bootstrap import paired_bootstrap_test  # noqa: E402
from punct.runner import dump, load_split  # noqa: E402


def split_by_article(flat: np.ndarray, lengths):
    out, off = [], 0
    for n in lengths:
        out.append(flat[off:off + n])
        off += n
    return out


def compare(gold, a, b, delta, n_boot, seed):
    """b relative to a: diff = F1_b - F1_a."""
    r = paired_bootstrap_test(gold, a, b, delta=delta, n_bootstrap=n_boot, seed=seed)
    return {k: (float(v) if isinstance(v, (np.floating, float)) else bool(v) if isinstance(v, (np.bool_, bool)) else v)
            for k, v in r.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--skip", type=int, default=0, help="articles skipped at the start (calibration part)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--serial_npz", default="", help="omit to skip H1 (needs E4 predictions)")
    ap.add_argument("--serial_name", default="serial_teacher")
    ap.add_argument("--b_npz", default="")
    ap.add_argument("--b_name", default="parallel")
    ap.add_argument("--e7_npz", required=True)
    ap.add_argument("--delta", type=float, default=0.002)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, gold = load_split(args.data_dir, args.split)
    skipped_len = sum(len(g) for g in gold[:args.skip])
    total_len_all = sum(len(g) for g in gold)
    gold = gold[args.skip:]
    if args.limit:
        gold = gold[:args.limit]
    lengths = [len(g) for g in gold]

    def load(path, name):
        z = np.load(path)
        if name not in z.files:
            raise KeyError(f"{name!r} not in {path}; available: {z.files}")
        flat = z[name]
        # A prediction file written over the whole split (e.g. E4) is cut down to the
        # evaluated range; a file already written for the skipped range (e.g. E7) is used as is.
        if flat.shape[0] == total_len_all and total_len_all != sum(lengths):
            flat = flat[skipped_len:skipped_len + sum(lengths)]
        if flat.shape[0] != sum(lengths):
            raise ValueError(f"{path}:{name} has {flat.shape[0]} labels, expected {sum(lengths)} "
                             "(check --skip/--limit match the run that wrote it)")
        return split_by_article(flat, lengths)

    e7 = {k: load(args.e7_npz, k) for k in ("pass1", "c_calibrated", "c_random", "b_same_ter")}

    res = {"delta": args.delta, "n_articles": len(gold), "n_boot": args.n_boot, "hypotheses": {}}
    # H1: non-inferiority of B vs serial  (diff = B - serial, lower bound >= -delta)
    if args.serial_npz and args.b_npz:
        serial = load(args.serial_npz, args.serial_name)
        b = load(args.b_npz, args.b_name)
        res["hypotheses"]["H1_B_noninferior_to_serial"] = compare(gold, serial, b, args.delta, args.n_boot, args.seed)
    else:
        res["H1_skipped"] = "no serial/B prediction files given"
    # H2: C better than B at equal TER      (diff = C - B_same_TER, lower bound > 0)
    res["hypotheses"]["H2_C_better_than_B_same_TER"] = compare(
        gold, e7["b_same_ter"], e7["c_calibrated"], args.delta, args.n_boot, args.seed)
    # H3: C better than random patches      (diff = C - random, lower bound > 0)
    res["hypotheses"]["H3_C_better_than_random"] = compare(
        gold, e7["c_random"], e7["c_calibrated"], args.delta, args.n_boot, args.seed)
    # exploratory: does patching help at all?
    res["exploratory"] = {"C_vs_pass1": compare(gold, e7["pass1"], e7["c_calibrated"], args.delta, args.n_boot, args.seed)}

    for k, v in {**res["hypotheses"], **res["exploratory"]}.items():
        print(f"{k}: diff={v['diff']:+.4f} CI=[{v['ci_lower']:+.4f},{v['ci_upper']:+.4f}] "
              f"non_inferior={v['non_inferior']} superior={v['superior']}")
    dump(args.out, res)


if __name__ == "__main__":
    main()
