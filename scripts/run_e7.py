"""E7: ablation of the two-pass seam re-windowing (method C).

Configurations, all with the same fine-tuned checkpoint:
  pass1            first pass only (method B with the given window/overlap)
  c_calibrated     C, positions chosen by calibrated confidence
  c_uncalibrated   C, positions chosen by raw confidence
  c_random         C's budget (number of selected positions), positions drawn at random
  b_same_ter       B with the overlap increased until its TER matches c_calibrated

Calibration (temperature + class bias) is fitted on --cal_split articles and
never on the evaluated articles. Each configuration is scored on all
articles and on the multi-window subset (articles longer than the window).

    python -m scripts.run_e7 --ckpt checkpoints/T1_varlen/best.pt \
        --cal_split dev --cal_limit 500 --eval_split dev --out results/E7_dev.json
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.calibrate import calibrate_probs, calibrate_temperature_and_bias, expected_calibration_error  # noqa: E402
from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import decode_parallel, load_finetuned, make_logits_fn, pool_windows  # noqa: E402
from punct.runner import dump, load_split, score  # noqa: E402
from punct.two_pass import decode_two_pass  # noqa: E402


def ter_of(lengths, window, overlap, tail):
    total = max(sum(lengths), 1)
    return sum(w.end - w.start for w in pool_windows(lengths, window, overlap, tail)) / total


def overlap_for_ter(lengths, window, tail, target):
    """Overlap whose pass-1 TER is closest to `target` (TER grows with overlap)."""
    best, best_gap = 0, float("inf")
    for v in range(0, window):
        gap = abs(ter_of(lengths, window, v, tail) - target)
        if gap < best_gap:
            best, best_gap = v, gap
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--cal_split", default="dev")
    ap.add_argument("--cal_limit", type=int, default=500)
    ap.add_argument("--eval_split", default="dev")
    ap.add_argument("--eval_skip", type=int, default=0, help="skip the first N eval articles (use when cal and eval split coincide)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--tail", default="short", choices=["short", "aligned"])
    ap.add_argument("--mask_left", type=int, default=0)
    ap.add_argument("--mask_right", type=int, default=0)
    ap.add_argument("--conf_thr", type=float, default=0.9)
    ap.add_argument("--seam_margin", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_preds", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    special = (codec.cls_id, codec.sep_id, codec.pad_id)
    logits_fn = make_logits_fn(load_finetuned(args.bert_dir, args.ckpt, device))
    base = dict(window=args.window, overlap=args.overlap, tail=args.tail, mask_left=args.mask_left,
                mask_right=args.mask_right, batch_size=args.batch_size)

    # --- calibration fit -------------------------------------------------
    cal_chars, cal_gold = load_split(args.data_dir, args.cal_split)
    if args.cal_limit:
        cal_chars, cal_gold = cal_chars[:args.cal_limit], cal_gold[:args.cal_limit]
    _, _, cal_probs = decode_parallel(cal_chars, logits_fn, *special, device, return_probs=True, **base)
    cp, cg = np.concatenate(cal_probs), np.concatenate(cal_gold)
    T, bias = calibrate_temperature_and_bias(
        torch.from_numpy(np.log(cp + 1e-12)).float(), torch.from_numpy(cg).long(), n_classes=8)
    ece_before = expected_calibration_error(cp, cg)
    ece_after = expected_calibration_error(calibrate_probs(cp, T, bias), cg)
    print(f"calibration on {len(cal_chars)} articles: T={T:.4f}")

    # --- evaluation set --------------------------------------------------
    chars, gold = load_split(args.data_dir, args.eval_split)
    chars, gold = chars[args.eval_skip:], gold[args.eval_skip:]
    if args.limit:
        chars, gold = chars[:args.limit], gold[:args.limit]
    lengths = [len(a) for a in chars]
    multi = [i for i, n in enumerate(lengths) if n > args.window]
    result = {"ckpt": args.ckpt, "cal_split": args.cal_split, "cal_articles": len(cal_chars),
              "eval_split": args.eval_split, "eval_skip": args.eval_skip, "n_articles": len(chars),
              "n_multi_window": len(multi), "base": base, "conf_thr": args.conf_thr,
              "seam_margin": args.seam_margin,
              "calibration": {"temperature": T, "bias": bias.tolist(), "ece_before": ece_before,
                              "ece_after": ece_after},
              "configs": {}}
    preds = {}

    def record(name, labels, info):
        entry = {"info": info, "all": score(gold, labels)}
        if multi:
            entry["multi_window"] = score([gold[i] for i in multi], [labels[i] for i in multi])
        result["configs"][name] = entry
        preds[name] = np.concatenate(labels)
        print(f"{name}: F1={entry['all']['micro_f1']:.4f} TER={info['ter']:.3f}"
              + (f" multiF1={entry['multi_window']['micro_f1']:.4f}" if multi else ""), flush=True)
        dump(args.out, result)

    kw = dict(seam_margin=args.seam_margin, conf_thr=args.conf_thr, **base)
    labels, info = decode_two_pass(chars, logits_fn, special, device, patch=False, **kw)
    record("pass1", labels, info)
    labels, info_c = decode_two_pass(chars, logits_fn, special, device, calibration=(T, bias), **kw)
    record("c_calibrated", labels, info_c)
    labels, info = decode_two_pass(chars, logits_fn, special, device, **kw)
    record("c_uncalibrated", labels, info)
    labels, info = decode_two_pass(chars, logits_fn, special, device, select="random",
                                   rng=np.random.default_rng(args.seed), budget=info_c["n_selected"], **kw)
    record("c_random", labels, info)

    v = overlap_for_ter(lengths, args.window, args.tail, info_c["ter"])
    labels, st = decode_parallel(chars, logits_fn, *special, device,
                                 **{**base, "overlap": v})
    record("b_same_ter", labels, {"ter": st.ter, "overlap": v, "n_windows": st.n_windows})

    if args.save_preds:
        np.savez_compressed(args.save_preds, **preds)
    dump(args.out, result)


if __name__ == "__main__":
    main()
