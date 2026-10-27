"""Convert punct.data_prep output (data/{train,val,test}.npz) into the layout
train.py reads (prepared_data_complete-style *_articles_{char,label}.npy +
char_codec.json), re-ordering labels to the training order.

    python -m scripts.convert_clean_data --src data --out prepared_data_clean \
        --bert_dir bert-base-chinese

Label orders:
    data_prep / char_codec.PAUSE_LABELS : none ， 。 、 ； ： ？ ！
    training / decoding / metrics       : none ， 。 ？ ！ ； ： 、
The sentence-final ids (2, 3, 4) are hard-coded downstream, so the labels are
re-mapped here and nothing else has to change. Splits: val -> dev (file name
used by train.py).

Run data_prep first with the CURRENT CharCodec (the earlier 10-23 output was
written before the vocabulary-shift fix and must not be converted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec, PAUSE_LABELS  # noqa: E402

TRAIN_ORDER = [None, "，", "。", "？", "！", "；", "：", "、"]


def label_remap() -> np.ndarray:
    """remap[old_id] = new_id, derived from the two orders (not typed by hand)."""
    return np.array([TRAIN_ORDER.index(m) for m in PAUSE_LABELS], dtype=np.int64)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_split(npz_path: str, remap: np.ndarray):
    d = np.load(npz_path, allow_pickle=True)
    ids, labels, bounds = d["char_ids"], d["label_ids"].astype(np.int64), d["boundaries"]
    chars, labs = [], []
    for a in range(len(bounds) - 1):
        s, e = int(bounds[a]), int(bounds[a + 1])
        chars.append(ids[s:e].astype(np.int32))
        labs.append(remap[labels[s:e]].astype(np.int32))
    return chars, labs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    args = ap.parse_args()

    remap = label_remap()
    print("label remap old->new:", remap.tolist())
    os.makedirs(args.out, exist_ok=True)
    summary = {"label_remap": remap.tolist(), "splits": {}, "sha256": {}}
    for src_name, dst_name in (("train", "train"), ("val", "dev"), ("test", "test")):
        path = os.path.join(args.src, f"{src_name}.npz")
        chars, labs = convert_split(path, remap)
        for kind, arr in (("char", chars), ("label", labs)):
            out = os.path.join(args.out, f"{dst_name}_articles_{kind}.npy")
            np.save(out, np.array(arr, dtype=object), allow_pickle=True)
            summary["sha256"][os.path.basename(out)] = sha256_file(out)
        n = int(sum(len(c) for c in chars))
        summary["splits"][dst_name] = {"articles": len(chars), "chars": n,
                                       "marks": int(sum((l > 0).sum() for l in labs))}
        print(dst_name, summary["splits"][dst_name])
    codec = CharCodec.from_bert_vocab(args.bert_dir)
    codec.save(os.path.join(args.out, "char_codec.json"))
    with open(os.path.join(args.out, "convert_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
