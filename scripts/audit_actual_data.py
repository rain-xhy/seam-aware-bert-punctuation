"""Audit of the data actually used for training (prepared_data_complete/).

Writes logs/data_audit_actual.json: file SHA-256 of the raw corpus and of the
prepared arrays, per-split article/char/mark counts, label distribution, and
the most frequent non-CJK characters that remain in the input (to check which
symbols the simplified cleaning left in).
"""
import hashlib, json, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from punct.char_codec import CharCodec

DATA = "prepared_data_complete"
CORPUS = r"E:\desktop\北大软微\课程\大语言模型应用与实践\作业\8\1998-上半年-带音"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    codec = CharCodec.load(os.path.join(DATA, "char_codec.json"))
    out = {"raw_files_sha256": {}, "prepared_files_sha256": {}, "splits": {}}
    for fn in sorted(os.listdir(CORPUS)):
        if fn.endswith(".txt"):
            out["raw_files_sha256"][fn] = sha(os.path.join(CORPUS, fn))
    for fn in sorted(os.listdir(DATA)):
        out["prepared_files_sha256"][fn] = sha(os.path.join(DATA, fn))
    for split in ("train", "dev", "test"):
        ch = np.load(os.path.join(DATA, f"{split}_articles_char.npy"), allow_pickle=True)
        lb = np.load(os.path.join(DATA, f"{split}_articles_label.npy"), allow_pickle=True)
        allc = np.concatenate([np.asarray(a) for a in ch])
        alll = np.concatenate([np.asarray(a) for a in lb])
        cnt = Counter(allc.tolist())
        unk = cnt.get(codec.unk_id, 0)
        noncjk = Counter()
        for cid, n in cnt.items():
            c = codec.decode_char(int(cid))
            if len(c) == 1 and not ("\u4e00" <= c <= "\u9fff"):
                noncjk[c] += n
        out["splits"][split] = {
            "articles": len(ch), "chars": int(allc.size),
            "label_counts": np.bincount(alll, minlength=8).tolist(),
            "min_article_len": int(min(len(a) for a in ch)),
            "unk_chars": int(unk), "unk_rate": unk / allc.size,
            "top_non_cjk_chars": noncjk.most_common(25),
            "non_cjk_total": int(sum(noncjk.values())),
        }
    os.makedirs("logs", exist_ok=True)
    with open("logs/data_audit_actual.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for s, v in out["splits"].items():
        print(s, v["articles"], v["chars"], "non_cjk", v["non_cjk_total"], "unk", v["unk_chars"])
        print("  top:", v["top_non_cjk_chars"][:15])


if __name__ == "__main__":
    main()
