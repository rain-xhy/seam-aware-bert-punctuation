"""Task 2.2 probe: sizes of the monthly split and the cost of the last window.

Split: months 01-04 -> train, 05 -> val, 06 -> test (by file name).
Article = consecutive lines sharing the first three id fields, as before.
Length = input chars after removing every punctuation mark (same as padding_probe.py).

For the test split it also compares two tail rules for fixed windows (W, V):
  align : the last window is moved back so that it ends at the article end (length W)
  short : the last window starts one stride after the previous one and is shorter
TER = forward tokens / article chars (special tokens not counted).

Usage: python -X utf8 -I split_probe.py <corpus_dir>
"""
import sys, re, glob, os, unicodedata
from collections import defaultdict

ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
MONTH_RE = re.compile(r"1998-(\d\d)")
SPLIT = {"01": "train", "02": "train", "03": "train", "04": "train", "05": "val", "06": "test"}


def kept_len(raw):
    """Return (article_id, number of input chars after dropping all punctuation)."""
    m = ID_RE.match(raw)
    body = PINYIN_RE.sub("", ID_RE.sub("", raw, count=1)).replace(" ", "").replace("　", "")
    n = sum(1 for ch in body if not unicodedata.category(ch).startswith("P"))
    return (m.group(1) if m else None), n


def window_lengths(n, w, v, tail):
    """Lengths of the windows that cover n chars with window w and overlap v."""
    if n == 0:
        return []
    if n <= w:
        return [n]
    s = w - v
    starts = list(range(0, n - w + 1, s))
    lengths = [w] * len(starts)
    if starts[-1] + w < n:
        if tail == "align":
            lengths.append(w)
        else:
            lengths.append(n - (starts[-1] + s))
    return lengths


def main(corpus_dir):
    art = defaultdict(int)
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        split = SPLIT[MONTH_RE.search(os.path.basename(f)).group(1)]
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                a, n = kept_len(line)
                art[(split, os.path.basename(f), a)] += n
    by_split = defaultdict(list)
    for (split, _, _), n in art.items():
        if n > 0:
            by_split[split].append(n)
    for split in ("train", "val", "test"):
        v = by_split[split]
        print(f"{split}: articles={len(v)} chars={sum(v)} "
              f">510={sum(x > 510 for x in v)} >2000={sum(x > 2000 for x in v)} "
              f">5000={sum(x > 5000 for x in v)} max={max(v)}")
    test = by_split["test"]
    total = sum(test)
    print("test tail rule (W,V): windows / TER for align vs short")
    for w, v in ((64, 0), (125, 20), (128, 20), (400, 100), (510, 0), (510, 20), (510, 64)):
        row = []
        for tail in ("align", "short"):
            lens = [x for n in test for x in window_lengths(n, w, v, tail)]
            row.append(f"{tail}: windows={len(lens)} TER={sum(lens) / total:.4f}")
        print(f"  W={w} V={v}: " + " | ".join(row))


if __name__ == "__main__":
    main(sys.argv[1])
