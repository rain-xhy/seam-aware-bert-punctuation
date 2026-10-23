"""Layer-2 probe: how much compute would sequence packing save on the test split?

For each test article (crc32(file:article) % 10 == 0, same rule as design_probe.py)
we cut windows of length W with overlap V (stride S = W - V, last window aligned
to the end, articles shorter than W give one short window). Then we count token
slots under three batching schemes:
  naive   : every window padded to W
  sorted  : windows sorted by length, batches of B, each padded to batch max
  packed  : short windows bin-packed (first-fit decreasing) into rows of W
Real tokens / slots = utilisation. Prints numbers only; redirect to logs/.

Usage: python -X utf8 -I padding_probe.py <corpus_dir>
"""
import sys, re, glob, os, unicodedata, zlib
from collections import defaultdict

ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
PAUSE = "，。、；：？！"


def kept_len(raw):
    """Return (article_id, number of input chars after dropping all punctuation)."""
    m = ID_RE.match(raw)
    body = PINYIN_RE.sub("", ID_RE.sub("", raw, count=1)).replace(" ", "").replace("　", "")
    n = sum(1 for ch in body if ch not in PAUSE and not unicodedata.category(ch).startswith("P"))
    return (m.group(1) if m else None), n


def window_lengths(n, w, v):
    """Lengths of the windows covering a text of n chars."""
    if n == 0:
        return []
    if n <= w:
        return [n]
    s = w - v
    count = -(-(n - w) // s) + 1  # ceil((n - w) / s) + 1
    return [w] * count


def first_fit_decreasing(lengths, cap):
    """Number of rows of size cap needed to pack the given lengths."""
    rows = []
    for x in sorted(lengths, reverse=True):
        for i, free in enumerate(rows):
            if free >= x:
                rows[i] -= x
                break
        else:
            rows.append(cap - x)
    return len(rows)


def main(corpus_dir):
    art = defaultdict(int)
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                a, n = kept_len(line)
                art[os.path.basename(f) + ":" + str(a)] += n
    test = [n for k, n in art.items() if zlib.crc32(k.encode()) % 10 == 0 and n > 0]
    print(f"test articles={len(test)} chars={sum(test)}")
    batch = 16
    for w in (128, 256, 510):
        for v in (0, 20, 64):
            if v >= w:
                continue
            wins = [x for n in test for x in window_lengths(n, w, v)]
            real = sum(wins)
            naive = len(wins) * w
            srt = sorted(wins)
            sorted_slots = sum(max(srt[i:i + batch]) * len(srt[i:i + batch])
                               for i in range(0, len(srt), batch))
            full = [x for x in wins if x == w]
            short = [x for x in wins if x < w]
            packed_rows = len(full) + first_fit_decreasing(short, w)
            packed = packed_rows * w
            print(f"W={w} V={v}: windows={len(wins)} short_windows={len(short)} "
                  f"real_tokens={real} overlap_redundancy={real / sum(test):.4f} | "
                  f"util naive={real / naive:.4f} sorted(B={batch})={real / sorted_slots:.4f} "
                  f"packed={real / packed:.4f} rows naive={len(wins)} packed={packed_rows}")


if __name__ == "__main__":
    main(sys.argv[1])
