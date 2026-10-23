"""Count lines that have no characters left after cleaning (explains 123761 vs 123691)."""
import sys, re, glob, os, unicodedata
from collections import Counter
ID_RE = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
PAUSE = "，。、；：？！"
empty = Counter(); sample = []
n = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.txt"))):
    with open(f, "r", encoding="gb18030", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            n += 1
            body = PINYIN_RE.sub("", ID_RE.sub("", line, count=1)).replace(" ", "").replace("　", "")
            kept = [c for c in body if c not in PAUSE and not unicodedata.category(c).startswith("P")]
            if not kept:
                empty[bool(body)] += 1
                if len(sample) < 5:
                    sample.append(body[:30])
print(f"nonblank lines={n}")
print(f"lines with zero kept chars: {sum(empty.values())} (body non-empty but only punctuation={empty[True]}, body empty={empty[False]})")
print("samples:", sample)
