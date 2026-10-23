"""Task 1.3 probe: label-set design facts and long-document candidates.

Questions answered with numbers (all printed, redirect to logs/label_probe.log):
  1. If only the 7 pause marks are labelled and every other punctuation mark is
     dropped from the text, how are labels distributed over character positions?
  2. How often do two pause marks sit next to each other (label conflict)?
  3. After dropping all punctuation, how long are lines, and how long are
     articles when consecutive lines with the same article id are joined?
  4. Is the fullwidth full stop used as a decimal point (kept as a character)?

Usage: python -X utf8 -I label_probe.py <corpus_dir>
"""
import sys, re, glob, os, unicodedata
from collections import Counter, defaultdict

ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
PAUSE = "，。、；：？！"


def clean_body(raw):
    """Drop line id, pinyin braces and inter-word spaces; return (article_id, body)."""
    m = ID_RE.match(raw)
    article = m.group(1) if m else None
    body = ID_RE.sub("", raw, count=1)
    body = PINYIN_RE.sub("", body)
    return article, body.replace(" ", "").replace("　", "")


def to_chars_and_labels(body):
    """Return (chars, labels). A pause mark labels the character before it;
    every other punctuation mark (category P, plus the dash/ellipsis family)
    is removed. Adjacent pause marks keep the last one and are counted."""
    chars, labels, conflicts = [], [], 0
    for ch in body:
        if ch in PAUSE:
            if chars:
                if labels[-1] != "O":
                    conflicts += 1
                labels[-1] = ch
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        chars.append(ch)
        labels.append("O")
    return chars, labels, conflicts


def quantile(sorted_vals, p):
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def main(corpus_dir):
    label_count = Counter()
    conflicts_total = 0
    line_lens = []
    art_len = defaultdict(int)
    art_order = []
    decimal_dot = Counter()
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                article, body = clean_body(line)
                # full stop between two digits => decimal point
                for i, ch in enumerate(body):
                    if ch == "．":
                        left = body[i - 1] if i > 0 else ""
                        right = body[i + 1] if i + 1 < len(body) else ""
                        is_num = left.isdigit() and right.isdigit()
                        decimal_dot["between_digits" if is_num else "other"] += 1
                chars, labels, conf = to_chars_and_labels(body)
                conflicts_total += conf
                label_count.update(labels)
                line_lens.append(len(chars))
                key = (os.path.basename(f), article)
                if key not in art_len:
                    art_order.append(key)
                art_len[key] += len(chars)

    n_pos = sum(label_count.values())
    print(f"char positions={n_pos}")
    for lab, c in label_count.most_common():
        print(f"  label {lab!r}: {c} ({c / n_pos:.4%})")
    print(f"adjacent pause-mark conflicts (dropped extra mark) = {conflicts_total}")

    line_lens.sort()
    print(f"lines={len(line_lens)} input-length (no punctuation): "
          f"mean={sum(line_lens)/len(line_lens):.1f} median={quantile(line_lens,.5)} "
          f"p90={quantile(line_lens,.9)} p99={quantile(line_lens,.99)} max={line_lens[-1]}")
    for k in (128, 256, 400, 510):
        print(f"  lines with > {k} chars: {sum(1 for v in line_lens if v > k)}")

    arts = sorted(art_len.values())
    print(f"articles (file, date-page-article id)={len(arts)}: "
          f"median={quantile(arts,.5)} p90={quantile(arts,.9)} max={arts[-1]}")
    for k in (510, 2000, 5000, 10000):
        print(f"  articles with > {k} chars: {sum(1 for v in arts if v > k)}")
    print(f"fullwidth full stop usage: {dict(decimal_dot)}")


if __name__ == "__main__":
    main(sys.argv[1])
