"""Probe for the dynamic-overlap idea: does sentence length vary inside an article,
and is it locally predictable (so a stride could adapt to it)?

Sentences are cut at 。？！ inside each article (not across articles, unlike
design_probe.py). The trailing fragment after the last final mark is excluded.
Same cleaning and label rule as design_probe.py.

Prints:
  1. share of sentence-length variance that is within articles
  2. lag-1 autocorrelation of sentence length within articles
     (near 0 => neighbouring density does not predict the next sentence)
  3. fraction of sentences longer than an overlap of 10/20/30 % of 125 chars and 20
Same for clauses (cut at any pause mark).

Usage: python -X utf8 -I stride_probe.py <corpus_dir>
"""
import sys, re, glob, os, unicodedata
from collections import defaultdict

ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
PAUSE = "，。、；：？！"
FINAL = "。？！"


def labels_of(body):
    """Pause mark labels the previous char; other P-category chars dropped."""
    labels = []
    for ch in body:
        if ch in PAUSE:
            if labels:
                labels[-1] = ch
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        labels.append("O")
    return labels


def pieces(labels, marks):
    """Lengths of pieces ending at a char labelled with one of `marks`."""
    out, last = [], -1
    for i, lab in enumerate(labels):
        if lab in marks:
            out.append(i - last)
            last = i
    return out


def report(name, per_article, budget=125):
    allv = [x for seq in per_article for x in seq]
    n = len(allv)
    mean = sum(allv) / n
    total_ss = sum((x - mean) ** 2 for x in allv)
    within_ss, num, den_terms = 0.0, 0.0, []
    for seq in per_article:
        if len(seq) < 2:
            continue
        m = sum(seq) / len(seq)
        within_ss += sum((x - m) ** 2 for x in seq)
    # lag-1 autocorrelation, centred on each article's own mean
    num, den = 0.0, 0.0
    for seq in per_article:
        if len(seq) < 3:
            continue
        m = sum(seq) / len(seq)
        d = [x - m for x in seq]
        num += sum(d[i] * d[i + 1] for i in range(len(d) - 1))
        den += sum(v * v for v in d)
    used = sum(1 for s in per_article if len(s) >= 2)
    print(f"[{name}] count={n} mean={mean:.2f} articles_with>=2={used}")
    print(f"  within-article share of variance={within_ss / total_ss:.4f}")
    print(f"  lag-1 autocorrelation (within article)={num / den:.4f}")
    for v in (round(budget * 0.1), 20, round(budget * 0.2), round(budget * 0.3)):
        k = sum(1 for x in allv if x > v)
        print(f"  longer than overlap {v}: {k} ({k / n:.4%})")


def main(corpus_dir):
    art = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                m = ID_RE.match(line)
                body = PINYIN_RE.sub("", ID_RE.sub("", line, count=1))
                body = body.replace(" ", "").replace("　", "")
                art[os.path.basename(f) + ":" + str(m.group(1) if m else None)].extend(labels_of(body))
    sent = [pieces(l, FINAL) for l in art.values()]
    clause = [pieces(l, PAUSE) for l in art.values()]
    print(f"articles={len(art)}")
    report("sentence 。？！", sent)
    report("clause any pause mark", clause)


if __name__ == "__main__":
    main(sys.argv[1])
