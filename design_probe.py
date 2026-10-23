"""Task 1.3 probe 3: numbers that decide window/overlap/split/tokenisation choices.

Answers, with printed numbers (redirect to logs/design_probe.log):
  1. Which category-P characters exist besides the 7 pause marks (keep vs drop list)?
  2. How long are sentences (split at . ? !) and clauses (split at any pause mark)?
     This bounds how large the window overlap must be.
  3. How many lines end with a pause mark (train/inference end-of-window mismatch)?
  4. Does char-by-char vocab lookup work (NFKC+lower gives one char)?
  5. What does a 9:1 split by article id give (crc32 so it is reproducible)?

Usage: python -X utf8 -I design_probe.py <corpus_dir> <vocab_path>
"""
import sys, re, glob, os, unicodedata, zlib
from collections import Counter, defaultdict

ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
PAUSE = "，。、；：？！"
FINAL = "。？！"


def clean_body(raw):
    """Return (article_id, text) with id, pinyin braces and spaces removed."""
    m = ID_RE.match(raw)
    body = PINYIN_RE.sub("", ID_RE.sub("", raw, count=1))
    return (m.group(1) if m else None), body.replace(" ", "").replace("　", "")


def to_chars_labels(body):
    """Pause mark labels the previous char; all other P-category chars dropped."""
    chars, labels = [], []
    for ch in body:
        if ch in PAUSE:
            if chars:
                labels[-1] = ch
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        chars.append(ch)
        labels.append("O")
    return chars, labels


def quant(vals, ps=(0.5, 0.9, 0.99)):
    vals = sorted(vals)
    return {p: vals[int(p * (len(vals) - 1))] for p in ps}


def gaps(label_str, marks):
    """Lengths of the pieces cut after each char whose label is in `marks`."""
    out, last = [], -1
    for i, lab in enumerate(label_str):
        if lab in marks:
            out.append(i - last)
            last = i
    return out


def main(corpus_dir, vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as fh:
        vocab = set(l.rstrip("\n") for l in fh)
    punct_other = Counter()
    art_text, art_lab = defaultdict(list), defaultdict(list)
    line_end, n_lines = Counter(), 0
    multi, alnum_chars, alnum_runs, n_chars = Counter(), 0, 0, 0
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                art, body = clean_body(line)
                for ch in body:
                    if unicodedata.category(ch).startswith("P") and ch not in PAUSE:
                        punct_other[ch] += 1
                chars, labels = to_chars_labels(body)
                if not chars:
                    continue
                n_lines += 1
                line_end["pause" if labels[-1] != "O" else "none"] += 1
                line_end["final" if labels[-1] in FINAL else "not_final"] += 1
                prev_alnum = False
                for ch in chars:
                    n_chars += 1
                    norm = unicodedata.normalize("NFKC", ch).lower()
                    if len(norm) != 1:
                        multi[ch] += 1
                    is_al = len(norm) == 1 and norm.isascii() and norm.isalnum()
                    alnum_chars += is_al
                    alnum_runs += is_al and not prev_alnum
                    prev_alnum = is_al
                key = os.path.basename(f) + ":" + str(art)
                art_text[key].append("".join(chars))
                art_lab[key].append("".join(labels))
    print(f"[1] other category-P chars (not the 7 pause marks), top 30 of {len(punct_other)}:")
    for ch, c in punct_other.most_common(30):
        print(f"  {ch} U+{ord(ch):04X} {unicodedata.category(ch)} {unicodedata.name(ch, '?')} {c}")
    all_lab = "".join("".join(v) for v in art_lab.values())
    sent = gaps(all_lab, FINAL)
    clause = gaps(all_lab, PAUSE)
    print(f"[2] chars={len(all_lab)} pause marks={len(clause)} sentence-final marks={len(sent)}")
    print(f"  mean chars per clause={len(all_lab)/len(clause):.2f} q={quant(clause)} max={max(clause)}")
    print(f"  mean chars per sentence={len(all_lab)/len(sent):.2f} q={quant(sent)} max={max(sent)}")
    for k in (20, 50, 100, 200):
        print(f"  sentences longer than {k}: {sum(1 for v in sent if v > k)} "
              f"({sum(1 for v in sent if v > k)/len(sent):.4%})")
    print(f"[3] lines={n_lines} end with a pause mark: {line_end['pause']} "
          f"({line_end['pause']/n_lines:.4%}); end with . ? ! : {line_end['final']} "
          f"({line_end['final']/n_lines:.4%})")
    print(f"[4] input chars={n_chars}; NFKC+lower length != 1: {sum(multi.values())} "
          f"top={multi.most_common(5)}")
    print(f"  ascii letter/digit chars after NFKC={alnum_chars} in {alnum_runs} runs "
          f"({alnum_chars/n_chars:.4%} of chars)")
    split = {"train": [0, 0, 0, 0], "test": [0, 0, 0, 0]}
    for key, parts in art_text.items():
        n = sum(len(p) for p in parts)
        part = "test" if zlib.crc32(key.encode()) % 10 == 0 else "train"
        s = split[part]
        s[0] += 1; s[1] += n; s[2] += n > 510; s[3] += n > 2000
    print("[5] split by crc32(file:article)%10==0 -> test; fields: articles chars >510 >2000")
    for part, s in split.items():
        print(f"  {part}: articles={s[0]} chars={s[1]} articles>510={s[2]} articles>2000={s[3]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
