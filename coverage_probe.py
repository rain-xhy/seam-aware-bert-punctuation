"""Task 1.3 probe: vocab coverage of cleaned corpus, label-set coverage, long-line stats.

Usage: python -X utf8 coverage_probe.py <corpus_dir> <vocab_path>
Cleaning here is only for measurement; the real pipeline is written in stage 4.
"""
import sys, re, glob, os, unicodedata
from collections import Counter

ID_RE = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")


def load_vocab(path):
    with open(path, "r", encoding="utf-8") as fh:
        return set(line.rstrip("\n") for line in fh)


def clean_line(raw):
    """Strip line id, pinyin braces and inter-word spaces."""
    body = ID_RE.sub("", raw, count=1)
    body = PINYIN_RE.sub("", body)
    return body.replace(" ", "").replace("　", "")


def main(corpus_dir, vocab_path):
    vocab = load_vocab(vocab_path)
    chars = Counter()
    n_lines = 0
    lens_over = {128: 0, 256: 0, 510: 0}
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                body = clean_line(line)
                n_lines += 1
                chars.update(body)
                for k in lens_over:
                    if len(body) > k:
                        lens_over[k] += 1
    total = sum(chars.values())
    hit_raw = sum(c for ch, c in chars.items() if ch in vocab)
    hit_nfkc = sum(c for ch, c in chars.items() if unicodedata.normalize("NFKC", ch) in vocab)
    print(f"lines={n_lines} total_chars={total}")
    print(f"vocab_size={len(vocab)}")
    print(f"coverage raw={hit_raw/total:.5f} after_NFKC={hit_nfkc/total:.5f}")
    print("lines longer than N chars (cleaned, punctuation kept):", lens_over)
    miss = Counter({ch: c for ch, c in chars.items()
                    if unicodedata.normalize("NFKC", ch) not in vocab})
    print("top OOV chars after NFKC:")
    for ch, c in miss.most_common(15):
        print(f"  {ch!r} U+{ord(ch):04X} {c}")
    # Input-side view: the model input has punctuation removed, and BERT's
    # basic tokenizer lower-cases after NFKC, so measure that way.
    def norm(ch):
        return unicodedata.normalize("NFKC", ch).lower()
    inp = Counter({ch: c for ch, c in chars.items()
                   if not unicodedata.category(ch).startswith("P")})
    inp_total = sum(inp.values())
    inp_hit = sum(c for ch, c in inp.items() if norm(ch) in vocab)
    print(f"input-side (category P removed) chars={inp_total} "
          f"coverage after NFKC+lower={inp_hit/inp_total:.5f}")
    inp_miss = Counter({ch: c for ch, c in inp.items() if norm(ch) not in vocab})
    print("top input-side OOV chars after NFKC+lower:")
    for ch, c in inp_miss.most_common(10):
        print(f"  {ch!r} U+{ord(ch):04X} {c}")
    print("selected punctuation in vocab (raw / NFKC):")
    for ch in "，。、；：？！“”‘’（）《》—…·":
        print(f"  {ch} count={chars.get(ch,0)} raw_in_vocab={ch in vocab} nfkc_in_vocab={unicodedata.normalize('NFKC', ch) in vocab}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
