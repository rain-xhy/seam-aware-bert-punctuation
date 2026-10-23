"""Task 1.1 corpus statistics: encoding, line count, punctuation distribution, line length."""
import sys, re, glob, os
from collections import Counter

def main(corpus_dir):
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    bad_total = 0
    punct = Counter(); n_lines = 0; n_chars = 0; lens = []; han_total = 0
    id_re = re.compile(r"^\d{8}-\d{2}-\d{3}-\d{3}\s")
    for f in files:
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            fl = 0; bad0 = bad_total
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                body = id_re.sub("", line, count=1).replace(" ", "")
                bad_total += body.count("�")
                fl += 1; n_lines += 1; n_chars += len(body); lens.append(len(body))
                for ch in body:
                    if '一' <= ch <= '鿿':
                        han_total += 1
                    elif not ch.isalnum():
                        punct[ch] += 1
            print(f"{os.path.basename(f)}: lines={fl} replaced_bad_chars={bad_total - bad0}")
    print(f"total U+FFFD replacement chars={bad_total}")
    lens.sort()
    q = lambda p: lens[int(p * (len(lens) - 1))]
    print(f"total lines={n_lines} chars(no space)={n_chars} han={han_total}")
    print(f"line len: mean={sum(lens)/len(lens):.1f} median={q(.5)} p90={q(.9)} p99={q(.99)} max={lens[-1]}")
    print("top non-alnum chars:")
    for ch, c in punct.most_common(20):
        print(f"  {ch!r} U+{ord(ch):04X} {c}")

if __name__ == "__main__":
    main(sys.argv[1])
