"""Locate undecodable bytes per corpus file (GB18030)."""
import sys, glob, os
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.txt"))):
    raw = open(f, "rb").read()
    bad = 0; pos = 0; samples = []
    while True:
        try:
            raw[pos:].decode("gb18030"); break
        except UnicodeDecodeError as e:
            a = pos + e.start; bad += 1
            if len(samples) < 3:
                samples.append((a, raw[max(0, a-12):a+12]))
            pos = a + 1
    print(os.path.basename(f), "bad_spans=", bad)
    for a, ctx in samples:
        print("   at", a, ctx.decode("gb18030", errors="replace").replace("\r\n", "\n"))
