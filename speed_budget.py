"""Derived speed budget: arithmetic on numbers that were already measured (no GPU needed).

Reads logs/batch_knee_probe.log (inference throughput) and logs/split_probe.log (TER) and
prints what the measured throughput plateau implies for three decoding schemes, forward
pass only (tokenisation, python loops and merging are not included):

  teacher serial 400/300, one window per forward   TER >= 400/300 (it keeps at most 300 chars)
  cross-article serial 400/300, batched lock-step  same TER, plateau throughput
  parallel fixed windows, short tail, W=510 V=20   TER from split_probe.log, plateau throughput

W=400 was not timed, so the W=510 B=1 throughput stands in for the serial window. That
overstates the serial speed, hence understates the speed-up.

Usage: python -X utf8 speed_budget.py <batch_knee_probe.log> <split_probe.log>
"""
import re
import sys


def knee_means(path, window):
    """Mean chars/s per batch size for one window length, from batch_knee_probe.log."""
    inside, out = False, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            head = re.match(r"W=(\d+)", line)
            if head:
                inside = int(head.group(1)) == window
                continue
            row = re.match(r"\s+B=\s*(\d+): asc \d+ / desc \d+ chars/s \(mean (\d+)", line)
            if inside and row:
                out[int(row.group(1))] = int(row.group(2))
    return out


def ter_pair(path, window, overlap):
    """(aligned-tail TER, short-tail TER) for one window/overlap, from split_probe.log."""
    pattern = r"\s+W=%d V=%d: align: windows=\d+ TER=([\d.]+) \| short: windows=\d+ TER=([\d.]+)" % (window, overlap)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            hit = re.match(pattern, line)
            if hit:
                return float(hit.group(1)), float(hit.group(2))
    raise ValueError("W=%d V=%d not found in %s" % (window, overlap, path))


def main(knee_log, split_log):
    means = knee_means(knee_log, 510)
    thr_b1 = means[1]
    plateau = max(v for b, v in means.items() if b >= 8)
    aligned, short = ter_pair(split_log, 510, 20)
    serial_ter = 400 / 300
    t_serial = serial_ter / thr_b1           # seconds per output char, one window per forward
    t_cross = serial_ter / plateau           # same windows, but forwards are saturated
    t_ours = short / plateau                 # fixed windows, short tail
    t_aligned = aligned / plateau
    print("inputs: W=510 B=1 %d chars/s | plateau (best of B>=8) %d chars/s" % (thr_b1, plateau))
    print("inputs: TER aligned %.4f, short %.4f (W=510 V=20) | serial TER lower bound %.4f"
          % (aligned, short, serial_ter))
    print("batching alone at W=510: plateau / B=1 = %.2fx" % (plateau / thr_b1))
    print("forward-only speed-up of parallel short-tail vs teacher serial (B=1): %.2fx" % (t_serial / t_ours))
    print("forward-only speed-up of parallel short-tail vs cross-article serial: %.2fx" % (t_cross / t_ours))
    print("short tail vs aligned tail, same W/V: %.1f%% less time, %.1f%% higher throughput"
          % ((1 - t_ours / t_aligned) * 100, (t_aligned / t_ours - 1) * 100))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
