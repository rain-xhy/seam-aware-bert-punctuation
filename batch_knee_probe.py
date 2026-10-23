"""Stage-3 probe 2: re-measure inference throughput vs batch with more warm-up and repeats.

train_speed_probe.py showed an implausible jump between B=1 and B=2 at W=125 (sd 5 ms).
Here every configuration gets 20 warm-up steps and 40 timed steps, and the configurations
are visited in two orders (ascending, then descending) so that clock/thermal drift shows up.
Prints median ms, sd, chars/s, peak memory. Needs the model dir only; inputs are random
vocab ids, which is enough for a speed measurement (no label or accuracy is computed).

Usage: python -X utf8 batch_knee_probe.py <model_dir>
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys, time, statistics, gc
import torch
from transformers import BertForTokenClassification

WINDOWS = (64, 125, 256, 510)
BATCHES = (1, 2, 4, 8, 16, 32, 64)
WARMUP, STEPS = 20, 40


def run(model, window, batch, device):
    """Median seconds per forward and peak MiB, or None when the configuration runs out of memory."""
    try:
        torch.cuda.reset_peak_memory_stats()
        g = torch.Generator(device="cpu").manual_seed(window * 1000 + batch)
        x = torch.randint(1000, 20000, (batch, window + 2), generator=g).to(device)
        mask = torch.ones_like(x)
        times = []
        with torch.inference_mode():
            for i in range(WARMUP + STEPS):
                torch.cuda.synchronize()
                t = time.perf_counter()
                model(input_ids=x, attention_mask=mask)
                torch.cuda.synchronize()
                if i >= WARMUP:
                    times.append(time.perf_counter() - t)
        return statistics.median(times), statistics.pstdev(times), torch.cuda.max_memory_allocated() / 2 ** 20
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main(model_dir):
    device = torch.device("cuda")
    model = BertForTokenClassification.from_pretrained(model_dir, num_labels=8, local_files_only=True)
    model.to(device).to(torch.bfloat16).eval()
    for window in WINDOWS:
        table = {}
        for order in ("asc", "desc"):
            seq = BATCHES if order == "asc" else tuple(reversed(BATCHES))
            for batch in seq:
                table.setdefault(batch, []).append(run(model, window, batch, device))
        print("\nW=%d  (chars/s counts window characters only, excluding [CLS]/[SEP])" % window)
        prev = None
        for batch in BATCHES:
            runs = table[batch]
            if any(r is None for r in runs):
                print("  B=%2d: OOM (failed configuration)" % batch)
                continue
            cps = [batch * window / r[0] for r in runs]
            best = statistics.mean(cps)
            spread = abs(cps[0] - cps[1]) / best * 100
            gain = "" if prev is None else "  gain %+.1f%%" % ((best / prev - 1) * 100)
            print("  B=%2d: asc %.0f / desc %.0f chars/s (mean %.0f, asc-desc gap %.1f%%), "
                  "median %.1f ms, sd %.2f ms, peak %.0f MiB%s"
                  % (batch, cps[0], cps[1], best, spread, runs[0][0] * 1000, runs[0][1] * 1000, runs[0][2], gain))
            prev = best


if __name__ == "__main__":
    main(sys.argv[1])
