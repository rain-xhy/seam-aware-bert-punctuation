"""Stage-3 probe: real training and inference speed of bert-base-chinese on this GPU.

Training : forward + backward + AdamW step, bf16 autocast, windows cut from the real
           1-4 month text, several (window length L, micro-batch B) pairs.
Inference: forward only, pure bf16 weights, several (window W, batch B) pairs.
An out-of-memory error marks that configuration as failed and the probe moves on.
Prints environment versions first. Writes nothing to disk; redirect stdout to logs/.

Usage: python -X utf8 train_speed_probe.py <corpus_dir> <model_dir>
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys, re, glob, time, random, statistics, subprocess, unicodedata, gc
import torch
import transformers
from transformers import BertForTokenClassification

ID_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})-\d{3}\s")
PINYIN_RE = re.compile(r"\{[A-Za-z0-9]+\}")
PAUSE = "，。、；：？！"
LABEL_OF = {ch: i + 1 for i, ch in enumerate(PAUSE)}
TRAIN_CHARS = 7125456  # training months, punctuation removed (split_probe.log)


def smi():
    """GPU temperature, clocks and power, to see throttling."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as exc:  # report, do not hide
        return "nvidia-smi failed: %s" % exc


def load_vocab(model_dir):
    with open(os.path.join(model_dir, "vocab.txt"), encoding="utf-8") as fh:
        return {line.rstrip("\n"): i for i, line in enumerate(fh)}


def build_stream(corpus_dir, vocab, max_chars):
    """Token ids and 8-class labels of the concatenated training-month text."""
    unk = vocab["[UNK]"]
    ids, labels = [], []
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        month = re.search(r"1998-(\d\d)", os.path.basename(f)).group(1)
        if month not in ("01", "02", "03", "04"):
            continue
        with open(f, "r", encoding="gb18030", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                body = PINYIN_RE.sub("", ID_RE.sub("", line, count=1))
                body = body.replace(" ", "").replace("　", "")
                for ch in body:
                    if ch in LABEL_OF:
                        if labels:
                            labels[-1] = LABEL_OF[ch]
                        continue
                    if unicodedata.category(ch).startswith("P"):
                        continue
                    norm = unicodedata.normalize("NFKC", ch).lower()
                    ids.append(vocab.get(norm, unk) if len(norm) == 1 else unk)
                    labels.append(0)
                if len(ids) >= max_chars:
                    return ids, labels
    return ids, labels


def make_batches(ids, labels, vocab, length, batch, count, rng, device):
    cls, sep = vocab["[CLS]"], vocab["[SEP]"]
    out = []
    for _ in range(count):
        rows, labs = [], []
        for _ in range(batch):
            s = rng.randrange(0, len(ids) - length)
            rows.append([cls] + ids[s:s + length] + [sep])
            labs.append([-100] + labels[s:s + length] + [-100])
        x = torch.tensor(rows, device=device)
        out.append((x, torch.ones_like(x), torch.tensor(labs, device=device)))
    return out


def timed(step, warmup, steps):
    for i in range(warmup):
        step(i)
    torch.cuda.synchronize()
    times = []
    for i in range(steps):
        t = time.perf_counter()
        step(warmup + i)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return times


def report(tag, length, batch, times):
    med = statistics.median(times)
    cps = batch * length / med
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    sd = statistics.pstdev(times)
    print("%s L=%d B=%d: median %.1f ms (sd %.1f), %.0f chars/s, peak %.0f MiB | %s"
          % (tag, length, batch, med * 1000, sd * 1000, cps, peak, smi()))
    return cps


def main(corpus_dir, model_dir):
    device = torch.device("cuda")
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
          "| transformers", transformers.__version__)
    props = torch.cuda.get_device_properties(0)
    print("GPU", props.name, "| total MiB", props.total_memory // 2 ** 20,
          "| bf16 supported", torch.cuda.is_bf16_supported(), "| idle:", smi())
    vocab = load_vocab(model_dir)
    ids, labels = build_stream(corpus_dir, vocab, 600000)
    print("stream chars", len(ids), "| label counts", [labels.count(k) for k in range(8)])

    rng = random.Random(0)
    torch.manual_seed(0)
    model = BertForTokenClassification.from_pretrained(model_dir, num_labels=8, local_files_only=True)
    print("attention implementation:", getattr(model.config, "_attn_implementation", "n/a"))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    print("\n== training (fwd + bwd + AdamW step, bf16 autocast, no data-loader or validation cost) ==")
    model.train()
    for length, batch in ((128, 16), (256, 8), (256, 16), (510, 4), (510, 8)):
        failed = False
        try:
            torch.cuda.reset_peak_memory_stats()
            pool = make_batches(ids, labels, vocab, length, batch, 6, rng, device)

            def step(i):
                x, m, y = pool[i % len(pool)]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=x, attention_mask=m, labels=y).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

            cps = report("train", length, batch, timed(step, 3, 25))
            print("   -> one pass over the %d training chars: %.1f min" % (TRAIN_CHARS, TRAIN_CHARS / cps / 60))
        except torch.cuda.OutOfMemoryError:
            failed = True
        if failed:
            print("train L=%d B=%d: OOM (failed configuration)" % (length, batch))
            opt.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    print("\n== inference (forward only, pure bf16 weights, inference_mode) ==")
    model.eval()
    model.to(torch.bfloat16)
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    for window in (125, 510):
        prev = None
        for batch in (1, 2, 4, 8, 16, 32, 64):
            failed = False
            try:
                torch.cuda.reset_peak_memory_stats()
                pool = make_batches(ids, labels, vocab, window, batch, 4, rng, device)

                @torch.inference_mode()
                def step(i):
                    x, m, _ = pool[i % len(pool)]
                    model(input_ids=x, attention_mask=m)

                cps = report("infer", window, batch, timed(step, 2, 10))
                if prev:
                    print("   -> gain over previous batch: %+.1f%%" % ((cps / prev - 1) * 100))
                prev = cps
            except torch.cuda.OutOfMemoryError:
                failed = True
            if failed:
                print("infer W=%d B=%d: OOM (failed configuration)" % (window, batch))
            gc.collect()
            torch.cuda.empty_cache()
    print("\nnote: timings exclude tokenisation, data loading, validation and checkpoint saving.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
