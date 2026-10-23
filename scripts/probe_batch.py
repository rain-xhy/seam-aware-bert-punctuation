"""批量大小探测：测试不同batch size的吞吐、显存和OOM边界。

从batch_size=1开始，按2倍递增(1,2,4,8,16,32,64)，记录每个配置的：
- 吞吐量(字符/秒)
- 显存峰值
- 是否OOM

找到吞吐增幅<5%的拐点作为推荐batch size。

    python -m scripts.probe_batch --ckpt checkpoints/T1_varlen/best.pt \
        --split dev --limit 100 --window 400 --overlap 100 \
        --out results/batch_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.runner import load_split, run_method  # noqa: E402


def probe_one_batch(chars, gold, logits_fn, special, device, batch_size, **kw):
    """测试一个batch size，返回(成功, 吞吐, 显存峰值MB, 耗时)"""
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        pred, stats = run_method("parallel", chars, logits_fn, special, device,
                                batch_size=batch_size, **kw)
        elapsed = time.perf_counter() - t0

        throughput = stats["text_chars"] / max(elapsed, 1e-9)
        peak_mb = stats.get("peak_mem_mb", 0.0)

        return True, throughput, peak_mb, elapsed

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            return False, 0.0, 0.0, 0.0
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=100, help="use first N articles")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--max_batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("警告: 需要CUDA才能测量显存")

    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))
    chars, gold = load_split(args.data_dir, args.split)
    chars, gold = chars[: args.limit], gold[: args.limit]
    total_chars = sum(len(c) for c in chars)

    print(f"加载模型: {args.ckpt}")
    model = load_finetuned(args.bert_dir, args.ckpt, device)
    logits_fn = make_logits_fn(model)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)
    kw = dict(window=args.window, overlap=args.overlap, tail="short")

    print(f"\n探测batch size: 数据={args.limit}篇文章, {total_chars}字")
    print(f"窗口配置: W={args.window}, V={args.overlap}\n")

    results = []
    batch = 1
    last_throughput = 0.0

    while batch <= args.max_batch:
        print(f"测试 batch={batch}...", end=" ", flush=True)

        success, throughput, peak_mb, elapsed = probe_one_batch(
            chars, gold, logits_fn, special, device, batch, **kw
        )

        if not success:
            print("✗ OOM")
            results.append({
                "batch_size": batch,
                "success": False,
                "oom": True,
            })
            batch *= 2
            continue

        # 计算吞吐增幅
        gain = 0.0 if last_throughput == 0 else (throughput - last_throughput) / last_throughput
        results.append({
            "batch_size": batch,
            "success": True,
            "throughput": round(throughput, 1),
            "peak_mem_mb": round(peak_mb, 1),
            "elapsed_s": round(elapsed, 2),
            "gain_percent": round(gain * 100, 2),
        })

        print(f"✓ {throughput:.0f} 字/秒, {peak_mb:.0f} MiB, 增幅 {gain*100:+.1f}%")
        last_throughput = throughput
        batch *= 2

    # 找推荐batch: 增幅<5%的拐点
    recommend = 1
    for r in results:
        if r["success"] and r.get("gain_percent", 100) >= 5:
            recommend = r["batch_size"]

    output = {
        "config": {"window": args.window, "overlap": args.overlap, "limit": args.limit,
                   "total_chars": total_chars},
        "results": results,
        "recommended_batch": recommend,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n推荐batch size: {recommend}")
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
