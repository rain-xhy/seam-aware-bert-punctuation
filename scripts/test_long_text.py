"""长文本鲁棒性测试：合成10万字的文章，验证窗口覆盖完整且无OOM。

    python -m scripts.test_long_text --ckpt checkpoints/T1_varlen/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.char_codec import CharCodec  # noqa: E402
from punct.engine import load_finetuned, make_logits_fn  # noqa: E402
from punct.runner import run_method  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="prepared_data_complete")
    ap.add_argument("--bert_dir", default="bert-base-chinese")
    ap.add_argument("--length", type=int, default=100000)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = CharCodec.load(os.path.join(args.data_dir, "char_codec.json"))

    # 合成长文本：循环重复一段真实文本
    rng = np.random.default_rng(42)
    base_text = rng.integers(200, 5000, size=1000).tolist()  # 真实字符范围
    long_text = (base_text * (args.length // len(base_text) + 1))[:args.length]
    fake_gold = [rng.choice(8) for _ in range(args.length)]  # 随机标签

    print(f"测试长文本: {args.length} 字")
    print(f"窗口: {args.window}, 重叠: {args.overlap}")

    model = load_finetuned(args.bert_dir, args.ckpt, device)
    logits_fn = make_logits_fn(model)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)

    try:
        pred, stats = run_method("parallel", [long_text], logits_fn, special, device,
                                window=args.window, overlap=args.overlap, batch_size=8)

        assert len(pred[0]) == args.length, f"长度不匹配: {len(pred[0])} vs {args.length}"

        print(f"\n✓ 测试通过:")
        print(f"  窗口数: {stats['n_windows']}")
        print(f"  TER: {stats['ter']:.3f}")
        print(f"  OOM重试: {stats.get('n_oom_retries', 0)}")
        print(f"  耗时: {stats['seconds']:.1f}s")
        print(f"  吞吐: {stats['chars_per_second']:.0f} 字/秒")
        print(f"  峰值显存: {stats.get('peak_mem_mb', 0):.0f} MiB")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        raise


if __name__ == "__main__":
    main()
