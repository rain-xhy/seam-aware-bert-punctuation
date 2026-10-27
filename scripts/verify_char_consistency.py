"""逐字一致性验证
验证不同batch大小、不同配置下的预测结果是否逐字一致
"""
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from punct.char_codec import CharCodec
from punct.engine import load_finetuned, make_logits_fn
from punct.runner import load_split, run_method
import torch

def compare_predictions(pred1, pred2, name1, name2):
    """比较两组预测的逐字一致性"""
    # 拼接成一维数组
    flat1 = np.concatenate(pred1)
    flat2 = np.concatenate(pred2)

    if len(flat1) != len(flat2):
        return {
            'consistent': False,
            'reason': f'length mismatch: {len(flat1)} vs {len(flat2)}',
            'agreement_rate': 0.0
        }

    matches = (flat1 == flat2).sum()
    total = len(flat1)
    rate = matches / total if total > 0 else 0.0

    return {
        'consistent': (matches == total),
        'agreement_rate': rate,
        'matches': int(matches),
        'total': int(total),
        'name1': name1,
        'name2': name2
    }

def main():
    ap = argparse.ArgumentParser(description="验证不同batch大小的逐字一致性")
    ap.add_argument("--ckpt", required=True, help="模型检查点")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=100, help="测试文章数")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16],
                    help="要测试的batch大小列表")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args()

    print("="*60)
    print("逐字一致性验证")
    print("="*60)
    print(f"检查点: {args.ckpt}")
    print(f"数据集: {args.split} (前 {args.limit} 篇)")
    print(f"Batch sizes: {args.batches}")
    print()

    # 加载数据
    chars, gold = load_split("prepared_data_complete", args.split)
    if args.limit:
        chars = chars[:args.limit]
        gold = gold[:args.limit]

    # 加载模型和codec
    from transformers import BertTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
    codec = CharCodec(tokenizer.vocab)
    model = load_finetuned("bert-base-chinese", args.ckpt, device, num_labels=8)
    logits_fn = make_logits_fn(model)
    special = (codec.cls_id, codec.sep_id, codec.pad_id)

    # 对每个batch size运行
    results = {}
    predictions = {}

    for batch_size in args.batches:
        print(f"运行 batch_size={batch_size}...", flush=True)
        kw = dict(window=args.window, overlap=args.overlap, tail="short", batch_size=batch_size)
        pred, stats = run_method("parallel", chars, logits_fn, special, device, **kw)

        predictions[batch_size] = pred
        results[batch_size] = {
            'batch_size': batch_size,
            'seconds': stats['seconds'],
            'chars_per_sec': stats['chars_per_second']
        }
        print(f"  耗时: {stats['seconds']:.2f}s, 速度: {stats['chars_per_second']:.0f} 字/秒")

    # 逐对比较
    print("\n" + "="*60)
    print("一致性检查")
    print("="*60)

    comparisons = []
    reference = args.batches[0]
    all_consistent = True

    for i, b1 in enumerate(args.batches):
        for b2 in args.batches[i+1:]:
            comp = compare_predictions(
                predictions[b1],
                predictions[b2],
                f"batch={b1}",
                f"batch={b2}"
            )
            comparisons.append(comp)

            status = "✅" if comp['consistent'] else "❌"
            print(f"{status} {comp['name1']} vs {comp['name2']}: "
                  f"一致率 {comp['agreement_rate']*100:.4f}% "
                  f"({comp['matches']}/{comp['total']})")

            if not comp['consistent']:
                all_consistent = False

    print("\n" + "="*60)
    if all_consistent:
        print("✅ 所有batch大小的预测结果完全一致")
    else:
        print("⚠️  存在不一致的预测结果")
    print("="*60)

    # 保存结果
    output = {
        'ckpt': args.ckpt,
        'split': args.split,
        'n_articles': len(chars),
        'window': args.window,
        'overlap': args.overlap,
        'batch_sizes': args.batches,
        'results': results,
        'comparisons': [{k: (int(v) if isinstance(v, np.integer) else
                            float(v) if isinstance(v, np.floating) else
                            bool(v) if isinstance(v, np.bool_) else v)
                        for k, v in c.items()} for c in comparisons],
        'all_consistent': bool(all_consistent)
    }

    import json
    out_file = Path("results") / "char_consistency_check.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {out_file}")

    return 0 if all_consistent else 1

if __name__ == "__main__":
    sys.exit(main())
