"""计算并验证数据文件的SHA-256哈希值，确保数据完整性。

    python -m scripts.verify_data_hash prepared_data_complete
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def compute_sha256(filepath: str) -> str:
    """计算文件的SHA-256哈希值"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="prepared data directory")
    ap.add_argument("--save", help="save hashes to this JSON file")
    ap.add_argument("--verify", help="verify against hashes in this JSON file")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return 1

    # 需要校验的文件
    files_to_check = [
        "char_codec.json",
        "train_articles_char.npy",
        "train_articles_label.npy",
        "dev_articles_char.npy",
        "dev_articles_label.npy",
        "test_articles_char.npy",
        "test_articles_label.npy",
    ]

    hashes = {}
    print(f"计算 {data_dir} 中文件的SHA-256哈希值...\n")

    for filename in files_to_check:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ⚠ 跳过(不存在): {filename}")
            continue

        hash_value = compute_sha256(str(filepath))
        size_mb = filepath.stat().st_size / 1024 / 1024
        hashes[filename] = hash_value
        print(f"  {filename:<30} {hash_value[:16]}... ({size_mb:.1f} MB)")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 哈希值已保存到: {args.save}")

    if args.verify:
        with open(args.verify, encoding="utf-8") as f:
            expected = json.load(f)

        print(f"\n验证哈希值(对照 {args.verify}):")
        all_match = True
        for filename, hash_value in hashes.items():
            if filename not in expected:
                print(f"  ⚠ {filename}: 参考文件中不存在")
                continue
            if hash_value == expected[filename]:
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename}: 不匹配!")
                print(f"    期望: {expected[filename]}")
                print(f"    实际: {hash_value}")
                all_match = False

        if all_match:
            print("\n✓ 所有文件哈希验证通过")
            return 0
        else:
            print("\n✗ 部分文件哈希不匹配")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
