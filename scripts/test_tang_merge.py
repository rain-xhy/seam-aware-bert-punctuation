"""验证Tang方法的merge_tang_drop_last_sentence实现正确性。

测试用例：
1. 单窗口：保留全部
2. 多窗口：每个窗口(除最后)截到最后一个句末标点
3. 边界case：窗口中没有句末标点

    python -m scripts.test_tang_merge
"""
from __future__ import annotations

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.merge import WindowProbs, merge_tang_drop_last_sentence  # noqa: E402
from punct.window_plan import Window  # noqa: E402

SENT_FINAL = (2, 3, 4)  # 。？！


def test_single_window():
    """单窗口：保留全部"""
    w = Window(0, 5)  # 窗口长度要和标签数量一致
    labels = np.array([0, 1, 2, 0, 1])  # 最后一个2是句末
    probs = np.eye(8)[labels]

    wp = [WindowProbs(w, probs)]
    result = merge_tang_drop_last_sentence(wp, 5, 8, [labels], SENT_FINAL)

    recovered = result.argmax(-1)
    assert np.array_equal(recovered, labels), f"期望{labels}，得到{recovered}"
    print("✓ 单窗口测试通过")


def test_multi_window_with_sentence_final():
    """多窗口，每个都有句末标点"""
    # 窗口1: [0,5), 标签[0,1,2,0,1]，最后的2(idx=2)是句末，保留[0,1,2]
    # 窗口2: [3,8), 标签[0,1,3,0,1]，最后的3(idx=2)是句末，保留[0,1,3]
    # 窗口3: [6,10), 标签[0,1,0,2]，是最后窗口，全保留

    w1 = Window(0, 5)
    labels1 = np.array([0, 1, 2, 0, 1])
    probs1 = np.eye(8)[labels1]

    w2 = Window(3, 8)
    labels2 = np.array([0, 1, 3, 0, 1])
    probs2 = np.eye(8)[labels2]

    w3 = Window(6, 10)
    labels3 = np.array([0, 1, 0, 2])
    probs3 = np.eye(8)[labels3]

    windows = [WindowProbs(w1, probs1), WindowProbs(w2, probs2), WindowProbs(w3, probs3)]
    result = merge_tang_drop_last_sentence(windows, 10, 8, [labels1, labels2, labels3], SENT_FINAL)

    recovered = result.argmax(-1)
    # 期望: w1保留[0,1,2], w2保留[0,1,3]覆盖[3,6), w3全保留覆盖[6,10)
    # 位置: 0,1,2,3,4,5,6,7,8,9
    # 来源: w1,w1,w1,w2,w2,w2,w3,w3,w3,w3
    expected = np.array([0, 1, 2, 0, 1, 3, 0, 1, 0, 2])

    assert np.array_equal(recovered, expected), f"期望{expected}，得到{recovered}"
    print("✓ 多窗口句末截断测试通过")


def test_no_sentence_final():
    """窗口中没有句末标点：保留全部"""
    w1 = Window(0, 3)
    labels1 = np.array([0, 1, 0])  # 无句末
    probs1 = np.eye(8)[labels1]

    w2 = Window(2, 5)
    labels2 = np.array([0, 1, 5])  # 最后窗口
    probs2 = np.eye(8)[labels2]

    windows = [WindowProbs(w1, probs1), WindowProbs(w2, probs2)]
    result = merge_tang_drop_last_sentence(windows, 5, 8, [labels1, labels2], SENT_FINAL)

    recovered = result.argmax(-1)
    # w1无句末，保留全部[0,1,0]; w2是最后，全保留[0,1,5]
    # 位置0,1,2,3,4被w1,w1,w1/w2,w2,w2覆盖
    expected = np.array([0, 1, 0, 1, 5])

    assert np.array_equal(recovered, expected), f"期望{expected}，得到{recovered}"
    print("✓ 无句末标点测试通过")


def main():
    print("测试Tang方法merge_tang_drop_last_sentence...\n")

    try:
        test_single_window()
        test_multi_window_with_sentence_final()
        test_no_sentence_final()

        print("\n✅ 所有测试通过！Tang方法实现正确。")
        return 0

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
