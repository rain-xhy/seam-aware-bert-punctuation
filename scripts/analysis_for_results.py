"""Derived numbers for section 5 of the paper, computed from files already on disk.

Nothing here runs the model. Every block reads results/*.json, results/*.npz and the
prepared data, and writes results/analysis_for_results.json.

    python -m scripts.analysis_for_results

Blocks
  seam_exposure   share of the fixed seam positions (400k +- 8) that lie within 8 characters
                  of an internal window edge, per decoding configuration (validation set)
  edep_reading    how the mask-width reading m_l / m_r moves with smoothing, tolerance and rule
  e7_changes      for the second-pass configurations, what happened to the positions whose label
                  differs from the first pass (wrong->right, right->wrong, wrong->wrong)
  test_groups     test-set F1 of every decoding method on articles longer than the window
  confusions      largest off-diagonal cells of the parallel decoder's confusion matrix
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from punct.metrics import punctuation_prf  # noqa: E402
from punct.runner import load_split  # noqa: E402
from punct.window_plan import plan_windows  # noqa: E402

DATA = "prepared_data_complete"
NAMES = ["无标点", "逗号", "句号", "问号", "叹号", "分号", "冒号", "顿号"]


def seam_mask(n, stride=400, margin=8):
    m = np.zeros(n, dtype=bool)
    for b in range(stride, n, stride):
        m[max(0, b - margin):min(n, b + margin)] = True
    return m


def seam_exposure():
    _, gold = load_split(DATA, "dev")
    lengths = [len(g) for g in gold]
    out = []
    for W in (128, 256, 400, 510):
        for V in (0, 50, 100, 200):
            if V >= W or (W == 128 and V == 200) or (W == 256 and V == 0 and False):
                continue
            n_seam = n_any = n_all = 0
            for n in lengths:
                sm = seam_mask(n)
                if not sm.any():
                    continue
                cover = np.zeros(n, dtype=np.int32)
                edge = np.zeros(n, dtype=np.int32)
                for w in plan_windows(n, W, V, "short"):
                    zone = np.zeros(w.end - w.start, dtype=bool)
                    if w.start > 0:
                        zone[:8] = True
                    if w.end < n:
                        zone[-8:] = True
                    cover[w.start:w.end] += 1
                    edge[w.start:w.end] += zone
                n_seam += int(sm.sum())
                n_any += int(((edge > 0) & sm).sum())
                n_all += int(((edge == cover) & (edge > 0) & sm).sum())
            out.append({"window": W, "overlap": V, "seam_positions": n_seam,
                        "any_edge": n_any / n_seam, "all_edge": n_all / n_seam})
    return out


def first_within(curve, thr, smooth):
    c = np.asarray(curve, dtype=float)
    if smooth > 1:
        k = smooth // 2
        c = np.array([c[max(0, i - k):i + k + 1].mean() for i in range(len(c))])
    return c, thr


def edep_reading():
    ed = json.load(open("results/E3_dev.json", encoding="utf-8"))["edep"]
    base = ed["centre_error"]
    rows = []
    for smooth in (1, 5, 9):
        for tol in (0.002, 0.005, 0.01):
            row = {"smooth": smooth, "tol": tol}
            for side in ("left", "right"):
                c, _ = first_within(ed[side], tol, smooth)
                ok = c <= base + tol
                first = int(np.argmax(ok)) if ok.any() else len(c) - 1
                bad = np.where(~ok)[0]
                last = int(bad[-1] + 1) if bad.size else 0
                row[side] = {"first": first, "last_exceed": last}
            rows.append(row)
    return {"centre_error": base, "left0": ed["left"][0], "right0": ed["right"][0],
            "left_1_5": ed["left"][1:6], "right_1_5": ed["right"][1:6],
            "window": ed["window"], "overlap": ed["overlap"], "rows": rows}


def e7_changes():
    _, gold = load_split(DATA, "dev")
    g = np.concatenate(gold[500:])
    z = np.load("results/E7_dev_full_preds.npz")
    p1 = z["pass1"]
    out = {"positions": int(g.size), "pass1_errors": int((p1 != g).sum())}
    for name in ("c_calibrated", "c_uncalibrated", "c_random", "b_same_ter"):
        q = z[name]
        d = p1 != q
        out[name] = {"changed": int(d.sum()),
                     "wrong_to_right": int((d & (p1 != g) & (q == g)).sum()),
                     "right_to_wrong": int((d & (p1 == g) & (q != g)).sum()),
                     "wrong_to_wrong": int((d & (p1 != g) & (q != g)).sum()),
                     "errors_after": int((q != g).sum())}
    return out


def test_groups():
    chars, gold = load_split(DATA, "test")
    lengths = np.array([len(x) for x in gold])
    z = np.load("results/E4_test_preds.npz")
    offs = np.concatenate([[0], np.cumsum(lengths)])
    g = np.concatenate(gold)
    multi = np.concatenate([np.full(n, n > 400) for n in lengths])
    out = {"n_multi_window_articles": int((lengths > 400).sum()), "n_single": int((lengths <= 400).sum()),
           "multi_positions": int(multi.sum()), "methods": {}}
    for name in z.files:
        p = z[name]
        out["methods"][name] = {
            "multi_f1": punctuation_prf(g[multi], p[multi], 8, 0)["micro_f1"],
            "single_f1": punctuation_prf(g[~multi], p[~multi], 8, 0)["micro_f1"]}
    return out


def confusions():
    cm = np.array(json.load(open("results/E4_test.json", encoding="utf-8"))["methods"]["parallel"]["quality"]["confusion"])
    cells = [(int(cm[i, j]), NAMES[i], NAMES[j]) for i in range(8) for j in range(8) if i != j]
    cells.sort(reverse=True)
    return {"total_off_diagonal": int(cm.sum() - np.trace(cm)),
            "top": [{"gold": a, "pred": b, "count": n} for n, a, b in cells[:8]]}


def main():
    res = {"seam_exposure": seam_exposure(), "edep_reading": edep_reading(), "e7_changes": e7_changes(),
           "test_groups": test_groups(), "confusions": confusions()}
    with open("results/analysis_for_results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
