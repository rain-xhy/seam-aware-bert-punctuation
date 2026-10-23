# data_prep.py 正式产出 对照 split_probe.log 探针数字

核对时间：阶段 4，`punct.data_prep` 首次在真实语料（6 个文件）上跑完之后。

| | 探针 split_probe.log（丢全部标点，无小数点） | 正式 data_prep（保留数字间小数点） | 差值 | 差值来源 |
|---|---|---|---|---|
| train 文章数 | 12128 | 12128 | 0 | 一致 |
| train 字数 | 7125456 | 7133268 | +7812 | 恰好等于 data_audit.json 里 train.decimal_points_kept=7812 |
| val 文章数 | 3223 | 3223 | 0 | 一致 |
| val 字数 | 1846905 | 1848738 | +1833 | 恰好等于 val.decimal_points_kept=1833 |
| test 文章数 | 3280 | 3280 | 0 | 一致 |
| test 字数 | 1863769 | 1865549 | +1780 | 恰好等于 test.decimal_points_kept=1780 |

结论：三个划分的文章数与探针完全一致，字数的全部差额由"数字间小数点保留为输入字符"这一条清洗规则解释（`代码/完善方案.md` §2 已说明探针会有此出入），不是实现错误。

另：`check_no_cross_split_leakage` 在正式产出上返回 0 个跨集合重复的文章键，三个月份划分之间没有文章泄漏。

UNK 回退：`vocab_unk_chars_seen = 1798`（三个划分合计约 1085 万字符的 0.0165%），量级与此前 `design_probe.log` 记录的 NFKC 多字符展开（120 个，如 ℃）及其他未登录字一致。
