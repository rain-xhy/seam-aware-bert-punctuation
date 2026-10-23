# 实验运行清单 - Windows PowerShell版本
# 训练完成后按顺序执行，预计总耗时: 1-2小时

$CKPT = "checkpoints/T1_varlen/best.pt"
$DATA = "prepared_data_complete"
$PYTHON = "E:/anaconda/app/envs/yolov8/python.exe"

Write-Host "=== 作业8实验运行清单 ===" -ForegroundColor Green
Write-Host "检查点: $CKPT"
Write-Host ""

# E0: 基础验证
Write-Host "1. E0: 编码器权重检查" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.check_encoder_weights $CKPT

# 频率基线
Write-Host "`n2. 频率基线实验" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.run_baseline --data_dir $DATA `
    --out results/baseline_frequency.json

# E3: 窗口扫描 + EDEP
Write-Host "`n3. E3: 窗口/重叠/尾部扫描 + EDEP" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.run_sweep --ckpt $CKPT --split dev `
    --windows 128 256 400 510 --overlaps 0 50 100 200 `
    --tails short aligned --edep --out results/E3_dev.json

# E4: 方法对比
Write-Host "`n3. E4: 方法对比 (6种)" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.run_methods --ckpt $CKPT --split dev `
    --methods direct_cut serial_yu serial_tang serial_teacher serial_cross parallel `
    --out results/E4_dev.json --save_preds results/E4_dev_preds.npz

# E4: 计时实验
Write-Host "`n4. E4: 计时benchmark (交错协议)" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.run_e4 --ckpt $CKPT --split dev --limit 300 `
    --repeats 5 --out results/E4_timing.json

# E5: 合并方式对比
Write-Host "`n5. E5: 合并方式对比 (hard/gauss/EDEP)" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.run_e5_merge --ckpt $CKPT --split dev --limit 300 `
    --out results/E5_merge.json

# 固定核心实验
Write-Host "`n6. 固定核心隔离实验" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.run_fixed_core --ckpt $CKPT --split dev --limit 200 `
    --core 192 --contexts 0 14 31 63 --out results/fixed_core.json

# Batch探测
Write-Host "`n7. Batch大小探测" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.probe_batch --ckpt $CKPT --split dev --limit 100 `
    --out results/batch_probe.json

# 温度校准
Write-Host "`n8. 温度校准实验" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.run_calibration --ckpt $CKPT --cal_split dev --test_split test `
    --cal_limit 500 --out results/calibration.json

# 长文本测试
Write-Host "`n9. 长文本鲁棒性测试 (10万字)" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.test_long_text --ckpt $CKPT --length 100000

# 独立验证
Write-Host "`n10. 独立验证" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.recompute --result results/E4_dev.json `
    --preds results/E4_dev_preds.npz --data_dir $DATA

# Bootstrap检验
Write-Host "`n11. Bootstrap显著性检验" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.bootstrap_compare --result results/E4_dev.json `
    --preds results/E4_dev_preds.npz --method_a parallel `
    --method_b serial_teacher --n_bootstrap 10000

# 误差样例
Write-Host "`n12. 误差样例展示" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.show_errors --result results/E4_dev.json `
    --preds results/E4_dev_preds.npz --method parallel --limit 30

# 多窗口子集
Write-Host "`n13. 多窗口子集评测" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.eval_multiwindow_subset --result results/E4_dev.json `
    --preds results/E4_dev_preds.npz --window 400 --out results/E4_multiwindow.json

# 数据校验
Write-Host "`n14. 数据完整性验证" -ForegroundColor Cyan
& $PYTHON -X utf8 -m scripts.verify_data_hash $DATA --save results/data_hashes.json

Write-Host "`n=== 所有实验完成 ===" -ForegroundColor Green
Write-Host "结果保存在 results/ 目录"
Write-Host ""
Write-Host "下一步: 生成图表和撰写报告"
