"""汇总所有训练模型的结果"""
import json
import os
from pathlib import Path

models = [
    ("T1_varlen", "变长随机裁剪（主模型）"),
    ("T2_center_only", "仅中心区监督对照"),
    ("T3_noedge", "无边缘感知对照"),
    ("T4_fixed256", "固定长度256对照"),
    ("T5_no_class_weight", "无类别权重对照"),
]

print("="*80)
print("训练结果汇总表")
print("="*80)
print()

results = []
for model_name, desc in models:
    history_path = f"checkpoints/{model_name}/history.json"
    if not os.path.exists(history_path):
        print(f"❌ {model_name}: history.json 不存在")
        continue
    
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if not history.get('epochs'):
        print(f"⚠️  {model_name}: 未完成任何 epoch")
        continue
    
    epochs = history['epochs']
    best_f1 = max(e['micro_f1'] for e in epochs)
    best_epoch = [e for e in epochs if e['micro_f1'] == best_f1][0]
    
    results.append({
        'name': model_name,
        'desc': desc,
        'n_epochs': len(epochs),
        'best_f1': best_f1,
        'best_epoch': best_epoch['epoch'],
        'best_val_loss': best_epoch['val_loss'],
        'best_train_loss': best_epoch['train_loss'],
        'macro_f1': best_epoch['macro_f1'],
        'time': best_epoch['seconds']
    })

# 按 F1 排序
results.sort(key=lambda x: x['best_f1'], reverse=True)

# 打印表格
print(f"{'模型':<20} {'描述':<30} {'Epochs':<8} {'Best Epoch':<12} {'Micro-F1':<10} {'Macro-F1':<10} {'Val Loss':<10}")
print("-"*120)

medals = ['🥇', '🥈', '🥉', '  ', '  ']
for i, r in enumerate(results):
    medal = medals[min(i, 4)]
    print(f"{medal} {r['name']:<18} {r['desc']:<30} {r['n_epochs']:<8} {r['best_epoch']:<12} {r['best_f1']:<10.4f} {r['macro_f1']:<10.4f} {r['best_val_loss']:<10.4f}")

print()
print("="*80)
print(f"✅ 共完成 {len(results)} 个模型的训练")
print(f"🥇 最佳模型: {results[0]['name']} (F1={results[0]['best_f1']:.4f})")
print("="*80)

# 保存为 JSON
summary = {
    'timestamp': str(Path('checkpoints/T1_varlen/best.pt').stat().st_mtime if os.path.exists('checkpoints/T1_varlen/best.pt') else ''),
    'models': results
}

with open('results/training_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n结果已保存到: results/training_summary.json")
