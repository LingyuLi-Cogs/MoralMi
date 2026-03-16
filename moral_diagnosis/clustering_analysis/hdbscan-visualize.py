import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# ==================== 1. 配置路径与模型列表 ====================

BASE_PATH = Path('./moral_representation/clustering/hdbscan_results_samples20_size100')
OUTPUT_PATH = BASE_PATH / 'manual_group_comparison'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --- 人工模型分类 (Lists provided by user) ---

# 注意：这些名字必须与 cluster_summary.pt 中保存的 model_name 一致
GROUP_DEFINITIONS = {
    'Base': [
        "Llama-3.1-8B", "Llama-4-Scout-17B-16E", 
        "Qwen3-0.6B-Base", "Qwen3-4B-Base", "Qwen3-8B-Base", 
        "Qwen3-14B-Base", "Qwen3-235B-A22B"
    ],
    'Instruct': [
        "gpt-oss-20b", "gpt-oss-120b", "Llama-3.1-8B-Instruct", "Llama-4-Scout-17B-16E-Instruct",
        "Qwen3-0.6B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", 
        "Qwen3-32B", "Qwen3-235B-A22B-Instruct-2507"
    ],
    'Guard': [
        "gpt-oss-safeguard-20b", "gpt-oss-safeguard-120b", 
        "Llama-Guard-3-8B", 
        "Guard-Gen-0.6B", "Guard-Gen-4B", "Guard-Gen-8B"
    ]
}

# --- 配色方案 (按照模型家族) ---

MODEL_FAMILIES = [
    ["gpt-oss-20b", "gpt-oss-safeguard-20b"],
    ["gpt-oss-120b", "gpt-oss-safeguard-120b"],
    ["Llama-3.1-8B", "Llama-3.1-8B-Instruct", "Llama-Guard-3-8B"],
    ["Llama-4-Scout-17B-16E", "Llama-4-Scout-17B-16E-Instruct"],
    ["Qwen3-0.6B-Base", "Qwen3-0.6B", "Guard-Gen-0.6B", "Qwen3Guard-Gen-0.6B"], # 包含可能的别名
    ["Qwen3-4B-Base", "Qwen3-4B", "Guard-Gen-4B", "Qwen3Guard-Gen-4B"],
    ["Qwen3-8B-Base", "Qwen3-8B", "Guard-Gen-8B", "Qwen3Guard-Gen-8B"],
    ["Qwen3-14B-Base", "Qwen3-14B"],
    ["Qwen3-32B"],
    ["Qwen3-235B-A22B", "Qwen3-235B-A22B-Instruct-2507"]
]

def get_model_color_map():
    """
    生成配色字典：Modern Academic (低饱和度、高辨识度)
    """
    academic_palette = [
        "#ECAD9E",  # Deep Blue-ish/Pink
        "#F4606C",  # Terracotta
        "#E6CEAC",  # Muted Green/Beige
        "#D1BA74",  # Muted Red/Gold
        "#D6D587",  # Deep Purple/Lime mix
        "#BEEDC7",  # Pale Green
        "#DA8BC3",  # Orchid
        "#8C8C8C",  # Gray
        "#CCB974",  # Olive
        "#64B5CD"   # Teal
    ]
    
    color_map = {}
    for family_idx, family_models in enumerate(MODEL_FAMILIES):
        # 循环使用调色板，防止家族数量超过颜色数量
        color = academic_palette[family_idx % len(academic_palette)]
        for model in family_models:
            color_map[model] = color
            
    return color_map

# 全局颜色映射
COLOR_MAP = get_model_color_map()

# ==================== 2. 数据加载与处理 ====================

# ==================== 2. 数据加载与处理 ====================

def load_all_summaries(base_path):
    """递归查找并加载所有 cluster_summary.pt"""
    summaries = {}
    files = list(base_path.rglob('cluster_summary.pt'))
    
    print(f"找到 {len(files)} 个汇总文件，开始加载...")
    
    for f in files:
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            model_name = data.get('model_name', 'Unknown')
            summaries[model_name] = data
            # print(f"  已加载: {model_name}")
        except Exception as e:
            print(f"  [Error] 加载失败 {f}: {e}")
    print(f"成功加载 {len(summaries)} 个模型数据。")      
    return summaries

def normalize_layers(layers, values):
    """将层数归一化到 0-1"""
    if not layers:
        return [], []
    max_layer = max(layers)
    if max_layer == 0:
        return [0], values
    return [l / max_layer for l in layers], values

# ==================== 3. 绘图逻辑 ====================

def plot_metric_comparison(all_summaries, metric_key, output_filename, y_label, title_suffix, y_limit=None):
    """通用绘图函数"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    group_order = ['Base', 'Instruct', 'Guard']
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    plotted_models = set()
    
    for ax, group_name in zip(axes, group_order):
        target_models = GROUP_DEFINITIONS.get(group_name, [])
        
        has_data = False
        for model_name in target_models:
            if model_name not in all_summaries:
                continue
                
            summary = all_summaries[model_name]
            
            # 提取数据并处理不同的 metric_key
            layers = summary['layers']
            try:
                if metric_key == 'n_clusters':
                    values = summary['by_layer']['n_clusters']
                elif metric_key == 'noise_ratio':
                    values = summary['by_layer']['noise_ratio']
                elif metric_key == 'ari_vs_mft5':
                    values = summary['by_layer']['ari_vs_mft5']
                # === 新增部分 ===
                elif metric_key == 'ari_vs_polarity3':
                    values = summary['by_layer']['ari_vs_polarity3']
                elif metric_key == 'ari_vs_mft15':
                    values = summary['by_layer']['ari_vs_mft15']
                else:
                    print(f"未知的指标 key: {metric_key}")
                    continue
            except KeyError as e:
                print(f"  [Warning] 模型 {model_name} 缺少指标数据 {e}")
                continue
                
            norm_layers, norm_values = normalize_layers(layers, values)
            color = COLOR_MAP.get(model_name, '#333333')
            
            ax.plot(norm_layers, norm_values, 
                   marker='o', 
                   markersize=4, 
                   linewidth=1.5,
                   color=color,
                   alpha=0.8,
                   label=model_name)
            
            has_data = True
            plotted_models.add(model_name)
            
        ax.set_title(f"{group_name} Models", fontsize=14, pad=10)
        ax.set_xlabel('Relative Depth (0=Input, 1=Output)', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        ax.set_xlim(-0.05, 1.05)
        if y_limit:
            ax.set_ylim(y_limit)
        elif metric_key == 'noise_ratio':
            ax.set_ylim(0, 1.05)
        elif 'ari' in metric_key:
            ax.set_ylim(0, 1.0)
            
        #if has_data:
        #    ax.legend(fontsize=10, loc='best', frameon=True, framealpha=0.9)
    
    plt.suptitle(f'Cross-Model Comparison: {title_suffix}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    save_path = OUTPUT_PATH / output_filename
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    print(f"图表已保存: {save_path}")
    plt.close()

# ==================== 4. 主程序 ====================

def main():
    print("="*60)
    print("生成跨模型对比图 (含 Polarity3 & MFT15)")
    print("="*60)
    
    summaries = load_all_summaries(BASE_PATH)
    
    if not summaries:
        print("未找到任何数据，请检查路径。")
        return

    # 检查缺失模型
    loaded_keys = set(summaries.keys())
    config_keys = set([m for group in GROUP_DEFINITIONS.values() for m in group])
    missing = loaded_keys - config_keys
    if missing:
        print(f"\n[警告] 以下加载的模型将被忽略(不在配置列表中):\n{missing}")

    # --- 1. 基础指标 ---
    
    plot_metric_comparison(
        summaries, 'n_clusters', 'comparison_n_clusters.png', 
        'Number of Clusters', 'Cluster Count by Layer'
    )
    
    plot_metric_comparison(
        summaries, 'noise_ratio', 'comparison_noise_ratio.png', 
        'Noise Ratio', 'Cluster Noise Ratio (HDBSCAN)',
        y_limit=(0, 1.05)
    )

    # --- 2. 道德内容指标 (MFT) ---
    
    print("绘制 MFT5 (5个道德维度)...")
    plot_metric_comparison(
        summaries, 'ari_vs_mft5', 'comparison_ari_mft5.png', 
        'ARI Score (vs MFT5)', 'Alignment with 5 Moral Foundations',
        y_limit=(0, 0.2)
    )

    # --- 3. 情感极性指标 (Polarity) ---
    
    print("绘制 Polarity3 (善/恶/中性)...")
    plot_metric_comparison(
        summaries, 'ari_vs_polarity3', 'comparison_ari_polarity3.png', 
        'ARI Score (vs Polarity3)', 'Alignment with Virtue/Vice/Neutral',
        y_limit=(0, 0.6)
    )

    # --- 4. 组合指标 (MFT15) ---
    
    print("绘制 MFT15 (5维度 x 3极性)...")
    plot_metric_comparison(
        summaries, 'ari_vs_mft15', 'comparison_ari_mft15.png', 
        'ARI Score (vs MFT15)', 'Alignment with Detailed Moral Concepts (MFT15)',
        y_limit=(0, 0.4)
    )

    print("\n" + "="*60)
    print(f"全部完成! 结果保存在: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()