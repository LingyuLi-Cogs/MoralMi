import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_sae_analysis_results(output_dir="./moral_surgery/outputs/sae_finetune_20251223_maxl0_100_hidden_65536_epoch_200"):
    output_path = Path(output_dir)
    summary_file = output_path / "summary.json"
    
    if not summary_file.exists():
        print(f"Error: {summary_file} 不存在，请先运行分析脚本。")
        return

    # 加载汇总数据
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    df = pd.DataFrame(summary['per_layer_summary'])
    mft_dimensions = [
        'care', 'harm', 'fairness', 'cheating', 'loyalty', 
        'betrayal', 'authority', 'subversion', 'sanctity', 'degradation'
    ]

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4)

    # --- 图 1: 跨层趋势分析 (单语义率 vs 典型性) ---
    ax1 = plt.subplot(2, 2, 1)
    ax1_twin = ax1.twinx()
    
    lns1 = ax1.plot(df['layer'], df['monosemantic_ratio'], 'b-o', label='Monosemantic Ratio', linewidth=2)
    lns2 = ax1_twin.plot(df['layer'], df['mean_prototypicality_spearman'], 'r-s', label='Moral Prototypicality', linewidth=2)
    
    ax1.set_xlabel('Model Layer')
    ax1.set_ylabel('Ratio of Monosemantic Features', color='b')
    ax1_twin.set_ylabel('Mean Spearman Correlation', color='r')
    ax1.set_title("Evolution of Moral Features across Layers", fontsize=14)
    
    # 合并图例
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left')

    # --- 图 2: 极性塌缩检测 (余弦相似度) ---
    ax2 = plt.subplot(2, 2, 2)
    sns.lineplot(data=df, x='layer', y='mean_pair_cosine', ax=ax2, marker='D', color='purple')
    ax2.axhline(0.5, ls='--', color='gray', alpha=0.5)
    ax2.set_title("Polarity Collapse (Lower is better separation)", fontsize=14)
    ax2.set_ylabel("Mean Cosine Similarity between Opposites")

    # --- 图 3: 特定层特征-维度相关性热力图 ---
    # 我们选择得分最高的层进行展示
    best_layer = summary['cross_layer_analysis']['best_layers'][0]['layer']
    corr_matrix_path = output_path / f"layer_{best_layer}" / "feature_dim_correlations.npy"
    
    if corr_matrix_path.exists():
        ax3 = plt.subplot(2, 1, 2)
        corr_data = np.load(corr_matrix_path)
        
        # 筛选出该层相关性最强的 Top 50 个特征进行可视化
        max_corrs = np.abs(corr_data).max(axis=1)
        top_indices = np.argsort(max_corrs)[-50:]
        subset_corr = corr_data[top_indices]
        
        sns.heatmap(subset_corr.T, cmap='RdBu_r', center=0, ax=ax3,
                    xticklabels=False, yticklabels=mft_dimensions)
        ax3.set_title(f"Top 50 Features Correlation Heatmap (Layer {best_layer})", fontsize=14)
        ax3.set_xlabel("SAE Feature Index (Top Sorted)")
        ax3.set_ylabel("MFT Dimensions")

    plt.tight_layout()
    plt.savefig(output_path / "sae_moral_analysis_viz.png", dpi=300)
    print(f"可视化完成！结果已保存至: {output_path / 'sae_moral_analysis_viz.png'}")
    plt.show()

if __name__ == "__main__":
    plot_sae_analysis_results()