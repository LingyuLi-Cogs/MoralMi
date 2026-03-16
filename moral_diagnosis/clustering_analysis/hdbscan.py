import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import json

# 科学计算
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

# 聚类
import hdbscan

# 降维可视化
import umap

# 绘图
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

warnings.filterwarnings('ignore')

# ==================== 配置 ====================

BASE_PATH = Path('./moral_representation/extracted_activations')
OUTPUT_PATH = Path('./moral_representation/clustering/hdbscan_results_samples20_size100')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    ('activations_gpt-oss-safeguard-20b', 'gpt-oss-safeguard-20b'),
]

# MFT五维度
MFT5_CATEGORIES = [
    'care-harm',
    'fairness-cheating', 
    'loyalty-betrayal',
    'authority-subversion',
    'sanctity-degradation'
]

# 颜色配置
MFT5_COLORS = {
    'care-harm': '#E41A1C',           # 红
    'fairness-cheating': '#377EB8',    # 蓝
    'loyalty-betrayal': '#4DAF4A',     # 绿
    'authority-subversion': '#984EA3', # 紫
    'sanctity-degradation': '#FF7F00'  # 橙
}

POLARITY3_COLORS = {
    'virtue': '#2ECC71',   # 绿
    'vice': '#E74C3C',     # 红
    'neutral': '#95A5A6'   # 灰
}

# 透明度配置
POLARITY_ALPHA = {
    'virtue': 1.0,
    'vice': 0.6,
    'neutral': 0.3
}

# HDBSCAN参数
HDBSCAN_MIN_CLUSTER_SIZE = 100
HDBSCAN_MIN_SAMPLES = 20

# PCA参数
PCA_VARIANCE_THRESHOLD = 0.95


# ==================== 辅助函数 ====================

def normalize_polarity(sample_type):
    """归一化sample_type到三类"""
    if sample_type in ['virtue', 'virtue_typical']:
        return 'virtue'
    elif sample_type in ['vice', 'vice_typical']:
        return 'vice'
    else:
        return 'neutral'


def load_activations(model_folder, model_name):
    """加载激活数据"""
    file_path = BASE_PATH / model_folder / 'activations_merged.pt'
    
    if not file_path.exists():
        print(f"  [警告] 文件不存在: {file_path}")
        return None
    
    print(f"  加载: {file_path}")
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    print(f"  样本数: {len(data)}")
    
    return data


def extract_metadata(data):
    """提取元数据和标签"""
    metadata = []
    
    for sample in data:
        polarity3 = normalize_polarity(sample['sample_type'])
        mft5 = sample['sampled_dimension']
        mft15 = f"{mft5}_{polarity3}"
        
        metadata.append({
            'id': sample['id'],
            'mft5': mft5,
            'polarity3': polarity3,
            'mft15': mft15,
            'original_sample_type': sample['sample_type']
        })
    
    return pd.DataFrame(metadata)


def get_layer_representations(data, layer):
    """提取指定层的mean_pooling表征"""
    representations = []
    
    for sample in data:
        if layer in sample['mean_pooling']:
            rep = sample['mean_pooling'][layer]
            if isinstance(rep, torch.Tensor):
                rep = rep.numpy()
            representations.append(rep.flatten())
    
    return np.array(representations)


def compute_clustering_metrics(cluster_labels, true_labels):
    """计算聚类评估指标"""
    # 过滤掉噪声样本（标签为-1）
    mask = cluster_labels != -1
    
    if mask.sum() < 10:  # 太少有效样本
        return {
            'ari': np.nan,
            'nmi': np.nan,
            'homogeneity': np.nan,
            'completeness': np.nan,
            'v_measure': np.nan
        }
    
    cl = cluster_labels[mask]
    tl = np.array(true_labels)[mask]
    
    return {
        'ari': adjusted_rand_score(tl, cl),
        'nmi': normalized_mutual_info_score(tl, cl),
        'homogeneity': homogeneity_score(tl, cl),
        'completeness': completeness_score(tl, cl),
        'v_measure': v_measure_score(tl, cl)
    }


def analyze_cluster_composition(cluster_labels, metadata):
    """分析每个簇的组成"""
    composition = {}
    
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_meta = metadata[mask]
        
        composition[int(cluster_id)] = {
            'size': int(mask.sum()),
            'by_mft5': cluster_meta['mft5'].value_counts().to_dict(),
            'by_polarity3': cluster_meta['polarity3'].value_counts().to_dict(),
            'by_mft15': cluster_meta['mft15'].value_counts().to_dict()
        }
    
    return composition


def create_umap_visualization(X_umap, metadata, cluster_labels, layer, model_name, output_dir):
    """生成四种UMAP可视化"""
    
    fig_size = (10, 8)
    point_size = 5
    
    # 图1: MFT15着色（MFT颜色 + polarity透明度）
    fig, ax = plt.subplots(figsize=fig_size)
    
    for mft5 in MFT5_CATEGORIES:
        for polarity in ['virtue', 'vice', 'neutral']:
            mask = (metadata['mft5'] == mft5) & (metadata['polarity3'] == polarity)
            if mask.sum() > 0:
                ax.scatter(
                    X_umap[mask, 0], X_umap[mask, 1],
                    c=MFT5_COLORS[mft5],
                    alpha=POLARITY_ALPHA[polarity],
                    s=point_size,
                    label=f'{mft5}_{polarity}'
                )
    
    ax.set_title(f'{model_name} - Layer {layer} - MFT15')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    # 图例太多，放在外面或不显示
    plt.tight_layout()
    plt.savefig(output_dir / f'umap_layer{layer}_mft15.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图2: MFT5着色
    fig, ax = plt.subplots(figsize=fig_size)
    
    for mft5 in MFT5_CATEGORIES:
        mask = metadata['mft5'] == mft5
        if mask.sum() > 0:
            ax.scatter(
                X_umap[mask, 0], X_umap[mask, 1],
                c=MFT5_COLORS[mft5],
                alpha=0.6,
                s=point_size,
                label=mft5
            )
    
    ax.set_title(f'{model_name} - Layer {layer} - MFT5 Dimensions')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best', markerscale=3)
    plt.tight_layout()
    plt.savefig(output_dir / f'umap_layer{layer}_mft5.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图3: Polarity3着色
    fig, ax = plt.subplots(figsize=fig_size)
    
    for polarity in ['virtue', 'vice', 'neutral']:
        mask = metadata['polarity3'] == polarity
        if mask.sum() > 0:
            ax.scatter(
                X_umap[mask, 0], X_umap[mask, 1],
                c=POLARITY3_COLORS[polarity],
                alpha=0.6,
                s=point_size,
                label=polarity
            )
    
    ax.set_title(f'{model_name} - Layer {layer} - Polarity')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best', markerscale=3)
    plt.tight_layout()
    plt.savefig(output_dir / f'umap_layer{layer}_polarity3.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图4: 聚类结果着色
    fig, ax = plt.subplots(figsize=fig_size)
    
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len([c for c in unique_clusters if c != -1])
    
    # 为聚类生成颜色
    if n_clusters > 0:
        cluster_cmap = plt.cm.get_cmap('tab20', max(n_clusters, 1))
    
    color_idx = 0
    for cluster_id in sorted(unique_clusters):
        mask = cluster_labels == cluster_id
        if cluster_id == -1:
            # 噪声用灰色
            ax.scatter(
                X_umap[mask, 0], X_umap[mask, 1],
                c='lightgray',
                alpha=0.3,
                s=point_size,
                label=f'Noise ({mask.sum()})'
            )
        else:
            ax.scatter(
                X_umap[mask, 0], X_umap[mask, 1],
                c=[cluster_cmap(color_idx)],
                alpha=0.6,
                s=point_size,
                label=f'Cluster {cluster_id} ({mask.sum()})'
            )
            color_idx += 1
    
    ax.set_title(f'{model_name} - Layer {layer} - Clusters (n={n_clusters})')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    if len(unique_clusters) <= 15:
        ax.legend(loc='best', markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f'umap_layer{layer}_clusters.png', dpi=150, bbox_inches='tight')
    plt.close()


def process_single_layer(data, metadata, layer, model_name, output_dir):
    """处理单层的完整流程"""
    
    print(f"    Layer {layer}...")
    
    # Step 1: 提取表征
    X_raw = get_layer_representations(data, layer)
    n_samples, hidden_size = X_raw.shape
    
    # Step 2: 中心化
    global_mean = X_raw.mean(axis=0)
    X_centered = X_raw - global_mean
    
    # Step 3: PCA降维
    pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver='full')
    X_pca = pca.fit_transform(X_centered)
    n_pca_components = X_pca.shape[1]
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.sum()
    
    # Step 5: HDBSCAN聚类
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean',
        core_dist_n_jobs=-1
    )
    cluster_labels = clusterer.fit_predict(X_pca)
    probabilities = clusterer.probabilities_
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    noise_ratio = n_noise / n_samples
    
    # Step 6: UMAP降维
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
        n_jobs=-1
    )
    X_umap = reducer.fit_transform(X_pca)
    
    # Step 7: 评估指标
    metrics = {
        'vs_mft5': compute_clustering_metrics(cluster_labels, metadata['mft5'].values),
        'vs_polarity3': compute_clustering_metrics(cluster_labels, metadata['polarity3'].values),
        'vs_mft15': compute_clustering_metrics(cluster_labels, metadata['mft15'].values)
    }
    
    # 簇组成分析
    cluster_composition = analyze_cluster_composition(cluster_labels, metadata)
    
    # 生成可视化
    create_umap_visualization(X_umap, metadata, cluster_labels, layer, model_name, output_dir)
    
    # 构建样本级结果
    sample_results = []
    for i in range(n_samples):
        sample_results.append({
            'id': metadata.iloc[i]['id'],
            'mft5': metadata.iloc[i]['mft5'],
            'polarity3': metadata.iloc[i]['polarity3'],
            'mft15': metadata.iloc[i]['mft15'],
            'cluster_label': int(cluster_labels[i]),
            'cluster_probability': float(probabilities[i]),
            'umap_x': float(X_umap[i, 0]),
            'umap_y': float(X_umap[i, 1])
        })
    
    # 构建层级结果
    layer_result = {
        'model_name': model_name,
        'layer': layer,
        'n_samples': n_samples,
        'hidden_size': hidden_size,
        'global_mean': global_mean,
        'n_pca_components': n_pca_components,
        'explained_variance_ratio': explained_variance,
        'cumulative_variance': float(cumulative_variance),
        'hdbscan_params': {
            'min_cluster_size': HDBSCAN_MIN_CLUSTER_SIZE,
            'min_samples': HDBSCAN_MIN_SAMPLES,
            'metric': 'euclidean'
        },
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': float(noise_ratio),
        'metrics': metrics,
        'cluster_composition': cluster_composition,
        'sample_results': sample_results
    }
    
    # 保存层级结果
    torch.save(layer_result, output_dir / f'cluster_results_layer_{layer}.pt')
    
    return {
        'layer': layer,
        'n_clusters': n_clusters,
        'noise_ratio': noise_ratio,
        'n_pca_components': n_pca_components,
        'metrics': metrics
    }


def process_model(model_folder, model_name):
    """处理单个模型的所有层"""
    
    print(f"\n{'='*60}")
    print(f"处理模型: {model_name}")
    print(f"{'='*60}")
    
    # 加载数据
    data = load_activations(model_folder, model_name)
    if data is None:
        return None
    
    # 提取元数据
    metadata = extract_metadata(data)
    print(f"  标签分布:")
    print(f"    MFT5: {metadata['mft5'].value_counts().to_dict()}")
    print(f"    Polarity3: {metadata['polarity3'].value_counts().to_dict()}")
    
    # 确定层数
    sample = data[0]
    layers = sorted(sample['mean_pooling'].keys())
    print(f"  层数: {len(layers)} (0 to {max(layers)})")
    
    # 创建输出目录
    output_dir = OUTPUT_PATH / model_folder.replace('activations_', '')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每一层
    layer_summaries = []
    for layer in layers:
        summary = process_single_layer(data, metadata, layer, model_name, output_dir)
        layer_summaries.append(summary)
        
        # 打印进度
        print(f"      n_clusters={summary['n_clusters']}, "
              f"noise={summary['noise_ratio']:.2%}, "
              f"ARI(MFT5)={summary['metrics']['vs_mft5']['ari']:.3f}")
    
    # 构建跨层汇总
    summary = {
        'model_name': model_name,
        'layers': layers,
        'by_layer': {
            'n_clusters': [s['n_clusters'] for s in layer_summaries],
            'noise_ratio': [s['noise_ratio'] for s in layer_summaries],
            'n_pca_components': [s['n_pca_components'] for s in layer_summaries],
            'ari_vs_mft5': [s['metrics']['vs_mft5']['ari'] for s in layer_summaries],
            'nmi_vs_mft5': [s['metrics']['vs_mft5']['nmi'] for s in layer_summaries],
            'ari_vs_polarity3': [s['metrics']['vs_polarity3']['ari'] for s in layer_summaries],
            'nmi_vs_polarity3': [s['metrics']['vs_polarity3']['nmi'] for s in layer_summaries],
            'ari_vs_mft15': [s['metrics']['vs_mft15']['ari'] for s in layer_summaries],
            'nmi_vs_mft15': [s['metrics']['vs_mft15']['nmi'] for s in layer_summaries],
        }
    }
    
    # 保存汇总
    torch.save(summary, output_dir / 'cluster_summary.pt')
    
    # 生成跨层可视化
    create_layer_summary_plots(summary, output_dir)
    
    print(f"  完成! 结果保存至: {output_dir}")
    
    return summary


def create_layer_summary_plots(summary, output_dir):
    """生成跨层汇总图"""
    
    layers = summary['layers']
    by_layer = summary['by_layer']
    model_name = summary['model_name']
    
    # 图1: 簇数量随层变化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, by_layer['n_clusters'], 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Clusters')
    ax.set_title(f'{model_name} - Cluster Count by Layer')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'n_clusters_by_layer.png', dpi=150)
    plt.close()
    
    # 图2: 噪声比例随层变化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, by_layer['noise_ratio'], 'o-', linewidth=2, markersize=6, color='red')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Noise Ratio')
    ax.set_title(f'{model_name} - Noise Ratio by Layer')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(by_layer['noise_ratio']) * 1.2 + 0.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'noise_ratio_by_layer.png', dpi=150)
    plt.close()
    
    # 图3: ARI随层变化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, by_layer['ari_vs_mft5'], 'o-', linewidth=2, markersize=6, label='vs MFT5')
    ax.plot(layers, by_layer['ari_vs_polarity3'], 's-', linewidth=2, markersize=6, label='vs Polarity3')
    ax.plot(layers, by_layer['ari_vs_mft15'], '^-', linewidth=2, markersize=6, label='vs MFT15')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Adjusted Rand Index')
    ax.set_title(f'{model_name} - ARI by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ari_by_layer.png', dpi=150)
    plt.close()
    
    # 图4: NMI随层变化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, by_layer['nmi_vs_mft5'], 'o-', linewidth=2, markersize=6, label='vs MFT5')
    ax.plot(layers, by_layer['nmi_vs_polarity3'], 's-', linewidth=2, markersize=6, label='vs Polarity3')
    ax.plot(layers, by_layer['nmi_vs_mft15'], '^-', linewidth=2, markersize=6, label='vs MFT15')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized Mutual Information')
    ax.set_title(f'{model_name} - NMI by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'nmi_by_layer.png', dpi=150)
    plt.close()
    
    # 图5: PCA保留维度随层变化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, by_layer['n_pca_components'], 'o-', linewidth=2, markersize=6, color='green')
    ax.set_xlabel('Layer')
    ax.set_ylabel('PCA Components (95% variance)')
    ax.set_title(f'{model_name} - PCA Dimensionality by Layer')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_components_by_layer.png', dpi=150)
    plt.close()


def main():
    """主函数"""
    
    print("=" * 60)
    print("聚类分析：探索LLM道德表征的内在结构")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    all_summaries = {}
    
    for model_folder, model_name in MODEL_CONFIGS:
        try:
            summary = process_model(model_folder, model_name)
            if summary is not None:
                all_summaries[model_name] = summary
        except Exception as e:
            print(f"  [错误] 处理 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    
    
    # 保存总汇总
    torch.save(all_summaries, OUTPUT_PATH / 'all_models_summary.pt')
    
    print("\n" + "=" * 60)
    print(f"分析完成! 结果保存至: {OUTPUT_PATH}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    main()

# 生成跨模型比较
    if len(all_summaries) > 1:
        create_cross_model_comparison(all_summaries)



