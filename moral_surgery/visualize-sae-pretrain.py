#!/usr/bin/env python3
"""
SAE Training Metrics Visualization
==================================
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


def load_metrics(output_dir: str, layers: Optional[List[int]] = None) -> Dict[int, List[dict]]:
    """
    加载所有层的训练指标
    
    Args:
        output_dir: SAE 训练输出目录
        layers: 要加载的层列表，None 表示加载所有层
        
    Returns:
        {layer_id: [metrics_dict, ...]}
    """
    output_path = Path(output_dir)
    all_metrics = {}
    
    # 自动发现所有层
    if layers is None:
        layers = []
        for layer_dir in sorted(output_path.iterdir()):
            if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
                layer_id = int(layer_dir.name.split("_")[1])
                layers.append(layer_id)
    
    for layer in layers:
        metrics_path = output_path / f"layer_{layer}" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                all_metrics[layer] = json.load(f)
            print(f"Loaded metrics for layer {layer}: {len(all_metrics[layer])} records")
        else:
            print(f"Warning: No metrics found for layer {layer}")
            
    return all_metrics


def extract_metric_series(metrics: List[dict], 
                          metric_name: str, 
                          split: str = "train") -> Tuple[List[int], List[float]]:
    """
    从指标列表中提取某个指标的时间序列
    
    Args:
        metrics: 指标列表
        metric_name: 指标名称
        split: 'train' 或 'val'
        
    Returns:
        (epochs, values)
    """
    epochs = []
    values = []
    
    for m in metrics:
        if m.get('split', 'train') == split and metric_name in m:
            epochs.append(m['epoch'])
            values.append(m[metric_name])
            
    return epochs, values


def plot_single_layer_metrics(metrics: List[dict], 
                              layer: int,
                              save_path: Optional[str] = None):
    """
    绘制单层的所有指标变化
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Layer {layer} Training Metrics', fontsize=14, fontweight='bold')
    
    # 1. 总损失
    ax = axes[0, 0]
    epochs_train, loss_train = extract_metric_series(metrics, 'loss', 'train')
    epochs_val, loss_val = extract_metric_series(metrics, 'loss', 'val')
    ax.plot(epochs_train, loss_train, 'b-', label='Train', alpha=0.7)
    if epochs_val:
        ax.plot(epochs_val, loss_val, 'r-o', label='Val', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 重建损失
    ax = axes[0, 1]
    epochs_train, rec_train = extract_metric_series(metrics, 'loss_rec', 'train')
    epochs_val, rec_val = extract_metric_series(metrics, 'loss_rec', 'val')
    ax.plot(epochs_train, rec_train, 'b-', label='Train', alpha=0.7)
    if epochs_val:
        ax.plot(epochs_val, rec_val, 'r-o', label='Val', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 稀疏损失
    ax = axes[0, 2]
    epochs_train, sparse_train = extract_metric_series(metrics, 'loss_sparse', 'train')
    ax.plot(epochs_train, sparse_train, 'g-', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L1 Loss')
    ax.set_title('Sparsity Loss')
    ax.grid(True, alpha=0.3)
    
    # 4. L0 (平均非零激活数)
    ax = axes[1, 0]
    epochs_train, l0_train = extract_metric_series(metrics, 'l0', 'train')
    epochs_val, l0_val = extract_metric_series(metrics, 'l0', 'val')
    ax.plot(epochs_train, l0_train, 'b-', label='Train', alpha=0.7)
    if epochs_val:
        ax.plot(epochs_val, l0_val, 'r-o', label='Val', markersize=4)
    ax.axhline(y=50, color='k', linestyle='--', alpha=0.5, label='Target L0')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('L0')
    ax.set_title('Average Active Features (L0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 活跃特征比例
    ax = axes[1, 1]
    epochs_train, alive_train = extract_metric_series(metrics, 'frac_alive', 'train')
    ax.plot(epochs_train, [v * 100 for v in alive_train], 'purple', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Alive Features (%)')
    ax.set_title('Fraction of Alive Features')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # 6. 稀疏系数变化
    ax = axes[1, 2]
    epochs_train, coef_train = extract_metric_series(metrics, 'sparsity_coef', 'train')
    ax.plot(epochs_train, coef_train, 'orange', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Coefficient')
    ax.set_title('Sparsity Coefficient (Dynamic)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_all_layers_comparison(all_metrics: Dict[int, List[dict]],
                               metric_name: str,
                               title: str,
                               ylabel: str,
                               split: str = "train",
                               save_path: Optional[str] = None,
                               log_scale: bool = False):
    """
    绘制所有层的某个指标对比图
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    layers = sorted(all_metrics.keys())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    
    for i, layer in enumerate(layers):
        epochs, values = extract_metric_series(all_metrics[layer], metric_name, split)
        if epochs:
            ax.plot(epochs, values, color=cmap[i], alpha=0.7, label=f'Layer {layer}')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if log_scale:
        ax.set_yscale('log')
    
    # 图例放在右侧
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
              ncol=2 if len(layers) > 18 else 1, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_final_metrics_heatmap(all_metrics: Dict[int, List[dict]],
                               save_path: Optional[str] = None):
    """
    绘制最终指标的热力图（按层）
    """
    layers = sorted(all_metrics.keys())
    metrics_names = ['loss', 'loss_rec', 'l0', 'frac_alive']
    display_names = ['Total Loss', 'Recon Loss', 'L0', 'Alive %']
    
    # 提取最终值
    data = np.zeros((len(layers), len(metrics_names)))
    
    for i, layer in enumerate(layers):
        metrics = all_metrics[layer]
        # 获取最后一个 train 记录
        train_metrics = [m for m in metrics if m.get('split', 'train') == 'train']
        if train_metrics:
            final = train_metrics[-1]
            for j, name in enumerate(metrics_names):
                val = final.get(name, 0)
                if name == 'frac_alive':
                    val *= 100  # 转换为百分比
                data[i, j] = val
    
    # 归一化用于可视化
    data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # 热力图
    ax = axes[0]
    im = ax.imshow(data_normalized, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    ax.set_title('Final Metrics Heatmap (Normalized)', fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(layers)):
        for j in range(len(metrics_names)):
            val = data[i, j]
            text = f'{val:.3f}' if val < 10 else f'{val:.1f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=7,
                   color='white' if data_normalized[i, j] > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 条形图：按层的 L0 分布
    ax = axes[1]
    l0_values = data[:, 2]  # L0 列
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    bars = ax.barh(range(len(layers)), l0_values, color=colors)
    ax.axvline(x=50, color='r', linestyle='--', alpha=0.7, label='Target L0')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'Layer {l}' for l in layers])
    ax.set_xlabel('L0 (Active Features)')
    ax.set_title('Final L0 by Layer', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_training_summary(all_metrics: Dict[int, List[dict]],
                          save_path: Optional[str] = None):
    """
    绘制训练总结图：收敛曲线、最终性能等
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    layers = sorted(all_metrics.keys())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    
    # 1. 所有层的重建损失
    ax1 = fig.add_subplot(gs[0, 0])
    for i, layer in enumerate(layers):
        epochs, values = extract_metric_series(all_metrics[layer], 'loss_rec', 'train')
        if epochs:
            ax1.plot(epochs, values, color=cmap[i], alpha=0.6, linewidth=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.set_title('Reconstruction Loss (All Layers)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. 所有层的 L0
    ax2 = fig.add_subplot(gs[0, 1])
    for i, layer in enumerate(layers):
        epochs, values = extract_metric_series(all_metrics[layer], 'l0', 'train')
        if epochs:
            ax2.plot(epochs, values, color=cmap[i], alpha=0.6, linewidth=0.8)
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Target')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L0')
    ax2.set_title('L0 Sparsity (All Layers)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 所有层的活跃特征比例
    ax3 = fig.add_subplot(gs[0, 2])
    for i, layer in enumerate(layers):
        epochs, values = extract_metric_series(all_metrics[layer], 'frac_alive', 'train')
        if epochs:
            ax3.plot(epochs, [v * 100 for v in values], color=cmap[i], alpha=0.6, linewidth=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Alive Features (%)')
    ax3.set_title('Feature Utilization (All Layers)')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终重建损失分布
    ax4 = fig.add_subplot(gs[1, 0])
    final_rec_losses = []
    for layer in layers:
        train_metrics = [m for m in all_metrics[layer] if m.get('split', 'train') == 'train']
        if train_metrics:
            final_rec_losses.append(train_metrics[-1].get('loss_rec', 0))
    ax4.bar(range(len(layers)), final_rec_losses, color=cmap)
    ax4.set_xticks(range(0, len(layers), 4))
    ax4.set_xticklabels([f'{layers[i]}' for i in range(0, len(layers), 4)])
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Final Recon Loss')
    ax4.set_title('Final Reconstruction Loss by Layer')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 最终 L0 分布
    ax5 = fig.add_subplot(gs[1, 1])
    final_l0s = []
    for layer in layers:
        train_metrics = [m for m in all_metrics[layer] if m.get('split', 'train') == 'train']
        if train_metrics:
            final_l0s.append(train_metrics[-1].get('l0', 0))
    ax5.bar(range(len(layers)), final_l0s, color=cmap)
    ax5.axhline(y=50, color='r', linestyle='--', alpha=0.7, linewidth=2)
    ax5.set_xticks(range(0, len(layers), 4))
    ax5.set_xticklabels([f'{layers[i]}' for i in range(0, len(layers), 4)])
    ax5.set_xlabel('Layer')
    ax5.set_ylabel('Final L0')
    ax5.set_title('Final L0 by Layer')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 稀疏系数最终值
    ax6 = fig.add_subplot(gs[1, 2])
    final_coefs = []
    for layer in layers:
        train_metrics = [m for m in all_metrics[layer] if m.get('split', 'train') == 'train']
        if train_metrics:
            final_coefs.append(train_metrics[-1].get('sparsity_coef', 0))
    ax6.bar(range(len(layers)), final_coefs, color=cmap)
    ax6.set_xticks(range(0, len(layers), 4))
    ax6.set_xticklabels([f'{layers[i]}' for i in range(0, len(layers), 4)])
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('Sparsity Coef')
    ax6.set_title('Final Sparsity Coefficient by Layer')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=min(layers), vmax=max(layers)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], location='right', shrink=0.8, pad=0.02)
    cbar.set_label('Layer')
    
    plt.suptitle('SAE Training Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_full_report(output_dir: str, 
                         report_dir: Optional[str] = None,
                         layers: Optional[List[int]] = None):
    """
    生成完整的可视化报告
    """
    output_path = Path(output_dir)
    
    if report_dir is None:
        report_dir = output_path / "visualizations"
    else:
        report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading metrics from {output_dir}...")
    all_metrics = load_metrics(output_dir, layers)
    
    if not all_metrics:
        print("No metrics found!")
        return
    
    print(f"\nFound metrics for {len(all_metrics)} layers")
    print(f"Saving visualizations to {report_dir}\n")
    
    # 1. 训练总结图
    print("Generating training summary...")
    plot_training_summary(all_metrics, save_path=report_dir / "training_summary.png")
    
    # 2. 各指标的全层对比图
    print("Generating metric comparisons...")
    plot_all_layers_comparison(
        all_metrics, 'loss_rec', 
        'Reconstruction Loss Across All Layers',
        'MSE Loss', 'train',
        save_path=report_dir / "all_layers_recon_loss.png",
        log_scale=True
    )
    
    plot_all_layers_comparison(
        all_metrics, 'l0',
        'L0 Sparsity Across All Layers',
        'L0 (Active Features)', 'train',
        save_path=report_dir / "all_layers_l0.png"
    )
    
    plot_all_layers_comparison(
        all_metrics, 'frac_alive',
        'Feature Utilization Across All Layers',
        'Fraction Alive', 'train',
        save_path=report_dir / "all_layers_alive.png"
    )
    
    # 3. 最终指标热力图
    print("Generating final metrics heatmap...")
    plot_final_metrics_heatmap(all_metrics, save_path=report_dir / "final_metrics_heatmap.png")
    
    # 4. 每层的详细图（保存到 PDF）
    print("Generating per-layer detailed plots...")
    pdf_path = report_dir / "per_layer_details.pdf"
    with PdfPages(pdf_path) as pdf:
        for layer in sorted(all_metrics.keys()):
            fig = plot_single_layer_metrics(all_metrics[layer], layer)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    print(f"Saved per-layer details to {pdf_path}")
    
    # 5. 也单独保存每层的 PNG（可选）
    per_layer_dir = report_dir / "per_layer"
    per_layer_dir.mkdir(exist_ok=True)
    for layer in sorted(all_metrics.keys()):
        plot_single_layer_metrics(
            all_metrics[layer], layer,
            save_path=per_layer_dir / f"layer_{layer}_metrics.png"
        )
        plt.close()
    
    # 6. 生成统计摘要
    print("\nGenerating statistics summary...")
    summary = generate_statistics_summary(all_metrics)
    summary_path = report_dir / "statistics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved statistics to {summary_path}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total layers trained: {summary['num_layers']}")
    print(f"Epochs per layer: {summary['epochs_per_layer']}")
    print(f"\nFinal Reconstruction Loss:")
    print(f"  Mean: {summary['final_recon_loss']['mean']:.6f}")
    print(f"  Std:  {summary['final_recon_loss']['std']:.6f}")
    print(f"  Min:  {summary['final_recon_loss']['min']:.6f} (Layer {summary['final_recon_loss']['min_layer']})")
    print(f"  Max:  {summary['final_recon_loss']['max']:.6f} (Layer {summary['final_recon_loss']['max_layer']})")
    print(f"\nFinal L0:")
    print(f"  Mean: {summary['final_l0']['mean']:.2f}")
    print(f"  Std:  {summary['final_l0']['std']:.2f}")
    print(f"\nFeature Utilization:")
    print(f"  Mean: {summary['final_alive']['mean']:.2%}")
    print(f"  Min:  {summary['final_alive']['min']:.2%} (Layer {summary['final_alive']['min_layer']})")
    print("=" * 60)
    
    print(f"\nAll visualizations saved to {report_dir}")


def generate_statistics_summary(all_metrics: Dict[int, List[dict]]) -> dict:
    """生成统计摘要"""
    layers = sorted(all_metrics.keys())
    
    final_rec_losses = {}
    final_l0s = {}
    final_alives = {}
    epochs_counts = {}
    
    for layer in layers:
        train_metrics = [m for m in all_metrics[layer] if m.get('split', 'train') == 'train']
        if train_metrics:
            final = train_metrics[-1]
            final_rec_losses[layer] = final.get('loss_rec', 0)
            final_l0s[layer] = final.get('l0', 0)
            final_alives[layer] = final.get('frac_alive', 0)
            epochs_counts[layer] = len(train_metrics)
    
    rec_values = list(final_rec_losses.values())
    l0_values = list(final_l0s.values())
    alive_values = list(final_alives.values())
    
    return {
        'num_layers': len(layers),
        'layers': layers,
        'epochs_per_layer': dict(epochs_counts),
        'final_recon_loss': {
            'mean': float(np.mean(rec_values)),
            'std': float(np.std(rec_values)),
            'min': float(np.min(rec_values)),
            'max': float(np.max(rec_values)),
            'min_layer': int(layers[np.argmin(rec_values)]),
            'max_layer': int(layers[np.argmax(rec_values)]),
            'per_layer': {int(k): float(v) for k, v in final_rec_losses.items()}
        },
        'final_l0': {
            'mean': float(np.mean(l0_values)),
            'std': float(np.std(l0_values)),
            'min': float(np.min(l0_values)),
            'max': float(np.max(l0_values)),
            'per_layer': {int(k): float(v) for k, v in final_l0s.items()}
        },
        'final_alive': {
            'mean': float(np.mean(alive_values)),
            'std': float(np.std(alive_values)),
            'min': float(np.min(alive_values)),
            'max': float(np.max(alive_values)),
            'min_layer': int(layers[np.argmin(alive_values)]),
            'max_layer': int(layers[np.argmax(alive_values)]),
            'per_layer': {int(k): float(v) for k, v in final_alives.items()}
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize SAE Training Metrics")
    parser.add_argument("--output_dir", type=str, default="./outputs/sae_pretrain_20251223",
                        help="SAE training output directory")
    parser.add_argument("--report_dir", type=str, default="./outputs/sae_pretrain_20251223",
                        help="Directory to save visualizations (default: output_dir/visualizations)")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layers to visualize (comma-separated, e.g., '0,1,2'). Default: all")
    parser.add_argument("--single_layer", type=int, default=None,
                        help="Only visualize a single layer")
    args = parser.parse_args()
    
    # 解析层
    layers = None
    if args.single_layer is not None:
        layers = [args.single_layer]
    elif args.layers is not None:
        layers = [int(x) for x in args.layers.split(",")]
    
    if args.single_layer is not None:
        # 单层可视化模式
        all_metrics = load_metrics(args.output_dir, layers)
        if args.single_layer in all_metrics:
            plot_single_layer_metrics(
                all_metrics[args.single_layer],
                args.single_layer,
                save_path=f"layer_{args.single_layer}_metrics.png"
            )
            plt.show()
        else:
            print(f"No metrics found for layer {args.single_layer}")
    else:
        # 完整报告模式
        generate_full_report(args.output_dir, args.report_dir, layers)


if __name__ == "__main__":
    main()