#!/usr/bin/env python3
"""
SAE Feature-MFT Label Association Analysis (Multi-GPU Version)
==============================================================
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import multiprocessing as mp
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import cosine
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

MFT_DIMENSIONS = [
    'care', 'harm', 'fairness', 'cheating', 'loyalty', 
    'betrayal', 'authority', 'subversion', 'sanctity', 'degradation'
]

MFT_PAIRS = [
    ('care', 'harm'),
    ('fairness', 'cheating'),
    ('loyalty', 'betrayal'),
    ('authority', 'subversion'),
    ('sanctity', 'degradation')
]

VIRTUE_DIMS = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']
VICE_DIMS = ['harm', 'cheating', 'betrayal', 'subversion', 'degradation']


@dataclass
class AnalysisConfig:
    """分析配置"""
    sae_dir: str = "./outputs/sae_pretrain"
    val_h5_path: str = "./data/activations_qwen3_8b/qwen3_8b_val_activations.h5"
    val_csv_path: str = "./data/train/social_chem_val_expanded.csv"
    output_dir: str = "./outputs/sae_analysis"
    
    # SAE 架构参数
    input_dim: int = 4096
    hidden_dim: int = 16384
    
    # 分析参数
    layers: List[int] = None
    correlation_threshold: float = 0.1
    top_k_features: int = 100
    
    # 多GPU参数
    num_gpus: int = 8
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = list(range(36))


# ============================================================================
# SAE Model
# ============================================================================

class SparseAutoencoder(nn.Module):
    """稀疏自编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


# ============================================================================
# Data Loading
# ============================================================================

def load_sae(sae_path: str, input_dim: int, hidden_dim: int, 
             device: torch.device) -> Tuple[SparseAutoencoder, torch.Tensor]:
    """
    加载 SAE checkpoint
    
    关键修改: 使用 map_location='cpu' 先加载到 CPU，然后再移动到目标设备
    这样可以避免 CUDA_VISIBLE_DEVICES 导致的设备映射问题
    """
    # 先加载到 CPU，避免设备映射问题
    checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
    
    sae = SparseAutoencoder(input_dim, hidden_dim)
    sae.load_state_dict(checkpoint['model_state'])
    sae = sae.to(device)
    sae.eval()
    
    center = checkpoint['center'].to(device)
    
    return sae, center


def load_validation_data(h5_path: str, layer: int) -> Dict:
    """加载验证集数据"""
    with h5py.File(h5_path, 'r') as f:
        activations = f[f'activations/layer_{layer}'][:].astype(np.float32)
        
        metadata = {
            'id': f['metadata/id'][:].astype(str),
            'row_id': f['metadata/row_id'][:],
            'm_virtue': f['metadata/m_virtue'][:],
            'm_vice': f['metadata/m_vice'][:],
            'moral_vector': f['metadata/moral_vector'][:],
            'target_dimension': f['metadata/target_dimension'][:].astype(str)
        }
    
    return {
        'activations': activations,
        'metadata': metadata
    }


def compute_sae_features(sae: SparseAutoencoder, 
                         activations: np.ndarray,
                         center: torch.Tensor,
                         device: torch.device,
                         batch_size: int = 2048,
                         desc: str = "Computing features",
                         position: int = 0) -> np.ndarray:
    """计算 SAE 特征激活"""
    n_samples = activations.shape[0]
    hidden_dim = sae.encoder.out_features
    
    all_features = np.zeros((n_samples, hidden_dim), dtype=np.float32)
    
    with torch.no_grad():
        for start in tqdm(range(0, n_samples, batch_size), 
                          desc=desc, leave=False, position=position):
            end = min(start + batch_size, n_samples)
            batch = torch.from_numpy(activations[start:end]).to(device)
            
            batch = batch - center.unsqueeze(0)
            z = sae.encode(batch)
            all_features[start:end] = z.cpu().numpy()
    
    return all_features


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_dimension_alignment(features: np.ndarray, 
                                moral_vectors: np.ndarray,
                                m_virtue: np.ndarray,
                                m_vice: np.ndarray,
                                layer: int,
                                position: int) -> Dict:
    """分析特征与 MFT 维度的对齐性"""
    n_features = features.shape[1]
    n_dims = moral_vectors.shape[1]
    
    feature_dim_corr = np.zeros((n_features, n_dims))
    feature_dim_pval = np.zeros((n_features, n_dims))
    feature_virtue_corr = np.zeros(n_features)
    feature_vice_corr = np.zeros(n_features)
    
    for i in tqdm(range(n_features), 
                  desc=f"Layer {layer} correlations",
                  position=position,
                  leave=False):
        feat = features[:, i]
        
        if feat.std() < 1e-8:
            continue
            
        for j in range(n_dims):
            dim_vals = moral_vectors[:, j]
            if dim_vals.std() < 1e-8:
                continue
            corr, pval = stats.pearsonr(feat, dim_vals)
            feature_dim_corr[i, j] = corr
            feature_dim_pval[i, j] = pval
        
        if m_virtue.std() > 1e-8:
            feature_virtue_corr[i], _ = stats.pearsonr(feat, m_virtue)
        if m_vice.std() > 1e-8:
            feature_vice_corr[i], _ = stats.pearsonr(feat, m_vice)
    
    # 特征分类
    feature_types = []
    monosemantic_features = []
    polysemantic_features = []
    
    for i in range(n_features):
        corrs = feature_dim_corr[i]
        abs_corrs = np.abs(corrs)
        
        n_significant = np.sum(abs_corrs > 0.1)
        max_corr = abs_corrs.max()
        
        if max_corr < 0.05:
            feature_types.append('weak')
        elif n_significant == 1:
            feature_types.append('monosemantic')
            monosemantic_features.append(i)
        elif n_significant <= 2:
            top_dims = np.argsort(abs_corrs)[-2:]
            dim_names = [MFT_DIMENSIONS[d] for d in top_dims]
            is_same_pair = any(
                (dim_names[0] in pair and dim_names[1] in pair)
                for pair in MFT_PAIRS
            )
            if is_same_pair:
                feature_types.append('monosemantic')
                monosemantic_features.append(i)
            else:
                feature_types.append('polysemantic')
                polysemantic_features.append(i)
        else:
            feature_types.append('polysemantic')
            polysemantic_features.append(i)
    
    # 每个维度的 top 特征
    dimension_top_features = {}
    for j, dim_name in enumerate(MFT_DIMENSIONS):
        corrs = feature_dim_corr[:, j]
        
        top_pos_idx = np.argsort(corrs)[-50:][::-1]
        top_pos = [(int(idx), float(corrs[idx])) for idx in top_pos_idx if corrs[idx] > 0.05]
        
        top_neg_idx = np.argsort(corrs)[:50]
        top_neg = [(int(idx), float(corrs[idx])) for idx in top_neg_idx if corrs[idx] < -0.05]
        
        dimension_top_features[dim_name] = {
            'positive': top_pos[:20],
            'negative': top_neg[:20]
        }
    
    type_counts = {
        'monosemantic': len(monosemantic_features),
        'polysemantic': len(polysemantic_features),
        'weak': feature_types.count('weak')
    }
    
    return {
        'feature_dim_corr': feature_dim_corr,
        'feature_dim_pval': feature_dim_pval,
        'feature_virtue_corr': feature_virtue_corr,
        'feature_vice_corr': feature_vice_corr,
        'feature_types': feature_types,
        'type_counts': type_counts,
        'monosemantic_features': monosemantic_features,
        'polysemantic_features': polysemantic_features,
        'dimension_top_features': dimension_top_features
    }


def analyze_prototypicality(features: np.ndarray,
                            moral_vectors: np.ndarray,
                            dimension_top_features: Dict) -> Dict:
    """分析特征激活是否反映人类的道德典型性"""
    results = {}
    
    for dim_idx, dim_name in enumerate(MFT_DIMENSIONS):
        dim_results = {
            'spearman_correlations': [],
            'pearson_correlations': [],
            'feature_details': []
        }
        
        dim_values = moral_vectors[:, dim_idx]
        nonzero_mask = dim_values > 0
        
        if nonzero_mask.sum() < 50:
            results[dim_name] = {
                'status': 'insufficient_samples',
                'n_samples': int(nonzero_mask.sum())
            }
            continue
        
        dim_values_nonzero = dim_values[nonzero_mask]
        top_features = dimension_top_features.get(dim_name, {}).get('positive', [])
        
        for feat_idx, feat_corr in top_features[:20]:
            feat_values = features[nonzero_mask, feat_idx]
            
            if feat_values.std() < 1e-8:
                continue
            
            spearman_r, spearman_p = stats.spearmanr(feat_values, dim_values_nonzero)
            pearson_r, pearson_p = stats.pearsonr(feat_values, dim_values_nonzero)
            
            dim_results['spearman_correlations'].append(spearman_r)
            dim_results['pearson_correlations'].append(pearson_r)
            dim_results['feature_details'].append({
                'feature_idx': feat_idx,
                'overall_corr': feat_corr,
                'prototypicality_spearman': float(spearman_r),
                'prototypicality_pearson': float(pearson_r),
                'spearman_pval': float(spearman_p),
                'pearson_pval': float(pearson_p)
            })
        
        if dim_results['spearman_correlations']:
            dim_results['mean_spearman'] = float(np.mean(dim_results['spearman_correlations']))
            dim_results['mean_pearson'] = float(np.mean(dim_results['pearson_correlations']))
            dim_results['n_features_analyzed'] = len(dim_results['spearman_correlations'])
            dim_results['n_samples'] = int(nonzero_mask.sum())
        
        results[dim_name] = dim_results
    
    return results


def analyze_polarity_collapse(features: np.ndarray,
                              moral_vectors: np.ndarray,
                              m_virtue: np.ndarray,
                              m_vice: np.ndarray,
                              feature_dim_corr: np.ndarray) -> Dict:
    """检测道德极性是否塌缩"""
    n_features = features.shape[1]
    
    coactivation_features = []
    virtue_only_features = []
    vice_only_features = []
    
    high_virtue_mask = m_virtue > 0.5
    high_vice_mask = m_vice > 0.5
    
    if high_virtue_mask.sum() > 10 and high_vice_mask.sum() > 10:
        for i in range(n_features):
            feat = features[:, i]
            if feat.std() < 1e-8:
                continue
            
            virtue_activation = feat[high_virtue_mask].mean()
            vice_activation = feat[high_vice_mask].mean()
            baseline_activation = feat.mean()
            
            virtue_lift = virtue_activation - baseline_activation
            vice_lift = vice_activation - baseline_activation
            
            if virtue_lift > 0.1 and vice_lift > 0.1:
                coactivation_features.append({
                    'feature_idx': i,
                    'virtue_lift': float(virtue_lift),
                    'vice_lift': float(vice_lift)
                })
            elif virtue_lift > 0.1 and vice_lift < 0.05:
                virtue_only_features.append({
                    'feature_idx': i,
                    'virtue_lift': float(virtue_lift),
                    'vice_lift': float(vice_lift)
                })
            elif vice_lift > 0.1 and virtue_lift < 0.05:
                vice_only_features.append({
                    'feature_idx': i,
                    'virtue_lift': float(virtue_lift),
                    'vice_lift': float(vice_lift)
                })
    
    # 对立维度分析
    pair_analysis = {}
    
    for virtue_dim, vice_dim in MFT_PAIRS:
        virtue_idx = MFT_DIMENSIONS.index(virtue_dim)
        vice_idx = MFT_DIMENSIONS.index(vice_dim)
        
        virtue_corrs = feature_dim_corr[:, virtue_idx]
        vice_corrs = feature_dim_corr[:, vice_idx]
        
        norm_virtue = np.linalg.norm(virtue_corrs)
        norm_vice = np.linalg.norm(vice_corrs)
        
        if norm_virtue > 1e-8 and norm_vice > 1e-8:
            cosine_sim = np.dot(virtue_corrs, vice_corrs) / (norm_virtue * norm_vice)
        else:
            cosine_sim = 0.0
        
        virtue_top = set(np.argsort(np.abs(virtue_corrs))[-100:])
        vice_top = set(np.argsort(np.abs(vice_corrs))[-100:])
        overlap = len(virtue_top & vice_top)
        
        shared_features = list(virtue_top & vice_top)
        if shared_features:
            same_sign = sum(
                1 for f in shared_features
                if virtue_corrs[f] * vice_corrs[f] > 0
            )
            opposite_sign = len(shared_features) - same_sign
        else:
            same_sign = 0
            opposite_sign = 0
        
        pair_analysis[f"{virtue_dim}-{vice_dim}"] = {
            'cosine_similarity': float(cosine_sim),
            'top100_overlap': overlap,
            'shared_same_sign': same_sign,
            'shared_opposite_sign': opposite_sign,
            'interpretation': 'collapsed' if cosine_sim > 0.5 else (
                'partially_separated' if cosine_sim > 0 else 'well_separated'
            )
        }
    
    # 极性区分能力
    distinguishing_features = []
    polarity_label = (m_virtue > m_vice).astype(float)
    has_moral = (m_virtue > 0) | (m_vice > 0)
    
    if has_moral.sum() >= 100:
        for i in range(n_features):
            feat = features[:, i]
            if feat.std() < 1e-8:
                continue
            
            r, p = stats.pointbiserialr(polarity_label[has_moral], feat[has_moral])
            
            if abs(r) > 0.1 and p < 0.01:
                distinguishing_features.append({
                    'feature_idx': i,
                    'point_biserial_r': float(r),
                    'pval': float(p)
                })
    
    return {
        'coactivation_analysis': {
            'n_coactivation_features': len(coactivation_features),
            'n_virtue_only_features': len(virtue_only_features),
            'n_vice_only_features': len(vice_only_features),
            'coactivation_examples': coactivation_features[:20],
            'virtue_only_examples': virtue_only_features[:20],
            'vice_only_examples': vice_only_features[:20]
        },
        'pair_analysis': pair_analysis,
        'polarity_discrimination': {
            'n_distinguishing_features': len(distinguishing_features),
            'top_distinguishing_features': sorted(
                distinguishing_features, 
                key=lambda x: abs(x['point_biserial_r']),
                reverse=True
            )[:20]
        }
    }


# ============================================================================
# GPU Worker Process
# ============================================================================

def gpu_worker_process(gpu_id: int, 
                       layers: List[int], 
                       config_dict: Dict,
                       result_queue: mp.Queue,
                       status_queue: mp.Queue):
    """
    单个 GPU 的 worker 进程
    
    在进程启动时设置 CUDA_VISIBLE_DEVICES，使得该进程只能看到一个 GPU
    """
    # 设置环境变量，让此进程只能看到指定的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 现在 cuda:0 就是我们要用的 GPU
    device = torch.device("cuda:0")
    
    # 重建配置
    config = AnalysisConfig(**config_dict)
    
    # 用于进度条的位置
    position = gpu_id
    
    for layer in layers:
        status_queue.put((gpu_id, layer, 'started'))
        
        try:
            # 1. 加载 SAE
            sae_path = Path(config.sae_dir) / f"layer_{layer}" / "sae_finetuned.pt"
            if not sae_path.exists():
                result_queue.put({
                    'layer': layer,
                    'status': 'error',
                    'error': f'SAE not found: {sae_path}'
                })
                status_queue.put((gpu_id, layer, 'error'))
                continue
            
            # 关键: 使用 map_location='cpu' 加载，然后移动到设备
            sae, center = load_sae(str(sae_path), config.input_dim, config.hidden_dim, device)
            
            # 2. 加载验证集数据
            val_data = load_validation_data(config.val_h5_path, layer)
            
            # 3. 计算 SAE 特征
            features = compute_sae_features(
                sae, val_data['activations'], center, device,
                desc=f"GPU{gpu_id} L{layer} encode",
                position=position
            )
            
            moral_vectors = val_data['metadata']['moral_vector']
            m_virtue = val_data['metadata']['m_virtue']
            m_vice = val_data['metadata']['m_vice']
            
            # 4. 维度对齐分析
            alignment_results = analyze_dimension_alignment(
                features, moral_vectors, m_virtue, m_vice, layer, position
            )
            
            # 5. 原型性验证
            proto_results = analyze_prototypicality(
                features, moral_vectors, alignment_results['dimension_top_features']
            )
            
            # 6. 极性塌缩检测
            polarity_results = analyze_polarity_collapse(
                features, moral_vectors, m_virtue, m_vice,
                alignment_results['feature_dim_corr']
            )
            
            # 7. 保存相关性矩阵
            layer_output_dir = Path(config.output_dir) / f"layer_{layer}"
            layer_output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(
                layer_output_dir / "feature_dim_correlations.npy",
                alignment_results['feature_dim_corr']
            )
            np.save(
                layer_output_dir / "feature_virtue_corr.npy",
                alignment_results['feature_virtue_corr']
            )
            np.save(
                layer_output_dir / "feature_vice_corr.npy",
                alignment_results['feature_vice_corr']
            )
            
            # 8. 汇总结果
            coact = polarity_results['coactivation_analysis']
            pol_disc = polarity_results['polarity_discrimination']
            
            results = {
                'layer': layer,
                'status': 'success',
                'gpu_id': gpu_id,
                'n_features': config.hidden_dim,
                'n_samples': features.shape[0],
                'avg_l0': float((features > 0).sum(axis=1).mean()),
                'alignment': {
                    'type_counts': alignment_results['type_counts'],
                    'dimension_top_features': {
                        dim: {
                            'positive': alignment_results['dimension_top_features'][dim]['positive'][:10],
                            'negative': alignment_results['dimension_top_features'][dim]['negative'][:10]
                        }
                        for dim in MFT_DIMENSIONS
                    }
                },
                'prototypicality': {
                    dim: {
                        'mean_spearman': proto_results[dim].get('mean_spearman'),
                        'mean_pearson': proto_results[dim].get('mean_pearson'),
                        'n_samples': proto_results[dim].get('n_samples'),
                        'top_features': proto_results[dim].get('feature_details', [])[:5]
                    }
                    for dim in MFT_DIMENSIONS
                },
                'polarity_collapse': {
                    'coactivation_summary': {
                        'n_coactivation': coact['n_coactivation_features'],
                        'n_virtue_only': coact['n_virtue_only_features'],
                        'n_vice_only': coact['n_vice_only_features']
                    },
                    'pair_analysis': polarity_results['pair_analysis'],
                    'n_distinguishing_features': pol_disc['n_distinguishing_features']
                }
            }
            
            # 保存层结果
            with open(layer_output_dir / "analysis_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            result_queue.put(results)
            status_queue.put((gpu_id, layer, 'done'))
            
            # 释放 GPU 内存
            del sae, center, features, val_data
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback
            result_queue.put({
                'layer': layer,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            status_queue.put((gpu_id, layer, 'error'))
    
    status_queue.put((gpu_id, -1, 'finished'))


# ============================================================================
# Main Parallel Analysis
# ============================================================================

def distribute_layers_to_gpus(layers: List[int], num_gpus: int) -> Dict[int, List[int]]:
    """将层分配给各个 GPU"""
    gpu_assignments = {i: [] for i in range(num_gpus)}
    
    for i, layer in enumerate(layers):
        gpu_id = i % num_gpus
        gpu_assignments[gpu_id].append(layer)
    
    return gpu_assignments


def run_parallel_analysis(config: AnalysisConfig, 
                          logger: logging.Logger) -> List[Dict]:
    """使用多 GPU 并行分析所有层"""
    layers = config.layers
    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    
    logger.info(f"Running parallel analysis on {num_gpus} GPUs")
    logger.info(f"Total layers to analyze: {len(layers)}")
    
    # 分配层到各 GPU
    gpu_assignments = distribute_layers_to_gpus(layers, num_gpus)
    
    for gpu_id, assigned_layers in gpu_assignments.items():
        logger.info(f"  GPU {gpu_id}: layers {assigned_layers}")
    
    # 转换配置为字典
    config_dict = {
        'sae_dir': config.sae_dir,
        'val_h5_path': config.val_h5_path,
        'val_csv_path': config.val_csv_path,
        'output_dir': config.output_dir,
        'input_dim': config.input_dim,
        'hidden_dim': config.hidden_dim,
        'layers': config.layers,
        'correlation_threshold': config.correlation_threshold,
        'top_k_features': config.top_k_features,
        'num_gpus': config.num_gpus
    }
    
    # 创建队列
    result_queue = mp.Queue()
    status_queue = mp.Queue()
    
    # 启动 worker 进程
    processes = []
    for gpu_id, assigned_layers in gpu_assignments.items():
        if assigned_layers:
            p = mp.Process(
                target=gpu_worker_process,
                args=(gpu_id, assigned_layers, config_dict, result_queue, status_queue)
            )
            p.start()
            processes.append((gpu_id, p))
            logger.info(f"Started worker on GPU {gpu_id}")
    
    # 收集结果
    all_results = []
    completed = 0
    total = len(layers)
    finished_gpus = set()
    
    # 简单的进度显示
    print(f"\n{'='*60}")
    print(f"Processing {total} layers on {num_gpus} GPUs...")
    print(f"{'='*60}\n")
    
    while completed < total or len(finished_gpus) < len(processes):
        # 检查状态更新
        while not status_queue.empty():
            try:
                gpu_id, layer, status = status_queue.get_nowait()
                if status == 'finished':
                    finished_gpus.add(gpu_id)
                    logger.info(f"GPU {gpu_id} finished all tasks")
                elif status == 'started':
                    logger.info(f"[GPU {gpu_id}] Starting layer {layer}")
                elif status == 'done':
                    logger.info(f"[GPU {gpu_id}] Completed layer {layer}")
                elif status == 'error':
                    logger.error(f"[GPU {gpu_id}] Error on layer {layer}")
            except:
                break
        
        # 检查结果
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                all_results.append(result)
                completed += 1
                
                layer = result['layer']
                if result['status'] == 'success':
                    type_counts = result['alignment']['type_counts']
                    print(f"[{completed}/{total}] Layer {layer}: "
                          f"mono={type_counts['monosemantic']}, "
                          f"poly={type_counts['polysemantic']}, "
                          f"weak={type_counts['weak']}")
                else:
                    print(f"[{completed}/{total}] Layer {layer}: FAILED - {result.get('error', 'Unknown')[:50]}")
            except:
                break
        
        time.sleep(0.2)
    
    # 等待所有进程结束
    for gpu_id, p in processes:
        p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"Force terminating GPU {gpu_id} worker")
            p.terminate()
    
    # 按层排序
    all_results.sort(key=lambda x: x['layer'])
    
    return all_results


def create_summary_report(all_results: List[Dict], 
                          config: AnalysisConfig,
                          logger: logging.Logger) -> Dict:
    """创建跨层汇总报告"""
    
    logger.info("\n" + "="*60)
    logger.info("Creating Cross-Layer Summary Report")
    logger.info("="*60)
    
    successful_results = [r for r in all_results if r.get('status') == 'success']
    
    summary = {
        'config': {
            'sae_dir': config.sae_dir,
            'val_h5_path': config.val_h5_path,
            'output_dir': config.output_dir,
            'input_dim': config.input_dim,
            'hidden_dim': config.hidden_dim,
            'num_gpus': config.num_gpus
        },
        'timestamp': datetime.now().isoformat(),
        'n_layers_analyzed': len(successful_results),
        'n_layers_failed': len(all_results) - len(successful_results),
        'per_layer_summary': [],
        'cross_layer_analysis': {}
    }
    
    if not successful_results:
        logger.error("No successful results to summarize!")
        return summary
    
    monosemantic_by_layer = []
    polysemantic_by_layer = []
    distinguishing_by_layer = []
    
    for result in successful_results:
        layer = result['layer']
        type_counts = result['alignment']['type_counts']
        
        layer_summary = {
            'layer': layer,
            'monosemantic_ratio': type_counts['monosemantic'] / config.hidden_dim,
            'polysemantic_ratio': type_counts['polysemantic'] / config.hidden_dim,
            'weak_ratio': type_counts['weak'] / config.hidden_dim,
            'n_distinguishing': result['polarity_collapse']['n_distinguishing_features'],
            'avg_l0': result['avg_l0']
        }
        
        proto_spearman = []
        for dim in MFT_DIMENSIONS:
            if result['prototypicality'][dim]['mean_spearman'] is not None:
                proto_spearman.append(result['prototypicality'][dim]['mean_spearman'])
        
        layer_summary['mean_prototypicality_spearman'] = (
            float(np.mean(proto_spearman)) if proto_spearman else None
        )
        
        pair_cosines = [
            v['cosine_similarity'] 
            for v in result['polarity_collapse']['pair_analysis'].values()
        ]
        layer_summary['mean_pair_cosine'] = float(np.mean(pair_cosines))
        
        summary['per_layer_summary'].append(layer_summary)
        
        monosemantic_by_layer.append(layer_summary['monosemantic_ratio'])
        polysemantic_by_layer.append(layer_summary['polysemantic_ratio'])
        distinguishing_by_layer.append(layer_summary['n_distinguishing'])
    
    layers = [r['layer'] for r in successful_results]
    
    summary['cross_layer_analysis'] = {
        'monosemantic_trend': {
            'mean': float(np.mean(monosemantic_by_layer)),
            'std': float(np.std(monosemantic_by_layer)),
            'max_layer': int(layers[np.argmax(monosemantic_by_layer)]),
            'max_value': float(max(monosemantic_by_layer))
        },
        'polysemantic_trend': {
            'mean': float(np.mean(polysemantic_by_layer)),
            'std': float(np.std(polysemantic_by_layer))
        },
        'distinguishing_trend': {
            'mean': float(np.mean(distinguishing_by_layer)),
            'std': float(np.std(distinguishing_by_layer)),
            'max_layer': int(layers[np.argmax(distinguishing_by_layer)]),
            'max_value': int(max(distinguishing_by_layer))
        }
    }
    
    # 最佳层识别
    scores = []
    for layer_sum in summary['per_layer_summary']:
        score = (
            layer_sum['monosemantic_ratio'] * 0.3 +
            (layer_sum.get('mean_prototypicality_spearman', 0) or 0) * 0.3 +
            (1 - layer_sum['mean_pair_cosine']) * 0.2 +
            layer_sum['n_distinguishing'] / config.hidden_dim * 0.2
        )
        scores.append((layer_sum['layer'], score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    summary['cross_layer_analysis']['best_layers'] = [
        {'layer': l, 'score': s} for l, s in scores[:5]
    ]
    
    logger.info("\nCross-Layer Summary:")
    logger.info(f"  Monosemantic features: {summary['cross_layer_analysis']['monosemantic_trend']['mean']*100:.1f}% "
                f"± {summary['cross_layer_analysis']['monosemantic_trend']['std']*100:.1f}%")
    logger.info(f"  Best layer for monosemantic: {summary['cross_layer_analysis']['monosemantic_trend']['max_layer']}")
    logger.info(f"  Mean distinguishing features: {summary['cross_layer_analysis']['distinguishing_trend']['mean']:.0f}")
    logger.info(f"  Best layer for distinguishing: {summary['cross_layer_analysis']['distinguishing_trend']['max_layer']}")
    
    logger.info("\nTop 5 Layers (composite score):")
    for item in summary['cross_layer_analysis']['best_layers']:
        logger.info(f"  Layer {item['layer']}: score = {item['score']:.4f}")
    
    return summary


def main():
    # 设置多进程启动方式
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="SAE Feature-MFT Label Association Analysis (Multi-GPU)"
    )
    parser.add_argument("--sae_dir", type=str, 
                        default="./outputs/sae_pretrain",
                        help="SAE checkpoints directory")
    parser.add_argument("--val_h5", type=str,
                        default="./data/activations_qwen3_8b/qwen3_8b_val_activations.h5",
                        help="Validation activations H5 file")
    parser.add_argument("--val_csv", type=str,
                        default="./data/train/social_chem_val_expanded.csv",
                        help="Validation CSV file")
    parser.add_argument("--output_dir", type=str,
                        default="./outputs/sae_analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to analyze: 'all' or comma-separated like '0,1,2'")
    parser.add_argument("--hidden_dim", type=int, default=16384,
                        help="SAE hidden dimension")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # 解析层
    if args.layers == "all":
        layers = list(range(36))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    
    # 检查可用 GPU 数量
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    
    if num_gpus == 0:
        raise RuntimeError("No GPU available!")
    
    print(f"Using {num_gpus} GPUs (available: {available_gpus})")
    
    # 创建配置
    config = AnalysisConfig(
        sae_dir=args.sae_dir,
        val_h5_path=args.val_h5,
        val_csv_path=args.val_csv,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        layers=layers,
        num_gpus=num_gpus
    )
    
    # 创建输出目录
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_path = Path(config.output_dir) / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("analysis")
    
    logger.info("SAE Feature-MFT Association Analysis (Multi-GPU Version)")
    logger.info(f"Layers: {layers}")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"Output: {config.output_dir}")
    
    # 运行并行分析
    all_results = run_parallel_analysis(config, logger)
    
    # 创建汇总报告
    summary = create_summary_report(all_results, config, logger)
    
    # 保存汇总
    summary_path = Path(config.output_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nAnalysis complete. Results saved to {config.output_dir}")
    logger.info(f"Summary saved to {summary_path}")
    
    # 打印失败的层
    failed = [r for r in all_results if r.get('status') != 'success']
    if failed:
        logger.warning(f"\nFailed layers: {[r['layer'] for r in failed]}")
        for r in failed:
            logger.warning(f"  Layer {r['layer']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()