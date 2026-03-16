#!/usr/bin/env python3
"""
SAE Moral Alignment Finetuning (Multi-GPU Version)
===================================================
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
import multiprocessing as mp
import time
import traceback

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm


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

DIM_TO_IDX = {dim: idx for idx, dim in enumerate(MFT_DIMENSIONS)}


@dataclass
class FinetuneConfig:
    """微调配置"""
    # 路径
    pretrained_sae_dir: str = "./outputs/sae_pretrain"
    analysis_dir: str = "./outputs/sae_analysis"
    train_h5_path: str = "./data/activations_qwen3_8b/qwen3_8b_train_activations.h5"
    val_h5_path: str = "./data/activations_qwen3_8b/qwen3_8b_val_activations.h5"
    bucket_path: str = "./data/train/train_buckets.json"
    output_dir: str = "./outputs/sae_finetune"
    
    # 模型参数
    input_dim: int = 4096
    hidden_dim: int = 16384
    
    # 训练参数
    batch_size: int = 512
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # 损失权重
    lambda_recon: float = 1.0
    lambda_sparse: float = 1e-4
    lambda_align: float = 0.5
    lambda_polarity: float = 0.3
    lambda_proto: float = 0.2
    lambda_mono: float = 0.1
    
    # 对比学习参数
    temperature: float = 0.07
    polarity_margin: float = 0.5
    
    # 特征选择
    top_k_aligned_features: int = 50
    alignment_threshold: float = 0.1
    
    # 早停
    patience: int = 10
    min_delta: float = 1e-4
    
    # 多GPU参数
    num_gpus: int = 8
    num_workers: int = 4
    
    # 层配置
    layers: List[int] = field(default_factory=lambda: list(range(36)))


# ============================================================================
# SAE Model with Selective Parameter Freezing
# ============================================================================

class SparseAutoencoder(nn.Module):
    """稀疏自编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
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
    
    def normalize_decoder(self, feature_indices: Optional[torch.Tensor] = None):
        """
        归一化 decoder 列向量
        
        Args:
            feature_indices: 如果指定，只归一化这些特征对应的列
        """
        with torch.no_grad():
            if feature_indices is None:
                norms = self.decoder.weight.norm(dim=0, keepdim=True)
                self.decoder.weight.div_(norms.clamp(min=1e-8))
            else:
                # 只归一化指定的特征
                norms = self.decoder.weight[:, feature_indices].norm(dim=0, keepdim=True)
                self.decoder.weight[:, feature_indices] = (
                    self.decoder.weight[:, feature_indices] / norms.clamp(min=1e-8)
                )


class MonosemanticSAEWrapper(nn.Module):
    """
    单义神经元微调包装器
    
    只允许更新与单义特征相关的编码器行和解码器列，
    其他参数保持冻结。
    """
    
    def __init__(self, sae: SparseAutoencoder, monosemantic_indices: Set[int],
                 device: torch.device, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.sae = sae
        self.device = device
        self.logger = logger
        
        # 转换为排序列表
        self.mono_indices = sorted(list(monosemantic_indices))
        self.n_mono = len(self.mono_indices)
        
        # 创建索引张量
        self.mono_indices_tensor = torch.tensor(
            self.mono_indices, dtype=torch.long, device=device
        )
        
        if logger:
            logger.info(f"MonosemanticSAEWrapper: {self.n_mono} features to finetune")
        
        # 冻结所有参数
        for param in self.sae.parameters():
            param.requires_grad = False
        
        # 创建可训练的参数：只包含单义特征的权重
        # encoder.weight: (hidden_dim, input_dim) - 每行对应一个隐藏特征
        # encoder.bias: (hidden_dim,) - 每个元素对应一个隐藏特征
        # decoder.weight: (input_dim, hidden_dim) - 每列对应一个隐藏特征
        # decoder.bias: (input_dim,) - 重建偏置，全部冻结
        
        # 提取单义特征的编码器权重
        self.encoder_weight_mono = nn.Parameter(
            sae.encoder.weight[self.mono_indices].clone()
        )  # (n_mono, input_dim)
        
        self.encoder_bias_mono = nn.Parameter(
            sae.encoder.bias[self.mono_indices].clone()
        )  # (n_mono,)
        
        # 提取单义特征的解码器权重
        self.decoder_weight_mono = nn.Parameter(
            sae.decoder.weight[:, self.mono_indices].clone()
        )  # (input_dim, n_mono)
        
        # 记录初始权重用于计算变化
        self.register_buffer(
            'init_encoder_weight', self.encoder_weight_mono.data.clone()
        )
        self.register_buffer(
            'init_decoder_weight', self.decoder_weight_mono.data.clone()
        )
    
    def _sync_weights_to_sae(self):
        """将可训练参数同步回SAE"""
        with torch.no_grad():
            self.sae.encoder.weight[self.mono_indices] = self.encoder_weight_mono.data
            self.sae.encoder.bias[self.mono_indices] = self.encoder_bias_mono.data
            self.sae.decoder.weight[:, self.mono_indices] = self.decoder_weight_mono.data
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        对于非单义特征使用冻结的原始权重，
        对于单义特征使用可训练的权重。
        """
        batch_size = x.shape[0]
        
        # 完整的编码器权重和偏置
        encoder_weight = self.sae.encoder.weight.clone()
        encoder_bias = self.sae.encoder.bias.clone()
        
        # 替换单义特征的权重
        encoder_weight[self.mono_indices] = self.encoder_weight_mono
        encoder_bias[self.mono_indices] = self.encoder_bias_mono
        
        # 编码
        z = F.relu(F.linear(x, encoder_weight, encoder_bias))
        
        # 完整的解码器权重
        decoder_weight = self.sae.decoder.weight.clone()
        decoder_weight[:, self.mono_indices] = self.decoder_weight_mono
        
        # 解码
        x_rec = F.linear(z, decoder_weight, self.sae.decoder.bias)
        
        return x_rec, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码"""
        encoder_weight = self.sae.encoder.weight.clone()
        encoder_bias = self.sae.encoder.bias.clone()
        encoder_weight[self.mono_indices] = self.encoder_weight_mono
        encoder_bias[self.mono_indices] = self.encoder_bias_mono
        return F.relu(F.linear(x, encoder_weight, encoder_bias))
    
    def normalize_decoder(self):
        """归一化单义特征的解码器列"""
        with torch.no_grad():
            norms = self.decoder_weight_mono.norm(dim=0, keepdim=True)
            self.decoder_weight_mono.div_(norms.clamp(min=1e-8))
    
    def get_weight_change_stats(self) -> Dict[str, float]:
        """计算权重变化统计"""
        with torch.no_grad():
            encoder_diff = (self.encoder_weight_mono - self.init_encoder_weight).abs()
            decoder_diff = (self.decoder_weight_mono - self.init_decoder_weight).abs()
            
            return {
                'encoder_weight_change_mean': encoder_diff.mean().item(),
                'encoder_weight_change_max': encoder_diff.max().item(),
                'decoder_weight_change_mean': decoder_diff.mean().item(),
                'decoder_weight_change_max': decoder_diff.max().item(),
            }
    
    def get_finetuned_sae(self) -> SparseAutoencoder:
        """获取微调后的完整SAE"""
        self._sync_weights_to_sae()
        return self.sae
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """获取可训练参数列表"""
        return [
            self.encoder_weight_mono,
            self.encoder_bias_mono,
            self.decoder_weight_mono
        ]


# ============================================================================
# Feature Alignment Manager
# ============================================================================

class FeatureAlignmentManager:
    """
    管理特征与道德维度的对齐关系
    """
    
    def __init__(self, analysis_dir: str, layer: int, 
                 top_k: int = 50, threshold: float = 0.1,
                 logger: Optional[logging.Logger] = None):
        self.layer = layer
        self.top_k = top_k
        self.threshold = threshold
        self.logger = logger
        
        layer_dir = Path(analysis_dir) / f"layer_{layer}"
        
        # 加载相关性矩阵
        corr_path = layer_dir / "feature_dim_correlations.npy"
        if corr_path.exists():
            self.feature_dim_corr = np.load(corr_path)
        else:
            raise FileNotFoundError(f"Correlation matrix not found: {corr_path}")
        
        # 加载极性相关性
        virtue_corr_path = layer_dir / "feature_virtue_corr.npy"
        vice_corr_path = layer_dir / "feature_vice_corr.npy"
        
        if virtue_corr_path.exists():
            self.feature_virtue_corr = np.load(virtue_corr_path)
        else:
            self.feature_virtue_corr = None
            
        if vice_corr_path.exists():
            self.feature_vice_corr = np.load(vice_corr_path)
        else:
            self.feature_vice_corr = None
        
        self._build_alignment_masks()
        
    def _build_alignment_masks(self):
        """构建特征对齐掩码"""
        n_features = self.feature_dim_corr.shape[0]
        
        # 1. 每个维度的 top 特征
        self.dim_to_features = {}
        
        for dim_idx, dim_name in enumerate(MFT_DIMENSIONS):
            corrs = self.feature_dim_corr[:, dim_idx]
            
            pos_indices = np.where(corrs > self.threshold)[0]
            pos_features = [(int(idx), float(corrs[idx])) for idx in pos_indices]
            pos_features.sort(key=lambda x: x[1], reverse=True)
            
            neg_indices = np.where(corrs < -self.threshold)[0]
            neg_features = [(int(idx), float(corrs[idx])) for idx in neg_indices]
            neg_features.sort(key=lambda x: x[1])
            
            self.dim_to_features[dim_name] = {
                'positive': pos_features[:self.top_k],
                'negative': neg_features[:self.top_k]
            }
        
        # 2. 单义特征识别
        self.feature_primary_dim = {}
        self.monosemantic_indices = set()
        
        for feat_idx in range(n_features):
            corrs = self.feature_dim_corr[feat_idx]
            abs_corrs = np.abs(corrs)
            
            max_idx = abs_corrs.argmax()
            max_corr = corrs[max_idx]
            
            sorted_abs = np.sort(abs_corrs)[::-1]
            if abs(max_corr) > self.threshold and len(sorted_abs) > 1:
                if sorted_abs[1] < 1e-8 or sorted_abs[0] > 1.5 * sorted_abs[1]:
                    dim_name = MFT_DIMENSIONS[max_idx]
                    polarity = 'positive' if max_corr > 0 else 'negative'
                    self.feature_primary_dim[feat_idx] = (dim_name, polarity, float(max_corr))
                    self.monosemantic_indices.add(feat_idx)
        
        # 3. 添加每个维度的 top 特征到单义集合（即使它们不严格单义）
        for dim_name in MFT_DIMENSIONS:
            for feat_idx, corr in self.dim_to_features[dim_name]['positive'][:self.top_k]:
                self.monosemantic_indices.add(feat_idx)
        
        # 4. 极性对比对
        self.polarity_contrast_pairs = []
        
        for virtue_dim, vice_dim in MFT_PAIRS:
            virtue_features = [f[0] for f in self.dim_to_features[virtue_dim]['positive'][:20]]
            vice_features = [f[0] for f in self.dim_to_features[vice_dim]['positive'][:20]]
            
            if virtue_features and vice_features:
                self.polarity_contrast_pairs.append({
                    'virtue_dim': virtue_dim,
                    'vice_dim': vice_dim,
                    'virtue_features': virtue_features,
                    'vice_features': vice_features
                })
                # 添加到单义集合
                self.monosemantic_indices.update(virtue_features)
                self.monosemantic_indices.update(vice_features)
        
        if self.logger:
            self.logger.info(f"Layer {self.layer}: {len(self.feature_primary_dim)} strictly monosemantic features")
            self.logger.info(f"Layer {self.layer}: {len(self.monosemantic_indices)} total features to finetune")
            self.logger.info(f"Layer {self.layer}: {len(self.polarity_contrast_pairs)} polarity pairs")
    
    def get_monosemantic_indices(self) -> Set[int]:
        """获取所有需要微调的特征索引"""
        return self.monosemantic_indices
    
    def get_dimension_feature_mask(self, device: torch.device) -> torch.Tensor:
        """获取维度-特征掩码矩阵"""
        n_features = self.feature_dim_corr.shape[0]
        mask = torch.zeros(len(MFT_DIMENSIONS), n_features, device=device)
        
        for dim_idx, dim_name in enumerate(MFT_DIMENSIONS):
            for feat_idx, corr in self.dim_to_features[dim_name]['positive']:
                mask[dim_idx, feat_idx] = abs(corr)
        
        # 归一化
        row_sums = mask.sum(dim=1, keepdim=True)
        mask = mask / (row_sums + 1e-8)
        
        return mask
    
    def get_polarity_feature_mask(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取极性特征掩码"""
        n_features = self.feature_dim_corr.shape[0]
        n_pairs = len(self.polarity_contrast_pairs)
        
        virtue_mask = torch.zeros(n_pairs, n_features, device=device)
        vice_mask = torch.zeros(n_pairs, n_features, device=device)
        
        for pair_idx, pair in enumerate(self.polarity_contrast_pairs):
            for feat_idx in pair['virtue_features']:
                virtue_mask[pair_idx, feat_idx] = 1.0
            for feat_idx in pair['vice_features']:
                vice_mask[pair_idx, feat_idx] = 1.0
        
        # 归一化
        virtue_mask = virtue_mask / (virtue_mask.sum(dim=1, keepdim=True) + 1e-8)
        vice_mask = vice_mask / (vice_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        return virtue_mask, vice_mask
    
    def get_monosemantic_feature_info(self, device: torch.device) -> Dict:
        """获取单义特征信息用于正则化"""
        n_features = self.feature_dim_corr.shape[0]
        
        mono_mask = torch.zeros(n_features, device=device)
        target_dims = torch.zeros(n_features, dtype=torch.long, device=device)
        target_signs = torch.zeros(n_features, device=device)
        
        for feat_idx, (dim_name, polarity, corr) in self.feature_primary_dim.items():
            mono_mask[feat_idx] = 1.0
            target_dims[feat_idx] = DIM_TO_IDX[dim_name]
            target_signs[feat_idx] = 1.0 if polarity == 'positive' else -1.0
        
        return {
            'mask': mono_mask,
            'target_dims': target_dims,
            'target_signs': target_signs,
            'n_monosemantic': int(mono_mask.sum().item())
        }


# ============================================================================
# Dataset
# ============================================================================

class MoralActivationDataset(Dataset):
    """道德激活数据集"""
    
    def __init__(self, h5_path: str, layer: int, 
                 center: torch.Tensor,
                 bucket_path: Optional[str] = None,
                 preload: bool = True):
        self.h5_path = h5_path
        self.layer = layer
        self.center = center
        self.preload = preload
        
        with h5py.File(h5_path, 'r') as f:
            if preload:
                self.activations = torch.from_numpy(
                    f[f'activations/layer_{layer}'][:].astype(np.float32)
                )
                self.moral_vectors = torch.from_numpy(
                    f['metadata/moral_vector'][:].astype(np.float32)
                )
                self.m_virtue = torch.from_numpy(
                    f['metadata/m_virtue'][:].astype(np.float32)
                )
                self.m_vice = torch.from_numpy(
                    f['metadata/m_vice'][:].astype(np.float32)
                )
                self.target_dimensions = f['metadata/target_dimension'][:].astype(str)
                
                # 应用中心化
                self.activations = self.activations - center.unsqueeze(0)
            else:
                self.n_samples = f[f'activations/layer_{layer}'].shape[0]
                self.activations = None
        
        if preload:
            self._compute_sample_weights()
    
    def _compute_sample_weights(self):
        """计算平衡采样权重"""
        n_samples = len(self.activations)
        weights = torch.ones(n_samples)
        
        typicality = torch.maximum(self.m_virtue, self.m_vice)
        
        weights[typicality > 0.75] = 3.0
        weights[typicality > 0.5] = 2.0
        weights[(typicality > 0.25) & (typicality <= 0.5)] = 1.5
        weights[typicality == 0] = 0.3
        
        virtue_high = self.m_virtue > self.m_vice
        vice_high = self.m_vice > self.m_virtue
        
        n_virtue = virtue_high.sum().float()
        n_vice = vice_high.sum().float()
        
        if n_virtue > 0 and n_vice > 0:
            ratio = n_virtue / n_vice
            if ratio > 1.5:
                weights[vice_high] *= min(ratio, 3.0)
            elif ratio < 0.67:
                weights[virtue_high] *= min(1/ratio, 3.0)
        
        self.sample_weights = weights
    
    def __len__(self) -> int:
        return len(self.activations) if self.preload else self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.preload:
            return {
                'activation': self.activations[idx],
                'moral_vector': self.moral_vectors[idx],
                'm_virtue': self.m_virtue[idx],
                'm_vice': self.m_vice[idx],
                'idx': idx
            }
        else:
            raise NotImplementedError("Lazy loading not implemented for finetuning")


# ============================================================================
# Loss Functions
# ============================================================================

class AlignmentLoss(nn.Module):
    """维度对齐损失"""
    
    def __init__(self, feature_manager: FeatureAlignmentManager, device: torch.device):
        super().__init__()
        self.dim_mask = feature_manager.get_dimension_feature_mask(device)
        self.register_buffer('_dim_mask', self.dim_mask)
    
    def forward(self, z: torch.Tensor, moral_vectors: torch.Tensor) -> torch.Tensor:
        dim_activations = torch.mm(z, self._dim_mask.t())
        valid_mask = moral_vectors > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)
        
        dim_activations_scaled = torch.sigmoid(dim_activations)
        
        loss = F.mse_loss(
            dim_activations_scaled[valid_mask],
            moral_vectors[valid_mask]
        )
        
        return loss


class PolarityContrastLoss(nn.Module):
    """极性对比损失"""
    
    def __init__(self, feature_manager: FeatureAlignmentManager, 
                 device: torch.device, 
                 margin: float = 0.5):
        super().__init__()
        virtue_mask, vice_mask = feature_manager.get_polarity_feature_mask(device)
        self.register_buffer('virtue_mask', virtue_mask)
        self.register_buffer('vice_mask', vice_mask)
        self.margin = margin
        self.n_pairs = virtue_mask.shape[0]
    
    def forward(self, z: torch.Tensor, m_virtue: torch.Tensor, 
                m_vice: torch.Tensor) -> torch.Tensor:
        if self.n_pairs == 0:
            return torch.tensor(0.0, device=z.device)
        
        high_virtue_mask = m_virtue > 0.3
        high_vice_mask = m_vice > 0.3
        
        n_virtue = high_virtue_mask.sum()
        n_vice = high_vice_mask.sum()
        
        if n_virtue < 4 or n_vice < 4:
            return torch.tensor(0.0, device=z.device)
        
        virtue_samples = z[high_virtue_mask]
        vice_samples = z[high_vice_mask]
        
        total_loss = 0.0
        
        for pair_idx in range(self.n_pairs):
            v_mask = self.virtue_mask[pair_idx]
            c_mask = self.vice_mask[pair_idx]
            
            v_on_virtue = (virtue_samples * v_mask).sum(dim=1)
            v_on_vice = (virtue_samples * c_mask).sum(dim=1)
            
            c_on_virtue = (vice_samples * v_mask).sum(dim=1)
            c_on_vice = (vice_samples * c_mask).sum(dim=1)
            
            loss_virtue = F.relu(self.margin - v_on_virtue + v_on_vice).mean()
            loss_vice = F.relu(self.margin - c_on_vice + c_on_virtue).mean()
            
            total_loss = total_loss + loss_virtue + loss_vice
        
        return total_loss / (2 * self.n_pairs)


class PrototypicalityLoss(nn.Module):
    """典型性保持损失"""
    
    def __init__(self, feature_manager: FeatureAlignmentManager, device: torch.device):
        super().__init__()
        self.dim_mask = feature_manager.get_dimension_feature_mask(device)
        self.register_buffer('_dim_mask', self.dim_mask)
    
    def forward(self, z: torch.Tensor, moral_vectors: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        count = 0
        
        for dim_idx in range(10):
            dim_values = moral_vectors[:, dim_idx]
            valid_mask = dim_values > 0
            
            if valid_mask.sum() < 16:
                continue
            
            valid_z = z[valid_mask]
            valid_values = dim_values[valid_mask]
            
            dim_features = self._dim_mask[dim_idx]
            feature_activation = (valid_z * dim_features).sum(dim=1)
            
            n_valid = valid_mask.sum().item()
            n_pairs = min(100, n_valid * (n_valid - 1) // 2)
            
            if n_pairs < 10:
                continue
            
            indices = torch.randperm(n_valid, device=z.device)
            idx1 = indices[:n_pairs]
            idx2 = indices[n_pairs:2*n_pairs] if 2*n_pairs <= n_valid else indices[:n_pairs].roll(1)
            
            value_diff = valid_values[idx1] - valid_values[idx2]
            activation_diff = feature_activation[idx1] - feature_activation[idx2]
            
            sign_agreement = value_diff * activation_diff
            loss = F.relu(-sign_agreement + 0.1).mean()
            
            total_loss = total_loss + loss
            count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=z.device)
        
        return total_loss / count


class MonosemanticRegularization(nn.Module):
    """单义特征正则化"""
    
    def __init__(self, feature_manager: FeatureAlignmentManager, device: torch.device):
        super().__init__()
        mono_info = feature_manager.get_monosemantic_feature_info(device)
        self.register_buffer('mono_mask', mono_info['mask'])
        self.register_buffer('target_dims', mono_info['target_dims'])
        self.n_mono = mono_info['n_monosemantic']
        self.dim_mask = feature_manager.get_dimension_feature_mask(device)
    
    def forward(self, z: torch.Tensor, moral_vectors: torch.Tensor) -> torch.Tensor:
        if self.n_mono == 0:
            return torch.tensor(0.0, device=z.device)
        
        mono_indices = torch.where(self.mono_mask > 0)[0]
        mono_activations = z[:, mono_indices]
        
        sample_primary_dim = moral_vectors.argmax(dim=1)
        feature_target_dims = self.target_dims[mono_indices]
        
        match_matrix = (sample_primary_dim.unsqueeze(1) == feature_target_dims.unsqueeze(0)).float()
        
        mismatch_activation = mono_activations * (1 - match_matrix)
        loss = mismatch_activation.pow(2).mean()
        
        return loss


# ============================================================================
# Trainer
# ============================================================================

class SAEFinetuner:
    """SAE 微调训练器 - 只微调单义神经元"""
    
    def __init__(self, 
                 sae_wrapper: MonosemanticSAEWrapper,
                 feature_manager: FeatureAlignmentManager,
                 config: FinetuneConfig,
                 layer: int,
                 device: torch.device,
                 logger: logging.Logger):
        self.sae_wrapper = sae_wrapper
        self.feature_manager = feature_manager
        self.config = config
        self.layer = layer
        self.device = device
        self.logger = logger
        
        # 优化器 - 只优化单义特征的参数
        self.optimizer = torch.optim.AdamW(
            sae_wrapper.get_trainable_params(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 损失函数
        self.align_loss_fn = AlignmentLoss(feature_manager, device)
        self.polarity_loss_fn = PolarityContrastLoss(
            feature_manager, device, margin=config.polarity_margin
        )
        self.proto_loss_fn = PrototypicalityLoss(feature_manager, device)
        self.mono_reg_fn = MonosemanticRegularization(feature_manager, device)
        
        # 混合精度
        self.scaler = GradScaler('cuda')
        
        # 早停
        self.best_val_loss = float('inf')
        self.best_val_recon = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # 学习率调度器
        self.scheduler = None
        
    def setup_scheduler(self, num_training_steps: int):
        """设置学习率调度器"""
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=num_training_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算所有损失"""
        x = batch['activation'].to(self.device)
        moral_vector = batch['moral_vector'].to(self.device)
        m_virtue = batch['m_virtue'].to(self.device)
        m_vice = batch['m_vice'].to(self.device)
        
        # 前向传播
        x_rec, z = self.sae_wrapper(x)
        
        # 1. 重建损失
        loss_recon = F.mse_loss(x_rec, x)
        
        # 2. 稀疏损失
        loss_sparse = z.abs().mean()
        
        # 3. 对齐损失
        loss_align = self.align_loss_fn(z, moral_vector)
        
        # 4. 极性对比损失
        loss_polarity = self.polarity_loss_fn(z, m_virtue, m_vice)
        
        # 5. 典型性损失
        loss_proto = self.proto_loss_fn(z, moral_vector)
        
        # 6. 单义特征正则化
        loss_mono = self.mono_reg_fn(z, moral_vector)
        
        # 总损失
        total_loss = (
            self.config.lambda_recon * loss_recon +
            self.config.lambda_sparse * loss_sparse +
            self.config.lambda_align * loss_align +
            self.config.lambda_polarity * loss_polarity +
            self.config.lambda_proto * loss_proto +
            self.config.lambda_mono * loss_mono
        )
        
        # 统计
        with torch.no_grad():
            l0 = (z > 0).float().sum(dim=1).mean().item()
            frac_alive = (z > 0).any(dim=0).float().mean().item()
        
        metrics = {
            'total_loss': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_sparse': loss_sparse.item(),
            'loss_align': loss_align.item(),
            'loss_polarity': loss_polarity.item(),
            'loss_proto': loss_proto.item(),
            'loss_mono': loss_mono.item(),
            'l0': l0,
            'frac_alive': frac_alive
        }
        
        return total_loss, metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.sae_wrapper.train()
        all_metrics = []
    
        pbar = tqdm(dataloader, desc=f"Layer {self.layer} Epoch {epoch}", 
                    leave=False, position=1)
    
        for batch in pbar:
            self.optimizer.zero_grad()
        
            with autocast('cuda'):
                loss, metrics = self.compute_loss(batch)
        
            self.scaler.scale(loss).backward()
        
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.sae_wrapper.get_trainable_params(), 
                self.config.gradient_clip
            )
        
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
        
            # 只有当optimizer真正更新时才step scheduler
            if self.scheduler is not None and old_scale <= new_scale:
                self.scheduler.step()
        
            # 归一化单义特征的 decoder
            self.sae_wrapper.normalize_decoder()
        
            all_metrics.append(metrics)
            
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'rec': f"{metrics['loss_recon']:.4f}",
                'align': f"{metrics['loss_align']:.4f}"
            })
        
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证集评估"""
        self.sae_wrapper.eval()
        all_metrics = []
        
        for batch in dataloader:
            _, metrics = self.compute_loss(batch)
            all_metrics.append(metrics)
        
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
        
        # 添加权重变化统计
        weight_stats = self.sae_wrapper.get_weight_change_stats()
        avg_metrics.update(weight_stats)
        
        return avg_metrics
    
    def check_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """检查是否早停"""
        val_loss = val_metrics['total_loss']
        val_recon = val_metrics['loss_recon']
        
        improved = False
        
        if val_loss < self.best_val_loss - self.config.min_delta:
            self.best_val_loss = val_loss
            improved = True
        
        if val_recon < self.best_val_recon:
            self.best_val_recon = val_recon
        
        if improved:
            self.patience_counter = 0
            # 保存可训练参数的状态
            self.best_model_state = {
                'encoder_weight_mono': self.sae_wrapper.encoder_weight_mono.cpu().clone(),
                'encoder_bias_mono': self.sae_wrapper.encoder_bias_mono.cpu().clone(),
                'decoder_weight_mono': self.sae_wrapper.decoder_weight_mono.cpu().clone(),
            }
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def load_best_model(self):
        """加载最佳模型"""
        if self.best_model_state is not None:
            self.sae_wrapper.encoder_weight_mono.data = self.best_model_state['encoder_weight_mono'].to(self.device)
            self.sae_wrapper.encoder_bias_mono.data = self.best_model_state['encoder_bias_mono'].to(self.device)
            self.sae_wrapper.decoder_weight_mono.data = self.best_model_state['decoder_weight_mono'].to(self.device)


# ============================================================================
# Single Layer Training
# ============================================================================

def finetune_single_layer(layer: int,
                          config: FinetuneConfig,
                          device: torch.device,
                          logger: logging.Logger) -> Dict:
    """微调单层 SAE"""
    
    logger.info(f"{'='*60}")
    logger.info(f"Finetuning SAE for Layer {layer}")
    logger.info(f"{'='*60}")
    
    output_dir = Path(config.output_dir) / f"layer_{layer}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载预训练 SAE
    pretrained_path = Path(config.pretrained_sae_dir) / f"layer_{layer}" / "sae_final.pt"
    if not pretrained_path.exists():
        logger.error(f"Pretrained SAE not found: {pretrained_path}")
        return {'layer': layer, 'status': 'error', 'error': 'SAE not found'}
    
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    
    sae = SparseAutoencoder(config.input_dim, config.hidden_dim)
    sae.load_state_dict(checkpoint['model_state'])
    center = checkpoint['center']
    
    pretrain_metrics = checkpoint.get('final_metrics', {})
    logger.info(f"Loaded pretrained SAE from {pretrained_path}")
    if pretrain_metrics:
        logger.info(f"Pretrain recon: {pretrain_metrics.get('val', {}).get('loss_rec', 'N/A')}")
    
    # 2. 加载特征对齐管理器
    try:
        feature_manager = FeatureAlignmentManager(
            config.analysis_dir, layer,
            top_k=config.top_k_aligned_features,
            threshold=config.alignment_threshold,
            logger=logger
        )
    except FileNotFoundError as e:
        logger.error(f"Analysis results not found: {e}")
        return {'layer': layer, 'status': 'error', 'error': str(e)}
    
    # 3. 获取单义特征索引并创建包装器
    mono_indices = feature_manager.get_monosemantic_indices()
    
    if len(mono_indices) == 0:
        logger.warning(f"No monosemantic features found for layer {layer}, skipping")
        return {
            'layer': layer,
            'status': 'skipped',
            'reason': 'No monosemantic features found'
        }
    
    logger.info(f"Creating wrapper with {len(mono_indices)} monosemantic features")
    
    sae = sae.to(device)
    sae_wrapper = MonosemanticSAEWrapper(sae, mono_indices, device, logger)
    
    # 4. 创建数据集
    logger.info("Loading datasets...")
    train_dataset = MoralActivationDataset(
        config.train_h5_path, layer, center,
        bucket_path=config.bucket_path
    )
    val_dataset = MoralActivationDataset(
        config.val_h5_path, layer, center
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    sampler = WeightedRandomSampler(
        train_dataset.sample_weights.tolist(),
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 5. 创建训练器
    trainer = SAEFinetuner(sae_wrapper, feature_manager, config, layer, device, logger)
    
    num_training_steps = len(train_loader) * config.num_epochs
    trainer.setup_scheduler(num_training_steps)
    
    # 6. 训练循环
    all_metrics = []
    best_epoch = 0
    early_stopped = False
    
    init_val_metrics = trainer.evaluate(val_loader)
    logger.info(f"Initial val - recon: {init_val_metrics['loss_recon']:.6f}, "
                f"align: {init_val_metrics['loss_align']:.4f}, "
                f"polarity: {init_val_metrics['loss_polarity']:.4f}")
    
    for epoch in range(config.num_epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        train_metrics['epoch'] = epoch
        train_metrics['split'] = 'train'
        all_metrics.append(train_metrics.copy())
        
        val_metrics = trainer.evaluate(val_loader)
        val_metrics['epoch'] = epoch
        val_metrics['split'] = 'val'
        all_metrics.append(val_metrics.copy())
        
        logger.info(
            f"Epoch {epoch}: "
            f"train={train_metrics['total_loss']:.4f}, "
            f"val={val_metrics['total_loss']:.4f}, "
            f"recon={val_metrics['loss_recon']:.6f}, "
            f"align={val_metrics['loss_align']:.4f}, "
            f"polar={val_metrics['loss_polarity']:.4f}, "
            f"l0={val_metrics['l0']:.1f}, "
            f"enc_Δ={val_metrics.get('encoder_weight_change_mean', 0):.6f}"
        )
        
        if trainer.best_model_state is not None:
            best_epoch = epoch
        
        if trainer.check_early_stop(val_metrics):
            logger.info(f"Early stopping at epoch {epoch}")
            early_stopped = True
            break
    
    # 7. 加载最佳模型并保存
    trainer.load_best_model()
    final_val_metrics = trainer.evaluate(val_loader)
    
    # 获取微调后的完整 SAE
    finetuned_sae = sae_wrapper.get_finetuned_sae()
    
    # 保存模型
    torch.save({
        'model_state': finetuned_sae.state_dict(),
        'center': center,
        'config': asdict(config),
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'initial_metrics': init_val_metrics,
        'final_metrics': final_val_metrics,
        'pretrain_metrics': pretrain_metrics,
        'monosemantic_indices': sorted(list(mono_indices)),
        'n_finetuned_features': len(mono_indices),
        'weight_change_stats': sae_wrapper.get_weight_change_stats()
    }, output_dir / "sae_finetuned.pt")
    
    # 保存训练历史
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # 计算改进
    recon_change = final_val_metrics['loss_recon'] - init_val_metrics['loss_recon']
    align_change = final_val_metrics['loss_align'] - init_val_metrics['loss_align']
    polar_change = final_val_metrics['loss_polarity'] - init_val_metrics['loss_polarity']
    
    logger.info(f"Layer {layer} finetuning complete.")
    logger.info(f"  Features finetuned: {len(mono_indices)} / {config.hidden_dim}")
    logger.info(f"  Best epoch: {best_epoch}, Early stopped: {early_stopped}")
    logger.info(f"  Recon: {init_val_metrics['loss_recon']:.6f} -> {final_val_metrics['loss_recon']:.6f} ({recon_change:+.6f})")
    logger.info(f"  Align: {init_val_metrics['loss_align']:.4f} -> {final_val_metrics['loss_align']:.4f} ({align_change:+.4f})")
    logger.info(f"  Polar: {init_val_metrics['loss_polarity']:.4f} -> {final_val_metrics['loss_polarity']:.4f} ({polar_change:+.4f})")
    
    weight_stats = sae_wrapper.get_weight_change_stats()
    logger.info(f"  Weight changes - encoder: {weight_stats['encoder_weight_change_mean']:.6f} (max: {weight_stats['encoder_weight_change_max']:.6f})")
    logger.info(f"  Weight changes - decoder: {weight_stats['decoder_weight_change_mean']:.6f} (max: {weight_stats['decoder_weight_change_max']:.6f})")
    
    return {
        'layer': layer,
        'status': 'success',
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'n_finetuned_features': len(mono_indices),
        'initial_metrics': init_val_metrics,
        'final_metrics': final_val_metrics,
        'weight_change_stats': weight_stats,
        'improvements': {
            'recon_change': recon_change,
            'align_change': align_change,
            'polarity_change': polar_change
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
    """单个 GPU 的 worker 进程"""
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    
    log_dir = Path(config_dict['output_dir']) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'[GPU {gpu_id}] %(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"gpu_{gpu_id}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f"GPU_{gpu_id}")
    
    config = FinetuneConfig(**config_dict)
    
    logger.info(f"GPU {gpu_id} will finetune layers: {layers}")
    status_queue.put((gpu_id, -1, 'started', f"Processing layers: {layers}"))
    
    for layer in layers:
        status_queue.put((gpu_id, layer, 'started', None))
        
        try:
            result = finetune_single_layer(layer, config, device, logger)
            result_queue.put(result)
            
            if result['status'] == 'success':
                status_msg = f"recon={result['final_metrics']['loss_recon']:.6f}, n_feat={result['n_finetuned_features']}"
            else:
                status_msg = result.get('reason', 'Unknown')
            status_queue.put((gpu_id, layer, 'done', status_msg))
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"Error finetuning layer {layer}: {error_msg}")
            
            result_queue.put({
                'layer': layer,
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            status_queue.put((gpu_id, layer, 'error', str(e)[:100]))
            
            torch.cuda.empty_cache()
    
    status_queue.put((gpu_id, -1, 'finished', None))


# ============================================================================
# Main Parallel Training
# ============================================================================

def distribute_layers_to_gpus(layers: List[int], num_gpus: int) -> Dict[int, List[int]]:
    """将层分配给各个 GPU"""
    gpu_assignments = {i: [] for i in range(num_gpus)}
    
    for i, layer in enumerate(layers):
        gpu_id = i % num_gpus
        gpu_assignments[gpu_id].append(layer)
    
    return gpu_assignments


def run_parallel_finetuning(config: FinetuneConfig, 
                            logger: logging.Logger) -> List[Dict]:
    """使用多 GPU 并行微调所有层"""
    
    layers = config.layers
    num_gpus = min(config.num_gpus, torch.cuda.device_count())
    
    logger.info(f"Running parallel finetuning on {num_gpus} GPUs")
    logger.info(f"Total layers to finetune: {len(layers)}")
    
    gpu_assignments = distribute_layers_to_gpus(layers, num_gpus)
    
    for gpu_id, assigned_layers in gpu_assignments.items():
        logger.info(f"  GPU {gpu_id}: layers {assigned_layers}")
    
    config_dict = asdict(config)
    
    result_queue = mp.Queue()
    status_queue = mp.Queue()
    
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
    
    all_results = []
    completed = 0
    total = len(layers)
    finished_gpus = set()
    
    print(f"\n{'='*70}")
    print(f"Finetuning {total} layers on {num_gpus} GPUs (Monosemantic features only)")
    print(f"{'='*70}\n")
    
    while completed < total or len(finished_gpus) < len(processes):
        while not status_queue.empty():
            try:
                gpu_id, layer, status, msg = status_queue.get_nowait()
                
                if status == 'finished':
                    finished_gpus.add(gpu_id)
                    logger.info(f"GPU {gpu_id} finished all tasks")
                elif status == 'started' and layer >= 0:
                    logger.info(f"[GPU {gpu_id}] Starting layer {layer}")
                elif status == 'done':
                    logger.info(f"[GPU {gpu_id}] Completed layer {layer}: {msg}")
                elif status == 'error':
                    logger.error(f"[GPU {gpu_id}] Error on layer {layer}: {msg}")
            except:
                break
        
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                all_results.append(result)
                completed += 1
                
                layer = result['layer']
                if result['status'] == 'success':
                    imp = result.get('improvements', {})
                    n_feat = result.get('n_finetuned_features', 0)
                    print(f"[{completed:2d}/{total}] Layer {layer:2d}: "
                          f"n_feat={n_feat:4d}, "
                          f"recon Δ={imp.get('recon_change', 0):+.6f}, "
                          f"align Δ={imp.get('align_change', 0):+.4f}, "
                          f"polar Δ={imp.get('polarity_change', 0):+.4f}")
                elif result['status'] == 'skipped':
                    print(f"[{completed:2d}/{total}] Layer {layer:2d}: SKIPPED - {result.get('reason', 'Unknown')}")
                else:
                    print(f"[{completed:2d}/{total}] Layer {layer:2d}: FAILED - {result.get('error', 'Unknown')[:50]}")
            except:
                break
        
        time.sleep(0.5)
    
    for gpu_id, p in processes:
        p.join(timeout=30)
        if p.is_alive():
            logger.warning(f"Force terminating GPU {gpu_id} worker")
            p.terminate()
    
    all_results.sort(key=lambda x: x['layer'])
    
    return all_results


def create_summary_report(all_results: List[Dict], 
                          config: FinetuneConfig,
                          logger: logging.Logger) -> Dict:
    """创建汇总报告"""
    
    logger.info("\n" + "="*70)
    logger.info("Creating Summary Report")
    logger.info("="*70)
    
    successful = [r for r in all_results if r.get('status') == 'success']
    skipped = [r for r in all_results if r.get('status') == 'skipped']
    failed = [r for r in all_results if r.get('status') == 'error']
    
    summary = {
        'config': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'n_layers_total': len(all_results),
        'n_layers_success': len(successful),
        'n_layers_skipped': len(skipped),
        'n_layers_failed': len(failed),
        'skipped_layers': [r['layer'] for r in skipped],
        'failed_layers': [r['layer'] for r in failed],
        'per_layer_results': []
    }
    
    if successful:
        recon_changes = [r['improvements']['recon_change'] for r in successful]
        align_changes = [r['improvements']['align_change'] for r in successful]
        polar_changes = [r['improvements']['polarity_change'] for r in successful]
        n_features = [r['n_finetuned_features'] for r in successful]
        
        summary['aggregate_stats'] = {
            'total_finetuned_features': sum(n_features),
            'avg_finetuned_features_per_layer': float(np.mean(n_features)),
            'recon_change_mean': float(np.mean(recon_changes)),
            'recon_change_std': float(np.std(recon_changes)),
            'align_change_mean': float(np.mean(align_changes)),
            'align_change_std': float(np.std(align_changes)),
            'polarity_change_mean': float(np.mean(polar_changes)),
            'polarity_change_std': float(np.std(polar_changes))
        }
        
        align_improvements = [(r['layer'], -r['improvements']['align_change'], r['n_finetuned_features']) 
                              for r in successful]
        align_improvements.sort(key=lambda x: x[1], reverse=True)
        
        summary['best_align_improvement_layers'] = align_improvements[:5]
        summary['worst_align_improvement_layers'] = align_improvements[-5:]
        
        for r in successful:
            summary['per_layer_results'].append({
                'layer': r['layer'],
                'best_epoch': r['best_epoch'],
                'early_stopped': r['early_stopped'],
                'n_finetuned_features': r['n_finetuned_features'],
                'initial_recon': r['initial_metrics']['loss_recon'],
                'final_recon': r['final_metrics']['loss_recon'],
                'initial_align': r['initial_metrics']['loss_align'],
                'final_align': r['final_metrics']['loss_align'],
                'initial_polarity': r['initial_metrics']['loss_polarity'],
                'final_polarity': r['final_metrics']['loss_polarity'],
                'weight_change_stats': r.get('weight_change_stats', {}),
                'improvements': r['improvements']
            })
        
        logger.info(f"\nAggregate Statistics (n={len(successful)} layers):")
        logger.info(f"  Total finetuned features: {summary['aggregate_stats']['total_finetuned_features']}")
        logger.info(f"  Avg features per layer:   {summary['aggregate_stats']['avg_finetuned_features_per_layer']:.1f}")
        logger.info(f"  Recon change:    {summary['aggregate_stats']['recon_change_mean']:+.6f} ± {summary['aggregate_stats']['recon_change_std']:.6f}")
        logger.info(f"  Align change:    {summary['aggregate_stats']['align_change_mean']:+.4f} ± {summary['aggregate_stats']['align_change_std']:.4f}")
        logger.info(f"  Polarity change: {summary['aggregate_stats']['polarity_change_mean']:+.4f} ± {summary['aggregate_stats']['polarity_change_std']:.4f}")
        
        logger.info(f"\nTop 5 layers by alignment improvement:")
        for layer, imp, n_feat in summary['best_align_improvement_layers']:
            logger.info(f"  Layer {layer}: {imp:+.4f} ({n_feat} features)")
    
    if skipped:
        logger.warning(f"\nSkipped layers (no monosemantic features): {[r['layer'] for r in skipped]}")
    
    if failed:
        logger.warning(f"\nFailed layers: {[r['layer'] for r in failed]}")
        for r in failed:
            logger.warning(f"  Layer {r['layer']}: {r.get('error', 'Unknown error')}")
    
    return summary


def main():
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="SAE Moral Alignment Finetuning (Multi-GPU, Monosemantic Features Only)"
    )
    
    # 路径参数
    parser.add_argument("--pretrained_dir", type=str, default="./outputs/sae_pretrain")
    parser.add_argument("--analysis_dir", type=str, default="./outputs/sae_analysis")
    parser.add_argument("--train_h5", type=str, 
                        default="./data/activations_qwen3_8b/qwen3_8b_train_activations.h5")
    parser.add_argument("--val_h5", type=str,
                        default="./data/activations_qwen3_8b/qwen3_8b_val_activations.h5")
    parser.add_argument("--bucket_path", type=str, default="./data/train/train_buckets.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/sae_finetune")
    
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=16384)
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    
    # 损失权重
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=1e-4)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_polarity", type=float, default=0.3)
    parser.add_argument("--lambda_proto", type=float, default=0.2)
    parser.add_argument("--lambda_mono", type=float, default=0.1)
    
    # 特征选择
    parser.add_argument("--top_k_features", type=int, default=50)
    parser.add_argument("--align_threshold", type=float, default=0.1)
    
    # 并行配置
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    if args.layers == "all":
        layers = list(range(36))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No GPU available!")
    
    num_gpus = min(args.num_gpus, available_gpus)
    print(f"Available GPUs: {available_gpus}, Using: {num_gpus}")
    
    config = FinetuneConfig(
        pretrained_sae_dir=args.pretrained_dir,
        analysis_dir=args.analysis_dir,
        train_h5_path=args.train_h5,
        val_h5_path=args.val_h5,
        bucket_path=args.bucket_path,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        lambda_recon=args.lambda_recon,
        lambda_sparse=args.lambda_sparse,
        lambda_align=args.lambda_align,
        lambda_polarity=args.lambda_polarity,
        lambda_proto=args.lambda_proto,
        lambda_mono=args.lambda_mono,
        top_k_aligned_features=args.top_k_features,
        alignment_threshold=args.align_threshold,
        num_gpus=num_gpus,
        num_workers=args.num_workers,
        layers=layers
    )
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    log_path = Path(config.output_dir) / "finetune_main.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("main")
    
    logger.info("SAE Moral Alignment Finetuning (Monosemantic Features Only)")
    logger.info(f"Layers: {layers}")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Config: {json.dumps(asdict(config), indent=2)}")
    
    with open(Path(config.output_dir) / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # 多卡并行微调
    all_results = run_parallel_finetuning(config, logger)
    
    # 创建汇总报告
    summary = create_summary_report(all_results, config, logger)
    
    summary_path = Path(config.output_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    results_path = Path(config.output_dir) / "all_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*70}")
    logger.info("Finetuning Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {config.output_dir}")
    logger.info(f"Summary: {summary_path}")
    
    print(f"\n{'='*70}")
    print("Final Statistics")
    print(f"{'='*70}")
    print(f"Successful: {summary['n_layers_success']}/{summary['n_layers_total']}")
    print(f"Skipped:    {summary['n_layers_skipped']}/{summary['n_layers_total']}")
    print(f"Failed:     {summary['n_layers_failed']}/{summary['n_layers_total']}")
    
    if summary.get('aggregate_stats'):
        print(f"\nAggregate Statistics:")
        print(f"  Total finetuned features: {summary['aggregate_stats']['total_finetuned_features']}")
        print(f"  Avg per layer:            {summary['aggregate_stats']['avg_finetuned_features_per_layer']:.1f}")
        print(f"\nAverage Improvements:")
        print(f"  Reconstruction: {summary['aggregate_stats']['recon_change_mean']:+.6f}")
        print(f"  Alignment:      {summary['aggregate_stats']['align_change_mean']:+.4f}")
        print(f"  Polarity:       {summary['aggregate_stats']['polarity_change_mean']:+.4f}")


if __name__ == "__main__":
    main()