#!/usr/bin/env python3
"""
SAE Pretraining Script with Activation Centering
=================================================
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SAEConfig:
    """SAE 架构配置"""
    input_dim: int = 4096
    hidden_dim: int = 16384  # 4x expansion
    activation: str = "relu"  # relu or gelu
    tied_weights: bool = False  # 是否共享 encoder/decoder 权重
    normalize_decoder: bool = True  # 是否归一化 decoder 列向量
    
@dataclass
class TrainConfig:
    """训练配置"""
    # 数据路径
    train_h5_path: str = "./data/activations_qwen3_8b/qwen3_8b_train_activations.h5"
    val_h5_path: str = "./data/activations_qwen3_8b/qwen3_8b_val_activations.h5"
    output_dir: str = "./outputs/sae_pretrain"
    
    # 训练超参
    batch_size: int = 4096
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    warmup_epochs: int = 5
    
    # 稀疏约束
    sparsity_coef: float = 1e-3  # L1 系数
    target_l0: float = 50.0  # 目标平均激活数（用于动态调整 sparsity_coef）
    
    # 早停配置
    early_stop_recon_threshold: float = 0.001  # 重建损失阈值
    early_stop_l0_tolerance: float = 0.1  # L0 容差比例 (target_l0 * (1 ± tolerance))
    early_stop_patience: int = 5  # 连续多少个 epoch 满足条件才停止
    
    # 中心化
    compute_center: bool = True  # 是否计算中心
    
    # 系统配置
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    
    # 日志
    log_interval: int = 50
    eval_interval: int = 1  # 每 N epochs 评估
    
    # 层配置
    layers: List[int] = field(default_factory=lambda: list(range(36)))


# ============================================================================
# Activation Centering
# ============================================================================

def compute_global_center(h5_path: str, layer: int, 
                          batch_size: int = 10000) -> torch.Tensor:
    """
    预计算某层的全局激活中心（一次遍历）
    
    Args:
        h5_path: H5 文件路径
        layer: 层号
        batch_size: 批处理大小
        
    Returns:
        center: (dim,) 全局中心向量
    """
    with h5py.File(h5_path, 'r') as f:
        data = f[f'activations/layer_{layer}']
        n_samples = data.shape[0]
        dim = data.shape[1]
        
        # 使用 Welford's online algorithm 计算稳定的均值
        mean = np.zeros(dim, dtype=np.float64)
        count = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = data[start:end][:].astype(np.float64)
            batch_size_actual = batch.shape[0]
            
            delta = batch.mean(axis=0) - mean
            count += batch_size_actual
            mean += delta * batch_size_actual / count
            
    return torch.from_numpy(mean.astype(np.float32))


# ============================================================================
# SAE Architecture
# ============================================================================

class SparseAutoencoder(nn.Module):
    """
    稀疏自编码器
    
    架构: x -> Encoder -> ReLU/GELU -> Decoder -> x_rec
    
    特点:
    - L1 稀疏约束
    - 可选的 decoder 权重归一化
    - 可选的权重绑定 (decoder = encoder.T)
    """
    
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        
        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        
        # Decoder: hidden_dim -> input_dim
        if config.tied_weights:
            self.decoder = None  # 使用 encoder 的转置
        else:
            self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=True)
            
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """Kaiming 初始化"""
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        
        if self.decoder is not None:
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')
            nn.init.zeros_(self.decoder.bias)
            
    def normalize_decoder(self):
        """归一化 decoder 的列向量（特征方向）"""
        if self.decoder is not None and self.config.normalize_decoder:
            with torch.no_grad():
                # decoder.weight: (input_dim, hidden_dim)
                # 每列对应一个隐藏特征的重建方向
                norms = self.decoder.weight.norm(dim=0, keepdim=True)
                self.decoder.weight.div_(norms.clamp(min=1e-8))
                
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到稀疏表示"""
        z = self.encoder(x)
        
        if self.config.activation == "relu":
            z = F.relu(z)
        elif self.config.activation == "gelu":
            z = F.gelu(z)
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")
            
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从稀疏表示重建"""
        if self.config.tied_weights:
            return F.linear(z, self.encoder.weight.t(), bias=None)
        else:
            return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            x_rec: 重建激活
            z: 稀疏表示
        """
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z
    
    def get_features(self) -> torch.Tensor:
        """获取 decoder 的特征向量（每列是一个特征）"""
        if self.config.tied_weights:
            return self.encoder.weight.t()  # (input_dim, hidden_dim)
        else:
            return self.decoder.weight  # (input_dim, hidden_dim)


# ============================================================================
# Dataset
# ============================================================================

class ActivationDataset(Dataset):
    """
    激活数据集
    
    支持两种模式:
    1. 预加载到内存（适合数据量不大时）
    2. 按需从 H5 读取（适合数据量很大时）
    """
    
    def __init__(self, h5_path: str, layer: int, 
                 center: Optional[torch.Tensor] = None,
                 preload: bool = True):
        self.h5_path = h5_path
        self.layer = layer
        self.center = center
        self.preload = preload
        
        # 获取数据信息
        with h5py.File(h5_path, 'r') as f:
            self.n_samples = f[f'activations/layer_{layer}'].shape[0]
            self.dim = f[f'activations/layer_{layer}'].shape[1]
            
        # 预加载数据
        if preload:
            with h5py.File(h5_path, 'r') as f:
                self.data = torch.from_numpy(
                    f[f'activations/layer_{layer}'][:].astype(np.float32)
                )
                # 应用中心化
                if self.center is not None:
                    self.data = self.data - self.center.unsqueeze(0)
        else:
            self.data = None
            self._h5_file = None
            
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.preload:
            return self.data[idx]
        else:
            # 懒加载模式
            if self._h5_file is None:
                self._h5_file = h5py.File(self.h5_path, 'r')
            x = torch.from_numpy(
                self._h5_file[f'activations/layer_{self.layer}'][idx].astype(np.float32)
            )
            if self.center is not None:
                x = x - self.center
            return x


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """
    早停机制
    
    当重建损失 < threshold 且 L0 在目标范围内时，
    连续 patience 个 epoch 满足条件则停止训练
    """
    
    def __init__(self, 
                 recon_threshold: float,
                 target_l0: float,
                 l0_tolerance: float = 0.1,
                 patience: int = 5):
        self.recon_threshold = recon_threshold
        self.target_l0 = target_l0
        self.l0_tolerance = l0_tolerance
        self.patience = patience
        
        self.l0_lower = target_l0 * (1 - l0_tolerance)
        self.l0_upper = target_l0 * (1 + l0_tolerance)
        
        self.counter = 0
        self.should_stop = False
        
    def check(self, recon_loss: float, l0: float) -> bool:
        """
        检查是否应该停止
        
        Returns:
            True if training should stop
        """
        recon_ok = recon_loss < self.recon_threshold
        l0_ok = self.l0_lower <= l0 <= self.l0_upper
        
        if recon_ok and l0_ok:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.counter = 0
            
        return self.should_stop
    
    def get_status(self) -> str:
        """获取当前状态描述"""
        return f"EarlyStop: {self.counter}/{self.patience}"


# ============================================================================
# Training Logic
# ============================================================================

class SAETrainer:
    """SAE 训练器"""
    
    def __init__(self, 
                 sae: SparseAutoencoder,
                 config: TrainConfig,
                 layer: int,
                 device: torch.device,
                 logger: logging.Logger):
        self.sae = sae.to(device)
        self.config = config
        self.layer = layer
        self.device = device
        self.logger = logger
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = None  # 在 setup_scheduler 中初始化
        
        # 混合精度
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # 动态稀疏系数
        self.sparsity_coef = config.sparsity_coef
        
        # 早停
        self.early_stopping = EarlyStopping(
            recon_threshold=config.early_stop_recon_threshold,
            target_l0=config.target_l0,
            l0_tolerance=config.early_stop_l0_tolerance,
            patience=config.early_stop_patience
        )
        
        # 训练统计
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.metrics_history = []
        
    def setup_scheduler(self, num_training_steps: int):
        """设置学习率调度器"""
        warmup_steps = self.config.warmup_epochs * (num_training_steps // self.config.num_epochs)
        
        if self.config.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=self.config.learning_rate * 0.01
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        elif self.config.lr_scheduler == "constant":
            self.scheduler = None
            
    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失"""
        x_rec, z = self.sae(x)
        
        # 重建损失
        loss_rec = F.mse_loss(x_rec, x)
        
        # L1 稀疏损失
        loss_sparse = z.abs().mean()
            
        # 总损失
        loss = loss_rec + self.sparsity_coef * loss_sparse
        
        # 统计指标
        with torch.no_grad():
            l0 = (z > 0).float().sum(dim=-1).mean().item()  # 平均非零激活数
            frac_alive = ((z > 0).any(dim=0).float().mean().item())  # 活跃特征比例
            
        metrics = {
            'loss': loss.item(),
            'loss_rec': loss_rec.item(),
            'loss_sparse': loss_sparse.item(),
            'l0': l0,
            'frac_alive': frac_alive,
            'sparsity_coef': self.sparsity_coef
        }
        
        return loss, metrics
    
    def adjust_sparsity_coef(self, current_l0: float):
        """动态调整稀疏系数以达到目标 L0"""
        target = self.config.target_l0
        ratio = current_l0 / target

        if ratio > 2.0:
            self.sparsity_coef *= 1.2
        elif ratio > 1.1:
            self.sparsity_coef *= 1.05
        elif ratio < 0.9:
            self.sparsity_coef *= 0.95
            
        # 限制范围
        self.sparsity_coef = max(1e-6, min(100.0, self.sparsity_coef))
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.sae.train()
        epoch_metrics = []
        
        pbar = tqdm(dataloader, desc=f"Layer {self.layer} Epoch {self.epoch}",
                    disable=not self.is_main_process())
        
        for batch in pbar:
            x = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    loss, metrics = self.compute_loss(x)
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.sae.parameters(), 
                        self.config.gradient_clip
                    )
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, metrics = self.compute_loss(x)
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.sae.parameters(),
                        self.config.gradient_clip
                    )
                    
                self.optimizer.step()
            
            # 归一化 decoder
            self.sae.normalize_decoder()
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
                
            epoch_metrics.append(metrics)
            self.global_step += 1
            
            # 更新进度条
            if self.global_step % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'rec': f"{metrics['loss_rec']:.4f}",
                    'l0': f"{metrics['l0']:.1f}",
                    'alive': f"{metrics['frac_alive']:.2%}"
                })
                
        # 汇总 epoch 指标
        avg_metrics = {
            k: np.mean([m[k] for m in epoch_metrics])
            for k in epoch_metrics[0].keys()
        }
        
        # 动态调整稀疏系数
        self.adjust_sparsity_coef(avg_metrics['l0'])
        
        self.epoch += 1
        return avg_metrics
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证集评估"""
        self.sae.eval()
        all_metrics = []
        
        for batch in dataloader:
            x = batch.to(self.device)
            _, metrics = self.compute_loss(x)
            all_metrics.append(metrics)
            
        avg_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in all_metrics[0].keys()
        }
        
        return avg_metrics
    
    def check_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        return self.early_stopping.check(
            recon_loss=val_metrics['loss_rec'],
            l0=val_metrics['l0']
        )
        
    def is_main_process(self) -> bool:
        """是否是主进程（用于分布式训练）"""
        return True  # 单机多卡时每个进程都是独立的


# ============================================================================
# Main Training Loop
# ============================================================================

def train_single_layer(layer: int, 
                       sae_config: SAEConfig,
                       train_config: TrainConfig,
                       device: torch.device,
                       logger: logging.Logger) -> Dict:
    """训练单层的 SAE"""
    
    logger.info(f"=" * 60)
    logger.info(f"Training SAE for Layer {layer}")
    logger.info(f"=" * 60)
    
    output_dir = Path(train_config.output_dir) / f"layer_{layer}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 计算或加载全局中心
    center_path = output_dir / "center.pt"
    if center_path.exists():
        logger.info("Loading precomputed center...")
        center = torch.load(center_path)
    else:
        logger.info("Computing global activation center...")
        center = compute_global_center(train_config.train_h5_path, layer)
        torch.save(center, center_path)
    
    logger.info(f"Center norm: {center.norm().item():.4f}")
    logger.info(f"Center mean: {center.mean().item():.6f}")
    logger.info(f"Center std: {center.std().item():.6f}")
    
    # 2. 创建数据集和数据加载器
    logger.info("Creating datasets...")
    train_dataset = ActivationDataset(
        train_config.train_h5_path, 
        layer, 
        center=center,
        preload=True
    )
    val_dataset = ActivationDataset(
        train_config.val_h5_path,
        layer,
        center=center,
        preload=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # 3. 创建模型和训练器
    sae = SparseAutoencoder(sae_config)
    trainer = SAETrainer(sae, train_config, layer, device, logger)
    
    num_training_steps = len(train_loader) * train_config.num_epochs
    trainer.setup_scheduler(num_training_steps)
        
    # 4. 保存配置
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'sae_config': asdict(sae_config),
            'train_config': asdict(train_config),
            'center_norm': center.norm().item()
        }, f, indent=2)
    
    # 5. 训练循环
    logger.info("Starting training...")
    logger.info(f"Early stop conditions: recon < {train_config.early_stop_recon_threshold}, "
                f"L0 in [{train_config.target_l0 * (1 - train_config.early_stop_l0_tolerance):.1f}, "
                f"{train_config.target_l0 * (1 + train_config.early_stop_l0_tolerance):.1f}]")
    
    all_metrics = []
    final_epoch = 0
    early_stopped = False
    
    for epoch in range(train_config.num_epochs):
        # 训练
        train_metrics = trainer.train_epoch(train_loader)
        train_metrics['epoch'] = epoch
        train_metrics['split'] = 'train'
        all_metrics.append(train_metrics)
        
        logger.info(
            f"Epoch {epoch} Train: "
            f"loss={train_metrics['loss']:.4f}, "
            f"rec={train_metrics['loss_rec']:.6f}, "
            f"l0={train_metrics['l0']:.1f}, "
            f"alive={train_metrics['frac_alive']:.2%}, "
            f"λ={trainer.sparsity_coef:.2e}"
        )
        
        # 验证
        if (epoch + 1) % train_config.eval_interval == 0:
            val_metrics = trainer.evaluate(val_loader)
            val_metrics['epoch'] = epoch
            val_metrics['split'] = 'val'
            all_metrics.append(val_metrics)
            
            logger.info(
                f"Epoch {epoch} Val: "
                f"loss={val_metrics['loss']:.4f}, "
                f"rec={val_metrics['loss_rec']:.6f}, "
                f"l0={val_metrics['l0']:.1f}, "
                f"{trainer.early_stopping.get_status()}"
            )
            
            # 检查早停
            if trainer.check_early_stop(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}!")
                final_epoch = epoch
                early_stopped = True
                break
                
        final_epoch = epoch
        
    # 6. 保存最终模型和指标
    stop_reason = "early_stop" if early_stopped else "max_epochs"
    final_model_path = output_dir / "sae_final.pt"
    
    # 获取最终验证指标
    final_val_metrics = trainer.evaluate(val_loader)
    
    torch.save({
        'model_state': sae.state_dict(),
        'center': center,
        'sae_config': asdict(sae_config),
        'final_epoch': final_epoch,
        'stop_reason': stop_reason,
        'final_metrics': {
            'train': all_metrics[-2] if len(all_metrics) >= 2 else all_metrics[-1],
            'val': final_val_metrics
        }
    }, final_model_path)
    
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'history': all_metrics,
            'final_epoch': final_epoch,
            'stop_reason': stop_reason,
            'early_stopped': early_stopped
        }, f, indent=2)
    
    logger.info(f"Layer {layer} training complete.")
    logger.info(f"  Stop reason: {stop_reason} at epoch {final_epoch}")
    logger.info(f"  Final val loss_rec: {final_val_metrics['loss_rec']:.6f}")
    logger.info(f"  Final val L0: {final_val_metrics['l0']:.1f}")
    logger.info(f"  Model saved to {final_model_path}")
    
    return {
        'layer': layer,
        'final_epoch': final_epoch,
        'stop_reason': stop_reason,
        'final_metrics': final_val_metrics
    }


def train_parallel_worker(rank: int, 
                          world_size: int,
                          layer_assignments: List[List[int]],
                          sae_config: SAEConfig,
                          train_config: TrainConfig):
    """多卡并行训练的 worker 函数"""
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # 设置日志
    log_dir = Path(train_config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'[GPU {rank}] %(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"gpu_{rank}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f"GPU_{rank}")
    
    # 训练分配给这个 GPU 的层
    layers = layer_assignments[rank]
    logger.info(f"GPU {rank} will train layers: {layers}")
    
    results = []
    for layer in layers:
        try:
            result = train_single_layer(layer, sae_config, train_config, device, logger)
            results.append(result)
        except Exception as e:
            logger.error(f"Error training layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            
    # 保存该 GPU 的汇总结果
    summary_path = Path(train_config.output_dir) / f"summary_gpu_{rank}.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SAE Pretraining with L1 Sparsity and Early Stopping")
    parser.add_argument("--train_h5", type=str, 
                        default="./data/activations_qwen3_8b/qwen3_8b_train_activations.h5")
    parser.add_argument("--val_h5", type=str,
                        default="./data/activations_qwen3_8b/qwen3_8b_val_activations.h5")
    parser.add_argument("--output_dir", type=str, 
                        default="./outputs/sae_pretrain")
    parser.add_argument("--hidden_dim", type=int, default=16384,
                        help="SAE hidden dimension (expansion factor)")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--sparsity_coef", type=float, default=1e-3,
                        help="Initial L1 sparsity coefficient")
    parser.add_argument("--target_l0", type=float, default=50.0,
                        help="Target average L0 (number of active features)")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "gelu"])
    
    # 早停参数
    parser.add_argument("--early_stop_recon", type=float, default=0.001,
                        help="Early stop when recon loss < this threshold")
    parser.add_argument("--early_stop_l0_tol", type=float, default=0.1,
                        help="L0 tolerance ratio for early stopping")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Number of epochs to wait before early stopping")
    
    parser.add_argument("--layers", type=str, default="all",
                        help="Layers to train: 'all' or comma-separated list like '0,1,2'")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--single_gpu", type=int, default=None,
                        help="Train on single GPU (for debugging)")
    args = parser.parse_args()
    
    # 解析层
    if args.layers == "all":
        layers = list(range(36))
    else:
        layers = [int(x) for x in args.layers.split(",")]
    
    # 创建配置
    sae_config = SAEConfig(
        input_dim=4096,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        normalize_decoder=True
    )
    
    train_config = TrainConfig(
        train_h5_path=args.train_h5,
        val_h5_path=args.val_h5,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        sparsity_coef=args.sparsity_coef,
        target_l0=args.target_l0,
        early_stop_recon_threshold=args.early_stop_recon,
        early_stop_l0_tolerance=args.early_stop_l0_tol,
        early_stop_patience=args.early_stop_patience,
        layers=layers
    )
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存全局配置
    with open(Path(args.output_dir) / "global_config.json", 'w') as f:
        json.dump({
            'sae_config': asdict(sae_config),
            'train_config': asdict(train_config),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    if args.single_gpu is not None:
        # 单卡调试模式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logger = logging.getLogger("main")
        device = torch.device(f"cuda:{args.single_gpu}")
        
        results = []
        for layer in layers:
            result = train_single_layer(layer, sae_config, train_config, device, logger)
            results.append(result)
            
        # 保存汇总
        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # 打印汇总
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        for r in results:
            print(f"Layer {r['layer']:2d}: epoch={r['final_epoch']:3d}, "
                  f"stop={r['stop_reason']:10s}, "
                  f"rec={r['final_metrics']['loss_rec']:.6f}, "
                  f"l0={r['final_metrics']['l0']:.1f}")
    else:
        # 多卡并行模式
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        print(f"Using {num_gpus} GPUs for parallel training")
        
        # 分配层到各个 GPU
        layer_assignments = [[] for _ in range(num_gpus)]
        for i, layer in enumerate(layers):
            layer_assignments[i % num_gpus].append(layer)
            
        print("Layer assignments:")
        for gpu_id, assigned_layers in enumerate(layer_assignments):
            print(f"  GPU {gpu_id}: {assigned_layers}")
        
        # 启动多进程训练
        mp.spawn(
            train_parallel_worker,
            args=(num_gpus, layer_assignments, sae_config, train_config),
            nprocs=num_gpus,
            join=True
        )
        
        # 合并所有 GPU 的结果
        all_results = []
        for gpu_id in range(num_gpus):
            summary_path = Path(args.output_dir) / f"summary_gpu_{gpu_id}.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    all_results.extend(json.load(f))
                    
        # 按层排序并保存
        all_results.sort(key=lambda x: x['layer'])
        summary_path = Path(args.output_dir) / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        for r in all_results:
            print(f"Layer {r['layer']:2d}: epoch={r['final_epoch']:3d}, "
                  f"stop={r['stop_reason']:10s}, "
                  f"rec={r['final_metrics']['loss_rec']:.6f}, "
                  f"l0={r['final_metrics']['l0']:.1f}")
    
    print("\nAll layers trained successfully!")


if __name__ == "__main__":
    main()