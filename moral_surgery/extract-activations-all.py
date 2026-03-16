"""
Activation Extraction Script for Moral Aligner Training
========================================================
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import tempfile
import shutil

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [GPU %(process)d] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for activation extraction."""
    model_path: str
    data_dir: str
    output_dir: str
    layers: List[int] = None  # None means all layers
    batch_size: int = 64
    max_length: int = 128
    num_workers: int = 2  # Per-GPU workers
    prompt_template: str = "{action} is morally"
    dtype: str = "bfloat16"
    num_gpus: int = 4
    

class MoralActionDataset(Dataset):
    """Dataset for loading moral actions from CSV."""
    
    def __init__(
        self, 
        csv_path: str, 
        tokenizer, 
        max_length: int = 128,
        prompt_template: str = "{action}"
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        
        # Parse moral_vector from string to list
        self.df['moral_vector_parsed'] = self.df['moral_vector'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Format prompt
        text = self.prompt_template.format(action=row['action'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'idx': idx,  # Original index for ordering
            'row_id': row['row_id'],
            'id': row['id'],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'moral_vector': torch.tensor(row['moral_vector_parsed'], dtype=torch.float32),
            'target_dimension': row['target_dimension'],
            'm_virtue': row['m_virtue'],
            'm_vice': row['m_vice']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching."""
    return {
        'idx': [item['idx'] for item in batch],
        'row_id': [item['row_id'] for item in batch],
        'id': [item['id'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'moral_vector': torch.stack([item['moral_vector'] for item in batch]),
        'target_dimension': [item['target_dimension'] for item in batch],
        'm_virtue': torch.tensor([item['m_virtue'] for item in batch]),
        'm_vice': torch.tensor([item['m_vice'] for item in batch])
    }


class ActivationExtractor:
    """Extracts and stores activations from transformer layers."""
    
    def __init__(
        self,
        model: nn.Module,
        layers: List[int],
        pooling: str = 'mean'  # 'mean' or 'last'
    ):
        self.model = model
        self.layers = layers
        self.pooling = pooling
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _get_activation_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Store on CPU to save GPU memory
            self.activations[layer_idx] = hidden_states.detach()
        
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        # For Qwen3, the layers are typically at model.model.layers[i]
        for layer_idx in self.layers:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._get_activation_hook(layer_idx))
            self.hooks.append(hook)
        
        logger.info(f"Registered hooks on layers: {self.layers}")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and extract activations.
        
        Returns:
            Dict mapping layer_idx to pooled activation tensor [batch_size, hidden_dim]
        """
        self.activations = {}
        
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
        
        # Apply pooling
        pooled_activations = {}

        for layer_idx, hidden_states in self.activations.items():
            # hidden_states: [batch_size, seq_len, hidden_dim]
            
            if self.pooling == 'mean':
                # Move mask to same device as hidden_states
                mask = attention_mask.to(hidden_states.device).unsqueeze(-1).float()  # [batch, seq, 1]
                masked_hidden = hidden_states * mask
                summed = masked_hidden.sum(dim=1)  # [batch, hidden]
                lengths = mask.sum(dim=1).clamp(min=1)  # [batch, 1]
                pooled = summed / lengths  # [batch, hidden]
            
            elif self.pooling == 'last':
                # Get last non-padding token
                seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                seq_lengths = seq_lengths.to(hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]  # [batch, hidden]
            
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")
            
            pooled_activations[layer_idx] = pooled.cpu()
        
        return pooled_activations


class ShardHDF5Writer:
    """Writes activations to a shard HDF5 file for a single GPU."""
    
    def __init__(
        self,
        output_path: str,
        layers: List[int],
        hidden_dim: int,
        max_samples: int,  # Upper bound, will be resized later
        dtype: np.dtype = np.float16
    ):
        self.output_path = output_path
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.max_samples = max_samples
        self.dtype = dtype
        
        self.file = None
        self.datasets = {}
        self.current_idx = 0
        
        self._initialize()
    
    def _initialize(self):
        """Create HDF5 file and pre-allocate datasets."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.file = h5py.File(self.output_path, 'w')
        
        # Create dataset for each layer's activations
        for layer_idx in self.layers:
            self.datasets[f'layer_{layer_idx}'] = self.file.create_dataset(
                f'activations/layer_{layer_idx}',
                shape=(self.max_samples, self.hidden_dim),
                maxshape=(self.max_samples, self.hidden_dim),
                dtype=self.dtype,
                chunks=(min(1000, self.max_samples), self.hidden_dim)
            )
        
        # Create datasets for metadata
        self.datasets['idx'] = self.file.create_dataset(
            'metadata/idx',
            shape=(self.max_samples,),
            maxshape=(self.max_samples,),
            dtype=np.int64
        )
        
        self.datasets['row_id'] = self.file.create_dataset(
            'metadata/row_id',
            shape=(self.max_samples,),
            maxshape=(self.max_samples,),
            dtype=np.int64
        )
        
        self.datasets['moral_vector'] = self.file.create_dataset(
            'metadata/moral_vector',
            shape=(self.max_samples, 10),
            maxshape=(self.max_samples, 10),
            dtype=np.float32
        )
        
        self.datasets['m_virtue'] = self.file.create_dataset(
            'metadata/m_virtue',
            shape=(self.max_samples,),
            maxshape=(self.max_samples,),
            dtype=np.float32
        )
        
        self.datasets['m_vice'] = self.file.create_dataset(
            'metadata/m_vice',
            shape=(self.max_samples,),
            maxshape=(self.max_samples,),
            dtype=np.float32
        )
        
        # Store string metadata separately
        dt = h5py.special_dtype(vlen=str)
        self.datasets['id'] = self.file.create_dataset(
            'metadata/id',
            shape=(self.max_samples,),
            maxshape=(self.max_samples,),
            dtype=dt
        )
        
        self.datasets['target_dimension'] = self.file.create_dataset(
            'metadata/target_dimension',
            shape=(self.max_samples,),
            maxshape=(self.max_samples,),
            dtype=dt
        )
        
        # Store layer info as attributes
        self.file.attrs['layers'] = self.layers
        self.file.attrs['hidden_dim'] = self.hidden_dim
    
    def write_batch(
        self,
        activations: Dict[int, torch.Tensor],
        indices: List[int],
        row_ids: List[int],
        ids: List[str],
        moral_vectors: torch.Tensor,
        target_dimensions: List[str],
        m_virtues: torch.Tensor,
        m_vices: torch.Tensor
    ):
        """Write a batch of activations and metadata."""
        batch_size = len(row_ids)
        start_idx = self.current_idx
        end_idx = start_idx + batch_size
        
        # Write activations for each layer
        for layer_idx, acts in activations.items():
            self.datasets[f'layer_{layer_idx}'][start_idx:end_idx] = acts.numpy().astype(self.dtype)
        
        # Write metadata
        self.datasets['idx'][start_idx:end_idx] = np.array(indices)
        self.datasets['row_id'][start_idx:end_idx] = np.array(row_ids)
        self.datasets['moral_vector'][start_idx:end_idx] = moral_vectors.numpy()
        self.datasets['m_virtue'][start_idx:end_idx] = m_virtues.numpy()
        self.datasets['m_vice'][start_idx:end_idx] = m_vices.numpy()
        self.datasets['id'][start_idx:end_idx] = ids
        self.datasets['target_dimension'][start_idx:end_idx] = target_dimensions
        
        self.current_idx = end_idx
    
    def finalize(self):
        """Resize datasets to actual size and close."""
        if self.file is not None:
            # Resize all datasets to actual size
            actual_size = self.current_idx
            
            for layer_idx in self.layers:
                self.datasets[f'layer_{layer_idx}'].resize((actual_size, self.hidden_dim))
            
            self.datasets['idx'].resize((actual_size,))
            self.datasets['row_id'].resize((actual_size,))
            self.datasets['moral_vector'].resize((actual_size, 10))
            self.datasets['m_virtue'].resize((actual_size,))
            self.datasets['m_vice'].resize((actual_size,))
            self.datasets['id'].resize((actual_size,))
            self.datasets['target_dimension'].resize((actual_size,))
            
            self.file.attrs['total_samples'] = actual_size
            self.file.close()
            
            logger.info(f"Finalized shard with {actual_size} samples: {self.output_path}")


def get_model_hidden_dim(model: nn.Module) -> int:
    """Get the hidden dimension of the model."""
    return model.config.hidden_size


def get_num_layers(model: nn.Module) -> int:
    """Get the number of layers in the model."""
    return model.config.num_hidden_layers


def worker_process(
    gpu_id: int,
    config: ExtractionConfig,
    split: str,
    indices: List[int],
    layers: List[int],
    temp_dir: str,
    return_dict: dict
):
    """
    Worker process for a single GPU.
    Loads model, processes assigned data shard, saves to temp HDF5.
    """
    try:
        # Set GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        
        logger.info(f"GPU {gpu_id}: Starting worker, processing {len(indices)} samples")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model on this specific GPU
        logger.info(f"GPU {gpu_id}: Loading model...")
        
        torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch_dtype,
            device_map={'': device},  # Load entire model on this GPU
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        model.eval()
        
        hidden_dim = get_model_hidden_dim(model)
        logger.info(f"GPU {gpu_id}: Model loaded. Hidden dim: {hidden_dim}")
        
        # Create dataset
        csv_path = os.path.join(config.data_dir, f"social_chem_{split}_expanded.csv")
        full_dataset = MoralActionDataset(
            csv_path=csv_path,
            tokenizer=tokenizer,
            max_length=config.max_length,
            prompt_template=config.prompt_template
        )
        
        # Create subset for this GPU
        subset = Subset(full_dataset, indices)
        
        dataloader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Setup extractor
        extractor = ActivationExtractor(model, layers, pooling='mean')
        
        # Setup shard writer
        dtype = np.float16 if config.dtype == 'float16' else np.float16  # Save as float16 regardless
        shard_path = os.path.join(temp_dir, f"shard_gpu{gpu_id}.h5")
        
        writer = ShardHDF5Writer(
            output_path=shard_path,
            layers=layers,
            hidden_dim=hidden_dim,
            max_samples=len(indices),
            dtype=dtype
        )
        
        # Process batches
        logger.info(f"GPU {gpu_id}: Processing {len(subset)} samples...")
        
        for batch in tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Extract activations
            activations = extractor.extract(input_ids, attention_mask)
            
            # Write to HDF5
            writer.write_batch(
                activations=activations,
                indices=batch['idx'],
                row_ids=batch['row_id'],
                ids=batch['id'],
                moral_vectors=batch['moral_vector'],
                target_dimensions=batch['target_dimension'],
                m_virtues=batch['m_virtue'],
                m_vices=batch['m_vice']
            )
        
        extractor.remove_hooks()
        writer.finalize()
        
        # Clean up GPU memory
        del model
        del extractor
        torch.cuda.empty_cache()
        
        return_dict[gpu_id] = {
            'status': 'success',
            'shard_path': shard_path,
            'samples': len(indices)
        }
        
        logger.info(f"GPU {gpu_id}: Completed successfully")
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: Error - {str(e)}")
        import traceback
        traceback.print_exc()
        return_dict[gpu_id] = {
            'status': 'error',
            'error': str(e)
        }


def merge_shards(
    shard_paths: List[str],
    output_path: str,
    layers: List[int],
    hidden_dim: int,
    total_samples: int
):
    """Merge multiple shard HDF5 files into a single file, sorted by original index."""
    
    logger.info(f"Merging {len(shard_paths)} shards into {output_path}")
    
    # First, collect all data with indices for sorting
    all_data = {
        'idx': [],
        'row_id': [],
        'id': [],
        'moral_vector': [],
        'target_dimension': [],
        'm_virtue': [],
        'm_vice': []
    }
    
    for layer_idx in layers:
        all_data[f'layer_{layer_idx}'] = []
    
    # Read all shards
    for shard_path in shard_paths:
        with h5py.File(shard_path, 'r') as f:
            n = f.attrs['total_samples']
            
            all_data['idx'].append(f['metadata/idx'][:])
            all_data['row_id'].append(f['metadata/row_id'][:])
            all_data['id'].append(f['metadata/id'][:])
            all_data['moral_vector'].append(f['metadata/moral_vector'][:])
            all_data['target_dimension'].append(f['metadata/target_dimension'][:])
            all_data['m_virtue'].append(f['metadata/m_virtue'][:])
            all_data['m_vice'].append(f['metadata/m_vice'][:])
            
            for layer_idx in layers:
                all_data[f'layer_{layer_idx}'].append(f[f'activations/layer_{layer_idx}'][:])
    
    # Concatenate
    for key in all_data:
        all_data[key] = np.concatenate(all_data[key], axis=0)
    
    # Sort by original index
    sort_order = np.argsort(all_data['idx'])
    
    for key in all_data:
        all_data[key] = all_data[key][sort_order]
    
    # Write merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Write activations
        for layer_idx in layers:
            f.create_dataset(
                f'activations/layer_{layer_idx}',
                data=all_data[f'layer_{layer_idx}'],
                dtype=np.float16,
                chunks=(min(1000, total_samples), hidden_dim)
            )
        
        # Write metadata
        f.create_dataset('metadata/row_id', data=all_data['row_id'])
        f.create_dataset('metadata/moral_vector', data=all_data['moral_vector'])
        f.create_dataset('metadata/m_virtue', data=all_data['m_virtue'])
        f.create_dataset('metadata/m_vice', data=all_data['m_vice'])
        
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('metadata/id', data=all_data['id'], dtype=dt)
        f.create_dataset('metadata/target_dimension', data=all_data['target_dimension'], dtype=dt)
        
        # Attributes
        f.attrs['layers'] = layers
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['total_samples'] = total_samples
    
    logger.info(f"Merged file created: {output_path} ({total_samples} samples)")


def process_split_multiGPU(
    split: str,
    config: ExtractionConfig,
    layers: List[int],
    hidden_dim: int
):
    """Process a single data split using multiple GPUs in parallel."""
    
    csv_path = os.path.join(config.data_dir, f"social_chem_{split}_expanded.csv")
    output_path = os.path.join(
        config.output_dir, 
        f"qwen3_8b_{split}_activations.h5"
    )
    
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}, skipping...")
        return
    
    # Get total sample count
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    del df
    
    logger.info(f"Processing {split} split: {total_samples} samples on {config.num_gpus} GPUs")
    
    # Divide indices among GPUs
    all_indices = list(range(total_samples))
    indices_per_gpu = np.array_split(all_indices, config.num_gpus)
    
    # Create temp directory for shards
    temp_dir = tempfile.mkdtemp(prefix=f"activations_{split}_")
    logger.info(f"Temporary shard directory: {temp_dir}")
    
    try:
        # Use spawn method for CUDA compatibility
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Create shared dict for return values
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Launch worker processes
    processes = []
    for gpu_id in range(config.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                config,
                split,
                indices_per_gpu[gpu_id].tolist(),
                layers,
                temp_dir,
                return_dict
            )
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Check results
    shard_paths = []
    for gpu_id in range(config.num_gpus):
        result = return_dict.get(gpu_id, {'status': 'unknown'})
        if result['status'] == 'success':
            shard_paths.append(result['shard_path'])
            logger.info(f"GPU {gpu_id}: Processed {result['samples']} samples")
        else:
            logger.error(f"GPU {gpu_id}: Failed - {result.get('error', 'Unknown error')}")
            raise RuntimeError(f"GPU {gpu_id} failed")
    
    # Merge shards
    merge_shards(
        shard_paths=shard_paths,
        output_path=output_path,
        layers=layers,
        hidden_dim=hidden_dim,
        total_samples=total_samples
    )
    
    # Cleanup temp files
    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up temporary directory: {temp_dir}")
    
    logger.info(f"Completed {split} split. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract activations from Qwen3-8B (Multi-GPU)")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to local Qwen3-8B model"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/train",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/activations",
        help="Directory for output HDF5 files"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to extract (e.g., '16,20,24,28'). Default: all layers"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing (per GPU)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers per GPU"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="{action} is morally",
        help="Template for formatting actions"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type for model inference"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits to process"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    
    args = parser.parse_args()
    
    # Determine number of GPUs
    if args.num_gpus is None:
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")
    
    logger.info(f"Using {num_gpus} GPUs")
    for i in range(num_gpus):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Parse layers
    layers = None
    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(',')]
    
    # Create config
    config = ExtractionConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layers=layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        prompt_template=args.prompt_template,
        dtype=args.dtype,
        num_gpus=num_gpus
    )
    
    # Get model info (load briefly on CPU to get config)
    logger.info(f"Loading model config from {config.model_path}")
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config.model_path, trust_remote_code=True)
    hidden_dim = model_config.hidden_size
    num_layers = model_config.num_hidden_layers
    
    # Determine layers to extract
    if layers is None:
        layers = list(range(num_layers))
    
    logger.info(f"Model: hidden_dim={hidden_dim}, num_layers={num_layers}")
    logger.info(f"Extracting layers: {layers}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Process each split
    splits = [s.strip() for s in args.splits.split(',')]
    
    for split in splits:
        process_split_multiGPU(split, config, layers, hidden_dim)
    
    logger.info("All splits processed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Output directory: {config.output_dir}")
    for split in splits:
        h5_path = os.path.join(config.output_dir, f"qwen3_8b_{split}_activations.h5")
        if os.path.exists(h5_path):
            size_mb = os.path.getsize(h5_path) / (1024 * 1024)
            print(f"  {split}: {h5_path} ({size_mb:.1f} MB)")
    print("="*60)


if __name__ == "__main__":
    main()
