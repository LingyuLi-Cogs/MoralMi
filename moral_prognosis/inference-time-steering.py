import json
import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from functools import partial
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SteeringConfig:
    """Steering configuration."""
    # Model path — set via --model_path argument or edit here
    model_path: str = ""

    # SAE checkpoint directory
    sae_dir: str = "./outputs/sae_finetune"

    # Steering parameters
    steering_layers: List[int] = field(default_factory=lambda: [16, 17, 18, 19, 20])
    steering_strength: float = 1.0
    steering_mode: str = "add"  # "add", "replace", "blend"

    # Feature selection
    use_all_features: bool = True
    feature_scale: float = 1.0

    # Generation parameters
    max_new_tokens: int = 512
    do_sample: bool = True
    top_p: float = 0.9
    temperature: float = 0.7
    repetition_penalty: float = 1.1

    # Input/output
    input_file: str = "./moral_prognosis/Flames_1k_Chinese.jsonl"
    output_file: str = "./qwen3-8b-sae-steered.jsonl"

    # Parallel processing
    num_gpus: int = 8
    batch_size: int = 1  # samples per GPU per step

    # Precision
    dtype: str = "float16"


# ============================================================================
# SAE Model
# ============================================================================

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder."""

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


# ============================================================================
# SAE Manager
# ============================================================================

class SAEManager:
    """Manages loading and applying SAEs across multiple layers."""

    def __init__(self,
                 sae_dir: str,
                 layers: List[int],
                 device: torch.device,
                 dtype: torch.dtype):
        self.sae_dir = Path(sae_dir)
        self.layers = layers
        self.device = device
        self.dtype = dtype

        self.saes: Dict[int, SparseAutoencoder] = {}
        self.centers: Dict[int, torch.Tensor] = {}
        self.mono_indices: Dict[int, List[int]] = {}

        self._load_saes()

    def _load_saes(self):
        """Load SAEs for all specified layers."""
        for layer in self.layers:
            layer_dir = self.sae_dir / f"layer_{layer}"

            finetuned_path = layer_dir / "sae_finetuned.pt"
            pretrained_path = layer_dir / "sae_final.pt"

            if finetuned_path.exists():
                checkpoint_path = finetuned_path
            elif pretrained_path.exists():
                checkpoint_path = pretrained_path
            else:
                continue

            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            config = checkpoint.get('config', {})
            input_dim = config.get('input_dim', 4096)
            hidden_dim = config.get('hidden_dim', 16384)

            sae = SparseAutoencoder(input_dim, hidden_dim)
            sae.load_state_dict(checkpoint['model_state'])
            sae = sae.to(self.device).to(self.dtype)
            sae.eval()

            self.saes[layer] = sae

            center = checkpoint['center']
            if isinstance(center, torch.Tensor):
                self.centers[layer] = center.to(self.device).to(self.dtype)
            else:
                self.centers[layer] = torch.tensor(center, device=self.device, dtype=self.dtype)

            if 'monosemantic_indices' in checkpoint:
                self.mono_indices[layer] = checkpoint['monosemantic_indices']

    def apply_sae(self,
                  activation: torch.Tensor,
                  layer: int,
                  mode: str = "add",
                  strength: float = 1.0,
                  use_mono_only: bool = False,
                  feature_scale: float = 1.0) -> torch.Tensor:
        """Apply SAE-based steering to an activation tensor."""
        if layer not in self.saes:
            return activation

        sae = self.saes[layer]
        center = self.centers[layer]

        original_shape = activation.shape
        x = activation.view(-1, activation.shape[-1])
        x_centered = x - center.unsqueeze(0)

        with torch.no_grad():
            z = sae.encode(x_centered)

            if use_mono_only and layer in self.mono_indices:
                mask = torch.zeros(z.shape[-1], device=z.device, dtype=z.dtype)
                mask[self.mono_indices[layer]] = 1.0
                z = z * mask.unsqueeze(0)

            if feature_scale != 1.0:
                z = z * feature_scale

            x_rec = sae.decode(z)
            x_rec = x_rec + center.unsqueeze(0)

        if mode == "add":
            diff = x_rec - x
            x_steered = x + strength * diff
        elif mode == "replace":
            x_steered = x_rec
        elif mode == "blend":
            x_steered = (1 - strength) * x + strength * x_rec
        else:
            raise ValueError(f"Unknown steering mode: {mode}")

        return x_steered.view(original_shape)


# ============================================================================
# Worker Process
# ============================================================================

def worker_process(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    config_dict: dict,
    total_tasks: int
):
    """
    Worker process: loads the model on the assigned GPU and processes tasks.

    Args:
        gpu_id: GPU device ID
        task_queue: input task queue
        result_queue: output result queue
        config_dict: config as a plain dict
        total_tasks: total number of tasks (for progress display)
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    dtype = getattr(torch, config_dict['dtype'])

    # Config
    config = SteeringConfig(**config_dict)

    print(f"[GPU {gpu_id}] Loading model...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Load SAEs
    sae_manager = SAEManager(
        config.sae_dir,
        config.steering_layers,
        device,
        dtype
    )

    print(f"[GPU {gpu_id}] SAE loaded for layers: {list(sae_manager.saes.keys())}")

    # Identify model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Unknown model structure")

    # Steering state — use list to allow mutation in closure
    steering_enabled = [True]

    # Create hook
    def create_hook(layer_idx):
        def hook(module, input, output):
            if not steering_enabled[0]:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            steered = sae_manager.apply_sae(
                hidden_states,
                layer_idx,
                mode=config.steering_mode,
                strength=config.steering_strength,
                use_mono_only=not config.use_all_features,
                feature_scale=config.feature_scale
            )

            if rest is not None:
                return (steered,) + rest
            return steered
        return hook

    # Register hooks
    hooks = []
    for layer_idx in config.steering_layers:
        if layer_idx < len(layers) and layer_idx in sae_manager.saes:
            hook = layers[layer_idx].register_forward_hook(create_hook(layer_idx))
            hooks.append(hook)

    print(f"[GPU {gpu_id}] Ready, registered {len(hooks)} hooks")

    # Process tasks
    processed = 0
    while True:
        try:
            task = task_queue.get(timeout=5)
        except:
            # queue empty; check if more tasks remain
            if task_queue.empty():
                break
            continue

        if task is None:  # termination signal
            break

        task_id, data = task
        prompt = data.get("prompt", "")

        try:
            # Build chat messages
            # System prompt is intentionally Chinese for the FLAMES benchmark
            messages = [
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": prompt}
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                text = prompt

            # Tokenize
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            input_length = model_inputs.input_ids.shape[1]

            # Generate
            steering_enabled[0] = True
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    top_p=config.top_p,
                    temperature=config.temperature,
                    repetition_penalty=config.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode
            generated_ids = generated_ids[0][input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Build result
            result = {
                **data,
                'response': response,
                'steering_config': {
                    'layers': config.steering_layers,
                    'mode': config.steering_mode,
                    'strength': config.steering_strength,
                    'feature_scale': config.feature_scale
                }
            }

            result_queue.put((task_id, result, None))

        except Exception as e:
            error_result = {
                **data,
                'response': None,
                'error': str(e)
            }
            result_queue.put((task_id, error_result, str(e)))

        processed += 1
        if processed % 10 == 0:
            print(f"[GPU {gpu_id}] Processed {processed} tasks")

    # Cleanup
    for hook in hooks:
        hook.remove()

    print(f"[GPU {gpu_id}] Worker finished, processed {processed} tasks")


# ============================================================================
# Result Writer Process
# ============================================================================

def writer_process(
    result_queue: Queue,
    output_file: str,
    total_tasks: int,
    done_event
):
    """
    Writer process: reads from the result queue and writes to file.
    """
    results = {}
    errors = []

    pbar = tqdm(total=total_tasks, desc="Overall Progress")

    while len(results) + len(errors) < total_tasks:
        try:
            task_id, result, error = result_queue.get(timeout=10)

            if error:
                errors.append((task_id, error))
            else:
                results[task_id] = result

            pbar.update(1)

            # Periodic save
            if len(results) % 50 == 0:
                _save_results(results, output_file)

        except:
            continue

    pbar.close()

    # Final save
    _save_results(results, output_file)

    # Save error log
    if errors:
        error_file = Path(output_file).with_suffix('.errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"Errors saved to {error_file}")

    done_event.set()
    print(f"Writer finished. Results: {len(results)}, Errors: {len(errors)}")


def _save_results(results: dict, output_file: str):
    """Save results in sorted order."""
    sorted_results = [results[i] for i in sorted(results.keys())]

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in sorted_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ============================================================================
# Main Controller
# ============================================================================

def run_parallel_evaluation(config: SteeringConfig):
    """Run parallel evaluation."""

    print(f"Starting parallel evaluation with {config.num_gpus} GPUs")
    print(f"Input: {config.input_file}")
    print(f"Output: {config.output_file}")
    print(f"Steering layers: {config.steering_layers}")
    print(f"Steering mode: {config.steering_mode}, strength: {config.steering_strength}")

    # Read all tasks
    tasks = []
    with open(config.input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                tasks.append((idx, data))

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    # Check for already-completed tasks (resumable)
    completed_ids = set()
    output_path = Path(config.output_file)
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # use prompt as unique identifier
                        completed_ids.add(data.get('prompt', ''))
                    except:
                        pass
        print(f"Found {len(completed_ids)} completed tasks, resuming...")

    # Filter to pending tasks
    pending_tasks = [(idx, data) for idx, data in tasks if data.get('prompt', '') not in completed_ids]
    print(f"Pending tasks: {len(pending_tasks)}")

    if not pending_tasks:
        print("All tasks completed!")
        return

    # Create queues
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    done_event = manager.Event()

    # Fill task queue
    for task in pending_tasks:
        task_queue.put(task)

    # Add termination signals
    for _ in range(config.num_gpus):
        task_queue.put(None)

    # Convert config to dict
    config_dict = {
        'model_path': config.model_path,
        'sae_dir': config.sae_dir,
        'steering_layers': config.steering_layers,
        'steering_strength': config.steering_strength,
        'steering_mode': config.steering_mode,
        'use_all_features': config.use_all_features,
        'feature_scale': config.feature_scale,
        'max_new_tokens': config.max_new_tokens,
        'do_sample': config.do_sample,
        'top_p': config.top_p,
        'temperature': config.temperature,
        'repetition_penalty': config.repetition_penalty,
        'dtype': config.dtype
    }

    # Start worker processes
    workers = []
    for gpu_id in range(config.num_gpus):
        p = Process(
            target=worker_process,
            args=(gpu_id, task_queue, result_queue, config_dict, len(pending_tasks))
        )
        p.start()
        workers.append(p)
        print(f"Started worker on GPU {gpu_id}")

    # Start writer process
    writer = Process(
        target=writer_process,
        args=(result_queue, config.output_file, len(pending_tasks), done_event)
    )
    writer.start()

    # Wait for completion
    for p in workers:
        p.join()

    writer.join()

    print("All workers finished!")
    print(f"Results saved to {config.output_file}")


# ============================================================================
# Alternative: torch.multiprocessing with spawn
# ============================================================================

def run_parallel_evaluation_v2(config: SteeringConfig):
    """
    Alternative implementation using torch.multiprocessing.
    More compatible with certain CUDA environments.
    """
    import torch.multiprocessing as tmp
    tmp.set_start_method('spawn', force=True)

    print(f"Starting parallel evaluation (spawn mode) with {config.num_gpus} GPUs")

    # Read tasks
    tasks = []
    with open(config.input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                tasks.append((idx, data))

    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")

    # Distribute tasks across GPUs
    task_chunks = [[] for _ in range(config.num_gpus)]
    for i, task in enumerate(tasks):
        task_chunks[i % config.num_gpus].append(task)

    for i, chunk in enumerate(task_chunks):
        print(f"GPU {i}: {len(chunk)} tasks")

    # Shared results list
    manager = tmp.Manager()
    results_list = manager.list([None] * total_tasks)

    # Config dict
    config_dict = {
        'model_path': config.model_path,
        'sae_dir': config.sae_dir,
        'steering_layers': config.steering_layers,
        'steering_strength': config.steering_strength,
        'steering_mode': config.steering_mode,
        'use_all_features': config.use_all_features,
        'feature_scale': config.feature_scale,
        'max_new_tokens': config.max_new_tokens,
        'do_sample': config.do_sample,
        'top_p': config.top_p,
        'temperature': config.temperature,
        'repetition_penalty': config.repetition_penalty,
        'dtype': config.dtype
    }

    def worker_v2(gpu_id, task_list, results_list, config_dict):
        """V2 worker process."""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")
        dtype = getattr(torch, config_dict['dtype'])

        # Load model
        print(f"[GPU {gpu_id}] Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            config_dict['model_path'],
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config_dict['model_path'],
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        # Load SAEs
        sae_manager = SAEManager(
            config_dict['sae_dir'],
            config_dict['steering_layers'],
            device,
            dtype
        )

        # Get model layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            layers = model.transformer.h

        # Create hooks
        steering_enabled = [True]

        def create_hook(layer_idx):
            def hook(module, input, output):
                if not steering_enabled[0]:
                    return output
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    rest = output[1:]
                else:
                    hidden_states = output
                    rest = None

                steered = sae_manager.apply_sae(
                    hidden_states,
                    layer_idx,
                    mode=config_dict['steering_mode'],
                    strength=config_dict['steering_strength'],
                    use_mono_only=not config_dict['use_all_features'],
                    feature_scale=config_dict['feature_scale']
                )

                if rest is not None:
                    return (steered,) + rest
                return steered
            return hook

        hooks = []
        for layer_idx in config_dict['steering_layers']:
            if layer_idx < len(layers) and layer_idx in sae_manager.saes:
                hook = layers[layer_idx].register_forward_hook(create_hook(layer_idx))
                hooks.append(hook)

        print(f"[GPU {gpu_id}] Ready, processing {len(task_list)} tasks")

        for task_id, data in tqdm(task_list, desc=f"GPU {gpu_id}"):
            prompt = data.get("prompt", "")

            try:
                # System prompt is intentionally Chinese for the FLAMES benchmark
                messages = [
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": prompt}
                ]

                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    text = prompt

                model_inputs = tokenizer([text], return_tensors="pt").to(device)
                input_length = model_inputs.input_ids.shape[1]

                steering_enabled[0] = True
                with torch.no_grad():
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        attention_mask=model_inputs.attention_mask,
                        max_new_tokens=config_dict['max_new_tokens'],
                        do_sample=config_dict['do_sample'],
                        top_p=config_dict['top_p'],
                        temperature=config_dict['temperature'],
                        repetition_penalty=config_dict['repetition_penalty'],
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_ids = generated_ids[0][input_length:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                result = {
                    **data,
                    'response': response,
                    'steering_config': {
                        'layers': config_dict['steering_layers'],
                        'mode': config_dict['steering_mode'],
                        'strength': config_dict['steering_strength']
                    }
                }

            except Exception as e:
                result = {**data, 'response': None, 'error': str(e)}

            results_list[task_id] = result

        for hook in hooks:
            hook.remove()

        print(f"[GPU {gpu_id}] Finished")

    # Start processes
    processes = []
    for gpu_id in range(config.num_gpus):
        p = tmp.Process(
            target=worker_v2,
            args=(gpu_id, task_chunks[gpu_id], results_list, config_dict)
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    # Save results
    with open(config.output_file, 'w', encoding='utf-8') as f:
        for result in results_list:
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Results saved to {config.output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE Steering Evaluation - Multi-GPU")

    # Paths
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--sae_dir", type=str, default="./outputs/sae_finetune")
    parser.add_argument("--input_file", type=str,
                        default="./moral_prognosis/Flames_1k_Chinese.jsonl")
    parser.add_argument("--output_file", type=str, default="./qwen3-8b-sae-steered.jsonl")

    # Steering parameters
    parser.add_argument("--layers", type=str, default="16,17,18,19,20")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mode", type=str, default="add",
                        choices=["add", "replace", "blend"])
    parser.add_argument("--feature_scale", type=float, default=1.0)

    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Parallel processing parameters
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--use_spawn", action="store_true",
                        help="use spawn mode (more stable in some environments)")

    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Parse layer list
    layers = [int(x) for x in args.layers.split(",")]

    # Build config
    config = SteeringConfig(
        model_path=args.model_path,
        sae_dir=args.sae_dir,
        steering_layers=layers,
        steering_strength=args.strength,
        steering_mode=args.mode,
        feature_scale=args.feature_scale,
        input_file=args.input_file,
        output_file=args.output_file,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        num_gpus=args.num_gpus
    )

    # Run
    if args.use_spawn:
        run_parallel_evaluation_v2(config)
    else:
        run_parallel_evaluation(config)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
