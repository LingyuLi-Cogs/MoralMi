import os
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


# ============== Constants ==============

# Mapping from MFT dimension to virtue/vice category names
DIMENSION_TO_CATEGORIES = {
    'care-harm': {'virtue': 'care', 'vice': 'harm'},
    'fairness-cheating': {'virtue': 'fairness', 'vice': 'cheating'},
    'loyalty-betrayal': {'virtue': 'loyalty', 'vice': 'betrayal'},
    'authority-subversion': {'virtue': 'authority', 'vice': 'subversion'},
    'sanctity-degradation': {'virtue': 'sanctity', 'vice': 'degradation'}
}

# Mapping from category name to moral_vector index
CATEGORY_TO_INDEX = {
    'care': 0, 'harm': 1,
    'fairness': 2, 'cheating': 3,
    'loyalty': 4, 'betrayal': 5,
    'authority': 6, 'subversion': 7,
    'sanctity': 8, 'degradation': 9
}

# Model configurations: (activations_folder, model_name)
MODEL_CONFIGS = [
    ('activations_gpt-oss-20b', 'gpt-oss-20b'),
    ('activations_gpt-oss-120b', 'gpt-oss-120b'),
    ('activations_gpt-oss-safeguard-20b', 'gpt-oss-safeguard-20b'),
    ('activations_gpt-oss-safeguard-120b', 'gpt-oss-safeguard-120b'),
    ('activations_Llama-3.1-8B', 'Llama-3.1-8B'),
    ('activations_Llama-3.1-8B-Instruct', 'Llama-3.1-8B-Instruct'),
    ('activations_Llama-Guard-3-8B', 'Llama-Guard-3-8B'),
    ('activations_Llama-4-Scout-17B-16E', 'Llama-4-Scout-17B-16E'),
    ('activations_Llama-4-Scout-17B-16E-Instruct', 'Llama-4-Scout-17B-16E-Instruct'),
    ('activations_Qwen3-0.6B', 'Qwen3-0.6B'),
    ('activations_Qwen3-4B', 'Qwen3-4B'),
    ('activations_Qwen3-8B', 'Qwen3-8B'),
    ('activations_Qwen3-14B', 'Qwen3-14B'),
    ('activations_Qwen3-32B', 'Qwen3-32B'),
    ('activations_Qwen3-0.6B-Base', 'Qwen3-0.6B-Base'),
    ('activations_Qwen3-4B-Base', 'Qwen3-4B-Base'),
    ('activations_Qwen3-8B-Base', 'Qwen3-8B-Base'),
    ('activations_Qwen3-14B-Base', 'Qwen3-14B-Base'),
    ('activations_Qwen3Guard-Gen-0.6B', 'Guard-Gen-0.6B'),
    ('activations_Qwen3Guard-Gen-4B', 'Guard-Gen-4B'),
    ('activations_Qwen3Guard-Gen-8B', 'Guard-Gen-8B'),
    ('activations_Qwen3-235B-A22B', 'Qwen3-235B-A22B'),
    ('activations_Qwen3-235B-A22B-Instruct-2507', 'Qwen3-235B-A22B-Instruct-2507'),
]

# Path configuration
ACTIVATIONS_BASE_DIR = './moral_representation/extracted_activations'
CATEGORY_CENTERS_DIR = './moral_representation/category_centers/centers'
SIMILARITIES_DIR = './moral_representation/category_centers/similarities'


# ============== Utility Functions ==============

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def get_target_category(dimension, sample_type):
    """Determine the target category based on dimension and sample_type."""
    if 'virtue' in sample_type:
        return DIMENSION_TO_CATEGORIES[dimension]['virtue']
    else:
        return DIMENSION_TO_CATEGORIES[dimension]['vice']


def ensure_dir(path):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============== Anisotropy Correction ==============

def compute_global_mean(data, layers):
    """
    Compute the global mean vector for each layer across all samples.

    Args:
        data: list of all sample dicts
        layers: list of layer indices

    Returns:
        global_mean: {
            'mean_pooling': {layer: mean_vector, ...},
            'last_token': {layer: mean_vector, ...}
        }
    """
    print("  Computing global mean vectors (for anisotropy correction)...")

    global_mean = {
        'mean_pooling': {},
        'last_token': {}
    }

    n_samples = len(data)

    for layer in tqdm(layers, desc="  Computing per-layer mean"):
        # Collect representations for this layer across all samples
        vectors_mp = np.stack([np.asarray(sample['mean_pooling'][layer]) for sample in data])
        vectors_lt = np.stack([np.asarray(sample['last_token'][layer]) for sample in data])

        # Compute mean
        global_mean['mean_pooling'][layer] = vectors_mp.mean(axis=0)
        global_mean['last_token'][layer] = vectors_lt.mean(axis=0)

    print(f"  Global mean computed over {n_samples} samples")

    return global_mean


def subtract_mean(vector, mean_vector):
    """
    Subtract the global mean from a vector (anisotropy correction).

    Args:
        vector: original representation vector
        mean_vector: global mean vector

    Returns:
        corrected vector
    """
    return np.asarray(vector) - np.asarray(mean_vector)


# ============== Core Functions ==============

def build_category_centers(activations_path, output_path):
    """
    Build category center vectors from activations using typical samples,
    with anisotropy correction via global mean subtraction.

    Args:
        activations_path: path to merged activations file
        output_path: path to save the category centers

    Returns:
        centers: category center dict
        global_mean: global mean dict
        data: raw sample data (for downstream use)
    """
    print(f"  Loading activations: {activations_path}")
    data = torch.load(activations_path, weights_only=False)

    # Get layer indices
    first_sample = data[0]
    layers = sorted(list(first_sample['mean_pooling'].keys()))
    hidden_dim = len(first_sample['mean_pooling'][layers[0]])

    print(f"  Total samples: {len(data)}, layers: {len(layers)}, hidden dim: {hidden_dim}")

    # Step 1: Compute global mean over all samples
    global_mean = compute_global_mean(data, layers)

    # Step 2: Filter to typical samples
    typical_samples = [
        sample for sample in data
        if sample['sample_type'] in ['virtue_typical', 'vice_typical']
    ]

    print(f"  Found {len(typical_samples)} typical samples for building category centers")

    if len(typical_samples) == 0:
        raise ValueError("No typical samples found!")

    # Step 3: Group by category
    category_samples = defaultdict(list)

    for sample in typical_samples:
        dimension = sample['sampled_dimension']
        sample_type = sample['sample_type']
        category = get_target_category(dimension, sample_type)
        category_samples[category].append(sample)

    # Step 4: Compute per-layer category centers using corrected representations
    centers = {
        'mean_pooling': {layer: {} for layer in layers},
        'last_token': {layer: {} for layer in layers}
    }
    sample_counts = {}

    print("  Building anisotropy-corrected category centers...")

    for category, samples in category_samples.items():
        sample_counts[category] = len(samples)
        print(f"    Category '{category}': {len(samples)} samples")

        for layer in layers:
            # Mean-pooling center (corrected)
            vectors_mp = np.stack([
                subtract_mean(s['mean_pooling'][layer], global_mean['mean_pooling'][layer])
                for s in samples
            ])
            centers['mean_pooling'][layer][category] = vectors_mp.mean(axis=0)

            # Last-token center (corrected)
            vectors_lt = np.stack([
                subtract_mean(s['last_token'][layer], global_mean['last_token'][layer])
                for s in samples
            ])
            centers['last_token'][layer][category] = vectors_lt.mean(axis=0)

    # Step 5: Spot-check correction — print cosine similarities between selected centers
    print("\n  Spot-check: cosine similarities between category centers (after correction):")
    test_layer = layers[len(layers) // 2]  # use the middle layer
    print(f"    Test layer: {test_layer}")

    # care vs harm (should be opposing)
    care_center = centers['mean_pooling'][test_layer]['care']
    harm_center = centers['mean_pooling'][test_layer]['harm']
    print(f"    care vs harm (mean_pooling): {cosine_similarity(care_center, harm_center):.4f}")

    # care vs fairness (should be moderately correlated)
    fairness_center = centers['mean_pooling'][test_layer]['fairness']
    print(f"    care vs fairness (mean_pooling): {cosine_similarity(care_center, fairness_center):.4f}")

    # loyalty vs betrayal
    loyalty_center = centers['mean_pooling'][test_layer]['loyalty']
    betrayal_center = centers['mean_pooling'][test_layer]['betrayal']
    print(f"    loyalty vs betrayal (mean_pooling): {cosine_similarity(loyalty_center, betrayal_center):.4f}")

    # Build output dict
    output = {
        'num_layers': len(layers),
        'layer_indices': layers,
        'hidden_dim': hidden_dim,
        'categories': list(CATEGORY_TO_INDEX.keys()),
        'sample_counts': sample_counts,
        'mean_pooling': centers['mean_pooling'],
        'last_token': centers['last_token'],
        'global_mean': global_mean,  # saved for downstream use
        'correction_method': 'mean_subtraction'
    }

    ensure_dir(os.path.dirname(output_path))
    torch.save(output, output_path)
    print(f"\n  Category centers saved to: {output_path}")

    return output, global_mean, data


def compute_similarities(data, centers, global_mean, output_path):
    """
    Compute cosine similarity of each sample to its corresponding category center,
    with anisotropy correction applied.

    Args:
        data: list of all sample dicts
        centers: category centers dict
        global_mean: global mean dict
        output_path: path to save similarity results
    """
    layers = centers['layer_indices']
    results = []

    print(f"  Computing similarities for {len(data)} samples (anisotropy-corrected)...")

    for sample in tqdm(data, desc="  Processing samples"):
        dimension = sample['sampled_dimension']
        sample_type = sample['sample_type']

        if sample_type == 'neutral':
            result = process_neutral_sample(sample, dimension, centers, global_mean, layers)
        else:
            result = process_polarized_sample(sample, dimension, sample_type, centers, global_mean, layers)

        results.append(result)

    ensure_dir(os.path.dirname(output_path))
    torch.save(results, output_path)
    print(f"  Similarity results saved to: {output_path}")

    return results


def process_polarized_sample(sample, dimension, sample_type, centers, global_mean, layers):
    """Process a virtue or vice sample with anisotropy correction."""
    target_category = get_target_category(dimension, sample_type)

    cos_sim_mp = {}
    cos_sim_lt = {}

    for layer in layers:
        # Category centers are already anisotropy-corrected
        center_mp = centers['mean_pooling'][layer][target_category]
        center_lt = centers['last_token'][layer][target_category]

        # Apply anisotropy correction to sample
        sample_mp_corrected = subtract_mean(
            sample['mean_pooling'][layer],
            global_mean['mean_pooling'][layer]
        )
        sample_lt_corrected = subtract_mean(
            sample['last_token'][layer],
            global_mean['last_token'][layer]
        )

        cos_sim_mp[layer] = cosine_similarity(sample_mp_corrected, center_mp)
        cos_sim_lt[layer] = cosine_similarity(sample_lt_corrected, center_lt)

    return {
        'id': sample['id'],
        'moral_vector': sample['moral_vector'],
        'sampled_dimension': dimension,
        'sample_type': sample_type,
        'mean_pooling_cos_sim': cos_sim_mp,
        'last_token_cos_sim': cos_sim_lt
    }


def process_neutral_sample(sample, dimension, centers, global_mean, layers):
    """Process a neutral sample with anisotropy correction."""
    virtue_category = DIMENSION_TO_CATEGORIES[dimension]['virtue']
    vice_category = DIMENSION_TO_CATEGORIES[dimension]['vice']

    cos_sim_mp_virtue = {}
    cos_sim_lt_virtue = {}
    cos_sim_mp_vice = {}
    cos_sim_lt_vice = {}

    for layer in layers:
        # Apply anisotropy correction to sample
        sample_mp_corrected = subtract_mean(
            sample['mean_pooling'][layer],
            global_mean['mean_pooling'][layer]
        )
        sample_lt_corrected = subtract_mean(
            sample['last_token'][layer],
            global_mean['last_token'][layer]
        )

        # Virtue direction
        center_mp_v = centers['mean_pooling'][layer][virtue_category]
        center_lt_v = centers['last_token'][layer][virtue_category]
        cos_sim_mp_virtue[layer] = cosine_similarity(sample_mp_corrected, center_mp_v)
        cos_sim_lt_virtue[layer] = cosine_similarity(sample_lt_corrected, center_lt_v)

        # Vice direction
        center_mp_vi = centers['mean_pooling'][layer][vice_category]
        center_lt_vi = centers['last_token'][layer][vice_category]
        cos_sim_mp_vice[layer] = cosine_similarity(sample_mp_corrected, center_mp_vi)
        cos_sim_lt_vice[layer] = cosine_similarity(sample_lt_corrected, center_lt_vi)

    return {
        'id': sample['id'],
        'moral_vector': sample['moral_vector'],
        'sampled_dimension': dimension,
        'sample_type': 'neutral',
        'mean_pooling_cos_sim_virtue': cos_sim_mp_virtue,
        'last_token_cos_sim_virtue': cos_sim_lt_virtue,
        'mean_pooling_cos_sim_vice': cos_sim_mp_vice,
        'last_token_cos_sim_vice': cos_sim_lt_vice
    }


def process_single_model(folder_name, model_name):
    """Process a single model."""
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")

    activations_path = os.path.join(ACTIVATIONS_BASE_DIR, folder_name, 'activations_merged.pt')
    centers_path = os.path.join(CATEGORY_CENTERS_DIR, model_name, 'category_centers.pt')
    similarities_path = os.path.join(SIMILARITIES_DIR, f'{model_name}_similarities.pt')

    if not os.path.exists(activations_path):
        print(f"  Warning: activations file not found, skipping: {activations_path}")
        return False

    try:
        # Step 1: Build category centers (includes global mean computation)
        print("\n[Step 1] Building category centers (with anisotropy correction)...")
        centers, global_mean, data = build_category_centers(activations_path, centers_path)

        # Step 2: Compute cosine similarities
        print("\n[Step 2] Computing cosine similarities (anisotropy-corrected)...")
        compute_similarities(data, centers, global_mean, similarities_path)

        print(f"\n✓ Model {model_name} done!")
        return True

    except Exception as e:
        print(f"\n✗ Model {model_name} failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Process all models."""
    print("="*60)
    print("Category Center Construction (with Anisotropy Correction)")
    print("="*60)

    ensure_dir(CATEGORY_CENTERS_DIR)
    ensure_dir(SIMILARITIES_DIR)

    success_count = 0
    fail_count = 0
    skipped_count = 0

    for folder_name, model_name in MODEL_CONFIGS:
        activations_path = os.path.join(ACTIVATIONS_BASE_DIR, folder_name, 'activations_merged.pt')

        if not os.path.exists(activations_path):
            print(f"\nSkipping {model_name}: activations file not found")
            skipped_count += 1
            continue

        if process_single_model(folder_name, model_name):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*60)
    print("Done! Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Skipped: {skipped_count}")
    print("="*60)


def process_specific_model(model_name):
    """Process a single specified model."""
    folder_name = None
    for fn, mn in MODEL_CONFIGS:
        if mn == model_name:
            folder_name = fn
            break

    if folder_name is None:
        print(f"Model config not found: {model_name}")
        return False

    return process_single_model(folder_name, model_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build category centers with anisotropy correction.'
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Model name to process (e.g. Qwen3-0.6B). Processes all if omitted.')
    parser.add_argument('--list-models', action='store_true',
                        help='List all supported model names')

    args = parser.parse_args()

    if args.list_models:
        print("Supported models:")
        for folder_name, model_name in MODEL_CONFIGS:
            print(f"  - {model_name}")
    elif args.model:
        process_specific_model(args.model)
    else:
        main()
