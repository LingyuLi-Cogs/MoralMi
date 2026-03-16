import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def process_jsonl_vllm(input_file, output_file, model_path):
    """
    Run 8-GPU parallel inference with vLLM.
    """

    # 1. Initialize tokenizer for chat template
    print(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. Read all data and preprocess prompts
    print(f"Reading input file: {input_file}")
    original_data = []
    prompts = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            original_data.append(data)

            # Build chat template
            # System prompt is intentionally Chinese for the FLAMES benchmark
            messages = [
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": data.get("prompt", "")}
            ]

            # Apply chat template to format input for the model
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt_text)

    # 3. Initialize vLLM (8-GPU tensor parallel)
    # tensor_parallel_size=8 automatically shards the model across 8 GPUs
    print(f"Loading model and initializing vLLM (8-GPU parallel)...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,  # key: number of GPUs
        trust_remote_code=True,
        gpu_memory_utilization=0.9,  # adjust based on GPU memory; default is 0.9
        dtype="float16"  # or "bfloat16"
    )

    # 4. Set generation parameters
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.7,
    )

    # 5. Run batch inference
    print(f"Starting inference on {len(prompts)} items...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Save results
    print(f"Inference complete, saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, output in enumerate(outputs):
            # Extract generated text
            generated_text = output.outputs[0].text

            # Merge with original data
            result_data = original_data[i]
            result_data["response"] = generated_text

            f_out.write(json.dumps(result_data, ensure_ascii=False) + "\n")

    print("All done!")

if __name__ == "__main__":
    # Configure paths
    INPUT_JSONL = "./moral_prognosis/Flames_1k_Chinese.jsonl"
    OUTPUT_JSONL = "./moral_prognosis/outputs/qwen3-8b-zh-vllm.jsonl"
    MODEL_PATH = ""  # set your model path here

    process_jsonl_vllm(INPUT_JSONL, OUTPUT_JSONL, MODEL_PATH)
