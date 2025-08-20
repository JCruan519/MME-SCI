# MME-SCI: A Comprehensive Science Benchmark for Multimodal Large Language Models

Welcome to the MME-SCI repository! This benchmark is designed to rigorously evaluate the performance of multimodal large language models (MLLMs) on science-related tasks, providing a comprehensive testbed for assessing model capabilities in understanding and reasoning about scientific content across text and visual modalities.

[[📖 arXiv Paper](https://arxiv.org/pdf/2508.13938)] [[📊 Dataset](https://huggingface.co/datasets/JCruan/MME-SCI)]

## 🚀 Getting Started

Follow these steps to set up your environment and start using the MME-SCI benchmark.

### Environment Setup

First, configure the required dependencies using Conda:

```bash
# Navigate to the project root directory
cd MME_SCI/

# Create a dedicated Conda environment with Python 3.10
conda create -n mmesci python=3.10 -y

# Activate the environment
conda activate mmesci

# Install required packages
pip install -r requirements.txt
```


## 📊 Benchmark Data

The MME-SCI benchmark dataset is publicly available on Hugging Face Datasets. You can download it here:

- **Dataset Link**: [huggingface.co/datasets/JCruan/MME-SCI](https://huggingface.co/datasets/JCruan/MME-SCI)


## 🖥️ Model Deployment

Evaluation on MME-SCI requires deploying your model using **vllm** (a high-throughput LLM serving library). For full details, see the [vllm documentation](https://github.com/vllm-project/vllm).

> ⚠️ **Critical Note**: Set `--limit-mm-per-prompt` to `6` when deploying with vllm to ensure compatibility with the benchmark.

### Example Deployment Command

Here’s how to deploy a model (e.g., Qwen2.5-VL-7B-Instruct) for evaluation:

```bash
# Specify the model name/path (supports Hugging Face Hub IDs or local paths)
MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# Set visible GPUs (adjust based on your hardware configuration)
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# Launch the vllm server with optimal settings
vllm serve $MODEL \
    --port 12453 \                  # Port for API access
    --api-key token-abc123 \        # Authentication key (customize as needed)
    --dtype auto \                  # Automatically select data type (e.g., float16, bfloat16)
    --served-model-name qwen2_5_vl_7b_vllm \  # Model name for API references
    --gpu-memory-utilization 0.9 \  # Allocate 90% of GPU memory (adjust for your setup)
    --trust-remote-code \           # Trust code from the model repository
    --tensor-parallel-size 4 \      # Number of GPUs for tensor parallelism
    --limit-mm-per-prompt image=6   # Required: Allow up to 6 images per prompt
```


## 📈 Running Evaluation

Once your model is deployed, follow these steps to run the full evaluation pipeline.

### Preparations

Before starting, update the **API configuration** in the following files to match your deployed model:
- `vllm_localapi_eval.py`
- `get_judge_res.py`
- `run_vllm_api_eval_with_metrices.sh`

Adjust parameters like `model name`, `port`, and `API key` to ensure connectivity with your vllm server.

### One-Stop Evaluation

Run the following script to execute the complete evaluation pipeline, including data processing, model querying, and metric calculation:

```bash
# Navigate to the evaluation directory
cd model_eval/

# Execute the evaluation script
bash run_vllm_api_eval_with_metrices.sh
```

This script will automatically process the data, query the deployed model, and compute evaluation metrics for the MME-SCI benchmark.

## 📧 Contact

If you encounter issues or have questions, feel free to open an issue in the repository or contact the maintainers (jackchenruan@sjtu.edu.cn).
