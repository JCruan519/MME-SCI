# MME-SCI: A Comprehensive and Challenging Science Benchmark for Multimodal Large Language Models

This repository contains the code and resources for the MME-SCI benchmark, designed to evaluate the performance of multimodal large language models on science-related tasks.


## 0. Environment Setup

To get started, set up the required environment using the following commands:

```bash
# Navigate to the project root directory
cd MME_SCI/

# Create a conda environment with Python 3.10
conda create -n mmesci python=3.10 -y

# Activate the environment
conda activate mmesci

# Install dependencies
pip install -r requirements.txt
```


## 1. Benchmark Data

The MME-SCI benchmark could be download from [huggingface](https://github.com/JCruan519/MME-SCI)


## 2. Model Deployment

Evaluation on the MME-SCI benchmark relies on **vllm**. For detailed documentation, refer to the [url](https://github.com/vllm-project/vllm).

**NOTE**: The `--limit-mm-per-prompt` parameter of vllm should be set to 6.

### Example Deployment Command

Deploy a model (e.g., Qwen2.5-VL-7B-Instruct) with the following configuration:

```bash
# Specify the model name/path
MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# Set visible GPUs (adjust based on your hardware)
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# Launch the VLLM server
vllm serve $MODEL \
    --port 12453 \                  # Port for the API server
    --api-key token-abc123 \        # API key for authentication
    --dtype auto \                  
    --served-model-name qwen2_5_vl_7b_vllm \  # Name used to reference the model in API calls
    --gpu-memory-utilization 0.9 \ 
    --trust-remote-code \           
    --tensor-parallel-size 4 \  
    --limit-mm-per-prompt image=6   # Maximum number of images per prompt
```


## 3. Running Evaluation

Once the model is deployed, follow these steps to run the evaluation:

### Preparations
Modify the **API-related parameters** (e.g., model name, port, API key) in the following files to match your deployed model:
   - `vllm_localapi_eval.py`
   - `get_judge_res.py`
   - `run_vllm_api_eval_with_metrices.sh`

### One-Stop Evaluation
Execute the evaluation script to run the full benchmark pipeline:

```bash
# Navigate to the evaluation directory
cd model_eval/

# Run the evaluation script
bash run_vllm_api_eval_with_metrices.sh
```

This script will automatically process the data, query the deployed model, and compute evaluation metrics for the MME-SCI benchmark.
