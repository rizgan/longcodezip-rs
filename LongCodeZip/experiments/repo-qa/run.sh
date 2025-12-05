#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_PATH_NAME="qwencoder-7b-instruct"
BACKEND="vllm"
COMPRESSION_METHOD="code_compressor"
BASE_RESULT_DIR="code_compressor_exp_results"
BASE_LOG_DIR="logs-combinations"

mkdir -p ${BASE_LOG_DIR}
mkdir -p ${BASE_RESULT_DIR}

echo "Starting experiments for ${MODEL_NAME}"

# Configuration arrays
COMPRESSION_RATIOS=(0.1 0.2 0.3 0.4)
GPU_IDS=(0 1 2 3)

echo "--- Running CodeCompressor with various compression ratios ---"
for i in "${!COMPRESSION_RATIOS[@]}"; do
    ratio="${COMPRESSION_RATIOS[$i]}"
    gpu_id="${GPU_IDS[$i]}"
    
    echo "Running CodeCompressor: compression_ratio=${ratio} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python main.py \
        --model ${MODEL_NAME} \
        --backend ${BACKEND} \
        --compression-method ${COMPRESSION_METHOD} \
        --compression-ratio ${ratio} \
        --result-dir ${BASE_RESULT_DIR} \
        --rank-only > "${BASE_LOG_DIR}/7B_code_compressor_${ratio}_rank_only_true.log" 2>&1 &
    echo "Started CodeCompressor: compression_ratio=${ratio} on GPU ${gpu_id}"
done

echo "--- All CodeCompressor experiments started ---"