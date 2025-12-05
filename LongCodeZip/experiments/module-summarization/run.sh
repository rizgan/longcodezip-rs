export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_PATH_NAME="qwencoder-7b-instruct"
BASE_RESULT_DIR="results/${MODEL_PATH_NAME}"
BASE_LOG_DIR="logs/${MODEL_PATH_NAME}"

mkdir -p ${BASE_LOG_DIR}
mkdir -p ${BASE_RESULT_DIR}

echo "Starting experiments for ${MODEL_NAME} on GPU ${CUDA_VISIBLE_DEVICES}"

# --- CodeCompressor Method (Fine-grained with Beta) ---
TARGET_TOKENS=(4096)
FINE_RATIOS=(0.5)
BETAS=(0.5)

echo "--- Running CodeCompressor (Fine-grained with various Beta values) ---"
for ratio in "${FINE_RATIOS[@]}"; do
    for tokens in "${TARGET_TOKENS[@]}"; do
        if [[ "${ratio}" == "1.0" ]]; then
            # If fine_ratio is 1.0, only use default beta 0.0
            beta=0.0
            echo "Running CodeCompressor (Fine-grained): target_tokens=${tokens}, fine_ratio=${ratio}, beta=${beta}"
            python main.py \
                --gen_model ${MODEL_NAME} \
                --model_name ${MODEL_PATH_NAME} \
                --method code_compressor \
                --save_dir "${BASE_RESULT_DIR}" \
                --code_compressor_target_token ${tokens} \
                --code_compressor_fine_ratio ${ratio} \
                --importance_beta ${beta} > "${BASE_LOG_DIR}/code_compressor_t${tokens}_fr${ratio}_b${beta}.log" 2>&1
            echo "Finished CodeCompressor (Fine-grained): target_tokens=${tokens}, fine_ratio=${ratio}, beta=${beta}"
        else
            # For other fine_ratios, test different beta values
            for beta in "${BETAS[@]}"; do
                echo "Running CodeCompressor (Fine-grained): target_tokens=${tokens}, fine_ratio=${ratio}, beta=${beta}"
                python main.py \
                    --gen_model ${MODEL_NAME} \
                    --model_name ${MODEL_PATH_NAME} \
                    --method code_compressor \
                    --save_dir "${BASE_RESULT_DIR}" \
                    --code_compressor_target_token ${tokens} \
                    --code_compressor_fine_ratio ${ratio} \
                    --importance_beta ${beta} > "${BASE_LOG_DIR}/code_compressor_t${tokens}_fr${ratio}_b${beta}.log" 2>&1
                echo "Finished CodeCompressor (Fine-grained): target_tokens=${tokens}, fine_ratio=${ratio}, beta=${beta}"
            done
        fi
    done
done
echo "--- Finished CodeCompressor ---"