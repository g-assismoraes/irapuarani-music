#!/bin/bash
#BEFORE RUN IT, RUN ON TERMINAL ----> chmod +x run.sh

HF_TOKEN=""
INPUT_PATH='data/txt/'

# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_PATH="llama/"
# CUDA_VISIBLE_DEVICES=1 python3 gen.py --model_id $MODEL_ID --hf_token $HF_TOKEN --input_path $INPUT_PATH --output_path $OUTPUT_PATH

# MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
# OUTPUT_PATH="mistral/"
# python3 gen.py --model_id $MODEL_ID --hf_token $HF_TOKEN --input_path $INPUT_PATH --output_path $OUTPUT_PATH

# MODEL_ID="microsoft/Phi-3-small-8k-instruct"
# OUTPUT_PATH="phi/"
# python3 gen.py --model_id $MODEL_ID --hf_token $HF_TOKEN --input_path $INPUT_PATH --output_path $OUTPUT_PATH

MODEL_ID="google/gemma-2-9b-it"
OUTPUT_PATH="gemma/"
CUDA_VISIBLE_DEVICES=1 python3 gen.py --model_id $MODEL_ID --hf_token $HF_TOKEN --input_path $INPUT_PATH --output_path $OUTPUT_PATH

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH="qwen/"
CUDA_VISIBLE_DEVICES=1 python3 gen.py --model_id $MODEL_ID --hf_token $HF_TOKEN --input_path $INPUT_PATH --output_path $OUTPUT_PATH

