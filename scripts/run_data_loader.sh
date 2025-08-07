#!/bin/bash
# scripts/run_data_loader.sh

# 加载环境变量
# source scripts/env.sh
source activate nlp_env

export PROJECT_ROOT=$(pwd)
export ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"
export CONFIG_DIR="$PROJECT_ROOT/configs"

# API 密钥（从安全存储获取）
export OPENAI_API_KEY=
export DASHSCOPE_API_KEY='sk-8e9a3860f9bb47d6bc5a32aa294afa1d'

# 数据集路径
export RAW_DATA_DIR="$ARTIFACTS_DIR/raw"
export PROCESSED_DATA_DIR="$ARTIFACTS_DIR/processed"
export QA_DIR="$ARTIFACTS_DIR/qa"
export FINAL_DATASET_DIR="$ARTIFACTS_DIR/final_dataset.jsonl"

# 日志配置（可覆盖）
export LOG_LEVEL=${LOG_LEVEL:-"DEBUG"}
export LOG_DIR="$PROJECT_ROOT/logs/data"
# export LOG_FILE="$ARTIFACTS_DIR/pipeline.log"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 执行Python脚本
# BizFinLoader
    # --generate_pseudo_dialogue \
    # --is_step \
# python -m src.pipeline.data_loader.biz_loader \
#     "$RAW_DATA_DIR/biz_combined.jsonl" \
#     --output_dir "$PROCESSED_DATA_DIR" \
#     --log_level "$LOG_LEVEL" \
#     --model "qwen-plus-latest" \
#     --max_turns 10 \
#     --cache_dir "$PROCESSED_DATA_DIR/cache" \
#     --combine_size 4

# python -m src.pipeline.data_loader.locomo_loader \
#     "$RAW_DATA_DIR/locomo_demo.json" \
#     --output_dir "$PROCESSED_DATA_DIR" \
#     --log_level "$LOG_LEVEL"
# python -m src.pipeline.data_loader.medical_loader \
#   --input_dir artifacts/raw/medical \
#   --output_dir artifacts/med_processed \
#   --max_events 10 \
#   --time_window 4 \
#   --generate_pseudo_dialogue \
#   --model qwen-turbo-latest \
#   --cache_dir artifacts/med_processed/cache \
#   --max_turns 4 \
#   --is_step
# python -m src.pipeline.med_loader.main preprocess \
#   --input_dir artifacts/raw/medical \
#   --output_dir artifacts/preprocessed \
#   --max_events 8 \
#   --time_window 3

python -m src.pipeline.med_loader.main generate \
  --input_dir artifacts/preprocessed \
  --output_dir artifacts/med_processed \
  --model qwen-turbo-latest \
  --cache_dir artifacts/med_processed/cache \
  --max_turns 5 \
  --is_step