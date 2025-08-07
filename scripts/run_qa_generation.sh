#!/bin/bash
# scripts/run_qa_generation.sh

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
export LOG_DIR="$PROJECT_ROOT/logs/qa"
# export LOG_FILE="$ARTIFACTS_DIR/pipeline.log"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
# 执行Python脚本
# 不同数据集换不同cache dir
# python -m src.pipeline.question_generator \
#     "$PROCESSED_DATA_DIR/biz_demo_processed.json" \
#     --model "qwen-turbo-latest" \
#     --min_sessions 1 \
#     --max_sessions 4 \
#     --session_threshold 1 \
#     --min_evidences 5 \
#     --max_evidences 20 \
#     --easy 2 \
#     --medium 5 \
#     --hard 3 \
#     --cache_dir "$QA_DIR/v1/biz_cache/biz_cache0" \
#     --max_preferred_examples 3 \
#     --max_disliked_examples 5 \
#     --enable_validation \
#     --output_dir "$QA_DIR/v1"
python -m src.pipeline.question_generator \
    "artifacts/med_processed/processed_dataset.json" \
    --model "qwen-turbo-latest" \
    --min_sessions 1 \
    --max_sessions 4 \
    --session_threshold 1 \
    --min_evidences 1 \
    --max_evidences 20 \
    --easy 1 \
    --medium 0 \
    --hard 0 \
    --cache_dir "artifacts/med_qa/cache" \
    --max_preferred_examples 3 \
    --max_disliked_examples 5 \
    --output_dir "artifacts/med_qa/"
    # --enable_validation \
    