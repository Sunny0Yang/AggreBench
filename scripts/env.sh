#!/bin/bash
# 基础路径配置
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
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export LOG_DIR="$PROJECT_ROOT/logs"
# export LOG_FILE="$ARTIFACTS_DIR/pipeline.log"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"