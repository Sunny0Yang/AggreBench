#!/bin/bash
# scripts/run_data_loader.sh

# 加载环境变量
source scripts/env.sh

# 执行Python脚本
python -m src.pipeline.data_loader \
    "$RAW_DATA_DIR/locomo10.json" \
    --output_dir "$PROCESSED_DATA_DIR" \
    --log_level "$LOG_LEVEL"