#!/bin/bash
# scripts/run_qa_generation.sh

# 加载环境变量
source scripts/env.sh

# 执行Python脚本
python -m src.pipeline.question_generator \
    "$PROCESSED_DATA_DIR/locomo10_processed.json" \
    --model "qwen3-32b" \
    --min_sessions 2 \
    --max_sessions 5 \
    --min_evidences 3 \
    --max_evidences 8 \
    --num_qa 1 \
    --output_dir "$QA_DIR" \
    --log_level "$LOG_LEVEL"