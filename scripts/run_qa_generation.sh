#!/bin/bash
# scripts/run_qa_generation.sh

# 加载环境变量
source scripts/env.sh

# 执行Python脚本
python -m src.pipeline.question_generator \
    "$PROCESSED_DATA_DIR/locomo10_processed.json" \
    --model "qwen3-32b" \
    --min_sessions 5 \
    --max_sessions 10 \
    --session_threshold 2 \
    --min_evidences 5 \
    --max_evidences 15 \
    --num_qa 1 \
    --difficulty "hard" \
    --output_dir "$QA_DIR" \
    --log_level "$LOG_LEVEL"