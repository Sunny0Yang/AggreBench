#!/bin/bash
# scripts/run_qa_generation.sh

# 加载环境变量
source scripts/env.sh

# 执行Python脚本
python -m src.pipeline.question_generator \
    "$PROCESSED_DATA_DIR/locomo10.json" \
    --model "qwen3-32b" \
    --min_sessions 1 \
    --max_sessions 4 \
    --min_evidences 5 \
    --max_evidences 10 \
    --num_qa 1 \
    --output_dir "$QA_DIR" \
    --log_level "$LOG_LEVEL"