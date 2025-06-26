#!/bin/bash
# scripts/evaluate_mem0.sh

# 加载环境变量
source scripts/env.sh

# 执行Python脚本
python -m src.run_experiments \
    --technique "mem0" \
    --dataset "../artifacts/processed/locomo10_processed.json" \
    --qa_file "../artifacts/qa/locomo10_processed_qa.json" \
    --output_dir "results/" \
    --batch_size 2