#!/bin/bash
# scripts/evaluate.sh

# 加载环境变量
source scripts/env.sh

# 执行Python脚本
python -m src.evaluate \
    --input_file "results/mem0_locomo10_results.json" \
    --output_dir "scores/"