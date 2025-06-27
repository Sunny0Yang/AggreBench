# src/evaluation/evaluate.py
import json
import os
import re
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def calculate_evidence_coverage(memories_used: List[Dict], gold_evidence: List[str]) -> Tuple[float, float]:
    """
    计算证据覆盖率指标
    返回：(精确率, 召回率)
    """
    if not gold_evidence:
        return 0.0, 0.0
    
    # 提取预测记忆中的关键实体
    pred_entities = set(mem["dia_id"] for mem in memories_used)
    print(f"pred_entities:{pred_entities}")
    # 提取标准证据中的关键实体
    gold_entities = set(evid.split(":", 2)[:2] for evid in gold_evidence)
    print(f"glod_entities:{gold_entities}")
    # 计算交集
    common = pred_entities & gold_entities
    
    # 计算精确率和召回率
    precision = len(common) / len(pred_entities) if pred_entities else 0.0
    recall = len(common) / len(gold_entities) if gold_entities else 0.0
    
    return precision, recall

def evaluate_results(results: List[Dict]) -> Dict:
    """执行完整评估"""
    metrics = defaultdict(list)
    
    for res in results:
        # 基础答案指标
        correct = (res["gold"] == res["response"])
        # TODO 证据覆盖率指标
        # evidence_precision, evidence_recall = calculate_evidence_coverage(
        #     res["memories_used"], res["gold_evidence"]
        # )
        
        # 收集指标
        metrics["correct"].append(correct)
        # metrics["evidence_precision"].append(evidence_precision)
        # metrics["evidence_recall"].append(evidence_recall)
        metrics["latency"].append(res["latency"])
        metrics["tokens_used"].append(res["tokens_used"])
        
        # 添加指标到结果记录
        # res["metrics"] = {
        #     "evidence_precision": evidence_precision,
        #     "evidence_recall": evidence_recall
        # }
    
    # 计算汇总指标
    summary = {
        "num_samples": len(results),
        "acc": sum(metrics["correct"]) / len(results),
        # "evidence_precision": sum(metrics["evidence_precision"]) / len(results),
        # "evidence_recall": sum(metrics["evidence_recall"]) / len(results),
        "avg_latency": sum(metrics["latency"]) / len(results),
        "avg_tokens": sum(metrics["tokens_used"]) / len(results)
    }
    
    return summary, results

def save_evaluation(eval_summary: Dict, detailed_results: List[Dict], output_path: str):
    """保存评估结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存详细结果（包含指标）
    detailed_output = os.path.splitext(output_path)[0] + "_detailed.json"
    with open(detailed_output, 'w') as f:
        json.dump({
            "summary": eval_summary,
            "results": detailed_results
        }, f, indent=2)
    
    # 保存摘要报告
    with open(output_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    logger.info(f"评估结果已保存至: {output_path}")

def main():
    import argparse
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="评估记忆系统结果")
    parser.add_argument("--input_file", required=True, help="结果文件路径")
    parser.add_argument("--output_dir", default="scores", help="输出目录")
    
    args = parser.parse_args()
    
    # 加载结果
    try:
        with open(args.input_file, 'r') as f:
            results = json.load(f)
        logger.info(f"已加载 {len(results)} 个结果")
    except Exception as e:
        logger.error(f"加载结果失败: {str(e)}")
        return
    
    # 执行评估
    eval_summary, detailed_results = evaluate_results(results)
    
    # 打印摘要
    logger.info("\n评估摘要:")
    logger.info(f"样本数量: {eval_summary['num_samples']}")
    logger.info(f"准确率: {eval_summary['acc']:.3f}")
    # logger.info(f"证据精确率: {eval_summary['evidence_precision']:.3f}")
    # logger.info(f"证据召回率: {eval_summary['evidence_recall']:.3f}")
    logger.info(f"平均延迟: {eval_summary['avg_latency']:.3f} 秒")
    logger.info(f"平均Token使用: {eval_summary['avg_tokens']:.1f}")
    
    # 保存结果
    input_name = "_".join((os.path.basename(args.input_file).split('.')[0]).split("_")[:2])
    output_file = f"{input_name}_score.json"
    output_path = os.path.join(args.output_dir, output_file)
    
    save_evaluation(eval_summary, detailed_results, output_path)

if __name__ == "__main__":
    main()