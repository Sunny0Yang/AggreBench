# src/evaluation/evaluate.py
import json
import os
import re
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

def extract_answer(answer_str: str) -> str:
    """
    从答案字符串中提取核心答案部分
    - 识别并提取数值答案
    - 提取列表型答案的关键元素
    - 处理复杂答案的摘要提取
    """
    # 尝试提取数值答案
    num_match = re.search(r'The answer is:\s*([\d,.]+)', answer_str)
    if num_match:
        return num_match.group(1).replace(',', '')
    
    # 尝试提取列表型答案
    list_match = re.search(r'The answer is:\s*([\w\s,]+)', answer_str)
    if list_match:
        return list_match.group(1)
    
    # 尝试提取摘要型答案
    summary_match = re.search(r'The answer is:\s*(.*?)$', answer_str)
    if summary_match:
        return summary_match.group(1)
    
    # 回退方案：返回整个答案
    return answer_str

def calculate_em(pred: str, gold: str) -> float:
    """计算完全匹配率（Exact Match）"""
    pred_clean = extract_answer(pred).lower().strip()
    gold_clean = extract_answer(gold).lower().strip()
    return 1.0 if pred_clean == gold_clean else 0.0

def calculate_f1(pred: str, gold: str) -> float:
    """计算F1分数（基于词重叠）"""
    pred_words = set(re.split(r'\W+', extract_answer(pred).lower()))
    gold_words = set(re.split(r'\W+', extract_answer(gold).lower()))
    
    if not pred_words or not gold_words:
        return 0.0
    
    # 计算交集
    common = pred_words & gold_words
    precision = len(common) / len(pred_words)
    recall = len(common) / len(gold_words)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

def calculate_evidence_coverage(pred_memories: List[Dict], gold_evidence: List[str]) -> Tuple[float, float]:
    """
    计算证据覆盖率指标
    返回：(精确率, 召回率)
    """
    if not gold_evidence:
        return 0.0, 0.0
    
    # 提取预测记忆中的关键实体
    pred_entities = set()
    for mem in pred_memories:
        content = mem.get("memory", "").lower()
        # 提取实体（简化实现）
        entities = re.findall(r'\b(\w{4,})\b', content)
        pred_entities.update(entities)
    
    # 提取标准证据中的关键实体
    gold_entities = set()
    for evid in gold_evidence:
        # 提取证据中的关键部分
        key_part = evid.split(':', 1)[-1].strip().lower()
        entities = re.findall(r'\b(\w{4,})\b', key_part)
        gold_entities.update(entities)
    
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
        em = calculate_em(res["response"], res["gold"])
        f1 = calculate_f1(res["response"], res["gold"])
        
        # 证据覆盖率指标
        evidence_precision, evidence_recall = calculate_evidence_coverage(
            res["memories_used"], res["gold_evidence"]
        )
        
        # 收集指标
        metrics["em"].append(em)
        metrics["f1"].append(f1)
        metrics["evidence_precision"].append(evidence_precision)
        metrics["evidence_recall"].append(evidence_recall)
        metrics["latency"].append(res["latency"])
        metrics["tokens_used"].append(res["tokens_used"])
        
        # 添加指标到结果记录
        res["metrics"] = {
            "em": em,
            "f1": f1,
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall
        }
    
    # 计算汇总指标
    summary = {
        "num_samples": len(results),
        "em": sum(metrics["em"]) / len(results),
        "f1": sum(metrics["f1"]) / len(results),
        "evidence_precision": sum(metrics["evidence_precision"]) / len(results),
        "evidence_recall": sum(metrics["evidence_recall"]) / len(results),
        "avg_latency": sum(metrics["latency"]) / len(results),
        "avg_tokens": sum(metrics["tokens_used"]) / len(results),
        "per_category": {}
    }
    
    return summary, results

def analyze_by_category(results: List[Dict]) -> Dict:
    """按问题类别分析结果"""
    category_metrics = defaultdict(lambda: defaultdict(list))
    
    for res in results:
        # 提取问题类别（简化实现）
        category = "numeric" if re.search(r'\d+', res["gold"]) else "textual"
        
        # 收集指标
        em = res["metrics"]["em"]
        f1 = res["metrics"]["f1"]
        
        category_metrics[category]["em"].append(em)
        category_metrics[category]["f1"].append(f1)
    
    # 计算每类平均值
    summary = {}
    for cat, metrics in category_metrics.items():
        summary[cat] = {
            "num_samples": len(metrics["em"]),
            "avg_em": sum(metrics["em"]) / len(metrics["em"]),
            "avg_f1": sum(metrics["f1"]) / len(metrics["f1"])
        }
    
    return summary

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
    
    # 按类别分析
    category_summary = analyze_by_category(detailed_results)
    eval_summary["category_analysis"] = category_summary
    
    # 打印摘要
    logger.info("\n评估摘要:")
    logger.info(f"样本数量: {eval_summary['num_samples']}")
    logger.info(f"完全匹配率 (EM): {eval_summary['em']:.3f}")
    logger.info(f"F1分数: {eval_summary['f1']:.3f}")
    logger.info(f"证据精确率: {eval_summary['evidence_precision']:.3f}")
    logger.info(f"证据召回率: {eval_summary['evidence_recall']:.3f}")
    logger.info(f"平均延迟: {eval_summary['avg_latency']:.3f} 秒")
    logger.info(f"平均Token使用: {eval_summary['avg_tokens']:.1f}")
    
    # 保存结果
    input_name = os.path.basename(args.input_file).split('.')[0]
    output_file = f"{input_name}_score.json"
    output_path = os.path.join(args.output_dir, output_file)
    
    save_evaluation(eval_summary, detailed_results, output_path)

if __name__ == "__main__":
    main()