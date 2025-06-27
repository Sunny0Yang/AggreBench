import argparse
import os
import json
import logging
from datetime import datetime
from src.technique.base import MemoryManager
# from src.technique.langmem import LangMemManager
from src.technique.mem0 import Mem0Manager
# from src.technique.rag import RAGManager
# from src.technique.zep import ZepManager

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def load_json(file_path: str) -> dict:
    """加载JSON文件"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载JSON失败: {str(e)}")
        raise

def save_results(results: list, output_path: str) -> None:
    """保存结果到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")
        raise

def find_session_data(conversation_id: str, session_id: str, dataset: dict) -> dict:
    """在数据集中查找特定会话数据"""
    for conv in dataset:
        if conv['conversation_id'] == conversation_id:
            for session in conv['sessions']:
                if session['session_id'] == session_id:
                    return session
    logger.warning(f"未找到会话: {conversation_id}/{session_id}")
    return None

def add_memories_for_qa(manager: MemoryManager, qa: dict, dataset: dict):
    """为单个QA添加相关会话到记忆系统"""
    for session_id in qa['session_ids']:
        session_data = find_session_data(qa['conversation_id'], session_id, dataset)
        if session_data:
            logger.info(f"处理会话: {session_id}")
            manager.mem_add(session_data, qa['qa_index']) # qa_index作为用户唯一标识符，记忆不会冲突
        else:
            logger.error(f"无法添加会话: {session_id} (对话: {qa['conversation_id']})")

def process_qa(manager: MemoryManager, qa: dict, dataset: dict) -> dict:
    """处理单个QA对"""
    result = {
        "qa_id": qa.get("qa_index"),
        "conversation_id": qa['conversation_id'],
        "question": qa['question'],
        "gold": qa['answer'],
        "gold_evidence": qa['evidence'],
        "session_ids": qa['session_ids'],
        "technique": manager.technique_name,
        "response": "",
        "memories_used": [],
        "latency": 0,
        "tokens_used": 0
    }
    
    # try:
    #     # 添加相关会话到记忆
    #     add_memories_for_qa(manager, qa, dataset)
    # except Exception as e:
    #     logger.error(f"添加记忆失败{result['qa_id']}: {str(e)}")
    #     result["error"] = str(e)    
    try:    # 执行搜索和生成
        start_time = datetime.now()
        memories = manager.mem_search(qa)
        logger.info(f"########搜索完成#########")
        response = manager.generate_response(qa['question'], memories)
        latency = (datetime.now() - start_time).total_seconds()
        
        # 更新结果
        result.update({
            "response": response,
            "memories_used": memories,
            "latency": latency,
            "tokens_used": manager.get_token_count()
        })
    except Exception as e:
        logger.error(f"处理QA失败 {result['qa_id']}: {str(e)}")
        result["error"] = str(e)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--technique", default="mem0", choices=[
        "mem0", "rag", "langmem", "zep", "openai", "full-context"
    ], help="Memory technique to use")
    parser.add_argument("--dataset", default="artifacts/processed/locomo10_processed.json", 
                       help="Path to processed dataset")
    parser.add_argument("--qa_file", default="artifacts/qa/locomo10_processed_qa.json", 
                       help="Path to QA dataset")
    parser.add_argument("--output_dir", default="results", help="Output directory for results")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for RAG processing")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top memories to retrieve")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing")

    args = parser.parse_args()
    
    # 加载数据集
    dataset = load_json(args.dataset)
    qa_data = load_json(args.qa_file)
    logger.info(f"数据集加载成功: {len(dataset)}组对话, {len(qa_data)}个QA")
    
    # 初始化记忆管理器
    technique_config = {
        "batch_size": args.batch_size
    }
    
    manager = None
    if args.technique == "mem0":
        manager = Mem0Manager(**technique_config)
    # elif args.technique == "rag":
    #     manager = RAGManager(**technique_config)
    # elif args.technique == "langmem":
    #     manager = LangMemManager(**technique_config)
    # elif args.technique == "zep":
    #     manager = ZepManager(**technique_config)
    
    if not manager:
        logger.error(f"无效的记忆技术: {args.technique}")
        return
    
    # 处理所有QA对
    results = []
    for qa in qa_data:
        logger.info(f"处理QA: {qa.get('qa_index', 'unknown')} (对话: {qa['session_ids']})")
        result = process_qa(manager, qa, dataset)
        results.append(result)
    
    # 保存结果
    output_file = f"{args.technique}_{os.path.splitext(os.path.basename(args.dataset))[0]}_results.json"
    output_path = os.path.join(args.output_dir, output_file)
    save_results(results, output_path)
    
    logger.info(f"实验完成！生成 {len(results)} 个结果")

if __name__ == "__main__":
    main()