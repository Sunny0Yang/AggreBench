# src/pipeline/bizfin_loader.py

import json
import os
import re
import logging
from typing import List, Dict
from utils.struct import MultiModalTurn, Session, Conversation, ConversationDataset

logger = logging.getLogger(__name__)

class BizFinLoader:
    def __init__(self):
        self.logger = logger

    def _extract_tables(self, text_content: str) -> List[Dict]:
        """从文本内容中提取表格数据"""
        tables = []
        
        # 表格模式：匹配以 | 开头的行
        table_pattern = r'\|(.+?)\|\n\|([\s\-:]+)\|(.+?)\n([\s\S]+?)(?=\n\n|\n###|$)'
        matches = re.findall(table_pattern, text_content, re.MULTILINE)
        
        for match in matches:
            # 提取表头
            headers = [h.strip() for h in match[0].split('|') if h.strip()]
            
            # 提取数据行
            rows = []
            for row in match[3].split('\n'):
                if not row.strip() or row.startswith('|---'):
                    continue
                cols = [c.strip() for c in row.split('|') if c.strip()]
                if len(cols) == len(headers):
                    rows.append(dict(zip(headers, cols)))
            
            if rows:
                tables.append({
                    "headers": headers,
                    "rows": rows
                })
        
        return tables

    def _process_sample(self, sample: Dict, conversation_id: str) -> Conversation:
        """处理单个样本数据"""
        messages = sample.get("messages", [])
        sessions = []
        
        # 提取所有表格数据
        all_tables = []
        for msg in messages:
            if msg.get("role") == "user":
                for content in msg.get("content", []):
                    if content.get("type") == "text":
                        tables = self._extract_tables(content["text"])
                        all_tables.extend(tables)
        
        # 为每个表格创建一个会话
        for table_idx, table in enumerate(all_tables):
            session_id = f"{conversation_id}_table_{table_idx+1}"
            
            # 创建回合 - 每行数据作为一个回合
            turns = []
            for row_idx, row in enumerate(table["rows"]):
                # 将行数据格式化为字符串
                row_content = ", ".join([f"{k}: {v}" for k, v in row.items()])
                turns.append(MultiModalTurn(
                    turn_id=f"{session_id}_row_{row_idx+1}",
                    speaker="System",
                    content=f"Row {row_idx+1}: {row_content}"
                ))
            
            # 创建会话对象
            sessions.append(Session(
                session_id=session_id,
                time="N/A",  # 结构化数据没有时间信息
                participants=["System"],
                turns=turns
            ))
        
        # 提取问题
        questions = []
        for msg in messages:
            if msg.get("role") == "user":
                for content in msg.get("content", []):
                    if content.get("type") == "text":
                        # 提取问题
                        q_match = re.search(r'### 用户问题\nquestion: (.+?)\n', content["text"])
                        if q_match:
                            questions.append({
                                "question": q_match.group(1).strip(),
                                "answer": self._extract_answer(sample.get("choices", [])),
                                "evidence": [],
                                "qa_index": len(questions)
                            })
        
        return Conversation(
            conversation_id=conversation_id,
            speakers=["System", "User"],
            sessions=sessions
        )

    def _extract_answer(self, choices: List) -> str:
        """从choices中提取答案"""
        if choices:
            choice = choices[0]
            for content in choice.get("message", {}).get("content", []):
                if content.get("type") == "text":
                    return content["text"]
        return ""

    def load(self, file_path: str) -> ConversationDataset:
        """加载BizFinBench数据集文件并转换为ConversationDataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        logger.info(f"加载BizFinBench数据集: {file_path}")
        
        try:
            # 读取JSONL文件
            data_lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data_lines.append(json.loads(line))
            
            # 处理每个样本
            conversations = []
            for idx, sample in enumerate(data_lines):
                conversation_id = f"conv_{idx+1}"
                conversation = self._process_sample(sample, conversation_id)
                conversations.append(conversation)
            
            # 创建数据集对象
            logger.info(f"成功加载 {len(conversations)} 个对话")
            return ConversationDataset(conversations=conversations)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e.msg} (位置: {e.pos})")
            raise
        except Exception as e:
            logger.error(f"加载数据集时出错: {str(e)}")
            raise

    @staticmethod
    def save(dataset: ConversationDataset, output_path: str):
        """将数据集保存到文件"""
        serialized = []
        for conversation in dataset.conversations:
            conv_data = {
                "conversation_id": conversation.id,
                "speakers": conversation.speakers,
                "sessions": []
            }
            for session in conversation.sessions:
                session_data = {
                    "session_id": session.id,
                    "time": session.time,
                    "participants": session.participants,
                    "turns": [{
                        "turn_id": turn.id,
                        "speaker": turn.speaker,
                        "content": turn.content
                    } for turn in session.turns]
                }
                conv_data["sessions"].append(session_data)
            serialized.append(conv_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

def main():
    loader = BizFinLoader()
    dataset = loader.load(args.input_data)
    
    # 保存处理后的数据集
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{os.path.splitext(os.path.basename(args.input_data))[0]}_processed.json"
    output_path = os.path.join(args.output_dir, filename)
    loader.save(dataset, output_path)
    logger.info(f"处理后的数据已保存至: {output_path}")

if __name__ == "__main__":
    import argparse
    from utils.logger import setup_logging
    from utils.params import get_base_parser, data_loader_args
    
    logger = setup_logging()
    
    # 参数解析
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser = data_loader_args(parser)
    args = parser.parse_args()
    
    main()