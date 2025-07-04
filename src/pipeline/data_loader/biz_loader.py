# src/pipeline/bizfin_loader.py

import json
import os
import re
import logging
from typing import List, Dict
from utils.struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset
from utils.session_simulator import SessionSimulator
from utils.prompt_templates import PERSONA
logger = logging.getLogger(__name__)

class BizFinLoader:
    def __init__(self, model:str, max_turns:int, is_step:bool, cache_dir: str,
        combine_size = 10, generate_pseudo_dialogue=True
        ):
        self.logger = logger
        self.combine_size = combine_size
        self.session_simulator = SessionSimulator(model=model, max_turns=max_turns, is_step=is_step, cache_dir=cache_dir)
        self.generate_pseudo_dialogue = generate_pseudo_dialogue
        self.persona = PERSONA["financial"]

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

    def _create_combined_conversation(self, samples: List[Dict], conversation_id: str) -> Conversation:
        """创建组合对话对象"""
        sessions = []
        session_counter = 1  # 会话计数器
        
        # 处理每个样本
        for sample_idx, sample in enumerate(samples):
            # 提取样本中的会话
            sample_session = self._extract_session(sample, conversation_id, session_counter)
            sessions.append(sample_session)
            session_counter += 1
        
        return Conversation(
            conversation_id=conversation_id,
            speakers=["Assistant"],
            sessions=sessions,
        )

    def _table_to_evidences(self, table_objects: List[Table]) -> List[str]:
        """将表格转换为证据列表"""
        evidences = []
        for table in table_objects:
            for row in table.rows:
                stock_code = row.get("股票代码", "未知代码")
                stock_name = row.get("股票简称", "未知股票")
                prefix = f"{stock_name}({stock_code})"
                for key, value in row.items():
                    if key in ["股票代码", "股票简称"]:
                        continue
                    evidences.append(f"{prefix} {key}: {value}")
        evidences = list(set(evidences))
        return evidences

    def _extract_session(self, sample: Dict, conversation_id: str, start_index: int) -> Session:
        """从样本中提取会话并重新编号"""
        sessions = []
        messages = sample.get("messages", [])
        
        # 提取所有表格数据
        all_tables = []
        for msg in messages:
            if msg.get("role") == "user":
                for content in msg.get("content", []):
                    if content.get("type") == "text":
                        tables = self._extract_tables(content["text"])
                        all_tables.extend(tables)
        if not all_tables:
            return []

        session_id = f"{conversation_id}_session_{start_index}"

        # 创建Table对象列表
        table_objects = []
        for table_data in all_tables:
            table = Table(
                headers=table_data["headers"],
                rows=table_data["rows"]
            )
            table_objects.append(table)
        
        # 为每个表格创建一个session
        turns = []
        if self.generate_pseudo_dialogue:
            conv_id = conversation_id.split('_')[-1]
            if not( int(conv_id) > 2 and int(start_index) > 3): # DEBUG
                # 生成表格evidence
                evidences = self._table_to_evidences(table_objects)
                # 生成对话回合
                dialog = self.session_simulator.generate_dialog(
                    evidences=evidences,
                    persona=self.persona
                )
                
                # 转换为MultiModalTurn对象
                for turn_idx, turn in enumerate(dialog):
                    turns.append(MultiModalTurn(
                        turn_id=f"{session_id}_turn_{turn_idx+1}",
                        speaker=turn["speaker"],
                        content=turn["content"]
                    ))
        else:
            # 如果不需要伪对话，添加简单的表头信息
            turns.append(MultiModalTurn(
                turn_id=f"{session_id}_title",
                speaker="Assistant",
                content=f"Session contains {len(all_tables)} tables"
            ))
            
        session = Session(
            session_id=session_id,
            time="N/A",
            participants = ["User", "Assistant"] if self.generate_pseudo_dialogue else ["Assistant"],
            turns=turns,
            tables=table_objects
        )
        
        return session

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
            for group_idx in range(0, len(data_lines), self.combine_size):
                group_samples = data_lines[group_idx:group_idx+self.combine_size]
                if not group_samples:
                    continue
                    
                # 创建组合对话
                conversation_id = f"conv_{group_idx//self.combine_size+1}"
                conversation = self._create_combined_conversation(group_samples, conversation_id)
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
                    } for turn in session.turns],
                    "tables": [{
                        "headers": table.headers,
                        "rows": table.rows
                    } for table in session.tables],
                }
                conv_data["sessions"].append(session_data)
            serialized.append(conv_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

def main():
    loader = BizFinLoader(model=args.model, 
                        max_turns=args.max_turns, is_step= args.is_step,
                        cache_dir=args.cache_dir, combine_size=args.combine_size,
                        generate_pseudo_dialogue=args.generate_pseudo_dialogue)
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