import os
import json
from typing import List, Dict, Tuple, Any
from utils.data_struct import MultiModalTurn, Table, Session, Conversation, ConversationDataset
from utils.session_simulator import SessionSimulator
from utils.prompt_templates import PERSONA

import logging
logger = logging.getLogger(__name__)

class MedicalDialogueGenerator:
    def __init__(self, input_dir: str, output_dir: str, 
                 model: str, cache_dir: str,
                 max_turns: int = 5, is_step: bool = True):
        """
        医疗数据对话生成器
        
        参数:
        input_dir: 预处理数据目录路径
        output_dir: 最终输出目录
        model: 用于对话生成的模型
        cache_dir: 缓存生成对话的目录
        max_turns: 最大对话轮数 (默认: 5)
        is_step: 是否启用逐步生成 (默认: True)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model = model
        self.cache_dir = cache_dir
        self.max_turns = max_turns
        self.is_step = is_step
        
        # 初始化会话模拟器
        self.session_simulator = SessionSimulator(
            model=self.model, 
            max_turns=self.max_turns,
            is_step=self.is_step,
            cache_dir=self.cache_dir,
            domain="medical"
        )
        self.persona = PERSONA["medical"]
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def generate_dialogues(self):
        """加载预处理数据并生成伪对话"""
        logger.info(f"开始生成医疗数据伪对话，来源目录: {self.input_dir}")
        
        # 加载预处理数据
        dataset = self._load_preprocessed_data()
        if not dataset:
            logger.error("无法加载预处理数据")
            return
        
        # 为每个会话生成对话
        for conversation in dataset.conversations:
            for session in conversation.sessions:
                self._generate_dialogue_for_session(session)
        
        # 保存最终数据集
        self._save_final_dataset(dataset)
        logger.info("伪对话生成完成")
    
    def _load_preprocessed_data(self) -> ConversationDataset:
        """加载预处理数据"""
        input_path = os.path.join(self.input_dir, "filtered_data.json")
        if not os.path.exists(input_path):
            logger.error(f"预处理数据文件不存在: {input_path}")
            return None
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = []
            for conv_data in data:
                sessions = []
                for session_data in conv_data["sessions"]:
                    tables = []
                    for table_data in session_data["tables"]:
                        table = Table(
                            headers=table_data["headers"],
                            rows=table_data["rows"]
                        )
                        table.table_type = table_data.get("table_type", "Unknown")
                        tables.append(table)
                    
                    session = Session(
                        session_id=session_data["session_id"],
                        time=session_data["time"],
                        participants=session_data["participants"],
                        turns=session_data["turns"],
                        tables=tables
                    )
                    sessions.append(session)
                
                conversation = Conversation(
                    conversation_id=conv_data["conversation_id"],
                    speakers=conv_data["speakers"],
                    sessions=sessions
                )
                conversations.append(conversation)
            
            return ConversationDataset(conversations=conversations)
        except Exception as e:
            logger.error(f"加载预处理数据时出错: {str(e)}")
            return None
    
    def _generate_dialogue_for_session(self, session: Session):
        """为会话生成伪对话"""
        logger.info(f"为会话 {session.id} 生成伪对话")
        
        # 将表格转换为证据
        evidences = self._tables_to_evidences(session.tables)
        
        if not evidences:
            logger.warning(f"会话 {session.id} 没有有效证据")
            return
        
        # 生成对话
        logger.info(f"为会话 {session.id} 生成伪对话，共有 {len(evidences)} 条证据")
        
        # 生成对话
        dialog = self.session_simulator.generate_dialog(
            evidences=evidences,
            persona=self.persona
        )
        
        # 转换为回合格式
        turns = []
        for i, turn in enumerate(dialog):
            turns.append({
                "turn_id": f"{session.id}_turn_{i+1}",
                "speaker": turn["speaker"],
                "content": turn["content"],
                "mentioned_evidence": turn.get("mentioned_evidence", [])
            })
        
        # 更新会话的对话回合
        session.turns = turns
    
    def _tables_to_evidences(self, tables: List[Table]) -> List[Tuple]:
        """
        将表格转换为证据元组列表
        Evidence = Tuple[patient_id, timestamp, table_type, ...其他值]
        """
        evidences = []
        for table in tables:
            table_type = getattr(table, "table_type", "Unknown")
            for row in table.rows:
                try:
                    if isinstance(row, dict):
                        row_values = tuple(row.values())
                    else:
                        row_values = tuple(row)
                    
                    evidence = (row_values[:2] + (str(table_type),) + row_values[2:])
                    
                    evidences.append(evidence)
                    logger.debug(f"evidence:{evidence}")    
                except Exception as e:
                    logger.warning(f"创建证据时出错: {str(e)}")
                    logger.debug(f"问题行: {row}")
        return evidences
    
    def _save_final_dataset(self, dataset: ConversationDataset):
        """保存最终数据集"""
        output_path = os.path.join(self.output_dir, "medical_processed.json")
        
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
                    "turns": session.turns,
                    "tables": []
                }
                
                # 处理表格数据
                for table in session.tables:
                    table_data = {
                        "headers": table.headers,
                        "rows": table.rows,
                        "table_type": getattr(table, 'table_type', 'Unknown')
                    }
                    session_data["tables"].append(table_data)
                
                conv_data["sessions"].append(session_data)
            serialized.append(conv_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        
        logger.info(f"已保存最终数据集至: {output_path}")