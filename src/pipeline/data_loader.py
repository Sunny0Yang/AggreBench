# src/pipeline/data_loader.py
import json
import os
import re
import logging
from typing import List, Dict
from utils.struct import DialogueTurn, MultiModalTurn, Session, ConversationDataset
class LoCoMoLoader:
    def __init__(self):
        self.session_pattern = re.compile(r'session_(\d+)$')
        self.timestamp_pattern = re.compile(r'session_(\d+)_date_time$')

    def is_session_key(self, key: str) -> bool:
        """检查键是否为会话内容键"""
        return bool(self.session_pattern.match(key))

    def is_timestamp_key(self, key: str) -> bool:
        """检查键是否为会话时间戳键"""
        return bool(self.timestamp_pattern.match(key))

    def extract_session_index(self, key: str) -> int:
        """从键名提取会话索引"""
        match = self.session_pattern.match(key) or self.timestamp_pattern.match(key)
        if match:
            return int(match.group(1))
        return 0

    def _process_turn(self, turn_data: Dict, session_key: str, turn_idx: int, default_speaker: str) -> MultiModalTurn:
        """处理单个对话回合（支持多模态）"""
        return MultiModalTurn(
            turn_id=turn_data.get("dia_id", f"D{session_key}:{turn_idx+1}"),
            speaker=turn_data.get("speaker", default_speaker),
            content=turn_data.get("text", ""),
            img_urls=turn_data.get("img_url", []),
            blip_caption=turn_data.get("blip_caption"),
            query=turn_data.get("query")
        )

    def _process_conversation(self, conversation: Dict) -> List[Session]:
        """处理单个对话样本中的会话数据"""
        sessions = []
        session_map = {}
        
        # 提取主要说话人
        speaker_a = conversation.get("speaker_a", "Speaker A")
        speaker_b = conversation.get("speaker_b", "Speaker B")
        speakers = [speaker_a, speaker_b]
        default_speaker = speaker_a
        # 组织会话数据
        for key, value in conversation.items():
            if self.is_session_key(key):
                session_idx = self.extract_session_index(key)
                
                # 获取对应的时间戳
                timestamp_key = f"session_{session_idx}_date_time"
                timestamp = conversation.get(timestamp_key, "Unknown date/time")
                
                # 处理对话回合
                turns = []
                for turn_idx, turn_data in enumerate(value):
                    # 确保内容不为空
                    content = turn_data.get("text")
                    if not content:
                        logger.warning(f"跳过空对话回合: {session_idx}-{turn_data}")
                        continue
                    turn = self._process_turn(turn_data, key, turn_idx, default_speaker)
                    default_speaker = turn.speaker
                    turns.append(turn)
                
                # 创建会话对象
                session = Session(
                    session_id=f"{key}",
                    time=timestamp,
                    participants=speakers,
                    turns=turns
                )
                session_map[session_idx] = session
        
        # 按会话索引排序
        for idx in sorted(session_map.keys()):
            sessions.append(session_map[idx])
            
        return sessions

    def load(self, file_path: str) -> ConversationDataset:
        """加载LoCoMo数据集文件并转换为ConversationDataset"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        logger.info(f"加载LoCoMo数据集: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            # 提取和处理所有会话
            all_sessions = []
            for sample in raw_data:
                conversation = sample.get("conversation", {})
                if conversation:
                    sample_sessions = self._process_conversation(conversation)
                    all_sessions.extend(sample_sessions)
            
            # 创建数据集对象
            logger.info(f"成功加载 {len(all_sessions)} 个会话")
            return ConversationDataset(sessions=all_sessions)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e.msg} (位置: {e.pos})")
            raise
        except Exception as e:
            logger.error(f"加载数据集时出错: {str(e)}")
            raise

    @staticmethod
    def save(dataset: ConversationDataset, output_path: str):
        """将数据集保存到文件（用于调试）"""
        serialized = []
        for session in dataset.sessions:
            serialized.append({
                "session_id": session.id,
                "participants": session.participants,
                "time": session.time,
                "turns": [{
                    "turn_id": turn.id,
                    "speaker": turn.speaker,
                    "content": turn.content
                } for turn in session.turns]
            })
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

def main():
    loader = LoCoMoLoader()
    dataset = loader.load(args.input_data)
    # 保存处理后的数据集
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"{os.path.splitext(os.path.basename(args.input_data))[0]}.json"
    output_path = os.path.join(args.output_dir, filename)
    loader.save(dataset, output_path)

if __name__ == "__main__":
    import time
    import argparse
    from utils.logger import setup_logging
    from utils.params import get_base_parser, data_loader_args
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # os.environ['LOG_FILE'] = f"data_pre_{timestamp}.log"
    logger = setup_logging()
    # 参数解析
    base_parser = get_base_parser()
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser = data_loader_args(parser)
    args = parser.parse_args()
    main()