# src/utils/struct.py

from typing import List, Dict, Tuple
# 单个对话回合表示
Evidence = Tuple[str, str, str, float, str]

class DialogueTurn:
    def __init__(self, turn_id, speaker, content):
        self.id = turn_id
        self.speaker = speaker
        self.content = content

# 扩展的对话回合表示
class MultiModalTurn:
    def __init__(self, turn_id, speaker, content, 
                 img_urls=None, blip_caption=None, query=None, evidence=None):
        self.id = turn_id
        self.speaker = speaker
        self.text_content = content  # 原始文本内容
        self.img_urls = img_urls or []  # 图片URL列表
        self.blip_caption = blip_caption  # 图片描述
        self.query = query  # 搜索查询词
        # 生成综合内容（文本+图片描述(no url)）
        self.content = self._generate_combined_content()
        self.mentioned_evidence = evidence
    
    def _generate_combined_content(self) -> str:
        """生成组合文本内容，融合多模态信息"""
        content = self.text_content
        # 添加图片描述
        if self.blip_caption:
            content += f" [blip_caption: {self.blip_caption}]"
        # 添加搜索查询
        if self.query:
            content += f" [query: {self.query}]"
        return content
    
    def __repr__(self):
        return f"<MultiModalTurn {self.id}: {self.speaker} - {self.text_content[:30]}...>"

class Table:
    """表格数据结构"""
    def __init__(self, headers: List[str], rows: List[Dict[str, str]]):
        self.headers = headers  # 表头列表
        self.rows = rows        # 行数据列表（每行是一个字典）
    
    def __str__(self):
        return f"Table(headers={self.headers}, rows={len(self.rows)})"
        
# 会话表示
class Session:
    def __init__(self, session_id, time, participants, turns, type="conversation", tables: List[Table] = None):
        self.id = session_id
        self.time = time
        self.participants = participants
        self.turns = turns  # DialogueTurn列表
        self.tables = tables

# 新增：对话表示（包含多个会话）
class Conversation:
    def __init__(self, conversation_id, speakers, sessions):
        self.id = conversation_id
        self.speakers = speakers
        self.sessions = sessions  # Session对象列表

# 数据集表示
class ConversationDataset:
    def __init__(self, conversations):
        self.conversations = conversations  # Conversation对象列表

def load_data(input_path: str) -> ConversationDataset:
    """加载并转换数据为ConversationDataset对象"""
    import json
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        conversations = []
        for conv_data in raw_data:
            sessions = []
            for session_data in conv_data.get("sessions", []):
                turns = []
                for turn_data in session_data.get("turns", []):
                    turn = MultiModalTurn(
                        turn_id=turn_data.get("turn_id", f"turn_{len(turns)+1}"),
                        speaker=turn_data.get("speaker", "Unknown"),
                        content=turn_data.get("content", "")
                    )
                    turns.append(turn)
                
                tables = []
                for table_data in session_data.get("tables", []):
                    headers = table_data.get("headers", [])
                    rows = table_data.get("rows", [])
                    table = Table(headers=headers, rows=rows)
                    tables.append(table)

                session = Session(
                    session_id=session_data.get("session_id", f"session_{len(sessions)+1}"),
                    time=session_data.get("time", "Unknown"),
                    participants=session_data.get("participants", ["Participant A", "Participant B"]),
                    turns=turns,
                    type=session_data.get("type", "conversation"),
                    tables=tables
                )
                sessions.append(session)
            
            conversation = Conversation(
                conversation_id=conv_data.get("conversation_id", f"conv_{len(conversations)+1}"),
                speakers=conv_data.get("speakers", ["Speaker A", "Speaker B"]),
                sessions=sessions
            )
            conversations.append(conversation)
        
        return ConversationDataset(conversations=conversations)
    except Exception as e:
        raise RuntimeError(f"数据加载错误: {str(e)}")
        
def save_results(results: List[Dict], output_path: str):
    """保存生成的QA对结果"""
    import json
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"成功保存 {len(results)} 条QA对至: {output_path}")
    except Exception as e:
        raise Exception(f"保存结果到 {output_path} 时出错: {e}") from e
