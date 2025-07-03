# src/utils/struct.py

from typing import List, Dict
# 单个对话回合表示
class DialogueTurn:
    def __init__(self, turn_id, speaker, content):
        self.id = turn_id
        self.speaker = speaker
        self.content = content

# 扩展的对话回合表示
class MultiModalTurn:
    def __init__(self, turn_id, speaker, content, 
                 img_urls=None, blip_caption=None, query=None):
        self.id = turn_id
        self.speaker = speaker
        self.text_content = content  # 原始文本内容
        self.img_urls = img_urls or []  # 图片URL列表
        self.blip_caption = blip_caption  # 图片描述
        self.query = query  # 搜索查询词
        # 生成综合内容（文本+图片描述(no url)）
        self.content = self._generate_combined_content()
    
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
        self.tables = tables or []

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