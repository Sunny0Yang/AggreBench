from abc import ABC, abstractmethod
class MemoryManager(ABC):
    def __init__(self):
        self.token_count = 0
        self.technique_name = ""

    @abstractmethod
    def mem_add(self, conversation: dict):
        """添加对话记忆"""
        
    @abstractmethod
    def mem_search(self, qa_dict: dict) -> list:
        """搜索相关记忆"""
        
    @abstractmethod
    def generate_response(self, question: str, memories: list):
        """生成回答"""

    def get_token_count(self) -> int:
        """获取文本的token数量"""
        return self.token_count