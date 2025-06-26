from abc import ABC, abstractmethod
class MemoryManager(ABC):
    @abstractmethod
    def mem_add(self, conversation: dict):
        """添加对话记忆"""
        
    # @abstractmethod
    # def mem_search(self, query: str, conversation_id: str) -> list:
    #     """搜索相关记忆"""
        
    # @abstractmethod
    # def generate_response(self, question: str, memories: list) -> str:
    #     """生成回答"""