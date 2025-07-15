import json
import os
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal, Tuple
from pathlib import Path

# Common Difficulty Level type
DifficultyLevel = Literal["easy", "medium", "hard"]

class BaseCacheManager:
    """
    基类缓存管理器，提供通用的缓存加载、保存和路径管理功能
    子类应实现 specific_hash_key 方法来定义其独特的缓存键生成逻辑
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.current_cache_path: Path = None
        self.cache_data: Dict[str, Any] = {}

    @abstractmethod
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        生成其特定的缓存键
        """
        raise NotImplementedError("Subclasses must implement _generate_cache_key method.")

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """根据缓存键获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.json"

    def load_cache(self, *args, **kwargs) -> bool:
        """
        加载特定参数组合的缓存到self.cache_data
        如果文件不存在，则初始化空缓存数据
        返回 True 如果成功加载或初始化，否则 False
        """
        cache_key = self._generate_cache_key(*args, **kwargs)
        self.current_cache_path = self._get_cache_file_path(cache_key)
        
        if self.current_cache_path.exists():
            try:
                with open(self.current_cache_path, 'r', encoding='utf-8') as f:
                    self.cache_data = json.load(f)
                    self.logger.info(f"Loaded cache from {self.current_cache_path}")
                return True
            except json.JSONDecodeError as e:
                self.logger.warning(f"Error loading cache from {self.current_cache_path}: {e}. Initializing empty cache.")
                self.cache_data = self._initialize_empty_cache_data()
                return False
        else:
            self.logger.info(f"No cache found at {self.current_cache_path}. Initializing empty cache.")
            self.cache_data = self._initialize_empty_cache_data()
            return True

    def save_cache(self):
        """保存当前缓存数据到文件"""
        if self.current_cache_path:
            try:
                with open(self.current_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Cache saved to {self.current_cache_path}")
            except Exception as e:
                self.logger.error(f"Failed to save cache to {self.current_cache_path}: {e}")
        else:
            self.logger.warning("No current cache path set. Cannot save cache.")

    @abstractmethod
    def _initialize_empty_cache_data(self) -> Dict:
        """
        定义其空缓存的初始结构
        """
        raise NotImplementedError("Subclasses must implement _initialize_empty_cache_data method.")

class QACacheManager(BaseCacheManager):
    """
    QA 生成的缓存管理器
    缓存包含偏好 QA (preferred_qas) 和已生成 QA (generated_qas)
    缓存键基于排序后的 session_ids 和 difficulty
    """
    def __init__(self, cache_dir: str = "./qa_generation_cache"):
        super().__init__(cache_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _generate_cache_key(self, session_ids: List[str], difficulty: DifficultyLevel) -> str:
        """
        根据会话ID和难度生成唯一的缓存键
        会话ID列表应保持排序，以确保一致的哈希值
        """
        sorted_session_ids = sorted(session_ids)
        unique_string = f"{'_'.join(sorted_session_ids)}_{difficulty}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def _initialize_empty_cache_data(self) -> Dict:
        return {
            "preferred_qas": [],
            "generated_qas": []
        }

    def add_generated_qa(self, qa_pair: Dict):
        """添加一个已生成的QA对到缓存中"""
        question_text = qa_pair.get("question", "")
        if not any(q.get("question") == question_text for q in self.cache_data["generated_qas"]):
            self.cache_data["generated_qas"].append(qa_pair)
            self.logger.debug(f"Added new QA to cache: {question_text[:30]}...")
        else:
            self.logger.debug(f"Skipped adding duplicate QA to cache: {question_text[:30]}...")

    def get_preferred_qas(self) -> List[Dict]:
        """获取偏好QA列表"""
        return self.cache_data.get("preferred_qas", [])

    def get_generated_qas_count(self) -> int:
        """获取已生成QA的数量"""
        return len(self.cache_data.get("generated_qas", []))

    def get_generated_questions(self) -> List[str]:
        """获取所有已生成的问题文本列表，用于去重"""
        return [qa.get("question", "") for qa in self.cache_data.get("generated_qas", []) if qa.get("question")]

    def add_preferred_qa(self, qa_pair: Dict):
        """手动添加一个偏好QA到缓存中"""
        question_text = qa_pair.get("question", "")
        if not any(q.get("question") == question_text for q in self.cache_data["preferred_qas"]):
            self.cache_data["preferred_qas"].append(qa_pair)
            self.logger.info(f"Manually added preferred QA: {question_text[:30]}...")
            self.save_cache()

class DialogCacheManager(BaseCacheManager):
    """
    对话模拟器的缓存管理器
    缓存包含当前对话状态 (state) 和对话历史 (dialog)
    缓存键基于排序后的 evidences 和 persona
    """
    def __init__(self, cache_dir: str = "./dialog_cache"):
        super().__init__(cache_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _generate_cache_key(self, evidences: List[str], persona: str) -> str:
        """
        根据证据列表和角色生成唯一的缓存键
        证据列表应保持排序，以确保一致的哈希值
        """
        sorted_evidences = sorted(evidences)
        unique_string = f"{json.dumps(sorted_evidences, ensure_ascii=False)}_{persona}"
        return hashlib.md5(unique_string.encode("utf-8")).hexdigest()

    def _initialize_empty_cache_data(self) -> Dict:
        """初始化空对话缓存的结构"""
        return {
            "state": {
                "session_hash": "",
                "evidences": [],
                "persona": "",
                "turn_count": 0,
                "remaining_evidences": []
            },
            "dialog": []
        }

    def get_session_state(self) -> Dict:
        """获取当前会话状态"""
        return self.cache_data.get("state", self._initialize_empty_cache_data()["state"])

    def get_dialog_history(self) -> List[Dict]:
        """获取当前对话历史"""
        return self.cache_data.get("dialog", [])

    def update_cache_data(self, state: Dict, dialog: List[Dict]):
        """更新缓存中的会话状态和对话历史"""
        self.cache_data["state"] = state
        self.cache_data["dialog"] = dialog
        self.save_cache()