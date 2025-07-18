import json
import os
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal, Tuple
from pathlib import Path
from utils.struct import Evidence
# Common Difficulty Level type
DifficultyLevel = Literal["easy", "medium", "hard"]


class BaseCacheManager(ABC):
    """
    基类缓存管理器，提供通用的缓存加载、保存和路径管理功能
    子类应实现 _generate_cache_key 方法来定义其独特的缓存键生成逻辑
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
    缓存包含统一的 QA 列表，每个 QA 包含状态标签 (status: "liked", "generated", "disliked")
    使用单一全局缓存文件来存储所有生成的QA，实现断点恢复和偏好管理。
    """
    def __init__(self, cache_dir: str = "./qa_generation_cache"):
        super().__init__(cache_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_cache()
        
    def _generate_cache_key(self, *args, **kwargs) -> str:
        return "qa_cache"

    def _initialize_empty_cache_data(self) -> Dict:
        return {
            "questions": []
        }

    def generate_qa_id(self, qa_pair: Dict) -> str:
        """Generates a unique ID for a QA pair based on its content."""
        qa_content_string = f"{qa_pair.get('question_text', '')}"
        return hashlib.md5(qa_content_string.encode('utf-8')).hexdigest()

    def add_qa(self, qa_pair: Dict, status: str = "generated", sql_info: dict = {}) -> bool:
        """
        添加或更新一个QA对到缓存中。
        如果QA对已存在（通过qa_id判断），则更新其信息（特别是status/sql_status）。
        否则，添加新的QA对。
        status: liked / disliked / generated
        sql_status: match / skipped / *_not_match / not yet / failed
        优先保留 'liked' 或 'disliked' 状态，不被 'generated' 覆盖。
        """
        qa_id = qa_pair.get("qa_id")
        if not qa_id:
            qa_id = self.generate_qa_id(qa_pair)
            qa_pair["qa_id"] = qa_id

        existing_qa_index = -1
        for i, q in enumerate(self.cache_data["questions"]):
            if q.get("qa_id") == qa_id:
                existing_qa_index = i
                break

        qa_data_to_store = {
            "qa_id": qa_id,
            "question_text": qa_pair.get("question_text"),
            "answer_text": qa_pair.get("answer_text"),
            "evidence": qa_pair.get("evidence"),
            "conversation_id": qa_pair.get("conversation_id"),
            "session_ids": qa_pair.get("session_ids"),
            "timestamp": qa_pair.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S")),
            "difficulty": qa_pair.get("difficulty"),
            "status": status,
            "sql_info": sql_info
        }

        if existing_qa_index != -1:
            # Check existing status priority
            current_status = self.cache_data["questions"][existing_qa_index].get("status")
            if (current_status == "liked" and status != "liked") or \
               (current_status == "disliked" and status not in ["liked", "disliked"]):
                self.logger.debug(f"Skipped updating QA {qa_id}: new status '{status}' is lower priority than existing '{current_status}'")
                return False # Don't overwrite higher priority status
            
            # Update existing QA
            self.cache_data["questions"][existing_qa_index].update(qa_data_to_store)
            self.logger.debug(f"Updated existing QA in cache: {qa_id} (status: {status})")
            return False # Not a new addition
        else:
            self.cache_data["questions"].append(qa_data_to_store)
            self.logger.debug(f"Added new QA to cache: {qa_id} (status: {status})")
            return True # A new addition

    def get_preferred_qas(self, difficulty: DifficultyLevel = None) -> List[Dict]:
        """获取所有被标记为“liked”的QA列表，可按难度过滤。"""
        preferred_qas = [qa for qa in self.cache_data.get("questions", []) if qa.get("status") == "liked"]
        if difficulty:
            return [qa for qa in preferred_qas if qa.get("difficulty") == difficulty]
        return preferred_qas
    
    def get_disliked_qas(self, difficulty: DifficultyLevel = None) -> List[Dict]:
        """获取所有被标记为“disliked”的QA列表，可按难度过滤。"""
        disliked_qas = [qa for qa in self.cache_data.get("questions", []) if qa.get("status") == "disliked"]
        if difficulty:
            return [qa for qa in disliked_qas if qa.get("difficulty") == difficulty]
        return disliked_qas

    def get_all_questions_text(self, difficulty: DifficultyLevel = None) -> List[str]:
        """
        获取缓存中所有QA的question_text列表，可按难度过滤。
        用于去重和确保新问题具有辨识度。
        """
        all_qas = self.cache_data.get("questions", [])
        if difficulty:
            q_texts = [qa.get("question_text", "") for qa in all_qas if qa.get("difficulty") == difficulty]
        else:
            q_texts = [qa.get("question_text", "") for qa in all_qas]
        return [q for q in q_texts if q]

    def get_all_qas(self, difficulty: DifficultyLevel = None) -> List[Dict]:
        """
        获取缓存中所有QA列表，可按难度过滤。
        返回完整的QA字典对象。
        """
        all_qas = self.cache_data.get("questions", [])
        if difficulty:
            return [qa for qa in all_qas if qa.get("difficulty") == difficulty]
        return all_qas
        
    def get_exportable_qas(self) -> List[Dict]:
        """
        获取所有应导出到最终数据集的QA列表 (status为'liked'或'generated')。
        """
        exportable_qas = [qa for qa in self.cache_data.get("questions", []) if qa.get("status") in ["liked", "generated"]]
        difficulty_rank = {"hard": 2, "medium": 1, "easy": 0}

        return sorted(exportable_qas, key=lambda x: difficulty_rank.get(x.get("difficulty"), 3))

    def get_qa_by_id(self, qa_id: str) -> Dict | None:
        """根据QA ID获取特定的QA对"""
        for qa in self.cache_data.get("questions", []):
            if qa.get("qa_id") == qa_id:
                return qa
        return None
    
    def save_cache(self):
        """
        保存当前缓存数据到文件，并在保存前对所有问题按 difficulty 排序：
        """
        if not self.current_cache_path:
            self.logger.warning("No current cache path set. Cannot save cache.")
            return

        difficulty_rank = {"hard": 2, "medium": 1, "easy": 0}
        try:
            self.cache_data["questions"].sort(
                key=lambda q: (
                    difficulty_rank.get(q.get("difficulty", "easy"), 3),
                    q.get("timestamp", "")
                )
            )
            with open(self.current_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Cache (sorted by difficulty) saved to {self.current_cache_path}")
        except Exception as e:
            self.logger.error(f"Failed to save cache to {self.current_cache_path}: {e}")

class DialogCacheManager(BaseCacheManager):
    """
    对话模拟器的缓存管理器
    缓存包含当前对话状态 (state) 和对话历史 (dialog)
    缓存键基于排序后的 evidences 和 persona
    """
    def __init__(self, cache_dir: str = "./dialog_cache"):
        super().__init__(cache_dir)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _generate_cache_key(self, evidences: List[Evidence], persona: str) -> str:
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