# utils/session_simulator.py
import os
import json
import re
import hashlib
import logging
from typing import Dict, List, Tuple
from pathlib import Path
from utils.prompt_templates import SESSION_SIMULATOR_PROMPT
from client.llm_client import client

# 配置logger，日志保存到dialog_simulator.log文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='dialog_simulator.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

class SessionSimulator:
    def __init__(self,
                 model: str,
                 max_turns: int = 6,
                 is_step: bool = True,
                 cache_dir: str = "./dialog_cache"):
        """
        会话模拟器初始化

        :param model: LLM 模型名称
        :param max_turns: 最大对话轮次
        :param is_step: 是否启用暂停机制，True 为启用，每次轮次结束后暂停
        :param cache_dir: 对话缓存目录
        """
        self.model = model
        self.max_turns = max_turns
        self.cache_dir = Path(cache_dir)
        self.is_step = is_step
        os.makedirs(self.cache_dir, exist_ok=True)

        self.current_state: Dict = {}

        self.current_dialog: List[Dict] = []

    def generate_dialog(self,
                        evidences: List[str],
                        persona: str) -> List[Dict]:
        """
        生成伪对话

        :param evidences: 证据字符串列表，用户 LLM 需要在会话中透露出来的信息
        :param persona: 用户人格描述
        :return: 对话回合列表
        """
        # 创建一个哈希值作为该会话的唯一标识，用于缓存
        # 包含 evidences 和 persona，确保相同参数的会话能加载同一缓存
        sorted_evidences = sorted(evidences)
        session_hash = hashlib.md5(
            f"{json.dumps(sorted_evidences, ensure_ascii=False)}_{persona}".encode("utf-8")
        ).hexdigest()

        # 尝试从缓存加载会话状态和对话历史
        if self._load_session_state(session_hash):
            logger.info(f"从缓存恢复会话: {session_hash}")
            logger.info(f"当前轮次: {self.current_state['turn_count']}/{self.max_turns}")
        else:
            logger.info(f"创建新会话: {session_hash}")
            # 初始化对话状态
            self.current_state = {
                "session_hash": session_hash,
                "evidences": evidences,
                "persona": persona,
                "turn_count": 0,
                "paused": False,
                "remaining_evidences": list(evidences)
            }
            self.current_dialog = []
            # 初始化对话
            self.current_dialog.append({
                "id": 1,
                "speaker": "Assistant",
                "content": "Hi! How can I assist you today?",
            })
            self._save_session_state() # 保存初始状态和对话历史

        # 从加载的状态中获取当前轮次
        current_turn = self.current_state["turn_count"]
        # 进行对话轮次
        while current_turn < self.max_turns:
            # 如果启用了暂停机制，且当前不是会话开始的第一步
            if self.is_step and current_turn > 0:
                logger.info(f"\n--- 对话暂停，当前轮次: {current_turn}/{self.max_turns} ---")
                logger.info(f"您可以修改缓存文件{self._get_session_cache_file(session_hash)}中的对话历史，然后按回车键继续...")
                input("（按回车键继续）")
                if not self._load_session_state(session_hash):
                    logger.error("错误：无法加载暂停后的会话状态。")
                    break
                logger.info("继续对话...")
            if self.current_state["remaining_evidences"] == []:
                logger.info("所有信息都已被提及，对话结束。")
                break
            # 将当前对话历史转换为适合 LLM prompt 的字符串格式
            # 使用列表存储 Dict 结构的好处是方便序列化（json）和反序列化
            # 在转换为 prompt 时再进行格式化
            chat_history_str = self._format_chat_history(self.current_dialog)
                
            user_prompt = SESSION_SIMULATOR_PROMPT["user"].format(
                evidences="\n".join(f"- {e}" for e in self.current_state["remaining_evidences"]),
                persona=self.current_state["persona"],
                chat_history=chat_history_str
            )
            if current_turn == self.max_turns - 1 and self.current_state["remaining_evidences"]:
                user_prompt += "\nCRITICAL: Final turn - MUST cover ALL remaining evidence in one response"
            logger.debug(f"user_prompt: {user_prompt}")

            logger.info(f"\n--- User LLM (Turn {current_turn + 1}) ---")
            user_response_raw = self._llm_generate([{"role": "user", "content": user_prompt}])
            user_response_content, mentioned_by_user = self._extract_and_clean_llm_response(user_response_raw)
            
            self.current_dialog.append({
                "id": len(self.current_dialog) + 1,
                "speaker": "User",
                "content": user_response_content, # 存储清理后的内容
            })

            # 更新 remaining_evidences
            self.update_remaining_evidences(mentioned_by_user, 'user')

            # 生成助手响应
            # 助手 LLM 的 prompt 只需要当前对话历史和用户最新输入
            assistant_prompt = SESSION_SIMULATOR_PROMPT["assistant"].format(
                evidences="\n".join(f"- {e}" for e in self.current_state["remaining_evidences"]),
                chat_history=self._format_chat_history(self.current_dialog), # 传入更新后的历史
                user_input=user_response_content
            )
            logger.info(f"\n--- Assistant LLM (Turn {current_turn + 1}) ---")
            assistant_response_raw = self._llm_generate([{"role": "user", "content": assistant_prompt}])
            assistant_response_content, mentioned_by_assistant = self._extract_and_clean_llm_response(assistant_response_raw)

            self.current_dialog.append({
                "id": len(self.current_dialog) + 1,
                "speaker": "Assistant",
                "content": assistant_response_content, # 存储清理后的内容
            })

            # 更新 remaining_evidences
            self.update_remaining_evidences(mentioned_by_assistant,'assistant')
            # 更新状态
            self.current_state["turn_count"] += 1
            current_turn = self.current_state["turn_count"] # 更新当前轮次

            # 保存当前会话状态和对话历史到缓存
            self._save_session_state()

        logger.info(f"\n--- 对话结束，共进行 {self.current_state['turn_count']} 轮次 ---")
        return self.current_dialog

    def _llm_generate(self, messages: List[Dict]) -> str:
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                extra_body={"enable_thinking": True}
            )
            response_content = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_content += chunk.choices[0].delta.content
            logger.info(f"API response: {response_content}")
            return response_content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return "对不起，我暂时无法回应。"  # 返回一个默认错误信息

    def _extract_and_clean_llm_response(self, raw_llm_response: str) -> Tuple[str, List[str]]:
        """
        从原始 LLM 响应中提取标记的证据列表，并返回清理后的响应内容。
        Args:
            raw_llm_response: LLM 返回的原始字符串，可能包含证据标记部分。
        Returns:
            Tuple[str, List[str]]: 清理后的对话内容字符串 和 被标记的证据列表。
        """
        # 放宽正则匹配，允许前后多余空白，换行符兼容 \r\n 和 \n
        pattern = r"EVIDENCES_USED_IN_THIS_TURN:\s*\r?\n(.*?)(?=\r?\n---|$)"
        match = re.search(pattern, raw_llm_response, re.DOTALL)
    
        mentioned_evidences = []
        dialog_content = raw_llm_response  # 默认是完整内容
    
        if match:
            evidences_block = match.group(1).strip()
            # 移除证据标记部分，得到真正的对话内容
            dialog_content = raw_llm_response[:match.start()].strip()
    
            # 从证据块中提取每个证据字符串
            for line in evidences_block.splitlines():
                line = line.strip()
                if line.startswith('- ') and len(line) > 2:
                    mentioned_evidences.append(line[2:])  # 移除 "- " 前缀
    
        return dialog_content, mentioned_evidences

    def _get_session_cache_file(self, session_hash: str) -> Path:
        """获取特定会话的缓存文件路径"""
        return self.cache_dir / f"{session_hash}.json"

    def _load_session_state(self, session_hash: str) -> bool:
        """
        从缓存加载会话状态和对话历史。
        如果成功加载，则更新 self.current_state 和 self.current_dialog。
        """
        cache_file = self._get_session_cache_file(session_hash)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                self.current_state = cached_data.get("state", {})
                self.current_dialog = cached_data.get("dialog", [])
                if "remaining_evidences" not in self.current_state:
                    self.current_state["remaining_evidences"] = list(self.current_state.get("evidences", []))
                return True
            except json.JSONDecodeError as e:
                logger.error(f"缓存文件 {cache_file} 解析失败: {e}")
                return False
        return False

    def _save_session_state(self):
        """
        将会话状态和对话历史保存到缓存。
        """
        if not self.current_state:
            logger.warning("警告：没有当前会话状态可保存。")
            return

        session_hash = self.current_state.get("session_hash")
        if not session_hash:
            logger.error("错误：无法获取会话哈希值，无法保存缓存。")
            return

        cache_file = self._get_session_cache_file(session_hash)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "state": self.current_state,
                    "dialog": self.current_dialog
                }, f, ensure_ascii=False, indent=2)
            # logger.info(f"会话状态和对话历史已保存到: {cache_file}")
        except IOError as e:
            logger.error(f"保存缓存文件 {cache_file} 失败: {e}")

    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        """
        将列表结构存储的对话历史格式化为 LLM prompt 所需的字符串。
        """
        formatted_history = []
        for entry in chat_history:
            formatted_history.append(f"{entry['speaker']}: {entry['content']}")
        return "\n".join(formatted_history)

    def _filter_remaining_evidences(self, remaining_evidences: List[str], mentioned_evidences: List[str], role: str) -> List[str]:
        filtered_evidences = []
        for evidence in remaining_evidences:
            matched = False
            for mentioned in mentioned_evidences:
                # 优化匹配逻辑：忽略大小写，去除空白，部分匹配
                norm_evidence = re.sub(r"\s+", "", evidence).lower()
                norm_mentioned = re.sub(r"\s+", "", mentioned).lower()
                if norm_evidence in norm_mentioned or norm_mentioned in norm_evidence:
                    matched = True
                    break
            if not matched:
                filtered_evidences.append(evidence)
                logger.debug(f"{role} LLM 未标记提及信息: {evidence}")
            else:
                logger.info(f"{role} LLM 已标记提及信息: {evidence}")
        return filtered_evidences

    def update_remaining_evidences(self, mentioned: List[str], role:str):
        self.current_state["remaining_evidences"] = self._filter_remaining_evidences(
            self.current_state.get("remaining_evidences", []),
            mentioned,
            role=role
        )