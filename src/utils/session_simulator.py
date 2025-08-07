# utils/session_simulator.py
import os
import json
import ast
import re
import hashlib
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from utils.prompt_templates import SESSION_SIMULATOR_PROMPT
from client.llm_client import client
from utils.cache_manager import DialogCacheManager

logger = logging.getLogger(__name__)

class SessionSimulator:
    def __init__(self,
                 model: str,
                 max_turns: int = 6,
                 is_step: bool = True,
                 cache_dir: str = "./dialog_cache",
                 domain: str = "financial"):
        """
        会话模拟器初始化

        :param model: LLM 模型名称
        :param max_turns: 最大对话轮次
        :param is_step: 是否启用暂停机制，True 为启用，每次轮次结束后暂停
        :param cache_dir: 对话缓存目录
        :param domain: 领域类型，如 "financial" 或 "medical"，用于确定证据格式
        """
        self.model = model
        self.max_turns = max_turns
        self.is_step = is_step
        self.cache_manager = DialogCacheManager(cache_dir)
        self.current_state: Dict = {}
        self.current_dialog: List[Dict] = []
        self.domain = domain

    def generate_dialog(self,
                        evidences: List[Tuple],
                        persona: str) -> List[Dict]:
        """
        生成伪对话

        :param evidences: 证据列表，用户 LLM 需要在会话中透露出来的信息
        :param persona: 用户人格描述
        :return: 对话回合列表
        """
        logger.debug(f"Persona: {persona}")
        # Load cache based on evidences and persona
        self.cache_manager.load_cache(evidences, persona)
        
        # Get initial state and dialog from cache
        self.current_state = self.cache_manager.get_session_state()
        self.current_dialog = self.cache_manager.get_dialog_history()

        # FIX: Ensure evidences and remaining_evidences are tuples after loading from cache
        # JSON serialization converts tuples to lists. Convert them back to tuples for proper hashing.
        if "evidences" in self.current_state and isinstance(self.current_state["evidences"], list):
            self.current_state["evidences"] = [
                tuple(item) if isinstance(item, list) else item
                for item in self.current_state["evidences"]
            ]
        if "remaining_evidences" in self.current_state and isinstance(self.current_state["remaining_evidences"], list):
            self.current_state["remaining_evidences"] = [
                tuple(item) if isinstance(item, list) else item
                for item in self.current_state["remaining_evidences"]
            ]

        # If cache was empty, initialize state and starting dialog
        if not self.current_state.get("session_hash"): # Check if it's a freshly initialized empty state
            session_hash = self.cache_manager._generate_cache_key(evidences, persona)
            logger.info(f"创建新会话: {session_hash}")
            self.current_state = {
                "session_hash": session_hash,
                "evidences": evidences,
                "persona": persona,
                "turn_count": 0,
                "remaining_evidences": list(evidences),
                "domain": self.domain
            }
            self.current_dialog = [
                {
                    "id": 1,
                    "speaker": "Assistant",
                    "content": "Hi! How can I assist you today?",
                    "mentioned_evidence": [],
                }
            ]
            self.cache_manager.update_cache_data(self.current_state, self.current_dialog) # Save initial state
            logger.info(f"新会话已保存到缓存: {self.cache_manager.current_cache_path}")
        else:
            logger.info(f"从缓存恢复会话: {self.current_state['session_hash']}")
            logger.info(f"当前轮次: {self.current_state['turn_count']}/{self.max_turns}")
            # 如果缓存中没有domain字段，添加默认值
            if "domain" not in self.current_state:
                self.current_state["domain"] = self.domain

        # 从加载的状态中获取当前轮次
        current_turn = self.current_state["turn_count"]
        # 进行对话轮次
        while current_turn < self.max_turns:
            # If enabled and not the very first turn (0), pause
            if self.is_step and current_turn > 0:
                logger.info(f"\n--- 对话暂停，当前轮次: {current_turn}/{self.max_turns} ---")
                logger.info(f"您可以检查缓存文件 {self.cache_manager.current_cache_path} 中的对话历史，然后按回车键继续...")
                input("（按回车键继续）")
                # Reload cache to reflect potential manual changes
                self.cache_manager.load_cache(evidences, persona)
                # FIX: Re-apply conversion after reloading cache
                self.current_state = self.cache_manager.get_session_state()
                self.current_dialog = self.cache_manager.get_dialog_history()
                if "evidences" in self.current_state and isinstance(self.current_state["evidences"], list):
                    self.current_state["evidences"] = [
                        tuple(item) if isinstance(item, list) else item
                        for item in self.current_state["evidences"]
                    ]
                if "remaining_evidences" in self.current_state and isinstance(self.current_state["remaining_evidences"], list):
                    self.current_state["remaining_evidences"] = [
                        tuple(item) if isinstance(item, list) else item
                        for item in self.current_state["remaining_evidences"]
                    ]
                logger.info("继续对话...")
            
            # Check if all evidences have been discussed before generating next turn
            if not self.current_state["remaining_evidences"]:
                logger.info("所有信息都已被提及，对话结束。")
                break
            
            # --- Prepare context for User LLM ---
            last_turn_for_user_prompt = ""
            if self.current_dialog:
                last_turn_for_user_prompt = f"{self.current_dialog[-1]['speaker']}: {self.current_dialog[-1]['content']}"

            # 根据领域选择不同的提示模板
            user_prompt_template = SESSION_SIMULATOR_PROMPT[self.domain]["user"]
            
            # 格式化证据列表，根据领域不同有不同的格式
            evidences_str = self._format_evidences_for_prompt(self.current_state["remaining_evidences"], self.current_state.get("domain", "financial"))
            
            user_prompt = user_prompt_template.format(
                evidences=evidences_str,
                persona=self.current_state["persona"]["user"],
                last_turn_content=last_turn_for_user_prompt
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
                "content": user_response_content,
                "mentioned_evidence": mentioned_by_user,
            })

            # Update remaining evidences based on what user mentioned (which are now proper tuple objects)
            self.update_remaining_evidences(mentioned_by_user, 'user')

            # --- Prepare context for Assistant LLM ---
            # 根据领域选择不同的提示模板
            assistant_prompt_template = SESSION_SIMULATOR_PROMPT[self.domain]["assistant"]
            
            # 格式化证据列表，根据领域不同有不同的格式
            evidences_str = self._format_evidences_for_prompt(self.current_state["remaining_evidences"], self.current_state.get("domain", "financial"))
            
            assistant_prompt = assistant_prompt_template.format(
                persona=self.current_state["persona"]["assistant"],
                evidences=evidences_str,
                last_turn_content=user_response_content,
            )
            logger.debug(f"assistant_prompt: {assistant_prompt}")
            logger.info(f"\n--- Assistant LLM (Turn {current_turn + 1}) ---")
            assistant_response_raw = self._llm_generate([{"role": "user", "content": assistant_prompt}])
            assistant_response_content, mentioned_by_assistant = self._extract_and_clean_llm_response(assistant_response_raw)

            self.current_dialog.append({
                "id": len(self.current_dialog) + 1,
                "speaker": "Assistant",
                "content": assistant_response_content,
                "mentioned_evidence": mentioned_by_assistant,
            })

            # Update remaining evidences based on what assistant mentioned
            self.update_remaining_evidences(mentioned_by_assistant,'assistant')
            
            self.current_state["turn_count"] += 1
            current_turn = self.current_state["turn_count"]

            self.cache_manager.update_cache_data(self.current_state, self.current_dialog) # Save updated state and dialog

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
            logger.debug(f"API response: {response_content}")
            return response_content
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return "对不起，我暂时无法回应。"

    def _format_evidences_for_prompt(self, evidences: List[Tuple], domain: str) -> str:
        """
        根据不同领域格式化证据列表为提示词中的字符串
        
        :param evidences: 证据列表
        :param domain: 领域类型，如 "financial" 或 "medical"
        :return: 格式化后的证据字符串
        """
        if domain == "medical":
            # 医疗领域的证据格式化：(patient_id, timestamp, table_type, variable_name, value)
            return "Evidence format: (patient_id, timestamp, table_type, variable_name, value)\n" + \
                   "\n".join(f"- {e}" for e in evidences)
        else:
            # 金融领域的证据格式化：(code, sname, date, value, metric)#兼容原有逻辑
            return "\n".join(f"- {e}" for e in evidences)

    def _extract_and_clean_llm_response(self, raw: str) -> Tuple[str, List[Tuple]]:
        """
        清理对话内容 + 把 LLM 标记的证据解析成 Evidence 元组
        """
        pattern = r"EVIDENCES_USED_IN_THIS_TURN:\s*\r?\n(.*?)(?=\r?\n---|$)"
        match = re.search(pattern, raw, re.DOTALL)
    
        content = raw
        evidences: List[Tuple] = []
        if match:
            # Content is everything before the EVIDENCES_USED_IN_THIS_TURN block
            content = raw[:match.start()].strip()
            block = match.group(1).strip()
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    payload = line[2:].strip()
                    try:
                        # Directly evaluate the payload as a tuple
                        parsed_item = ast.literal_eval(payload)
                        # 确保它是一个元组
                        if isinstance(parsed_item, tuple):
                            # 对于医疗领域，确保是5元素元组
                            if self.current_state.get("domain") == "medical" and len(parsed_item) == 5:
                                evidences.append(parsed_item)
                            # 对于金融领域，确保是5元素元组
                            elif self.current_state.get("domain") == "financial" and len(parsed_item) == 5:
                                evidences.append(parsed_item)
                            # 对于其他领域或结构，只要是元组就接受
                            elif self.current_state.get("domain", "") not in ["medical", "financial"]:
                                evidences.append(parsed_item)
                            else:
                                logger.warning(f"Parsed item from LLM does not match expected structure for domain {self.current_state.get('domain')}, skipping: {parsed_item}")
                        else:
                            logger.warning(f"Parsed item from LLM is not a tuple, skipping: {parsed_item}")
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"无法解析证据字符串 '{payload}': {e}")
        return content, evidences

    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        """
        将列表结构存储的对话历史格式化为 LLM prompt 所需的字符串。
        """
        formatted_history = []
        for entry in chat_history:
            formatted_history.append(f"{entry['speaker']}: {entry['content']}")
        return "\n".join(formatted_history)

    def _filter_remaining_evidences(self, remaining_evidences: List[Tuple], mentioned_evidences: List[Tuple], role: str) -> List[Tuple]:
        mentioned_set = set(mentioned_evidences)
        filtered = [ev for ev in remaining_evidences if ev not in mentioned_set]
        for ev in remaining_evidences:
            if ev in mentioned_set:
                logger.info(f"{role} marked: {ev}")
            else:
                logger.debug(f"{role} missed: {ev}")
        return filtered

    def update_remaining_evidences(self, mentioned: List[str], role:str):
        self.current_state["remaining_evidences"] = self._filter_remaining_evidences(
            self.current_state.get("remaining_evidences", []),
            mentioned,
            role=role
        )