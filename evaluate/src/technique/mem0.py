import os
import time
import threading
import logging
from tqdm import tqdm
from mem0 import MemoryClient
from .base import MemoryManager
from ..client.llm_client import client
from ..utils.prompts_templates import RESPONSE_GENERATION_PROMPTS
logger = logging.getLogger(__name__)

class Mem0Manager(MemoryManager):  # 继承基类MemoryManager
    def __init__(self, batch_size=2):
        super().__init__()
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY")
        )
        self.batch_size = batch_size
        self.technique_name = "Mem0"
        self.token_count = 0

    def mem_add(self, session_data, qa_index):
        speaker_a, speaker_b = session_data['participants']
        speaker_a_user_id = f"{speaker_a}_{qa_index}"
        speaker_b_user_id = f"{speaker_b}_{qa_index}"
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)
        messages = []
        messages_reverse = []
        for turn in session_data["turns"]:
            if turn["speaker"] == speaker_a:
                messages.append({"role": "user", "content": f"{speaker_a}: {turn['content']}"})
                messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {turn['content']}"})
            elif turn["speaker"] == speaker_b:
                messages.append({"role": "assistant", "content": f"{speaker_b}: {turn['content']}"})
                messages_reverse.append({"role": "user", "content": f"{speaker_b}: {turn['content']}"})
            else:
                raise ValueError(f"Unknown speaker: {turn['speaker']}")
        for i in tqdm(range(0, len(messages), self.batch_size), desc="Memory adding"):
            batch_messages = messages[i : i + self.batch_size]
            batch_messages_reverse = messages_reverse[i : i + self.batch_size]
            logger.info(f"Adding batch {i}")
            # TODO 加metadata 记录dia_id
            self.mem0_client.add(
                    batch_messages, user_id=speaker_a_user_id
                )
            self.mem0_client.add(
                    batch_messages_reverse, user_id=speaker_b_user_id
                )

    def mem_search(self, qa_dict):
        query = qa_dict["question"]
        speaker_a, speaker_b = qa_dict["participants"]
        speaker_a_user_id = f"{speaker_a}_{qa_dict['qa_index']}"
        speaker_b_user_id = f"{speaker_b}_{qa_dict['qa_index']}"
        logger.info(f"搜索用户 {speaker_a_user_id} 的记忆: {query[:30]}...")
        
        # TODO 这里只做了一个用户的记忆搜索，担心搜索两个用户的对话记录会导致结果翻倍
        results = self.mem0_client.search(query, user_id=speaker_a_user_id)
        if not results:
            logger.error(f"无法搜索 {speaker_a_user_id} 的记忆")
            return []

        logger.info(f"搜索完成: {len(results)} 条结果")
        logger.info(f"结果示例: {results[:3]}")
        # TODO dia_id 逻辑添加
        semantic_mems = [
            {"memory": m["memory"], "score": round(m["score"], 2)}
            for m in results
        ]
        
        return semantic_mems

    def generate_response(self, question, memories):
        try:
            memory_context = self._build_memory_context(memories)
            prompt = RESPONSE_GENERATION_PROMPTS['template_en'].format(
                memory_context=memory_context,
                question=question
            )
            logger.info(f"生成回答")
            response = client.chat.completions.create(
                model=os.getenv("MODEL"),
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                extra_body={"enable_thinking": False} 
            )
            
            self.token_count = response.usage.total_tokens
            
            temp = response.choices[0].message.content.strip()
            import re
            match = re.search(r"^The answer is:\s*([0-9]+(?:\.[0-9]+)?)", temp)
            if match:
                return float(match.group(1))
            return temp
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return "生成回答时出错"

    def _build_memory_context(self, memories):
        memory_context = ""
        for i, mem in enumerate(memories):
            memory_context += f"{i+1}. {mem['memory']}\n"
        return memory_context
    
    def get_token_count(self):
        return self.token_count